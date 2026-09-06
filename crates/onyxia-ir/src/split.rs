//! Backend-driven splitting of oversized weight tables.
//!
//! GPUs cap the size of a single storage-buffer binding (Adreno 730 for
//! example allows 128 MiB) at a fraction of what a buffer may hold, and an
//! LLM embedding table (`[262144, 640]` fp32 is 671 MB) exceeds it. A backend
//! whose binding limit is smaller than a constant table asks for the table
//! to be split into row chunks *before* it uploads constants. The rewrite
//! uses nothing but existing primitives, so every backend and the
//! interpreter see the same module. Two consumers are handled — the two
//! ways an LLM reads its (tied) embedding table:
//!
//! `Gather{axis: 0}` over a constant `[R, ...]`, `n` chunks of `rpc` rows:
//!
//! ```text
//! idn     = ids < 0 ? ids + R : ids                (ONNX negative wrap)
//! g_i     = gather(chunk_i, clamp(idn - i*rpc, 0, rows_i - 1))
//! out     = select(idn >= (n-1)*rpc, g_{n-1}, ... select(idn >= rpc, g_1, g_0))
//! ```
//!
//! Out-of-range indices (which are an error in the interpreter) come out
//! clamped to the last row instead — this is a legalization for weights,
//! not a validator.
//!
//! `MatMul{trans_b: true}` against a constant `[N, K]` (the lm_head): each
//! row chunk is its own matmul and the results concatenate along `N`.

use crate::graph::{Module, Node, NodeId, NodeKind, Origin, SourceInfo, ValueDef, ValueId};
use crate::prim::{BinaryOp, CmpOp, Prim};
use crate::{DataType, DimExpr, Result, SymbolicShape, TensorType};
use std::collections::HashSet;

/// A constant table too large for one binding, and how to chunk it.
struct Table {
    cid: crate::graph::ConstId,
    dims: Vec<u64>,
    rows: usize,
    row_bytes: usize,
    /// Rows per chunk.
    rpc: usize,
    n_chunks: usize,
    name: String,
}

impl Table {
    /// The chunking for `value` if it is a constant of more than `max_bytes`
    /// (in the pool's storage layout) whose rows are contiguous and fit.
    fn of(module: &Module, value: ValueId, max_bytes: usize) -> Option<Self> {
        let Origin::Const(cid) = module.value(value).origin else {
            return None;
        };
        let total = module.consts.bytes(cid).len();
        if total <= max_bytes {
            return None;
        }
        let dims = module.consts.ty(cid).shape.as_static()?;
        let rows = dims[0] as usize;
        if rows == 0 || total % rows != 0 {
            return None;
        }
        let row_bytes = total / rows;
        if row_bytes > max_bytes {
            return None; // a single row does not fit; nothing row-wise can help
        }
        let rpc = max_bytes / row_bytes;
        let n_chunks = rows.div_ceil(rpc);
        debug_assert!(n_chunks >= 2);
        let name = module
            .value(value)
            .name
            .clone()
            .unwrap_or_else(|| "table".into());
        Some(Self {
            cid,
            dims,
            rows,
            row_bytes,
            rpc,
            n_chunks,
            name,
        })
    }

    /// Row range of chunk `i`.
    fn range(&self, i: usize) -> (usize, usize) {
        let off = i * self.rpc;
        (off, self.rpc.min(self.rows - off))
    }

    /// Add chunk `i` to the constant pool and return its value.
    fn chunk(&self, module: &mut Module, i: usize) -> Result<ValueId> {
        let (off, len) = self.range(i);
        let bytes = module.consts.bytes(self.cid)
            [off * self.row_bytes..(off + len) * self.row_bytes]
            .to_vec();
        let mut cdims = self.dims.clone();
        cdims[0] = len as u64;
        let dtype = module.consts.ty(self.cid).dtype;
        let cty = TensorType::of(dtype, &cdims);
        let ccid = module.consts.add(cty.clone(), bytes)?;
        Ok(module.add_value(ValueDef {
            name: Some(format!("{}.rows{off}", self.name)),
            ty: cty,
            origin: Origin::Const(ccid),
            content: None,
        }))
    }
}

/// Split every `Gather{axis: 0}` table and every `MatMul{trans_b}` weight
/// larger than `max_bytes` (bytes in the constant pool's storage layout).
/// Returns the number of tables split.
pub fn split_large_tables(module: &mut Module, max_bytes: usize) -> Result<usize> {
    let mut dead: HashSet<NodeId> = HashSet::new();
    let mut count = 0;

    for id in module.node_ids() {
        let node = module.node(id);
        let split = match node.kind {
            NodeKind::Prim(Prim::Gather { axis: 0 }) => {
                Table::of(module, node.inputs[0], max_bytes)
                    .map(|t| split_gather(module, id, t))
                    .transpose()?
            }
            NodeKind::Prim(Prim::MatMul { trans_b: true, .. })
                if module.value(node.inputs[1]).ty.shape.rank() == 2 =>
            {
                Table::of(module, node.inputs[1], max_bytes)
                    .map(|t| split_matmul(module, id, t))
                    .transpose()?
            }
            _ => None,
        };
        if split.is_some() {
            dead.insert(id);
            count += 1;
        }
    }

    if count > 0 {
        module.remove_nodes(&dead);
    }
    Ok(count)
}

/// `out = concat_N(a @ chunk_i^T)`.
fn split_matmul(module: &mut Module, id: NodeId, t: Table) -> Result<()> {
    let node = module.node(id);
    let NodeKind::Prim(Prim::MatMul { trans_a, .. }) = node.kind else {
        unreachable!()
    };
    let a = node.inputs[0];
    let out = node.outputs[0];
    let loc = node.loc.clone();
    let out_ty = module.value(out).ty.clone();
    let last = out_ty.shape.rank() - 1;
    let mut parts = Vec::with_capacity(t.n_chunks);
    for i in 0..t.n_chunks {
        let (_, len) = t.range(i);
        let chunk = t.chunk(module, i)?;
        let mut pdims = out_ty.shape.dims().to_vec();
        pdims[last] = DimExpr::constant(len as u64);
        let pty = TensorType::new(out_ty.dtype, SymbolicShape::from(pdims));
        parts.push(emit(
            module,
            Prim::MatMul {
                trans_a,
                trans_b: true,
            },
            &[a, chunk],
            &pty,
            &loc,
        ));
    }
    emit_into(module, Prim::Concat { axis: last }, &parts, out, &loc);
    Ok(())
}

/// Chunked gather with clamped local indices merged by range selects.
fn split_gather(module: &mut Module, id: NodeId, t: Table) -> Result<()> {
    let node = module.node(id);
    let ids = node.inputs[1];
    let out = node.outputs[0];
    let loc = node.loc.clone();
    let (rows, dims, n_chunks) = (t.rows, t.dims.clone(), t.n_chunks);
    {
        let ids_ty = module.value(ids).ty.clone();
        let out_ty = module.value(out).ty.clone();
        let bool_ids_ty = TensorType::new(DataType::Bool, ids_ty.shape.clone());
        // Condition broadcast against the gathered rows: ids dims ++ [1; ...].
        let mut cond_dims: Vec<DimExpr> = ids_ty.shape.dims().to_vec();
        cond_dims.extend((1..dims.len()).map(|_| DimExpr::constant(1)));
        let cond_shape = SymbolicShape::from(cond_dims);
        let bool_cond_ty = TensorType::new(DataType::Bool, cond_shape.clone());

        let zero = const_i64(module, 0)?;
        let r = const_i64(module, rows as i64)?;
        let neg = emit(
            module,
            Prim::Compare(CmpOp::Lt),
            &[ids, zero],
            &bool_ids_ty,
            &loc,
        );
        let wrapped = emit(
            module,
            Prim::Binary(BinaryOp::Add),
            &[ids, r],
            &ids_ty,
            &loc,
        );
        let idn = emit(module, Prim::Select, &[neg, wrapped, ids], &ids_ty, &loc);

        let mut acc: Option<ValueId> = None;
        for i in 0..n_chunks {
            let (off, len) = t.range(i);
            let cval = t.chunk(module, i)?;
            let off_c = const_i64(module, off as i64)?;
            let last_c = const_i64(module, len as i64 - 1)?;
            let local = emit(
                module,
                Prim::Binary(BinaryOp::Sub),
                &[idn, off_c],
                &ids_ty,
                &loc,
            );
            let lo = emit(
                module,
                Prim::Binary(BinaryOp::Max),
                &[local, zero],
                &ids_ty,
                &loc,
            );
            let cl = emit(
                module,
                Prim::Binary(BinaryOp::Min),
                &[lo, last_c],
                &ids_ty,
                &loc,
            );
            let g = emit(module, Prim::Gather { axis: 0 }, &[cval, cl], &out_ty, &loc);
            acc = Some(match acc {
                None => g,
                Some(prev) => {
                    let in_i = emit(
                        module,
                        Prim::Compare(CmpOp::Ge),
                        &[idn, off_c],
                        &bool_ids_ty,
                        &loc,
                    );
                    let cond = emit(
                        module,
                        Prim::Reshape {
                            shape: cond_shape.dims().to_vec(),
                        },
                        &[in_i],
                        &bool_cond_ty,
                        &loc,
                    );
                    if i + 1 == n_chunks {
                        emit_into(module, Prim::Select, &[cond, g, prev], out, &loc)
                    } else {
                        emit(module, Prim::Select, &[cond, g, prev], &out_ty, &loc)
                    }
                }
            });
        }
    }
    Ok(())
}

fn const_i64(module: &mut Module, v: i64) -> Result<ValueId> {
    let ty = TensorType::of(DataType::I64, &[]);
    let cid = module.consts.add(ty.clone(), v.to_le_bytes().to_vec())?;
    Ok(module.add_value(ValueDef {
        name: None,
        ty,
        origin: Origin::Const(cid),
        content: None,
    }))
}

/// Append a single-output primitive node producing a fresh value of `ty`.
fn emit(
    module: &mut Module,
    prim: Prim,
    inputs: &[ValueId],
    ty: &TensorType,
    loc: &SourceInfo,
) -> ValueId {
    let out = module.add_value(ValueDef {
        name: None,
        ty: ty.clone(),
        origin: Origin::Input, // patched below
        content: None,
    });
    emit_into(module, prim, inputs, out, loc)
}

/// Append a single-output primitive node writing an existing value.
fn emit_into(
    module: &mut Module,
    prim: Prim,
    inputs: &[ValueId],
    out: ValueId,
    loc: &SourceInfo,
) -> ValueId {
    let node = module.add_node(Node {
        kind: NodeKind::Prim(prim),
        inputs: inputs.to_vec(),
        outputs: vec![out],
        loc: loc.clone(),
    });
    module.value_mut(out).origin = Origin::Node { node, output: 0 };
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::interp::{Tensor, eval};

    fn build() -> Module {
        // table [10, 4] f32, row r = [r, r+0.25, r+0.5, r+0.75]
        let mut bld = GraphBuilder::new();
        let vals: Vec<f32> = (0..40)
            .map(|i| (i / 4) as f32 + (i % 4) as f32 * 0.25)
            .collect();
        let table = bld.const_f32(&vals, &[10, 4]).unwrap();
        let ids = bld.input("ids", TensorType::of(DataType::I64, &[2, 3]));
        let y = bld.gather(table, ids, 0).unwrap();
        bld.output("y", y);
        bld.finish().unwrap()
    }

    #[test]
    fn split_matches_unsplit() {
        let module = build();
        let mut split = module.clone();
        // 16-byte rows, 48-byte cap → 3 rows per chunk → chunks of 3,3,3,1.
        let n = split_large_tables(&mut split, 48).unwrap();
        assert_eq!(n, 1);
        crate::validate::validate(&split).unwrap();
        let chunks = split
            .values
            .iter()
            .filter(|v| v.name.as_deref().is_some_and(|n| n.contains(".rows")))
            .count();
        assert_eq!(chunks, 4);

        // Boundaries of every chunk, a negative (wraps to 9) and the last row.
        let ids = Tensor::from_i64(&[0, 2, 3, 5, 6, -1], &[2, 3]).unwrap();
        let e = eval(&module, &[("ids", ids.clone())]).unwrap();
        let g = eval(&split, &[("ids", ids)]).unwrap();
        assert_eq!(e[0].1.shape(), g[0].1.shape());
        assert_eq!(e[0].1.to_f32().unwrap(), g[0].1.to_f32().unwrap());
    }

    #[test]
    fn split_matmul_matches_unsplit() {
        // x [2, 4] @ table [10, 4]^T → [2, 10]
        let mut bld = GraphBuilder::new();
        let vals: Vec<f32> = (0..40).map(|i| (i as f32 * 0.37).sin()).collect();
        let table = bld.const_f32(&vals, &[10, 4]).unwrap();
        let x = bld.input("x", TensorType::of(DataType::F32, &[2, 4]));
        let y = bld
            .prim(
                Prim::MatMul {
                    trans_a: false,
                    trans_b: true,
                },
                &[x, table],
            )
            .unwrap();
        bld.output("y", y);
        let module = bld.finish().unwrap();
        let mut split = module.clone();
        assert_eq!(split_large_tables(&mut split, 48).unwrap(), 1);
        crate::validate::validate(&split).unwrap();
        let x = Tensor::from_f32(
            &(0..8).map(|i| i as f32 * 0.5 - 1.0).collect::<Vec<_>>(),
            &[2, 4],
        )
        .unwrap();
        let e = eval(&module, &[("x", x.clone())]).unwrap();
        let g = eval(&split, &[("x", x)]).unwrap();
        assert_eq!(e[0].1.shape(), g[0].1.shape());
        for (a, b) in e[0]
            .1
            .to_f32()
            .unwrap()
            .iter()
            .zip(g[0].1.to_f32().unwrap())
        {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn small_tables_are_left_alone() {
        let mut module = build();
        assert_eq!(split_large_tables(&mut module, 160).unwrap(), 0);
        assert_eq!(module.nodes.len(), 1);
    }
}
