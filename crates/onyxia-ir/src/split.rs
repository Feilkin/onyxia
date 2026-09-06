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
//! row chunk is its own matmul and the results concatenate along `N`. The
//! same for `com.microsoft.MatMulNBits` (the 4-bit lm_head), whose weights,
//! scales and zero points are all `N`-major and chunk together.
//!
//! Packed (4-bit) gather results cannot go through `Select`, so those are
//! merged in the 8-bit dtype and cast back.

use crate::graph::{
    Composite, Module, Node, NodeId, NodeKind, Origin, SourceInfo, ValueDef, ValueId,
};
use crate::prim::{BinaryOp, CmpOp, Prim};
use crate::{AttrValue, DataType, DimExpr, Result, SymbolicShape, TensorType};
use std::collections::HashSet;

const MATMUL_NBITS: &str = "com.microsoft.MatMulNBits";

/// A constant tensor viewed as `rows` contiguous rows.
struct Table {
    cid: crate::graph::ConstId,
    dims: Vec<u64>,
    rows: usize,
    row_bytes: usize,
    /// Leading-dim elements per row (`dims[0] / rows`; 1 for `[rows, ..]`,
    /// `n_blocks` for a flat `[rows * n_blocks]` scale vector).
    per_row: u64,
    name: String,
}

/// How a table (and its companions) are cut: `rpc` rows per chunk.
#[derive(Clone, Copy)]
struct Chunking {
    rows: usize,
    rpc: usize,
    n_chunks: usize,
}

impl Table {
    /// View constant `value` as `rows` rows, if it is one and divides evenly.
    fn new(module: &Module, value: ValueId, rows: usize) -> Option<Self> {
        let Origin::Const(cid) = module.value(value).origin else {
            return None;
        };
        let total = module.consts.bytes(cid).len();
        let dims = module.consts.ty(cid).shape.as_static()?;
        if rows == 0 || total % rows != 0 || dims[0] % rows as u64 != 0 {
            return None;
        }
        let name = module
            .value(value)
            .name
            .clone()
            .unwrap_or_else(|| "table".into());
        Some(Self {
            cid,
            per_row: dims[0] / rows as u64,
            dims,
            rows,
            row_bytes: total / rows,
            name,
        })
    }

    fn total_bytes(&self) -> usize {
        self.rows * self.row_bytes
    }

    /// The chunking that keeps every chunk of this table under `max_bytes`,
    /// or `None` if it already fits (or a single row does not).
    fn chunking(&self, max_bytes: usize) -> Option<Chunking> {
        if self.total_bytes() <= max_bytes || self.row_bytes > max_bytes {
            return None;
        }
        let mut rpc = max_bytes / self.row_bytes;
        if rpc >= 64 {
            rpc -= rpc % 64; // keep kernel row groups aligned
        }
        let n_chunks = self.rows.div_ceil(rpc);
        debug_assert!(n_chunks >= 2);
        Some(Chunking {
            rows: self.rows,
            rpc,
            n_chunks,
        })
    }

    /// Add chunk `i` to the constant pool and return its value.
    fn chunk(&self, module: &mut Module, c: Chunking, i: usize) -> Result<ValueId> {
        debug_assert_eq!(c.rows, self.rows);
        let (off, len) = c.range(i);
        let bytes = module.consts.bytes(self.cid)
            [off * self.row_bytes..(off + len) * self.row_bytes]
            .to_vec();
        let mut cdims = self.dims.clone();
        cdims[0] = len as u64 * self.per_row;
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

impl Chunking {
    /// Row range of chunk `i`.
    fn range(&self, i: usize) -> (usize, usize) {
        let off = i * self.rpc;
        (off, self.rpc.min(self.rows - off))
    }
}

/// The table behind `value` with its own leading dim as rows, if it needs
/// chunking under `max_bytes`.
fn oversized(module: &Module, value: ValueId, max_bytes: usize) -> Option<(Table, Chunking)> {
    let rows = module.consts.ty(match module.value(value).origin {
        Origin::Const(cid) => cid,
        _ => return None,
    });
    let rows = rows.shape.as_static()?[0] as usize;
    let t = Table::new(module, value, rows)?;
    let c = t.chunking(max_bytes)?;
    Some((t, c))
}

/// Split every `Gather{axis: 0}` table and every `MatMul{trans_b}` weight
/// larger than `max_bytes` (bytes in the constant pool's storage layout).
/// Returns the number of tables split.
pub fn split_large_tables(module: &mut Module, max_bytes: usize) -> Result<usize> {
    let mut dead: HashSet<NodeId> = HashSet::new();
    let mut count = 0;

    for id in module.node_ids() {
        let node = module.node(id);
        let split = match &node.kind {
            NodeKind::Prim(Prim::Gather { axis: 0 }) => {
                oversized(module, node.inputs[0], max_bytes)
                    .map(|(t, c)| split_gather(module, id, t, c))
                    .transpose()?
            }
            NodeKind::Prim(Prim::MatMul { trans_b: true, .. })
                if module.value(node.inputs[1]).ty.shape.rank() == 2 =>
            {
                oversized(module, node.inputs[1], max_bytes)
                    .map(|(t, c)| split_matmul(module, id, t, c))
                    .transpose()?
            }
            NodeKind::Composite(comp) if comp.name == MATMUL_NBITS => {
                split_matmul_nbits(module, id, max_bytes)?
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

/// The output type of a matmul chunk: `out_ty` with the last dim = `len`.
fn part_ty(out_ty: &TensorType, len: usize) -> TensorType {
    let mut pdims = out_ty.shape.dims().to_vec();
    let last = pdims.len() - 1;
    pdims[last] = DimExpr::constant(len as u64);
    TensorType::new(out_ty.dtype, SymbolicShape::from(pdims))
}

/// `out = concat_N(a @ chunk_i^T)`.
fn split_matmul(module: &mut Module, id: NodeId, t: Table, c: Chunking) -> Result<()> {
    let node = module.node(id);
    let NodeKind::Prim(Prim::MatMul { trans_a, .. }) = node.kind else {
        unreachable!()
    };
    let a = node.inputs[0];
    let out = node.outputs[0];
    let loc = node.loc.clone();
    let out_ty = module.value(out).ty.clone();
    let last = out_ty.shape.rank() - 1;
    let mut parts = Vec::with_capacity(c.n_chunks);
    for i in 0..c.n_chunks {
        let (_, len) = c.range(i);
        let chunk = t.chunk(module, c, i)?;
        parts.push(emit(
            module,
            Prim::MatMul {
                trans_a,
                trans_b: true,
            },
            &[a, chunk],
            &part_ty(&out_ty, len),
            &loc,
        ));
    }
    emit_into(module, Prim::Concat { axis: last }, &parts, out, &loc);
    Ok(())
}

/// `com.microsoft.MatMulNBits` with `N`-major weights `[N, nb, bs]`, scales
/// `[N * nb]` (or `[N, nb]`) and optional zero points `[N, nb]`: chunk all
/// three by `N` rows, one composite per chunk (attr `N` = chunk rows),
/// concatenated along `N`. `Some(())` if the weights needed splitting.
fn split_matmul_nbits(module: &mut Module, id: NodeId, max_bytes: usize) -> Result<Option<()>> {
    let node = module.node(id);
    let NodeKind::Composite(comp) = &node.kind else {
        unreachable!()
    };
    let attrs = comp.attrs.clone();
    let n = attrs.int("N")? as usize;
    let a = node.inputs[0];
    let out = node.outputs[0];
    let loc = node.loc.clone();
    let Some(weights) = Table::new(module, node.inputs[1], n) else {
        return Ok(None);
    };
    let Some(c) = weights.chunking(max_bytes) else {
        return Ok(None);
    };
    let mut companions = Vec::new();
    for &v in &node.inputs[2..] {
        let Some(t) = Table::new(module, v, n) else {
            return Ok(None); // non-constant scales/zero points: leave it
        };
        companions.push(t);
    }
    let out_ty = module.value(out).ty.clone();
    let last = out_ty.shape.rank() - 1;
    let mut parts = Vec::with_capacity(c.n_chunks);
    for i in 0..c.n_chunks {
        let (_, len) = c.range(i);
        let mut inputs = vec![a, weights.chunk(module, c, i)?];
        for t in &companions {
            inputs.push(t.chunk(module, c, i)?);
        }
        parts.push(emit(
            module,
            Prim::Concat { axis: 0 }, // placeholder kind, replaced below
            &inputs,
            &part_ty(&out_ty, len),
            &loc,
        ));
        let nid = match module.value(*parts.last().unwrap()).origin {
            Origin::Node { node, .. } => node,
            _ => unreachable!(),
        };
        module.nodes[nid.index()].kind = NodeKind::Composite(Composite {
            name: MATMUL_NBITS.into(),
            attrs: attrs.clone().with("N", AttrValue::Int(len as i64)),
        });
    }
    emit_into(module, Prim::Concat { axis: last }, &parts, out, &loc);
    Ok(Some(()))
}

/// Chunked gather with clamped local indices merged by range selects.
fn split_gather(module: &mut Module, id: NodeId, t: Table, c: Chunking) -> Result<()> {
    let node = module.node(id);
    let ids = node.inputs[1];
    let out = node.outputs[0];
    let loc = node.loc.clone();
    let (rows, dims, n_chunks) = (t.rows, t.dims.clone(), c.n_chunks);
    // Packed results are merged in the matching 8-bit dtype and cast back.
    let dtype = module.consts.ty(t.cid).dtype;
    let merge_dtype = match dtype {
        DataType::U4 => DataType::U8,
        DataType::I4 => DataType::I8,
        d => d,
    };
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

        let merge_ty = TensorType::new(merge_dtype, out_ty.shape.clone());
        let mut acc: Option<ValueId> = None;
        for i in 0..n_chunks {
            let (off, len) = c.range(i);
            let cval = t.chunk(module, c, i)?;
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
            let mut g = emit(module, Prim::Gather { axis: 0 }, &[cval, cl], &out_ty, &loc);
            if merge_dtype != dtype {
                g = emit(
                    module,
                    Prim::Cast { to: merge_dtype },
                    &[g],
                    &merge_ty,
                    &loc,
                );
            }
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
                    if i + 1 == n_chunks && merge_dtype == dtype {
                        emit_into(module, Prim::Select, &[cond, g, prev], out, &loc)
                    } else if i + 1 == n_chunks {
                        let m = emit(module, Prim::Select, &[cond, g, prev], &merge_ty, &loc);
                        emit_into(module, Prim::Cast { to: dtype }, &[m], out, &loc)
                    } else {
                        emit(module, Prim::Select, &[cond, g, prev], &merge_ty, &loc)
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

    /// Pack low-nibble-first u4 values.
    fn pack_u4(vals: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; vals.len().div_ceil(2)];
        for (i, &v) in vals.iter().enumerate() {
            out[i / 2] |= (v & 0xF) << ((i % 2) * 4);
        }
        out
    }

    #[test]
    fn split_u4_gather_matches_unsplit() {
        // table [10, 4] u4 (20 bytes), gathered then cast to f32.
        let mut bld = GraphBuilder::new();
        let vals: Vec<u8> = (0..40).map(|i| (i * 7 % 16) as u8).collect();
        let table = bld
            .constant(TensorType::of(DataType::U4, &[10, 4]), pack_u4(&vals))
            .unwrap();
        let ids = bld.input("ids", TensorType::of(DataType::I64, &[2, 3]));
        let y = bld.gather(table, ids, 0).unwrap();
        let y = bld.prim(Prim::Cast { to: DataType::F32 }, &[y]).unwrap();
        bld.output("y", y);
        let module = bld.finish().unwrap();
        let mut split = module.clone();
        // 2-byte rows, 6-byte cap → 3 rows per chunk → 4 chunks.
        assert_eq!(split_large_tables(&mut split, 6).unwrap(), 1);
        crate::validate::validate(&split).unwrap();
        let ids = Tensor::from_i64(&[0, 2, 3, 5, 6, -1], &[2, 3]).unwrap();
        let e = eval(&module, &[("ids", ids.clone())]).unwrap();
        let g = eval(&split, &[("ids", ids)]).unwrap();
        assert_eq!(e[0].1.to_f32().unwrap(), g[0].1.to_f32().unwrap());
    }

    #[test]
    fn split_matmul_nbits_matches_unsplit() {
        // K=8, N=10, block_size=4 → 2 blocks per row; scales flat [N * nb].
        let (k, n, bs, nb) = (8usize, 10usize, 4usize, 2usize);
        let q: Vec<u8> = (0..n * k).map(|i| (i * 5 % 16) as u8).collect();
        let scales: Vec<f32> = (0..n * nb).map(|i| 0.1 + i as f32 * 0.05).collect();
        let zp: Vec<u8> = (0..n * nb).map(|i| (i * 3 % 16) as u8).collect();
        let mut bld = GraphBuilder::new();
        let a = bld.input("a", TensorType::of(DataType::F32, &[2, k as u64]));
        let b = bld
            .constant(
                TensorType::of(DataType::U4, &[n as u64, nb as u64, bs as u64]),
                pack_u4(&q),
            )
            .unwrap();
        let s = bld.const_f32(&scales, &[(n * nb) as u64]).unwrap();
        let z = bld
            .constant(
                TensorType::of(DataType::U4, &[n as u64, nb as u64]),
                pack_u4(&zp),
            )
            .unwrap();
        let attrs = crate::Attrs::new()
            .with("K", AttrValue::Int(k as i64))
            .with("N", AttrValue::Int(n as i64))
            .with("bits", AttrValue::Int(4))
            .with("block_size", AttrValue::Int(bs as i64));
        let y = bld
            .composite(
                MATMUL_NBITS,
                attrs,
                &[a, b, s, z],
                vec![TensorType::of(DataType::F32, &[2, n as u64])],
            )
            .unwrap()[0];
        bld.output("y", y);
        let module = bld.finish().unwrap();
        let mut split = module.clone();
        // 4-byte weight rows, 12-byte cap → 3 rows per chunk → 4 chunks.
        assert_eq!(split_large_tables(&mut split, 12).unwrap(), 1);
        crate::validate::validate(&split).unwrap();
        let reg = crate::standard_decompositions();
        let plain = crate::inline_composites(module, &reg, &|_| false).unwrap();
        let split = crate::inline_composites(split, &reg, &|_| false).unwrap();
        let a = Tensor::from_f32(
            &(0..2 * k).map(|i| i as f32 * 0.3 - 1.0).collect::<Vec<_>>(),
            &[2, k],
        )
        .unwrap();
        let e = eval(&plain, &[("a", a.clone())]).unwrap();
        let g = eval(&split, &[("a", a)]).unwrap();
        assert_eq!(e[0].1.shape(), g[0].1.shape());
        for (x, y) in e[0]
            .1
            .to_f32()
            .unwrap()
            .iter()
            .zip(g[0].1.to_f32().unwrap())
        {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    #[test]
    fn small_tables_are_left_alone() {
        let mut module = build();
        assert_eq!(split_large_tables(&mut module, 160).unwrap(), 0);
        assert_eq!(module.nodes.len(), 1);
    }
}
