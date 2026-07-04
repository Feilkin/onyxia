//! Constant folding and the compile-time value domain.
//!
//! Three cooperating pieces:
//!
//! 1. **Constant folding** — a node whose inputs are all constants is
//!    evaluated with the [reference interpreter](crate::interp) at compile
//!    time; its result moves into the constant pool and the node
//!    disappears. Size-thresholded so folding never bloats the pool.
//! 2. **Symbolic content** — small integer values whose *elements* are
//!    known as [`DimExpr`]s (the output of `Shape` and the arithmetic on
//!    it). [`eval_content`] evaluates a primitive over contents without
//!    executing anything, so ONNX shape-arithmetic chains
//!    (`Shape → Gather → Concat → Reshape`) resolve at lowering into
//!    symbolic `Reshape` targets instead of runtime nodes. Lowering is the
//!    main caller; [`fold`] also propagates content across any such nodes
//!    remaining in the graph.
//! 3. **Dead-code elimination** — nodes whose outputs nobody consumes are
//!    removed (typically the shape plumbing made redundant by 2).

use crate::Result;
use crate::dim::{Bindings, DimExpr};
use crate::graph::{Module, NodeId, NodeKind, Origin, SymbolicContent, ValueDef};
use crate::interp::{Tensor, const_tensor, eval_prim};
use crate::prim::{BinaryOp, Prim, ReduceOp};
use crate::types::TensorType;
use std::collections::HashSet;

/// Options for [`fold`].
#[derive(Debug, Clone)]
pub struct FoldOptions {
    /// Do not fold nodes whose output would exceed this many bytes —
    /// folding is for shape plumbing and small algebra, not for baking
    /// giant tensors into the pool.
    pub max_output_bytes: usize,
    /// Do not even *consider* folding a node whose constant inputs total
    /// more than this many bytes. Checked before any data is materialized,
    /// so weight-scale constants are never cloned out of the pool or fed to
    /// the (naive, CPU) interpreter just to discover the result is too big.
    pub max_input_bytes: usize,
}

impl Default for FoldOptions {
    fn default() -> Self {
        Self {
            max_output_bytes: 1 << 20, // 1 MiB
            max_input_bytes: 1 << 20,  // 1 MiB
        }
    }
}

/// Run constant folding, content propagation, and DCE to a fixpoint.
///
/// Returns whether anything changed.
pub fn fold(module: &mut Module, opts: &FoldOptions) -> Result<bool> {
    let mut changed_any = false;
    loop {
        let mut changed = false;

        seed_content_from_consts(module);

        let mut dead: HashSet<NodeId> = HashSet::new();
        for node_id in module.topo_order()? {
            let node = module.node(node_id);
            let NodeKind::Prim(prim) = &node.kind else {
                continue; // composites fold only via their decompositions
            };

            // 1. Try full constant folding. Size gates run first, on pool
            //    metadata only — nothing is materialized for oversized
            //    candidates.
            let const_input_bytes: Option<usize> = node
                .inputs
                .iter()
                .map(|&v| match module.value(v).origin {
                    Origin::Const(cid) => Some(module.consts.bytes(cid).len()),
                    _ => None,
                })
                .sum();
            if const_input_bytes.is_some_and(|b| b <= opts.max_input_bytes)
                && let Some(tensors) = const_inputs(module, node_id)
            {
                let refs: Vec<&Tensor> = tensors.iter().collect();
                let mut bindings = Bindings::new();
                // Symbolic params (e.g. a symbolic Reshape target) make
                // eval fail — that's a "can't fold", not an error.
                if let Ok(out) = eval_prim(prim, &refs, &mut bindings) {
                    if out.bytes().len() <= opts.max_output_bytes {
                        let ty = TensorType::of(
                            out.dtype(),
                            &out.shape().iter().map(|&d| d as u64).collect::<Vec<_>>(),
                        );
                        let cid = module.consts.add(ty.clone(), out.bytes().to_vec())?;
                        let out_id = module.node(node_id).outputs[0];
                        let def = module.value_mut(out_id);
                        def.ty = ty;
                        def.origin = Origin::Const(cid);
                        dead.insert(node_id);
                        changed = true;
                        continue;
                    }
                }
            }

            // 2. Try symbolic content propagation.
            let node = module.node(node_id);
            let contents: Vec<Option<&SymbolicContent>> = node
                .inputs
                .iter()
                .map(|&v| module.value(v).content.as_ref())
                .collect();
            if let Some(content) = eval_content(prim, &contents) {
                let out_id = node.outputs[0];
                if module.value(out_id).content.as_ref() != Some(&content) {
                    module.value_mut(out_id).content = Some(content);
                    changed = true;
                }
            }
        }
        module.remove_nodes(&dead);

        changed |= eliminate_dead(module) > 0;

        changed_any |= changed;
        if !changed {
            return Ok(changed_any);
        }
    }
}

/// Fold a `Transpose` of a matmul operand into the matmul's
/// `trans_a`/`trans_b` flag, when the permutation swaps the last two dims
/// and is identity on the batch dims. The orphaned `Transpose` is left
/// for [`eliminate_dead`]. Returns whether anything changed.
///
/// This is a *runtime* optimization, not just cleanup: tied-embedding
/// exports (e.g. Gemma) feed `lm_head` as `MatMul(h, Transpose(W))`, and
/// without this fold every run re-materializes the transposed GiB-scale
/// weight on the device.
pub fn fold_transpose_into_matmul(module: &mut Module) -> bool {
    fn swaps_last_two(perm: &[usize]) -> bool {
        let r = perm.len();
        r >= 2
            && perm[..r - 2].iter().enumerate().all(|(i, &p)| p == i)
            && perm[r - 2] == r - 1
            && perm[r - 1] == r - 2
    }

    let mut changed = false;
    for id in module.node_ids() {
        let NodeKind::Prim(Prim::MatMul { .. }) = module.node(id).kind else {
            continue;
        };
        for side in 0..2 {
            let operand = module.node(id).inputs[side];
            let Origin::Node { node: producer, .. } = module.value(operand).origin else {
                continue;
            };
            let NodeKind::Prim(Prim::Transpose { perm }) = &module.node(producer).kind else {
                continue;
            };
            if !swaps_last_two(perm) {
                continue;
            }
            let src = module.node(producer).inputs[0];
            let node = &mut module.nodes[id.index()];
            node.inputs[side] = src;
            let NodeKind::Prim(Prim::MatMul { trans_a, trans_b }) = &mut node.kind else {
                unreachable!("checked above");
            };
            match side {
                0 => *trans_a = !*trans_a,
                _ => *trans_b = !*trans_b,
            }
            changed = true;
        }
    }
    changed
}

/// Remove nodes none of whose outputs are consumed (by nodes or by the
/// module's outputs). Returns the number of nodes removed.
pub fn eliminate_dead(module: &mut Module) -> usize {
    let mut removed_total = 0;
    loop {
        let mut used = vec![false; module.values.len()];
        for (_, id) in &module.outputs {
            used[id.index()] = true;
        }
        for node in &module.nodes {
            for &v in &node.inputs {
                used[v.index()] = true;
            }
        }
        let dead: HashSet<NodeId> = module
            .node_ids()
            .filter(|&id| module.node(id).outputs.iter().all(|&v| !used[v.index()]))
            .collect();
        if dead.is_empty() {
            return removed_total;
        }
        removed_total += dead.len();
        module.remove_nodes(&dead);
    }
}

/// Give small integer constants symbolic content, so shape arithmetic over
/// them participates in [`eval_content`].
fn seed_content_from_consts(module: &mut Module) {
    const MAX_SEED_ELEMS: usize = 64;
    for id in module.value_ids() {
        let def = module.value(id);
        if def.content.is_some() {
            continue;
        }
        let Origin::Const(cid) = def.origin else {
            continue;
        };
        if def.ty.dtype != crate::types::DataType::I64 || def.ty.shape.rank() > 1 {
            continue;
        }
        let Ok(tensor) = const_tensor(module, cid) else {
            continue;
        };
        if tensor.numel() > MAX_SEED_ELEMS {
            continue;
        }
        let Ok(vals) = tensor.to_i64() else { continue };
        // Negative values (e.g. Reshape's -1 sentinel) are still useful as
        // *constants* in content arithmetic; DimExpr supports them via
        // subtraction from zero.
        let elems: Vec<DimExpr> = vals
            .iter()
            .map(|&v| {
                if v >= 0 {
                    DimExpr::constant(v as u64)
                } else {
                    DimExpr::constant(0) - DimExpr::constant(v.unsigned_abs())
                }
            })
            .collect();
        let content = if tensor.shape().is_empty() {
            SymbolicContent::scalar(elems.into_iter().next().expect("scalar has one elem"))
        } else {
            SymbolicContent::vector(elems)
        };
        module.value_mut(id).content = Some(content);
    }
}

/// The tensors for a node's inputs if they are all pool constants.
fn const_inputs(module: &Module, node_id: NodeId) -> Option<Vec<Tensor>> {
    module
        .node(node_id)
        .inputs
        .iter()
        .map(|&v| match module.value(v).origin {
            Origin::Const(cid) => const_tensor(module, cid).ok(),
            _ => None,
        })
        .collect()
}

/// Evaluate a primitive over symbolic contents, without executing anything.
///
/// Returns `None` when the primitive/operands are outside the supported
/// fragment (rank ≤ 1 integer values, the ops ONNX shape plumbing uses).
/// This is the workhorse lowering calls to make shape chains disappear.
pub fn eval_content(prim: &Prim, inputs: &[Option<&SymbolicContent>]) -> Option<SymbolicContent> {
    let get = |i: usize| -> Option<&SymbolicContent> { inputs.get(i).copied().flatten() };
    match prim {
        // Shape-vector concatenation (axis 0 over rank-1 contents).
        Prim::Concat { axis: 0 } => {
            let mut elems = Vec::new();
            for c in inputs {
                let c = (*c)?;
                elems.extend(c.elems.iter().cloned());
            }
            Some(SymbolicContent::vector(elems))
        }

        // Picking dims out of a shape vector with constant indices.
        Prim::Gather { axis: 0 } => {
            let data = get(0)?;
            let indices = get(1)?;
            if data.shape.len() != 1 {
                return None;
            }
            let n = data.elems.len() as i64;
            let picked: Option<Vec<DimExpr>> = indices
                .elems
                .iter()
                .map(|e| {
                    let mut i = const_of(e)?;
                    if i < 0 {
                        i += n;
                    }
                    data.elems.get(usize::try_from(i).ok()?).cloned()
                })
                .collect();
            let picked = picked?;
            Some(if indices.shape.is_empty() {
                SymbolicContent::scalar(picked.into_iter().next()?)
            } else {
                SymbolicContent::vector(picked)
            })
        }

        // Shape arithmetic. Element-wise with scalar broadcast.
        Prim::Binary(op) => {
            let (a, b) = (get(0)?, get(1)?);
            let n = a.elems.len().max(b.elems.len());
            if (a.elems.len() != n && a.elems.len() != 1)
                || (b.elems.len() != n && b.elems.len() != 1)
            {
                return None;
            }
            let pick = |c: &SymbolicContent, i: usize| -> DimExpr {
                c.elems[if c.elems.len() == 1 { 0 } else { i }].clone()
            };
            let elems: Option<Vec<DimExpr>> = (0..n)
                .map(|i| {
                    let (x, y) = (pick(a, i), pick(b, i));
                    match op {
                        BinaryOp::Add => Some(x + y),
                        BinaryOp::Sub => Some(x - y),
                        BinaryOp::Mul => Some(x * y),
                        BinaryOp::Div => x.div_exact(&y),
                        _ => None,
                    }
                })
                .collect();
            let out_scalar = a.shape.is_empty() && b.shape.is_empty();
            let elems = elems?;
            Some(if out_scalar {
                SymbolicContent::scalar(elems.into_iter().next()?)
            } else {
                SymbolicContent::vector(elems)
            })
        }

        // Slicing a shape vector with constant step-±1 bounds.
        Prim::Slice { specs } => {
            let data = get(0)?;
            if data.shape.len() != 1 {
                return None;
            }
            let [spec] = specs.as_slice() else {
                return None;
            };
            if spec.axis != 0 || spec.step.abs() != 1 {
                return None;
            }
            let start = usize::try_from(spec.start.as_const()?).ok()?;
            let end = usize::try_from(spec.end.as_const()?).ok()?;
            let elems: Vec<DimExpr> = if spec.step == 1 {
                data.elems.get(start..end)?.to_vec()
            } else {
                // Reversed: [start, end) in the direction of travel.
                let lo = end + 1;
                let mut v = data.elems.get(lo..=start)?.to_vec();
                v.reverse();
                v
            };
            Some(SymbolicContent::vector(elems))
        }

        // Rank changes between scalar and [1] keep content.
        Prim::Reshape { shape } => {
            let data = get(0)?;
            match shape.len() {
                0 => Some(SymbolicContent::scalar(data.elems.first()?.clone())),
                1 => Some(SymbolicContent::vector(data.elems.clone())),
                _ => None,
            }
        }

        // Integer→integer casts keep content.
        Prim::Cast { to } if to.is_int() && !to.is_packed() => get(0).cloned(),

        // ReduceProd/Sum over a shape vector (numel computations).
        Prim::Reduce { op, axes, keepdims } => {
            let data = get(0)?;
            if data.shape.len() != 1 || axes != &[0] {
                return None;
            }
            let init = match op {
                ReduceOp::Prod => DimExpr::constant(1),
                ReduceOp::Sum => DimExpr::constant(0),
                _ => return None,
            };
            let acc = data.elems.iter().cloned().fold(init, |acc, e| match op {
                ReduceOp::Prod => acc * e,
                ReduceOp::Sum => acc + e,
                _ => unreachable!(),
            });
            Some(if *keepdims {
                SymbolicContent::vector(vec![acc])
            } else {
                SymbolicContent::scalar(acc)
            })
        }

        _ => None,
    }
}

fn const_of(e: &DimExpr) -> Option<i64> {
    // as_const rejects negatives; shape indices may be negative, so peek
    // via evaluation against no bindings when the expr is constant.
    if e.is_const() {
        // A constant expression evaluates without bindings unless negative;
        // reconstruct the signed value from the canonical form.
        e.eval(&Bindings::new()).map(|v| v as i64).ok().or_else(|| {
            // Negative constant: eval errors; recover via 0 - e.
            (DimExpr::constant(0) - e.clone())
                .eval(&Bindings::new())
                .ok()
                .map(|v| -(v as i64))
        })
    } else {
        None
    }
}

/// Convenience for lowering: build a [`ValueDef`] carrying content.
pub fn content_value(name: Option<String>, ty: TensorType, content: SymbolicContent) -> ValueDef {
    ValueDef {
        name,
        ty,
        origin: Origin::Input, // caller overwrites; placeholder
        content: Some(content),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::prim::UnaryOp;
    use crate::types::DataType;

    #[test]
    fn transpose_folds_into_matmul_trans_b() {
        // out = a[2,3] × transpose(w[4,3]) — the tied-embedding lm_head
        // pattern, with runtime inputs so constant folding can't hide it.
        let mut b = GraphBuilder::new();
        let a = b.input("a", TensorType::of(DataType::F32, &[2, 3]));
        let w = b.input("w", TensorType::of(DataType::F32, &[4, 3]));
        let wt = b.transpose(w, &[1, 0]).unwrap();
        let mm = b.matmul(a, wt).unwrap();
        b.output("out", mm);
        let mut m = b.finish().unwrap();

        let expect = crate::interp::eval(
            &m,
            &[
                (
                    "a",
                    Tensor::from_f32(&(0..6).map(|i| i as f32).collect::<Vec<_>>(), &[2, 3])
                        .unwrap(),
                ),
                (
                    "w",
                    Tensor::from_f32(&(0..12).map(|i| i as f32 * 0.5).collect::<Vec<_>>(), &[4, 3])
                        .unwrap(),
                ),
            ],
        )
        .unwrap();

        assert!(fold_transpose_into_matmul(&mut m));
        assert_eq!(eliminate_dead(&mut m), 1, "orphaned Transpose removed");
        assert_eq!(m.nodes.len(), 1);
        let NodeKind::Prim(Prim::MatMul { trans_a, trans_b }) = m.nodes[0].kind else {
            panic!("matmul survived");
        };
        assert!(!trans_a);
        assert!(trans_b);
        crate::validate::validate(&m).unwrap();

        // Semantics preserved.
        let got = crate::interp::eval(
            &m,
            &[
                (
                    "a",
                    Tensor::from_f32(&(0..6).map(|i| i as f32).collect::<Vec<_>>(), &[2, 3])
                        .unwrap(),
                ),
                (
                    "w",
                    Tensor::from_f32(&(0..12).map(|i| i as f32 * 0.5).collect::<Vec<_>>(), &[4, 3])
                        .unwrap(),
                ),
            ],
        )
        .unwrap();
        assert_eq!(expect[0].1.to_f32().unwrap(), got[0].1.to_f32().unwrap());
    }

    #[test]
    fn batch_permuting_transpose_does_not_fold() {
        // perm [1,0,2] moves a batch dim — must NOT fold.
        let mut b = GraphBuilder::new();
        let a = b.input("a", TensorType::of(DataType::F32, &[4, 2, 3]));
        let w = b.input("w", TensorType::of(DataType::F32, &[3, 4, 5]));
        let wt = b
            .prim(
                Prim::Transpose {
                    perm: vec![1, 0, 2],
                },
                &[w],
            )
            .unwrap();
        let mm = b.matmul(a, wt).unwrap();
        b.output("out", mm);
        let mut m = b.finish().unwrap();
        assert!(!fold_transpose_into_matmul(&mut m));
        assert_eq!(m.nodes.len(), 2);
    }

    #[test]
    fn folds_constant_subgraph_to_pool() {
        // out = neg(a + b) with a, b constants; plus a runtime add.
        let mut b = GraphBuilder::new();
        let x = b.input("x", TensorType::of(DataType::F32, &[2]));
        let ca = b.const_f32(&[1.0, 2.0], &[2]).unwrap();
        let cb = b.const_f32(&[10.0, 20.0], &[2]).unwrap();
        let sum = b.add(ca, cb).unwrap();
        let neg = b.unary(UnaryOp::Neg, sum).unwrap();
        let out = b.add(x, neg).unwrap();
        b.output("out", out);
        let mut m = b.finish().unwrap();
        assert_eq!(m.nodes.len(), 3);

        let changed = fold(&mut m, &FoldOptions::default()).unwrap();
        assert!(changed);
        // Only the runtime add remains; folded result lives in the pool.
        assert_eq!(m.nodes.len(), 1);
        crate::validate::validate(&m).unwrap();

        // And the module still computes the right thing.
        let x_in = crate::interp::Tensor::from_f32(&[100.0, 200.0], &[2]).unwrap();
        let outs = crate::interp::eval(&m, &[("x", x_in)]).unwrap();
        assert_eq!(outs[0].1.to_f32().unwrap(), vec![89.0, 178.0]);
    }

    #[test]
    fn respects_size_threshold() {
        let mut b = GraphBuilder::new();
        let ca = b.const_f32(&[1.0; 8], &[8]).unwrap();
        let cb = b.const_f32(&[2.0; 8], &[8]).unwrap();
        let sum = b.add(ca, cb).unwrap();
        b.output("out", sum);
        let mut m = b.finish().unwrap();
        let opts = FoldOptions {
            max_output_bytes: 16, // 8 f32 = 32 bytes
            ..Default::default()
        };
        fold(&mut m, &opts).unwrap();
        assert_eq!(m.nodes.len(), 1, "over-threshold node must not fold");

        // Same graph, gated on *input* size instead: nothing materializes.
        let opts = FoldOptions {
            max_input_bytes: 16, // two 32-byte inputs
            ..Default::default()
        };
        fold(&mut m, &opts).unwrap();
        assert_eq!(m.nodes.len(), 1, "over-threshold inputs must not fold");
    }

    #[test]
    fn dce_removes_unused_chain() {
        let mut b = GraphBuilder::new();
        let x = b.input("x", TensorType::of(DataType::F32, &[2]));
        let used = b.add(x, x).unwrap();
        let dead1 = b.mul(x, x).unwrap();
        let _dead2 = b.unary(UnaryOp::Neg, dead1).unwrap();
        b.output("out", used);
        let mut m = b.finish().unwrap();
        assert_eq!(m.nodes.len(), 3);
        assert_eq!(eliminate_dead(&mut m), 2);
        assert_eq!(m.nodes.len(), 1);
        crate::validate::validate(&m).unwrap();
    }

    /// The Gemma-style token-count chain
    /// `Shape(x) → Gather[1] → Concat with consts → (Reshape target)`
    /// evaluates symbolically into `[1, S, 4, 64]` — zero runtime nodes.
    #[test]
    fn shape_chain_evaluates_symbolically() {
        let mut tab = crate::dim::SymbolTable::new();
        let s = DimExpr::sym(tab.intern("S"));

        // Simulate lowering `Shape(x)` for x: [1, S, 256]:
        let shape_of_x = SymbolicContent::vector(vec![
            DimExpr::constant(1),
            s.clone(),
            DimExpr::constant(256),
        ]);

        // Gather index 1 -> scalar S.
        let idx = SymbolicContent::scalar(DimExpr::constant(1));
        let picked =
            eval_content(&Prim::Gather { axis: 0 }, &[Some(&shape_of_x), Some(&idx)]).unwrap();
        assert_eq!(picked.elems, vec![s.clone()]);
        assert!(picked.shape.is_empty());

        // Reshape scalar to [1] (ONNX Unsqueeze becomes this).
        let s_vec = eval_content(
            &Prim::Reshape {
                shape: vec![DimExpr::constant(1)],
            },
            &[Some(&picked)],
        )
        .unwrap();

        // Concat([1], [S], [4], [64]) -> the reshape target.
        let one = SymbolicContent::vector(vec![DimExpr::constant(1)]);
        let four = SymbolicContent::vector(vec![DimExpr::constant(4)]);
        let sixty_four = SymbolicContent::vector(vec![DimExpr::constant(64)]);
        let target = eval_content(
            &Prim::Concat { axis: 0 },
            &[Some(&one), Some(&s_vec), Some(&four), Some(&sixty_four)],
        )
        .unwrap();
        assert_eq!(
            target.elems,
            vec![
                DimExpr::constant(1),
                s.clone(),
                DimExpr::constant(4),
                DimExpr::constant(64)
            ]
        );
        // This vector plugs directly into Prim::Reshape { shape: target.elems }.
    }

    #[test]
    fn shape_arithmetic_content() {
        let mut tab = crate::dim::SymbolTable::new();
        let s = DimExpr::sym(tab.intern("S"));
        let t = DimExpr::sym(tab.intern("T"));

        // total = S + T (scalar contents)
        let a = SymbolicContent::scalar(s.clone());
        let b = SymbolicContent::scalar(t.clone());
        let sum = eval_content(&Prim::Binary(BinaryOp::Add), &[Some(&a), Some(&b)]).unwrap();
        assert_eq!(sum.elems, vec![s.clone() + t.clone()]);

        // numel = ReduceProd([1, S, 256]) = 256*S
        let shape_vec = SymbolicContent::vector(vec![
            DimExpr::constant(1),
            s.clone(),
            DimExpr::constant(256),
        ]);
        let numel = eval_content(
            &Prim::Reduce {
                op: ReduceOp::Prod,
                axes: vec![0],
                keepdims: false,
            },
            &[Some(&shape_vec)],
        )
        .unwrap();
        assert_eq!(numel.elems, vec![s.clone() * DimExpr::constant(256)]);

        // Slice [1..3) of the shape vector -> [S, 256]
        let sl = eval_content(
            &Prim::Slice {
                specs: vec![crate::prim::SliceSpec {
                    axis: 0,
                    start: DimExpr::constant(1),
                    end: DimExpr::constant(3),
                    step: 1,
                }],
            },
            &[Some(&shape_vec)],
        )
        .unwrap();
        assert_eq!(sl.elems, vec![s, DimExpr::constant(256)]);
    }

    #[test]
    fn fold_pass_propagates_content_through_graph_nodes() {
        // A real graph: i64 consts flow through Concat; fold folds it to a
        // pool constant (small, all-const) — and content seeding gives the
        // result symbolic content too.
        let mut b = GraphBuilder::new();
        let c1 = b.const_i64(&[1], &[1]).unwrap();
        let c2 = b.const_i64(&[4, 64], &[2]).unwrap();
        let cat = b.concat(&[c1, c2], 0).unwrap();
        b.output("target", cat);
        let mut m = b.finish().unwrap();
        fold(&mut m, &FoldOptions::default()).unwrap();
        assert_eq!(m.nodes.len(), 0, "all-const concat folds away");
        let (_, out_id) = &m.outputs[0];
        let content = m.value(*out_id).content.as_ref().unwrap();
        assert_eq!(
            content.elems,
            vec![
                DimExpr::constant(1),
                DimExpr::constant(4),
                DimExpr::constant(64)
            ]
        );
    }
}
