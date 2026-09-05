//! Backend-driven composite fusion.
//!
//! Lowering emits what the ONNX graph says: a residual `Add` feeding a
//! `SimplifiedLayerNormalization`, a `Gelu` feeding the gate `Mul`. A
//! backend with a kernel for the fused form asks for these patterns to
//! be rewritten into a single composite *before* legalization, the same
//! way it asks for composites it has kernels for to be left intact. The
//! fused composites have decompositions (back to the unfused nodes), so
//! backends without the kernels — and the interpreter — see the same
//! module semantics.
//!
//! Fusions (only applied when `supports(name)`):
//! - `onyxia.AddRmsNorm`: `sum = a + b; y = rmsnorm(sum) * w` →
//!   inputs `[a, b, w]`, outputs `[y, sum]`, attr `epsilon`. The sum stays
//!   a graph value (it is the residual stream).
//! - `onyxia.GeluMul`: `gelu(x) * u` → inputs `[x, u]`, output `[y]`,
//!   attr `approximate`. Only when the Gelu result has no other use.

use crate::graph::{Composite, Module, Node, NodeId, NodeKind, Origin, SourceInfo, ValueId};
use crate::prim::{BinaryOp, Prim};
use crate::{AttrValue, Attrs};
use std::collections::HashSet;

pub const ADD_RMS_NORM: &str = "onyxia.AddRmsNorm";
pub const GELU_MUL: &str = "onyxia.GeluMul";

/// Rewrite fusable patterns whose fused composite the backend `supports`.
/// Returns the number of fusions applied.
pub fn fuse_composites(module: &mut Module, supports: &dyn Fn(&str) -> bool) -> usize {
    let want_add_norm = supports(ADD_RMS_NORM);
    let want_gelu_mul = supports(GELU_MUL);
    if !want_add_norm && !want_gelu_mul {
        return 0;
    }

    // Use counts (node inputs + module outputs) for the single-use checks.
    let mut uses = vec![0usize; module.values.len()];
    for node in &module.nodes {
        for &v in &node.inputs {
            uses[v.index()] += 1;
        }
    }
    for (_, v) in &module.outputs {
        uses[v.index()] += 1;
    }

    let producer = |module: &Module, v: ValueId| -> Option<NodeId> {
        match module.value(v).origin {
            Origin::Node { node, output: 0 } => Some(node),
            _ => None,
        }
    };

    let mut dead: HashSet<NodeId> = HashSet::new();
    let mut new_nodes: Vec<Node> = Vec::new();
    // Rewire (value → new node index within `new_nodes`, output slot).
    let mut rewire: Vec<(ValueId, usize, usize)> = Vec::new();

    for id in module.node_ids() {
        if dead.contains(&id) {
            continue;
        }
        let node = module.node(id);
        match &node.kind {
            NodeKind::Composite(c) if want_add_norm && c.name == "SimplifiedLayerNormalization" => {
                let [sum, w] = node.inputs[..] else { continue };
                let Some(add_id) = producer(module, sum) else {
                    continue;
                };
                if dead.contains(&add_id) {
                    continue;
                }
                let add = module.node(add_id);
                if add.kind != NodeKind::Prim(Prim::Binary(BinaryOp::Add)) || add.outputs.len() != 1
                {
                    continue;
                }
                let [a, b] = add.inputs[..] else { continue };
                // Same static/symbolic shape on both addends and the sum:
                // the fused kernel does no broadcasting.
                let ty = &module.value(sum).ty;
                if module.value(a).ty != *ty || module.value(b).ty != *ty {
                    continue;
                }
                let eps = c.attrs.float_or("epsilon", 1e-5).unwrap_or(1e-5);
                let y = node.outputs[0];
                let idx = new_nodes.len();
                new_nodes.push(Node {
                    kind: NodeKind::Composite(Composite {
                        name: ADD_RMS_NORM.into(),
                        attrs: Attrs::new().with("epsilon", AttrValue::Float(eps)),
                    }),
                    inputs: vec![a, b, w],
                    outputs: vec![y, sum],
                    loc: SourceInfo {
                        name: node.loc.name.clone(),
                        op_type: Some("Add+SimplifiedLayerNormalization".into()),
                    },
                });
                rewire.push((y, idx, 0));
                rewire.push((sum, idx, 1));
                dead.insert(id);
                dead.insert(add_id);
            }
            NodeKind::Prim(Prim::Binary(BinaryOp::Mul)) if want_gelu_mul => {
                let [p, q] = node.inputs[..] else { continue };
                // Either operand may be the Gelu output.
                let pick = [(p, q), (q, p)].into_iter().find_map(|(g, u)| {
                    let gid = producer(module, g)?;
                    let gn = module.node(gid);
                    let NodeKind::Composite(gc) = &gn.kind else {
                        return None;
                    };
                    (gc.name == "Gelu"
                        && !dead.contains(&gid)
                        && uses[g.index()] == 1
                        && gn.outputs.len() == 1
                        && module.value(g).ty == module.value(u).ty
                        && module.value(u).ty == module.value(node.outputs[0]).ty)
                        .then(|| (gid, gn.inputs[0], u, gc.attrs.clone()))
                });
                let Some((gelu_id, x, u, gattrs)) = pick else {
                    continue;
                };
                let approximate = gattrs.str("approximate").unwrap_or("none").to_string();
                let y = node.outputs[0];
                let idx = new_nodes.len();
                new_nodes.push(Node {
                    kind: NodeKind::Composite(Composite {
                        name: GELU_MUL.into(),
                        attrs: Attrs::new().with("approximate", AttrValue::Str(approximate)),
                    }),
                    inputs: vec![x, u],
                    outputs: vec![y],
                    loc: SourceInfo {
                        name: node.loc.name.clone(),
                        op_type: Some("Gelu+Mul".into()),
                    },
                });
                rewire.push((y, idx, 0));
                dead.insert(id);
                dead.insert(gelu_id);
            }
            _ => {}
        }
    }

    let fused = new_nodes.len();
    if fused == 0 {
        return 0;
    }
    let ids: Vec<NodeId> = new_nodes.into_iter().map(|n| module.add_node(n)).collect();
    for (v, idx, output) in rewire {
        module.value_mut(v).origin = Origin::Node {
            node: ids[idx],
            output,
        };
    }
    module.remove_nodes(&dead);
    fused
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::interp::{Tensor, eval};
    use crate::{DataType, TensorType, inline_composites, standard_decompositions};

    fn f32s(n: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
        (0..n).map(f).collect()
    }

    fn build() -> Module {
        // residual = a + b; y = rmsnorm(residual) * w; z = gelu(y) * u;
        // out2 = residual + z  (the sum has a second consumer).
        let mut bld = GraphBuilder::new();
        let ty = TensorType::of(DataType::F32, &[2, 8]);
        let a = bld.input("a", ty.clone());
        let b = bld.input("b", ty.clone());
        let u = bld.input("u", ty.clone());
        let w = bld.input("w", TensorType::of(DataType::F32, &[8]));
        let sum = bld.add(a, b).unwrap();
        let y = bld
            .composite(
                "SimplifiedLayerNormalization",
                Attrs::new().with("epsilon", AttrValue::Float(1e-6)),
                &[sum, w],
                vec![ty.clone()],
            )
            .unwrap()[0];
        let g = bld
            .composite(
                "Gelu",
                Attrs::new().with("approximate", AttrValue::Str("tanh".into())),
                &[y],
                vec![ty.clone()],
            )
            .unwrap()[0];
        let z = bld.mul(g, u).unwrap();
        let out2 = bld.add(sum, z).unwrap();
        bld.output("z", z);
        bld.output("out2", out2);
        bld.finish().unwrap()
    }

    #[test]
    fn fuses_and_matches_unfused() {
        let module = build();
        let mut fused = module.clone();
        let n = fuse_composites(&mut fused, &|name| name.starts_with("onyxia."));
        assert_eq!(n, 2);
        let names: Vec<String> = fused
            .nodes
            .iter()
            .map(|n| match &n.kind {
                NodeKind::Composite(c) => c.name.clone(),
                NodeKind::Prim(p) => p.name().to_string(),
            })
            .collect();
        assert!(names.contains(&ADD_RMS_NORM.to_string()), "{names:?}");
        assert!(names.contains(&GELU_MUL.to_string()), "{names:?}");
        assert_eq!(names.len(), 3, "{names:?}"); // AddRmsNorm, GeluMul, final add
        crate::validate::validate(&fused).unwrap();

        let inputs = [
            (
                "a",
                Tensor::from_f32(&f32s(16, |i| (i as f32 * 0.7).sin()), &[2, 8]).unwrap(),
            ),
            (
                "b",
                Tensor::from_f32(&f32s(16, |i| (i as f32 * 0.3).cos()), &[2, 8]).unwrap(),
            ),
            (
                "u",
                Tensor::from_f32(&f32s(16, |i| 1.0 + i as f32 * 0.1), &[2, 8]).unwrap(),
            ),
            (
                "w",
                Tensor::from_f32(&f32s(8, |i| 0.5 + i as f32 * 0.05), &[8]).unwrap(),
            ),
        ];
        let reg = standard_decompositions();
        let plain = inline_composites(module, &reg, &|_| false).unwrap();
        let fused = inline_composites(fused, &reg, &|_| false).unwrap();
        let e = eval(&plain, &inputs).unwrap();
        let g = eval(&fused, &inputs).unwrap();
        for ((en, et), (gn, gt)) in e.iter().zip(&g) {
            assert_eq!(en, gn);
            for (x, y) in et.to_f32().unwrap().iter().zip(gt.to_f32().unwrap()) {
                assert!((x - y).abs() < 1e-5, "{en}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn unsupported_backend_leaves_module_alone() {
        let mut module = build();
        assert_eq!(fuse_composites(&mut module, &|_| false), 0);
        assert_eq!(module.nodes.len(), 5);
    }

    #[test]
    fn gelu_with_second_use_is_not_fused() {
        let mut bld = GraphBuilder::new();
        let ty = TensorType::of(DataType::F32, &[4]);
        let x = bld.input("x", ty.clone());
        let u = bld.input("u", ty.clone());
        let g = bld
            .composite("Gelu", Attrs::new(), &[x], vec![ty.clone()])
            .unwrap()[0];
        let z = bld.mul(g, u).unwrap();
        bld.output("z", z);
        bld.output("g", g);
        let mut module = bld.finish().unwrap();
        assert_eq!(fuse_composites(&mut module, &|_| true), 0);
    }
}
