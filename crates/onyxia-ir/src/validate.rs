//! Structural validation of a [`Module`].
//!
//! Checks everything the type system can't: id ranges, origin/output
//! back-pointer consistency, acyclicity, primitive arity, and — for
//! primitive nodes — that the declared output types match re-run shape
//! inference. Composites are checked structurally only (their types come
//! from the lowering registry, which this crate doesn't know about).

use crate::graph::{Module, NodeKind, Origin};
use crate::infer::infer_prim;
use crate::types::TensorType;
use crate::{Error, Result};

/// Validate a module, returning the first problem found.
pub fn validate(module: &Module) -> Result<()> {
    let num_values = module.values.len();
    let num_nodes = module.nodes.len();

    // Values: origins are in range and consistent.
    for (i, def) in module.values.iter().enumerate() {
        match def.origin {
            Origin::Input => {}
            Origin::Const(cid) => {
                if cid.index() >= module.consts.len() {
                    return Err(err_value(module, i, "const id out of range"));
                }
                let pool_ty = module.consts.ty(cid);
                if *pool_ty != def.ty {
                    return Err(err_value(
                        module,
                        i,
                        &format!("type {} disagrees with pool entry type {pool_ty}", def.ty),
                    ));
                }
            }
            Origin::Node { node, output } => {
                if node.index() >= num_nodes {
                    return Err(err_value(module, i, "producer node id out of range"));
                }
                let producer = module.node(node);
                let points_back = producer
                    .outputs
                    .get(output)
                    .is_some_and(|&v| v.index() == i);
                if !points_back {
                    return Err(err_value(
                        module,
                        i,
                        "producer node's outputs do not point back to this value",
                    ));
                }
            }
        }
    }

    // Nodes: ids in range, back-pointers, arity, inferred types.
    for (i, node) in module.nodes.iter().enumerate() {
        for &input in &node.inputs {
            if input.index() >= num_values {
                return Err(err_node(module, i, "input value id out of range"));
            }
        }
        for (out_idx, &output) in node.outputs.iter().enumerate() {
            if output.index() >= num_values {
                return Err(err_node(module, i, "output value id out of range"));
            }
            let back = module.value(output).origin;
            if back
                != (Origin::Node {
                    node: crate::graph::NodeId::from_index(i as u32),
                    output: out_idx,
                })
            {
                return Err(err_node(
                    module,
                    i,
                    &format!("output {out_idx} does not record this node as its origin"),
                ));
            }
        }

        if let NodeKind::Prim(prim) = &node.kind {
            let input_tys: Vec<&TensorType> =
                node.inputs.iter().map(|&v| &module.value(v).ty).collect();
            let inferred =
                infer_prim(prim, &input_tys).map_err(|e| err_node(module, i, &e.to_string()))?;
            if inferred.len() != node.outputs.len() {
                return Err(err_node(
                    module,
                    i,
                    &format!(
                        "produces {} outputs but inference expects {}",
                        node.outputs.len(),
                        inferred.len()
                    ),
                ));
            }
            for (out_idx, (ty, &out)) in inferred.iter().zip(&node.outputs).enumerate() {
                if *ty != module.value(out).ty {
                    return Err(err_node(
                        module,
                        i,
                        &format!(
                            "output {out_idx} declared as {} but inference says {ty}",
                            module.value(out).ty
                        ),
                    ));
                }
            }
        }
    }

    // I/O tables reference valid values with the right origins.
    for (name, id) in &module.inputs {
        if id.index() >= num_values {
            return Err(Error::InvalidGraph(format!(
                "input '{name}' has out-of-range value id"
            )));
        }
        if module.value(*id).origin != Origin::Input {
            return Err(Error::InvalidGraph(format!(
                "input '{name}' refers to a value whose origin is not Input"
            )));
        }
    }
    for (name, id) in &module.outputs {
        if id.index() >= num_values {
            return Err(Error::InvalidGraph(format!(
                "output '{name}' has out-of-range value id"
            )));
        }
    }

    // Acyclic.
    module.topo_order()?;

    Ok(())
}

fn err_value(module: &Module, index: usize, msg: &str) -> Error {
    let name = module.values[index].name.as_deref().unwrap_or("<unnamed>");
    Error::InvalidGraph(format!("value {index} ('{name}'): {msg}"))
}

fn err_node(module: &Module, index: usize, msg: &str) -> Error {
    let node = &module.nodes[index];
    let what = match &node.kind {
        NodeKind::Prim(p) => p.name().to_string(),
        NodeKind::Composite(c) => c.name.clone(),
    };
    let loc = node
        .loc
        .name
        .as_deref()
        .map(|n| format!(" (from {n})"))
        .unwrap_or_default();
    Error::InvalidGraph(format!("node {index} [{what}]{loc}: {msg}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::graph::{Node, NodeKind, SourceInfo, ValueDef};
    use crate::prim::{BinaryOp, Prim};
    use crate::types::DataType;

    #[test]
    fn valid_module_passes() {
        let mut b = GraphBuilder::new();
        let x = b.input("x", TensorType::of(DataType::F32, &[2]));
        let y = b.add(x, x).unwrap();
        b.output("y", y);
        // finish() validates internally; also validate explicitly.
        let m = b.finish().unwrap();
        assert!(validate(&m).is_ok());
    }

    #[test]
    fn detects_type_lies() {
        let mut b = GraphBuilder::new();
        let x = b.input("x", TensorType::of(DataType::F32, &[2]));
        let y = b.add(x, x).unwrap();
        b.output("y", y);
        let mut m = b.finish().unwrap();
        // Corrupt the declared output type behind inference's back.
        m.value_mut(y).ty = TensorType::of(DataType::F32, &[3]);
        let err = validate(&m).unwrap_err().to_string();
        assert!(err.contains("inference says"), "got: {err}");
    }

    #[test]
    fn detects_bad_backpointer() {
        let mut b = GraphBuilder::new();
        let x = b.input("x", TensorType::of(DataType::F32, &[2]));
        let y = b.add(x, x).unwrap();
        b.output("y", y);
        let mut m = b.finish().unwrap();
        // A value claiming to come from a node output that isn't it.
        m.add_value(ValueDef {
            name: Some("impostor".into()),
            ty: TensorType::of(DataType::F32, &[2]),
            origin: crate::graph::Origin::Node {
                node: crate::graph::NodeId::from_index(0),
                output: 0,
            },
            content: None,
        });
        let err = validate(&m).unwrap_err().to_string();
        assert!(err.contains("point back"), "got: {err}");
    }

    #[test]
    fn detects_arity_violation() {
        let mut b = GraphBuilder::new();
        let x = b.input("x", TensorType::of(DataType::F32, &[2]));
        let y = b.add(x, x).unwrap();
        b.output("y", y);
        let mut m = b.finish().unwrap();
        // Hand the Add node a third input.
        let Node { inputs, .. } = &mut m.nodes[0];
        inputs.push(x);
        assert!(validate(&m).is_err());
        // And make it a different prim with wrong arity entirely.
        m.nodes[0].inputs.pop();
        m.nodes[0].kind = NodeKind::Prim(Prim::Binary(BinaryOp::Add));
        assert!(validate(&m).is_ok());
        m.nodes[0].loc = SourceInfo {
            name: Some("/layer0/add".into()),
            op_type: Some("Add".into()),
        };
        // Error messages carry provenance.
        m.value_mut(y).ty = TensorType::of(DataType::F32, &[9]);
        let err = validate(&m).unwrap_err().to_string();
        assert!(err.contains("/layer0/add"), "got: {err}");
    }
}
