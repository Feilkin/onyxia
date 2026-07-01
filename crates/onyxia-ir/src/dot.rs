//! Graphviz DOT export for debugging and documentation.

use crate::graph::{Module, NodeKind, Origin};
use std::fmt::Write;

/// Render the module as a DOT digraph.
///
/// Inputs are ellipses, constants are grey boxes (elided unless small),
/// primitive nodes are boxes, composites are double boxes. Edges are
/// labeled with value types.
pub fn to_dot(module: &Module) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "digraph onyxia_ir {{");
    let _ = writeln!(out, "  rankdir=TB;");
    let _ = writeln!(out, "  node [fontname=\"monospace\", fontsize=10];");

    // Graph inputs.
    for (name, id) in &module.inputs {
        let _ = writeln!(
            out,
            "  v{} [label=\"{}\\n{}\", shape=ellipse];",
            id.index(),
            escape(name),
            module.value(*id).ty
        );
    }

    // Nodes.
    for (i, node) in module.nodes.iter().enumerate() {
        let (label, shape) = match &node.kind {
            NodeKind::Prim(p) => (p.name().to_string(), "box"),
            NodeKind::Composite(c) => (c.name.clone(), "Mdiamond"),
        };
        let src = node
            .loc
            .name
            .as_deref()
            .map(|n| format!("\\n{}", escape(n)))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "  n{i} [label=\"{}{}\", shape={shape}];",
            escape(&label),
            src
        );

        for &input in &node.inputs {
            let def = module.value(input);
            match def.origin {
                Origin::Node { node: producer, .. } => {
                    let _ = writeln!(
                        out,
                        "  n{} -> n{i} [label=\"{}\"];",
                        producer.index(),
                        def.ty
                    );
                }
                Origin::Input => {
                    let _ = writeln!(out, "  v{} -> n{i};", input.index());
                }
                Origin::Const(cid) => {
                    // Render constants inline, once per use, tersely.
                    let _ = writeln!(
                        out,
                        "  c{}_{i} [label=\"const {}\", shape=box, \
                         style=filled, fillcolor=lightgrey];",
                        cid.index(),
                        module.consts.ty(cid)
                    );
                    let _ = writeln!(out, "  c{}_{i} -> n{i};", cid.index());
                }
            }
        }
    }

    // Graph outputs.
    for (name, id) in &module.outputs {
        let _ = writeln!(
            out,
            "  out_{} [label=\"{}\", shape=ellipse, style=bold];",
            id.index(),
            escape(name)
        );
        if let Origin::Node { node, .. } = module.value(*id).origin {
            let _ = writeln!(
                out,
                "  n{} -> out_{} [label=\"{}\"];",
                node.index(),
                id.index(),
                module.value(*id).ty
            );
        }
    }

    let _ = writeln!(out, "}}");
    out
}

fn escape(s: &str) -> String {
    s.replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use crate::builder::GraphBuilder;
    use crate::types::{DataType, TensorType};

    #[test]
    fn renders_valid_dot() {
        let mut b = GraphBuilder::new();
        let x = b.input("x", TensorType::of(DataType::F32, &[2]));
        let c = b.const_f32(&[1.0, 1.0], &[2]).unwrap();
        let y = b.add(x, c).unwrap();
        b.output("y", y);
        let m = b.finish().unwrap();
        let dot = super::to_dot(&m);
        assert!(dot.starts_with("digraph"));
        assert!(dot.contains("add"));
        assert!(dot.contains("const f32[2]"));
        assert!(dot.ends_with("}\n"));
    }
}
