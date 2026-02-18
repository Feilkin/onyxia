//! DOT graph visualization for compiled dispatch models.
//!
//! Generates Graphviz DOT format showing the dispatch execution flow
//! with register routing information.

use crate::dispatch::CompiledModel;
use std::collections::HashMap;

/// Generate a DOT graph from a compiled dispatch model.
///
/// The graph shows:
/// - Each operation as a node labeled with op_type and node name
/// - Register routing: input/output registers for each operation
/// - Weight registers (pre-loaded constants)
/// - Model inputs/outputs
///
/// # Example
///
/// ```ignore
/// let dot = to_dispatch_dot(&compiled_model);
/// std::fs::write("dispatch.dot", dot)?;
/// // Render with: dot -Tpng dispatch.dot -o dispatch.png
/// ```
pub fn to_dispatch_dot(model: &CompiledModel) -> String {
    let mut dot = String::new();
    dot.push_str("digraph dispatch_graph {\n");
    dot.push_str("  rankdir=TB;\n");
    dot.push_str("  node [shape=box, style=rounded];\n\n");

    // Track which registers are used
    let mut register_info: HashMap<usize, Vec<String>> = HashMap::new();

    // Add model inputs
    for (name, reg) in &model.input_registers {
        register_info
            .entry(*reg)
            .or_default()
            .push(format!("INPUT({})", name));
    }

    // Add weight registers
    for weight in &model.weight_registers {
        register_info
            .entry(weight.register)
            .or_default()
            .push(format!("WEIGHT({:?}x{:?})", weight.dtype, weight.shape));
    }

    // Add model outputs
    for (name, reg) in &model.output_registers {
        register_info
            .entry(*reg)
            .or_default()
            .push(format!("OUTPUT({})", name));
    }

    // Create nodes for each operation
    for (idx, entry) in model.entries.iter().enumerate() {
        let op_id = format!("op_{}", idx);
        let label = if entry.node_name.is_empty() {
            format!("{} [#{}]", entry.name, idx)
        } else {
            format!("{} [#{}]\\n{}", entry.name, idx, entry.node_name)
        };

        // Build register info for this operation
        let mut reg_info = String::new();

        if !entry.input_regs.is_empty() {
            reg_info.push_str("\\nIN: ");
            reg_info.push_str(
                &entry
                    .input_regs
                    .iter()
                    .map(|r| format!("r{}", r))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
        }

        if !entry.output_regs.is_empty() {
            reg_info.push_str("\\nOUT: ");
            reg_info.push_str(
                &entry
                    .output_regs
                    .iter()
                    .map(|r| format!("r{}", r))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
        }

        dot.push_str(&format!("  {} [label=\"{}{}\"];\n", op_id, label, reg_info));

        // Record that this operation writes to its output registers
        for &out_reg in &entry.output_regs {
            register_info
                .entry(out_reg)
                .or_default()
                .push(format!("op_{}", idx));
        }
    }

    // Create edges based on register dependencies
    for (idx, entry) in model.entries.iter().enumerate() {
        let op_id = format!("op_{}", idx);

        // For each input register, find which operation produced it
        for &in_reg in &entry.input_regs {
            // Find the producer of this register
            let producer = find_register_producer(in_reg, idx, model);

            if let Some(producer_id) = producer {
                // Different edge styles based on what produced the register
                let edge_label = format!("r{}", in_reg);
                let edge_style = if model.weight_registers.iter().any(|w| w.register == in_reg) {
                    "style=dashed, color=blue"
                } else if model.input_registers.iter().any(|(_, r)| *r == in_reg) {
                    "style=bold, color=green"
                } else {
                    "color=black"
                };

                dot.push_str(&format!(
                    "  {} -> {} [label=\"{}\", {}];\n",
                    producer_id, op_id, edge_label, edge_style
                ));
            }
        }
    }

    // Add legend
    dot.push_str("\n  // Legend\n");
    dot.push_str("  subgraph cluster_legend {\n");
    dot.push_str("    label=\"Legend\";\n");
    dot.push_str("    style=dashed;\n");
    dot.push_str("    legend_op [label=\"Operation [#idx]\\nIN: input regs\\nOUT: output regs\", shape=box];\n");
    dot.push_str("    legend_in [label=\"Input\", style=bold, color=green];\n");
    dot.push_str("    legend_weight [label=\"Weight\", style=dashed, color=blue];\n");
    dot.push_str("    legend_compute [label=\"Computed\", color=black];\n");
    dot.push_str("  }\n");

    dot.push_str("}\n");
    dot
}

/// Find which operation or input produced a given register.
///
/// Returns:
/// - `Some("op_N")` if operation N wrote to this register
/// - `Some("input_NAME")` if it's a model input
/// - `Some("weight_N")` if it's a weight register
/// - `None` if the register is empty (error case)
fn find_register_producer(reg: usize, before_idx: usize, model: &CompiledModel) -> Option<String> {
    // Check if it's a model input
    if let Some((name, _)) = model.input_registers.iter().find(|(_, r)| *r == reg) {
        return Some(format!("input_{}", sanitize_name(name)));
    }

    // Check if it's a weight register
    if model.weight_registers.iter().any(|w| w.register == reg) {
        return Some(format!("weight_{}", reg));
    }

    // Find the most recent operation that wrote to this register
    for (idx, entry) in model.entries[..before_idx].iter().enumerate().rev() {
        if entry.output_regs.contains(&reg) {
            return Some(format!("op_{}", idx));
        }
    }

    None
}

/// Sanitize a name for use in DOT format.
fn sanitize_name(name: &str) -> String {
    name.replace('/', "_")
        .replace('.', "_")
        .replace(':', "_")
        .replace('-', "_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::{DispatchEntry, WeightRegister};
    use crate::types::DataType;

    // Mock OpDispatch for testing
    struct MockOp;
    impl crate::dispatch::OpDispatch for MockOp {
        fn dispatch(
            &self,
            _inputs: Vec<crate::dispatch::RuntimeTensor>,
            _ctx: &mut crate::dispatch::DispatchCtx,
        ) -> crate::Result<Vec<crate::dispatch::RuntimeTensor>> {
            Ok(vec![])
        }
    }

    #[test]
    fn test_to_dispatch_dot_generates_valid_output() {
        let mut model = CompiledModel {
            entries: vec![
                DispatchEntry {
                    op: Box::new(MockOp),
                    input_regs: vec![0, 1],
                    output_regs: vec![2],
                    name: "Add".to_string(),
                    node_name: "/model/add_0".to_string(),
                },
                DispatchEntry {
                    op: Box::new(MockOp),
                    input_regs: vec![2],
                    output_regs: vec![3],
                    name: "Mul".to_string(),
                    node_name: "/model/mul_0".to_string(),
                },
            ],
            num_registers: 4,
            input_registers: vec![("input".to_string(), 0)],
            output_registers: vec![("output".to_string(), 3)],
            weight_registers: vec![WeightRegister {
                register: 1,
                data: vec![0u8; 16],
                shape: vec![4],
                dtype: DataType::F32,
            }],
            metadata: Default::default(),
        };

        let dot = to_dispatch_dot(&model);

        // Verify basic structure
        assert!(dot.contains("digraph dispatch_graph"));
        assert!(dot.contains("op_0"));
        assert!(dot.contains("op_1"));
        assert!(dot.contains("Add"));
        assert!(dot.contains("Mul"));
        assert!(dot.contains("IN: r0, r1"));
        assert!(dot.contains("OUT: r2"));
    }

    #[test]
    fn test_sanitize_name() {
        assert_eq!(sanitize_name("/model/add:0"), "_model_add_0");
        assert_eq!(sanitize_name("input.tensor"), "input_tensor");
    }
}
