//! ConstantKernel implementation for ONNX Constant nodes.

use crate::error::Result;
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::Step;
use onyxia_onnx::TensorShape;

/// Kernel for ONNX Constant operator.
///
/// Constant nodes produce a tensor with fixed data that is known at plan time.
/// The parser already extracts this data into `TensorInfo.initializer`, which
/// the runtime automatically uploads during buffer allocation. Therefore, this
/// kernel emits zero GPU steps.
pub struct ConstantKernel;

impl OpKernel for ConstantKernel {
    fn name(&self) -> &str {
        "Constant"
    }

    fn infer_output_shapes(
        &self,
        node: &onyxia_onnx::Node,
        _input_shapes: &[TensorShape],
    ) -> Result<Vec<TensorShape>> {
        // Constant nodes have no inputs. The output shape is already set
        // by the parser when it extracted the tensor from the node's "value" attribute.
        // We just return it as-is.
        if node.outputs.is_empty() {
            return Err(crate::error::CodegenError::InvalidShape(
                "Constant node must have at least one output".to_string(),
            ));
        }

        // The shape is already in the graph's tensor registry.
        // Return Unknown here since the actual shape is already set during parsing.
        // The shape inference system will use the existing shape from the tensor registry.
        Ok(vec![TensorShape::Unknown])
    }

    fn plan(&self, _ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // No-op: constant data is in TensorInfo.initializer,
        // which the runtime uploads during buffer allocation.
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind};
    use std::collections::HashMap;

    #[test]
    fn test_constant_kernel_empty_steps() {
        // Create a minimal graph with a Constant node
        let mut graph = Graph::new();

        // Add constant output tensor with initializer data
        let constant_data: Vec<u8> = vec![0, 0, 0x80, 0x3f]; // 1.0f32 in little-endian
        let output_id = graph.add_tensor(TensorInfo {
            name: "constant_out".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1]),
            kind: TensorKind::Intermediate,
            initializer: Some(constant_data),
        });

        let node = Node {
            name: "Constant_0".to_string(),
            op_type: "Constant".to_string(),
            inputs: vec![],
            outputs: vec!["constant_out".to_string()],
            attributes: HashMap::new(),
            domain: String::new(),
        };

        let mut shaders = Vec::new();
        let output_ids = [output_id];
        let dynamic_dims: HashMap<String, usize> = HashMap::new();
        let mut ctx =
            PlanContext::for_test(&node, &graph, &[], &output_ids, &dynamic_dims, &mut shaders);

        let kernel = ConstantKernel;
        let steps = kernel.plan(&mut ctx).expect("plan should succeed");

        // Verify that no GPU steps are emitted
        assert_eq!(
            steps.len(),
            0,
            "ConstantKernel should emit zero steps (data is in initializer)"
        );
    }

    #[test]
    fn test_constant_kernel_shape_inference() {
        let node = Node {
            name: "Constant_0".to_string(),
            op_type: "Constant".to_string(),
            inputs: vec![],
            outputs: vec!["constant_out".to_string()],
            attributes: HashMap::new(),
            domain: String::new(),
        };

        let kernel = ConstantKernel;
        let shapes = kernel
            .infer_output_shapes(&node, &[])
            .expect("shape inference should succeed");

        // Should return one output shape (Unknown, since actual shape is in tensor registry)
        assert_eq!(shapes.len(), 1, "Constant should have one output");
        assert!(
            matches!(shapes[0], TensorShape::Unknown),
            "Shape should be Unknown (delegated to tensor registry)"
        );
    }
}
