//! CastKernel implementation for type conversion.

use crate::error::{CodegenError, Result};
use crate::kernel::{OpKernel, PlanContext};
use crate::plan::{BindingDesc, Step};
use naga_oil::compose::ShaderDefValue;
use onyxia_onnx::{DataType, TensorShape};
use std::collections::HashMap;

/// Kernel for type casting (ONNX Cast operator).
///
/// Converts tensor data between types (e.g., I64 → F32, F32 → I32).
/// The target type is specified by the "to" attribute in the ONNX node.
pub struct CastKernel;

impl OpKernel for CastKernel {
    fn name(&self) -> &str {
        "Cast"
    }

    fn infer_output_shapes(
        &self,
        _node: &onyxia_onnx::Node,
        input_shapes: &[TensorShape],
    ) -> Result<Vec<TensorShape>> {
        // Cast preserves shape, only changes data type
        if input_shapes.is_empty() {
            return Err(CodegenError::InvalidShape(
                "Cast requires at least one input".to_string(),
            ));
        }
        Ok(vec![input_shapes[0].clone()])
    }

    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
        // Get input and output info
        let input_info = ctx.input_info(0)?;
        let output_info = ctx.output_info(0)?;

        // Get source and target data types
        let source_dtype = input_info.dtype;
        let target_dtype = output_info.dtype;

        // Optional: Validate 'to' attribute if present (ONNX Cast spec)
        // The 'to' attribute is an int representing the target ONNX data type code
        // In practice, the output tensor dtype is already set correctly by the parser
        if let Ok(to_code) = ctx.node.attr::<i64>("to") {
            let expected_dtype = onnx_dtype_to_datatype(to_code);
            if expected_dtype != target_dtype {
                return Err(CodegenError::InvalidShape(format!(
                    "Cast 'to' attribute ({:?}) doesn't match output tensor dtype ({:?})",
                    expected_dtype, target_dtype
                )));
            }
        }

        // Calculate number of elements based on output shape
        let output_shape = ctx.static_shape(&output_info.shape)?;
        let num_elements: usize = output_shape.iter().product();

        // Determine which shader variant to use
        let (shader_label, shader_def) = match (source_dtype, target_dtype) {
            (DataType::I64, DataType::F32) => ("cast_i64_to_f32", "CAST_I64_TO_F32"),
            (DataType::I32, DataType::F32) => ("cast_i32_to_f32", "CAST_I32_TO_F32"),
            (DataType::F32, DataType::I32) => ("cast_f32_to_i32", "CAST_F32_TO_I32"),
            (DataType::F32, DataType::F16) => ("cast_f32_to_f16", "CAST_F32_TO_F16"),
            _ => {
                return Err(CodegenError::UnsupportedOp(format!(
                    "Cast from {:?} to {:?} is not yet supported",
                    source_dtype, target_dtype
                )));
            }
        };

        // Configure workgroup size
        let workgroup_size: u32 = 256;
        let num_workgroups = (num_elements as u32 + workgroup_size - 1) / workgroup_size;

        // Prepare shader definitions
        let mut shader_defs = HashMap::new();
        shader_defs.insert(
            "WORKGROUP_SIZE".to_string(),
            ShaderDefValue::UInt(workgroup_size),
        );
        shader_defs.insert(shader_def.to_string(), ShaderDefValue::Bool(true));

        // Compile shader
        let shader_index = ctx.compile_shader(
            shader_label,
            include_str!("../../shaders/elementwise/cast.wgsl"),
            shader_defs,
        )?;

        // Encode immediate data (must match ImmediateConstants struct in shader)
        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(num_elements as u32).to_le_bytes());

        // Create dispatch step with bindings and immediates
        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                BindingDesc {
                    buffer: ctx.input(0),
                    read_only: true,
                },
                BindingDesc {
                    buffer: ctx.output(0),
                    read_only: false,
                },
            ],
            workgroups: [num_workgroups, 1, 1],
            immediates: Some(immediates_data),
        }])
    }
}

/// Convert ONNX TensorProto::DataType code to internal DataType.
///
/// ONNX data type codes (from onnx.proto):
/// - FLOAT = 1
/// - UINT8 = 2
/// - INT32 = 6
/// - INT64 = 7
/// - BOOL = 9
/// - FLOAT16 = 10
/// - UINT32 = 12
fn onnx_dtype_to_datatype(code: i64) -> DataType {
    match code {
        1 => DataType::F32,
        2 => DataType::U8,
        6 => DataType::I32,
        7 => DataType::I64,
        9 => DataType::Bool,
        10 => DataType::F16,
        12 => DataType::U32,
        _ => DataType::F32, // Default fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::BufferRef;
    use onyxia_onnx::{AttributeValue, Graph, Node, TensorInfo, TensorKind, TensorShape};
    use std::collections::HashMap;

    fn create_cast_test_graph(source_dtype: DataType, target_dtype: DataType) -> Graph {
        let mut graph = Graph::new();

        // Add input tensor
        graph.add_tensor(TensorInfo {
            name: "input".to_string(),
            dtype: source_dtype,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add output tensor
        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: target_dtype,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Output,
            initializer: None,
        });

        graph.inputs = vec!["input".to_string()];
        graph.outputs = vec!["output".to_string()];

        graph
    }

    #[test]
    fn test_cast_kernel_i64_to_f32() {
        let graph = create_cast_test_graph(DataType::I64, DataType::F32);
        let mut node = Node::new("Cast");
        node.inputs = vec!["input".to_string()];
        node.outputs = vec!["output".to_string()];

        // Add "to" attribute (ONNX FLOAT = 1)
        node.attributes
            .insert("to".to_string(), AttributeValue::Int(1));

        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let steps = CastKernel.plan(&mut ctx).expect("Planning should succeed");

        // Verify we got exactly one dispatch step
        assert_eq!(steps.len(), 1);

        match &steps[0] {
            Step::Dispatch {
                shader_index,
                bindings,
                workgroups,
                ..
            } => {
                // Verify shader was compiled
                assert_eq!(*shader_index, 0);

                // Verify bindings: 1 read-only input + 1 read-write output
                assert_eq!(bindings.len(), 2);

                assert_eq!(bindings[0].buffer, BufferRef::Tensor(0));
                assert!(bindings[0].read_only);

                assert_eq!(bindings[1].buffer, BufferRef::Tensor(1));
                assert!(!bindings[1].read_only);

                // Verify workgroup count: ceil(4 / 256) = 1
                assert_eq!(*workgroups, [1, 1, 1]);
            }
            _ => panic!("Expected Dispatch step"),
        }

        // Verify shader was compiled
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].label, "cast_i64_to_f32");
        assert_eq!(shaders[0].entry_point, "main");
    }

    #[test]
    fn test_cast_kernel_i32_to_f32() {
        let graph = create_cast_test_graph(DataType::I32, DataType::F32);
        let mut node = Node::new("Cast");
        node.inputs = vec!["input".to_string()];
        node.outputs = vec!["output".to_string()];

        // Add "to" attribute (ONNX FLOAT = 1)
        node.attributes
            .insert("to".to_string(), AttributeValue::Int(1));

        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let steps = CastKernel.plan(&mut ctx).expect("Planning should succeed");

        assert_eq!(steps.len(), 1);

        // Verify shader label is correct
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].label, "cast_i32_to_f32");
    }

    #[test]
    fn test_cast_kernel_f32_to_i32() {
        let graph = create_cast_test_graph(DataType::F32, DataType::I32);
        let mut node = Node::new("Cast");
        node.inputs = vec!["input".to_string()];
        node.outputs = vec!["output".to_string()];

        // Add "to" attribute (ONNX INT32 = 6)
        node.attributes
            .insert("to".to_string(), AttributeValue::Int(6));

        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let steps = CastKernel.plan(&mut ctx).expect("Planning should succeed");

        assert_eq!(steps.len(), 1);

        // Verify shader label is correct
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].label, "cast_f32_to_i32");
    }

    #[test]
    fn test_cast_kernel_unsupported_conversion() {
        let graph = create_cast_test_graph(DataType::F32, DataType::U8);
        let mut node = Node::new("Cast");
        node.inputs = vec!["input".to_string()];
        node.outputs = vec!["output".to_string()];

        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let result = CastKernel.plan(&mut ctx);

        assert!(result.is_err(), "Should fail for unsupported conversion");
        match result {
            Err(CodegenError::UnsupportedOp(msg)) => {
                assert!(
                    msg.contains("Cast from"),
                    "Error message should mention Cast"
                );
            }
            _ => panic!("Expected UnsupportedOp error"),
        }
    }

    #[test]
    fn test_cast_kernel_larger_tensor() {
        let mut graph = Graph::new();

        // Create larger tensor (1024 elements)
        graph.add_tensor(TensorInfo {
            name: "input".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![32, 32]),
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![32, 32]),
            kind: TensorKind::Output,
            initializer: None,
        });

        let mut node = Node::new("Cast");
        node.inputs = vec!["input".to_string()];
        node.outputs = vec!["output".to_string()];
        node.attributes
            .insert("to".to_string(), AttributeValue::Int(1));

        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let steps = CastKernel.plan(&mut ctx).expect("Planning should succeed");

        match &steps[0] {
            Step::Dispatch { workgroups, .. } => {
                // 1024 elements / 256 = 4 workgroups
                assert_eq!(*workgroups, [4, 1, 1]);
            }
            _ => panic!("Expected Dispatch step"),
        }
    }

    #[test]
    fn test_cast_kernel_reads_to_attribute() {
        let graph = create_cast_test_graph(DataType::I64, DataType::F32);
        let mut node = Node::new("Cast");
        node.inputs = vec!["input".to_string()];
        node.outputs = vec!["output".to_string()];

        // Add "to" attribute matching the output dtype (ONNX FLOAT = 1)
        node.attributes
            .insert("to".to_string(), AttributeValue::Int(1));

        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        // Should succeed because 'to' attribute matches output dtype
        let result = CastKernel.plan(&mut ctx);
        assert!(
            result.is_ok(),
            "Planning should succeed when 'to' attribute matches output dtype"
        );
    }

    #[test]
    fn test_cast_kernel_validates_to_attribute() {
        let graph = create_cast_test_graph(DataType::I64, DataType::F32);
        let mut node = Node::new("Cast");
        node.inputs = vec!["input".to_string()];
        node.outputs = vec!["output".to_string()];

        // Add wrong "to" attribute (ONNX INT32 = 6, but output is F32)
        node.attributes
            .insert("to".to_string(), AttributeValue::Int(6));

        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        // Should fail because 'to' attribute doesn't match output dtype
        let result = CastKernel.plan(&mut ctx);
        assert!(
            result.is_err(),
            "Planning should fail when 'to' attribute doesn't match output dtype"
        );
    }
}
