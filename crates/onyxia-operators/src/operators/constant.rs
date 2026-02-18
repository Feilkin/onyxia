//! Constant operator.

use onyxia_core::{
    CompileCtx, DataType, DispatchCtx, Error, OpDispatch, Operator, Result, RuntimeTensor,
};

/// Constant operator - produces a constant tensor from ONNX attributes.
///
/// ONNX opset 25:
/// - **Inputs:** None
/// - **Outputs:** output (T) - Constant tensor value
/// - **Attributes:** value (tensor) or typed variants (value_float, value_ints, etc.)
///
/// The constant data is embedded in the ONNX model and extracted during parsing.
/// At runtime, the constant is uploaded to GPU on first dispatch.
pub struct ConstantOp;

/// Runtime dispatch for Constant - uploads constant data to GPU.
struct ConstantDispatch {
    /// Raw constant data bytes.
    data: Vec<u8>,
    /// Output tensor shape.
    shape: Vec<usize>,
    /// Output tensor data type.
    dtype: DataType,
}

impl OpDispatch for ConstantDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> Result<Vec<RuntimeTensor>> {
        // Constant has no inputs
        if !inputs.is_empty() {
            return Err(Error::Compilation(
                "Constant operator should have no inputs".into(),
            ));
        }

        // Upload constant data to GPU and return
        let tensor = ctx.upload_tensor(&self.data, &self.shape, self.dtype)?;
        Ok(vec![tensor])
    }
}

impl Operator for ConstantOp {
    fn name(&self) -> &str {
        "Constant"
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        // Constant nodes have no inputs, one output
        if ctx.input_count() != 0 {
            return Err(Error::Compilation(
                "Constant operator should have no inputs".into(),
            ));
        }
        if ctx.output_count() != 1 {
            return Err(Error::Compilation(
                "Constant operator should have exactly one output".into(),
            ));
        }

        // Get output tensor information
        let output_id =
            *ctx.node.outputs().first().ok_or_else(|| {
                Error::Compilation("Constant operator has no output edges".into())
            })?;
        let output_tensor = ctx.graph.tensor(output_id)?;

        // Extract constant data from the output tensor's EdgeData
        let data = match &output_tensor.data {
            onyxia_core::EdgeData::Constant(value) => value.to_bytes(),
            onyxia_core::EdgeData::Initializer(bytes) => bytes.clone(),
            onyxia_core::EdgeData::Runtime => {
                return Err(Error::Compilation(
                    "Constant node output must have constant or initializer data".into(),
                ));
            }
        };

        // Extract shape
        let shape = match output_tensor.shape.as_static() {
            Some(dims) => dims.to_vec(),
            None => {
                return Err(Error::Compilation(
                    "Constant node output must have static shape".into(),
                ));
            }
        };

        Ok(Box::new(ConstantDispatch {
            data,
            shape,
            dtype: output_tensor.dtype,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_core::IrGraph;
    use onyxia_onnx::{DataType as OnnxDataType, Graph, Node, TensorInfo, TensorKind};

    #[test]
    fn test_constant_dispatch_creation() {
        // Create a simple ONNX graph with a Constant node
        let mut onnx_graph = Graph::new();

        // Constant output with initializer data (1.0f32, 2.0f32, 3.0f32, 4.0f32)
        let constant_data = vec![
            0x00, 0x00, 0x80, 0x3f, // 1.0f32
            0x00, 0x00, 0x00, 0x40, // 2.0f32
            0x00, 0x00, 0x40, 0x40, // 3.0f32
            0x00, 0x00, 0x80, 0x40, // 4.0f32
        ];
        onnx_graph.add_tensor(TensorInfo {
            name: "const_output".to_string(),
            dtype: OnnxDataType::F32,
            shape: onyxia_onnx::TensorShape::Static(vec![4]),
            kind: TensorKind::Intermediate,
            initializer: Some(constant_data.clone()),
        });

        let mut node = Node::new("Constant");
        node.outputs = vec!["const_output".to_string()];
        onnx_graph.nodes.push(node);

        onnx_graph.outputs = vec!["const_output".to_string()];

        // Convert to IR
        let ir_graph = IrGraph::from_onnx(&onnx_graph).expect("IR conversion should succeed");

        // Verify the output tensor has initializer data
        let output_id = ir_graph.tensor_by_name("const_output").unwrap();
        let output_tensor = ir_graph.tensor(output_id).unwrap();

        assert!(matches!(
            output_tensor.data,
            onyxia_core::EdgeData::Initializer(_)
        ));
    }
}
