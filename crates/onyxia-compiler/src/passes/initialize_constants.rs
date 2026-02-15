//! Initialize constants pass.
//!
//! This pass runs before shape inference to populate tensor values from initializers.
//! This is necessary for operators like Reshape, Slice, Expand, etc. that need constant
//! shape/parameter inputs during shape inference.

use onyxia_core::{Error, IrGraph, OperatorRegistry, Pass, Result, Stage, TensorValue};

/// Pass that initializes tensor values from initializers.
///
/// This pass populates `TensorDef.value` from `TensorDef.initializer` for all
/// tensors that have initializer data. This happens before shape inference so that
/// operators requiring constant inputs (like Reshape) can access them during shape
/// inference.
pub struct InitializeConstantsPass;

impl InitializeConstantsPass {
    /// Create a new initialize constants pass.
    pub fn new() -> Self {
        Self
    }

    /// Initialize tensor values from initializers.
    fn initialize_constants(&self, graph: &mut IrGraph) -> Result<bool> {
        let mut changed = false;

        // Iterate over all tensors
        for i in 0..graph.tensor_count() {
            let tensor_id = onyxia_core::IrTensorId::new(i);
            let tensor = graph.tensor(tensor_id)?;

            // Skip if already has a value or no initializer
            if tensor.has_value() || !tensor.has_initializer() {
                continue;
            }

            // Parse initializer bytes into TensorValue
            let initializer = tensor.initializer.as_ref().unwrap();
            let shape = match &tensor.shape {
                onyxia_core::TensorShape::Static(dims) => dims.clone(),
                _ => {
                    // Can't fold tensors with non-static shapes at this stage
                    continue;
                }
            };

            let value = TensorValue::from_bytes(initializer, tensor.dtype, &shape)?;

            // Store the value
            graph.tensor_mut(tensor_id)?.value = Some(value);
            changed = true;
        }

        Ok(changed)
    }
}

impl Pass for InitializeConstantsPass {
    fn name(&self) -> &str {
        "initialize_constants"
    }

    fn stage(&self) -> Stage {
        Stage::Resolution // Run in the Resolution stage, before shape inference
    }

    fn run(&self, graph: &mut IrGraph, _registry: &OperatorRegistry) -> Result<bool> {
        self.initialize_constants(graph)
    }
}

impl Default for InitializeConstantsPass {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_core::{DataType, IrTensorId, TensorDef, TensorKind, TensorShape};

    #[test]
    fn test_initialize_from_initializer() {
        let mut graph = IrGraph::new();

        // Add a tensor with an initializer
        let shape_data: Vec<u8> = vec![6, 0, 0, 0, 0, 0, 0, 0]; // i64 value: 6
        let mut tensor = TensorDef::new(
            "shape".to_string(),
            DataType::I64,
            TensorShape::Static(vec![1]),
            TensorKind::Weight,
        );
        tensor.initializer = Some(shape_data);
        let tensor_id = graph.add_tensor(tensor);

        // Run the pass
        let registry = onyxia_core::OperatorRegistry::new();
        let pass = InitializeConstantsPass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();

        assert!(changed, "Pass should report changes");

        // Check that the tensor now has a value
        let tensor = graph.tensor(tensor_id).unwrap();
        assert!(
            tensor.has_value(),
            "Tensor should have a value after initialization"
        );

        let value = tensor.value.as_ref().unwrap();
        assert_eq!(value.as_i64(), Some(&[6][..]));
    }

    #[test]
    fn test_skip_already_initialized() {
        let mut graph = IrGraph::new();

        // Add a tensor with both initializer and value
        let shape_data: Vec<u8> = vec![6, 0, 0, 0, 0, 0, 0, 0];
        let mut tensor = TensorDef::new(
            "shape".to_string(),
            DataType::I64,
            TensorShape::Static(vec![1]),
            TensorKind::Weight,
        );
        tensor.initializer = Some(shape_data);
        tensor.value = Some(TensorValue::new(
            onyxia_core::TensorData::I64(vec![42]),
            vec![1],
            DataType::I64,
        ));
        graph.add_tensor(tensor);

        // Run the pass
        let registry = onyxia_core::OperatorRegistry::new();
        let pass = InitializeConstantsPass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();

        // Should not change anything
        assert!(
            !changed,
            "Pass should not change already initialized tensors"
        );
    }

    #[test]
    fn test_skip_non_static_shape() {
        let mut graph = IrGraph::new();

        // Add a tensor with symbolic shape
        let shape_data: Vec<u8> = vec![6, 0, 0, 0, 0, 0, 0, 0];
        let mut tensor = TensorDef::new(
            "shape".to_string(),
            DataType::I64,
            TensorShape::Symbolic(vec![onyxia_core::SymbolicDim::Fixed(1)]),
            TensorKind::Weight,
        );
        tensor.initializer = Some(shape_data);
        graph.add_tensor(tensor);

        // Run the pass
        let registry = onyxia_core::OperatorRegistry::new();
        let pass = InitializeConstantsPass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();

        // Should not fail, just skip
        assert!(!changed, "Pass should skip non-static shapes");
    }
}
