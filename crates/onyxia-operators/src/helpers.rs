//! Helper functions for operator implementation.
//!
//! Provides common utilities for shape inference and broadcasting.

use onyxia_core::{Error, InferenceCtx, Result, TensorShape};

/// Broadcast multiple shapes to a common output shape.
///
/// Implements NumPy-style broadcasting rules:
/// - Shapes are aligned from the rightmost dimension
/// - Dimensions match if they are equal or one of them is 1
/// - Missing dimensions in shorter shapes are treated as 1
///
/// # Example
///
/// ```text
/// [2, 3, 4] + [3, 4]    -> [2, 3, 4]
/// [2, 3, 4] + [2, 1, 4] -> [2, 3, 4]
/// [8, 1, 6, 1] + [7, 1, 5] -> [8, 7, 6, 5]
/// ```
pub fn broadcast_shapes(shapes: &[&[usize]]) -> Result<Vec<usize>> {
    if shapes.is_empty() {
        return Ok(vec![]);
    }

    // Find the maximum rank
    let max_rank = shapes.iter().map(|s| s.len()).max().unwrap();

    // Start with all dimensions as 1
    let mut result = vec![1; max_rank];

    // Process each shape
    for shape in shapes {
        let rank = shape.len();

        // Iterate from the rightmost dimension
        for i in 0..max_rank {
            // Get the dimension from the current shape (or 1 if out of bounds)
            let shape_dim = if i < rank { shape[rank - 1 - i] } else { 1 };

            // Get the current result dimension
            let result_idx = max_rank - 1 - i;
            let dim = result[result_idx];

            // Apply broadcasting rules
            if dim == 1 {
                result[result_idx] = shape_dim;
            } else if shape_dim != 1 && shape_dim != dim {
                return Err(Error::ShapeInference(format!(
                    "Cannot broadcast shapes: dimension mismatch at position {} (expected {} or 1, got {})",
                    result_idx, dim, shape_dim
                )));
            }
        }
    }

    Ok(result)
}

/// Infer output shape for elementwise operations via broadcasting.
///
/// Collects all non-absent input shapes and broadcasts them to a common output shape.
/// Returns `TensorShape::Unknown` if any input is unknown.
pub fn infer_elementwise_broadcast(ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
    // Collect all non-Absent input shapes
    let mut static_shapes = Vec::new();
    for i in 0..ctx.input_count() {
        let shape = ctx.input_shape(i)?;
        match shape {
            TensorShape::Static(dims) => static_shapes.push(dims),
            TensorShape::Symbolic(_) => {
                // Symbolic shapes should have been resolved by now, but we'll handle them
                return Err(Error::ShapeInference(
                    "Symbolic shapes should be resolved before shape inference".to_string(),
                ));
            }
            TensorShape::Unknown => {
                return Err(Error::ShapeInference(
                    "Unknown shapes should be resolved before shape inference".to_string(),
                ));
            }
            TensorShape::Absent => continue, // Skip absent (optional) inputs
        }
    }

    if static_shapes.is_empty() {
        return Err(Error::ShapeInference(
            "No non-absent inputs for elementwise operation".to_string(),
        ));
    }

    // Convert Vec<Vec<usize>> to Vec<&[usize]> for broadcast_shapes
    let shape_refs: Vec<&[usize]> = static_shapes.iter().map(|v| v.as_slice()).collect();
    let result_dims = broadcast_shapes(&shape_refs)?;
    Ok(vec![TensorShape::Static(result_dims)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shapes_same_rank() {
        // [2, 3, 4] and [2, 3, 4] -> [2, 3, 4]
        assert_eq!(
            broadcast_shapes(&[&[2, 3, 4], &[2, 3, 4]]).unwrap(),
            vec![2, 3, 4]
        );

        // [2, 3, 4] and [2, 1, 4] -> [2, 3, 4]
        assert_eq!(
            broadcast_shapes(&[&[2, 3, 4], &[2, 1, 4]]).unwrap(),
            vec![2, 3, 4]
        );

        // [2, 3, 4] and [1, 3, 1] -> [2, 3, 4]
        assert_eq!(
            broadcast_shapes(&[&[2, 3, 4], &[1, 3, 1]]).unwrap(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn test_broadcast_shapes_different_rank() {
        // [2, 3, 4, 5] and [5] -> [2, 3, 4, 5]
        assert_eq!(
            broadcast_shapes(&[&[2, 3, 4, 5], &[5]]).unwrap(),
            vec![2, 3, 4, 5]
        );

        // [1, 4, 5] and [2, 3, 1, 1] -> [2, 3, 4, 5]
        assert_eq!(
            broadcast_shapes(&[&[1, 4, 5], &[2, 3, 1, 1]]).unwrap(),
            vec![2, 3, 4, 5]
        );

        // [8, 1, 6, 1] and [7, 1, 5] -> [8, 7, 6, 5]
        assert_eq!(
            broadcast_shapes(&[&[8, 1, 6, 1], &[7, 1, 5]]).unwrap(),
            vec![8, 7, 6, 5]
        );
    }

    #[test]
    fn test_broadcast_shapes_scalar() {
        // [] and [2, 3] -> [2, 3] (scalar broadcasts to any shape)
        assert_eq!(broadcast_shapes(&[&[], &[2, 3]]).unwrap(), vec![2, 3]);

        // [2, 3] and [] -> [2, 3]
        assert_eq!(broadcast_shapes(&[&[2, 3], &[]]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn test_broadcast_shapes_incompatible() {
        // [3] and [4] -> error (neither is 1)
        assert!(broadcast_shapes(&[&[3], &[4]]).is_err());

        // [2, 3] and [2, 4] -> error
        assert!(broadcast_shapes(&[&[2, 3], &[2, 4]]).is_err());
    }

    #[test]
    fn test_broadcast_shapes_multiple_inputs() {
        // [2, 3, 4], [3, 4], [4] -> [2, 3, 4]
        assert_eq!(
            broadcast_shapes(&[&[2, 3, 4], &[3, 4], &[4]]).unwrap(),
            vec![2, 3, 4]
        );
    }
}
