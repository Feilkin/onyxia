//! Helper functions for operator implementation.
//!
//! Shape inference and broadcasting helpers have been removed.
//! These will be reimplemented for the dispatch model in task 043.

use onyxia_core::{Error, Result};

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
                return Err(Error::Compilation(format!(
                    "Cannot broadcast shapes: dimension mismatch at position {} (expected {} or 1, got {})",
                    result_idx, dim, shape_dim
                )));
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shapes() {
        // Test same-shape broadcasting
        assert_eq!(
            broadcast_shapes(&[&[2, 3, 4], &[2, 3, 4]]).unwrap(),
            vec![2, 3, 4]
        );

        // Test missing dimensions
        assert_eq!(
            broadcast_shapes(&[&[2, 3, 4], &[3, 4]]).unwrap(),
            vec![2, 3, 4]
        );

        // Test broadcasting dimension 1
        assert_eq!(
            broadcast_shapes(&[&[2, 3, 4], &[2, 1, 4]]).unwrap(),
            vec![2, 3, 4]
        );

        // Test multiple shapes
        assert_eq!(
            broadcast_shapes(&[&[8, 1, 6, 1], &[7, 1, 5]]).unwrap(),
            vec![8, 7, 6, 5]
        );
    }

    #[test]
    fn test_broadcast_incompatible() {
        // Test incompatible shapes
        let result = broadcast_shapes(&[&[2, 3], &[2, 4]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_multi() {
        // [2, 3, 4], [3, 4], [4] -> [2, 3, 4]
        assert_eq!(
            broadcast_shapes(&[&[2, 3, 4], &[3, 4], &[4]]).unwrap(),
            vec![2, 3, 4]
        );
    }
}
