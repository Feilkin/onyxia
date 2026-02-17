//! Broadcasting shape helper for runtime operations.

use crate::{Error, Result};

/// Compute NumPy-style broadcast output shape from two input shapes.
///
/// Implements NumPy broadcasting rules:
/// - Shapes are aligned from the rightmost dimension
/// - Dimensions match if they are equal or one of them is 1
/// - Missing dimensions in shorter shapes are treated as 1
///
/// # Example
///
/// ```text
/// broadcast_shape(&[2, 3, 4], &[3, 4])    -> [2, 3, 4]
/// broadcast_shape(&[2, 3, 4], &[2, 1, 4]) -> [2, 3, 4]
/// broadcast_shape(&[8, 1, 6, 1], &[7, 1, 5]) -> [8, 7, 6, 5]
/// ```
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let max_rank = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_rank);

    for i in 0..max_rank {
        let da = if i < max_rank - a.len() {
            1
        } else {
            a[i - (max_rank - a.len())]
        };
        let db = if i < max_rank - b.len() {
            1
        } else {
            b[i - (max_rank - b.len())]
        };

        if da == db {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else if db == 1 {
            result.push(da);
        } else {
            return Err(Error::Shape(format!(
                "Cannot broadcast shapes {:?} and {:?} at dimension {i}",
                a, b
            )));
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_same_shape() {
        assert_eq!(
            broadcast_shape(&[2, 3, 4], &[2, 3, 4]).unwrap(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn test_broadcast_missing_dims() {
        assert_eq!(broadcast_shape(&[2, 3, 4], &[3, 4]).unwrap(), vec![2, 3, 4]);
    }

    #[test]
    fn test_broadcast_ones() {
        assert_eq!(
            broadcast_shape(&[2, 3, 4], &[2, 1, 4]).unwrap(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn test_broadcast_complex() {
        assert_eq!(
            broadcast_shape(&[8, 1, 6, 1], &[7, 1, 5]).unwrap(),
            vec![8, 7, 6, 5]
        );
    }

    #[test]
    fn test_broadcast_incompatible() {
        let result = broadcast_shape(&[2, 3], &[2, 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_scalar() {
        assert_eq!(broadcast_shape(&[1], &[3, 4]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shape(&[5, 6], &[1]).unwrap(), vec![5, 6]);
    }
}
