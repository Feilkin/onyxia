//! Shape and type inference for primitives.
//!
//! One total function over the closed primitive set: [`infer_prim`]. There
//! is no default arm and no per-operator opt-in — adding a primitive without
//! a shape rule is a compile error. (An open operator trait would make every
//! analysis depend on every operator's cooperation; a closed enum makes
//! analyses total by construction.)
//!
//! Symbolic policy: two dimensions are considered equal when their canonical
//! polynomials are equal. Dims that *could* coincide at run time but are not
//! provably equal (`S` vs `T`) are inference errors — well-formed models use
//! the same symbol for dims that must match. The single exception is
//! `Reshape`'s element-count check, which downgrades to a run-time check
//! when either side involves symbols that prevent a proof (late-bound dims).

use crate::dim::{DimExpr, SymbolicShape};
use crate::prim::{BinaryOp, Prim, UnaryOp};
use crate::types::{DataType, TensorType};
use crate::{Error, Result};

/// Infer output types from input types for a primitive.
///
/// `inputs` must match the primitive's arity. Every primitive currently has
/// exactly one output; the return type is a `Vec` so that adding a
/// multi-output primitive later is not an API break.
pub fn infer_prim(prim: &Prim, inputs: &[&TensorType]) -> Result<Vec<TensorType>> {
    check_arity(prim, inputs.len())?;
    let out = match prim {
        Prim::Unary(op) => {
            let x = inputs[0];
            check_unary_dtype(*op, x.dtype)?;
            x.clone()
        }

        Prim::Binary(op) => {
            let (a, b) = (inputs[0], inputs[1]);
            require_same_dtype(a.dtype, b.dtype, prim.name())?;
            match op {
                BinaryOp::And | BinaryOp::Or | BinaryOp::Xor => require_bool(a.dtype, prim.name())?,
                _ => require_numeric(a.dtype, prim.name())?,
            }
            TensorType::new(a.dtype, broadcast_shapes(&a.shape, &b.shape)?)
        }

        Prim::Compare(_) => {
            let (a, b) = (inputs[0], inputs[1]);
            require_same_dtype(a.dtype, b.dtype, prim.name())?;
            require_unpacked(a.dtype, prim.name())?;
            TensorType::new(DataType::Bool, broadcast_shapes(&a.shape, &b.shape)?)
        }

        Prim::Select => {
            let (cond, a, b) = (inputs[0], inputs[1], inputs[2]);
            require_bool(cond.dtype, "select condition")?;
            require_same_dtype(a.dtype, b.dtype, "select branches")?;
            require_unpacked(a.dtype, "select")?;
            let shape = broadcast_shapes(&cond.shape, &a.shape)?;
            TensorType::new(a.dtype, broadcast_shapes(&shape, &b.shape)?)
        }

        Prim::Cast { to } => TensorType::new(*to, inputs[0].shape.clone()),

        Prim::MatMul { trans_a, trans_b } => {
            let (a, b) = (inputs[0], inputs[1]);
            require_same_dtype(a.dtype, b.dtype, "matmul")?;
            require_numeric(a.dtype, "matmul")?;
            if a.shape.rank() < 2 || b.shape.rank() < 2 {
                return Err(Error::Shape(format!(
                    "matmul requires rank >= 2 operands, got {} and {} \
                     (lowering handles rank-1 promotion)",
                    a.shape, b.shape
                )));
            }
            let (ad, bd) = (a.shape.dims(), b.shape.dims());
            let (m, ka) = trailing2(ad, *trans_a);
            let (kb, n) = trailing2(bd, *trans_b);
            if ka != kb {
                return Err(Error::Shape(format!(
                    "matmul contraction mismatch: {ka} vs {kb} (shapes {} x {})",
                    a.shape, b.shape
                )));
            }
            let batch = broadcast_dims(&ad[..ad.len() - 2], &bd[..bd.len() - 2])?;
            let mut dims = batch;
            dims.push(m.clone());
            dims.push(n.clone());
            TensorType::new(a.dtype, SymbolicShape(dims))
        }

        Prim::Reduce { axes, keepdims, .. } => {
            let x = inputs[0];
            require_numeric(x.dtype, "reduce")?;
            check_axes(axes, x.shape.rank(), "reduce")?;
            let mut dims = Vec::new();
            for (i, d) in x.shape.dims().iter().enumerate() {
                if axes.contains(&i) {
                    if *keepdims {
                        dims.push(DimExpr::constant(1));
                    }
                } else {
                    dims.push(d.clone());
                }
            }
            TensorType::new(x.dtype, SymbolicShape(dims))
        }

        Prim::Reshape { shape } => {
            let x = inputs[0];
            let out = SymbolicShape(shape.clone());
            let (in_n, out_n) = (x.shape.numel(), out.numel());
            if in_n != out_n {
                // Not provably equal. Hard error when both are constants;
                // otherwise defer to the run-time check at binding.
                if let (Some(a), Some(b)) = (in_n.as_const(), out_n.as_const()) {
                    return Err(Error::Shape(format!(
                        "reshape element count mismatch: {} ({a} elements) \
                         -> {out} ({b} elements)",
                        x.shape
                    )));
                }
            }
            TensorType::new(x.dtype, out)
        }

        Prim::Transpose { perm } => {
            let x = inputs[0];
            let rank = x.shape.rank();
            let mut seen = vec![false; rank];
            let is_permutation = perm.len() == rank
                && perm
                    .iter()
                    .all(|&p| p < rank && !std::mem::replace(&mut seen[p], true));
            if !is_permutation {
                return Err(Error::Shape(format!(
                    "transpose perm {perm:?} is not a permutation of rank {rank}"
                )));
            }
            let dims = perm.iter().map(|&p| x.shape.dims()[p].clone()).collect();
            TensorType::new(x.dtype, SymbolicShape(dims))
        }

        Prim::Broadcast { shape } => {
            let x = inputs[0];
            let target = SymbolicShape(shape.clone());
            TensorType::new(x.dtype, broadcast_shapes(&x.shape, &target)?)
        }

        Prim::Concat { axis } => {
            let first = inputs[0];
            let rank = first.shape.rank();
            if *axis >= rank {
                return Err(Error::Shape(format!(
                    "concat axis {axis} out of range for rank {rank}"
                )));
            }
            let mut axis_dim = DimExpr::constant(0);
            for x in inputs {
                require_same_dtype(first.dtype, x.dtype, "concat")?;
                if x.shape.rank() != rank {
                    return Err(Error::Shape(format!(
                        "concat rank mismatch: {} vs {}",
                        first.shape, x.shape
                    )));
                }
                for (i, (a, b)) in first.shape.dims().iter().zip(x.shape.dims()).enumerate() {
                    if i != *axis && a != b {
                        return Err(Error::Shape(format!(
                            "concat dim {i} mismatch: {} vs {}",
                            first.shape, x.shape
                        )));
                    }
                }
                axis_dim = axis_dim + x.shape.dims()[*axis].clone();
            }
            let mut dims = first.shape.dims().to_vec();
            dims[*axis] = axis_dim;
            TensorType::new(first.dtype, SymbolicShape(dims))
        }

        Prim::Slice { specs } => {
            let x = inputs[0];
            let rank = x.shape.rank();
            let mut dims = x.shape.dims().to_vec();
            let mut seen_axes = std::collections::HashSet::new();
            for spec in specs {
                if spec.axis >= rank {
                    return Err(Error::Shape(format!(
                        "slice axis {} out of range for rank {rank}",
                        spec.axis
                    )));
                }
                if !seen_axes.insert(spec.axis) {
                    return Err(Error::Shape(format!(
                        "slice axis {} specified twice",
                        spec.axis
                    )));
                }
                dims[spec.axis] = slice_out_dim(spec)?;
            }
            TensorType::new(x.dtype, SymbolicShape(dims))
        }

        Prim::Gather { axis } => {
            let (data, indices) = (inputs[0], inputs[1]);
            if indices.dtype != DataType::I64 {
                return Err(Error::DType(format!(
                    "gather indices must be i64, got {}",
                    indices.dtype
                )));
            }
            let rank = data.shape.rank();
            if rank == 0 || *axis >= rank {
                return Err(Error::Shape(format!(
                    "gather axis {axis} out of range for rank {rank}"
                )));
            }
            let mut dims: Vec<DimExpr> = data.shape.dims()[..*axis].to_vec();
            dims.extend(indices.shape.dims().iter().cloned());
            dims.extend(data.shape.dims()[*axis + 1..].iter().cloned());
            TensorType::new(data.dtype, SymbolicShape(dims))
        }

        Prim::Scatter => {
            let (data, indices, updates) = (inputs[0], inputs[1], inputs[2]);
            require_same_dtype(data.dtype, updates.dtype, "scatter")?;
            if indices.dtype != DataType::I64 {
                return Err(Error::DType(format!(
                    "scatter indices must be i64, got {}",
                    indices.dtype
                )));
            }
            let ir = indices.shape.rank();
            if ir == 0 {
                return Err(Error::Shape("scatter indices must have rank >= 1".into()));
            }
            let Some(k) = indices.shape.dims()[ir - 1].as_const() else {
                return Err(Error::Shape(format!(
                    "scatter requires a constant last indices dim, got {}",
                    indices.shape
                )));
            };
            let k = k as usize;
            let dr = data.shape.rank();
            if k > dr {
                return Err(Error::Shape(format!(
                    "scatter index depth {k} exceeds data rank {dr}"
                )));
            }
            // updates.shape must equal indices.shape[..-1] ++ data.shape[k..]
            let mut expect: Vec<DimExpr> = indices.shape.dims()[..ir - 1].to_vec();
            expect.extend(data.shape.dims()[k..].iter().cloned());
            if updates.shape.dims() != expect.as_slice() {
                return Err(Error::Shape(format!(
                    "scatter updates shape {} does not match expected {}",
                    updates.shape,
                    SymbolicShape(expect)
                )));
            }
            data.clone()
        }

        Prim::Iota { len, dtype } => {
            require_numeric(*dtype, "iota")?;
            TensorType::new(*dtype, SymbolicShape(vec![len.clone()]))
        }

        Prim::DimValues { exprs } => TensorType::of(DataType::I64, &[exprs.len() as u64]),

        Prim::Dequantize {
            block_size,
            out_shape,
        } => {
            let (data, scales) = (inputs[0], inputs[1]);
            if *block_size == 0 {
                return Err(Error::Shape("dequantize block_size must be > 0".into()));
            }
            if !matches!(data.dtype, DataType::U4 | DataType::I4 | DataType::U8) {
                return Err(Error::DType(format!(
                    "dequantize data must be u4/i4/u8, got {}",
                    data.dtype
                )));
            }
            if !scales.dtype.is_float() {
                return Err(Error::DType(format!(
                    "dequantize scales must be float, got {}",
                    scales.dtype
                )));
            }
            if let Some(zp) = inputs.get(2) {
                require_same_dtype(data.dtype, zp.dtype, "dequantize zero_points")?;
            }
            // data is [..., n_blocks, block_size] in logical elements.
            let last = data
                .shape
                .dims()
                .last()
                .ok_or_else(|| Error::Shape("dequantize data must have rank >= 1".into()))?;
            if let Some(l) = last.as_const() {
                if l as usize != *block_size {
                    return Err(Error::Shape(format!(
                        "dequantize data last dim {l} != block_size {block_size}"
                    )));
                }
            }
            let out = SymbolicShape(out_shape.clone());
            if let (Some(a), Some(b)) = (data.shape.numel().as_const(), out.numel().as_const()) {
                if a != b {
                    return Err(Error::Shape(format!(
                        "dequantize element count mismatch: data {} vs out {}",
                        data.shape, out
                    )));
                }
            }
            TensorType::new(scales.dtype, out)
        }
    };
    Ok(vec![out])
}

/// Broadcast two shapes following ONNX/numpy multidirectional rules.
///
/// Dims must be canonically equal, or one side must be the constant 1 (or
/// absent, for lower-rank operands). Symbolically ambiguous pairs are
/// errors — see the module docs.
pub fn broadcast_shapes(a: &SymbolicShape, b: &SymbolicShape) -> Result<SymbolicShape> {
    Ok(SymbolicShape(broadcast_dims(a.dims(), b.dims())?))
}

fn broadcast_dims(a: &[DimExpr], b: &[DimExpr]) -> Result<Vec<DimExpr>> {
    let rank = a.len().max(b.len());
    let one = DimExpr::constant(1);
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        // Align right.
        let da = (i + a.len()).checked_sub(rank).map(|j| &a[j]);
        let db = (i + b.len()).checked_sub(rank).map(|j| &b[j]);
        let dim = match (da, db) {
            (Some(x), Some(y)) => {
                if x == y {
                    x.clone()
                } else if *x == one {
                    y.clone()
                } else if *y == one {
                    x.clone()
                } else {
                    return Err(Error::Shape(format!(
                        "cannot broadcast dim {x} with {y} \
                         (dims that must match should share a symbol)"
                    )));
                }
            }
            (Some(x), None) => x.clone(),
            (None, Some(y)) => y.clone(),
            (None, None) => unreachable!(),
        };
        out.push(dim);
    }
    Ok(out)
}

/// Effective trailing (rows, cols) of a matmul operand, honoring transpose.
fn trailing2(dims: &[DimExpr], trans: bool) -> (&DimExpr, &DimExpr) {
    let (r, c) = (&dims[dims.len() - 2], &dims[dims.len() - 1]);
    if trans { (c, r) } else { (r, c) }
}

/// Output length of one slice spec.
fn slice_out_dim(spec: &crate::prim::SliceSpec) -> Result<DimExpr> {
    if spec.step == 0 {
        return Err(Error::Shape("slice step must be nonzero".into()));
    }
    if spec.step == 1 {
        return Ok(spec.end.clone() - spec.start.clone());
    }
    if spec.step == -1 {
        return Ok(spec.start.clone() - spec.end.clone());
    }
    // |step| != 1 requires constant bounds: ceil-division is not polynomial.
    let (Some(start), Some(end)) = (spec.start.as_const(), spec.end.as_const()) else {
        return Err(Error::Shape(format!(
            "slice with step {} requires constant bounds, got [{}, {})",
            spec.step, spec.start, spec.end
        )));
    };
    let (start, end) = (start as i64, end as i64);
    // ceil(num / d) for d > 0; num may be negative (caught below).
    let ceil_div = |num: i64, d: i64| (num + d - 1).div_euclid(d);
    let len = if spec.step > 0 {
        ceil_div(end - start, spec.step)
    } else {
        ceil_div(start - end, -spec.step)
    };
    if len < 0 {
        return Err(Error::Shape(format!(
            "slice [{start}, {end}) step {} has negative length",
            spec.step
        )));
    }
    Ok(DimExpr::constant(len as u64))
}

fn check_arity(prim: &Prim, got: usize) -> Result<()> {
    let (min, max) = prim.arity();
    if got < min || got > max {
        return Err(Error::InvalidGraph(format!(
            "{} expects {} inputs, got {got}",
            prim.name(),
            if min == max {
                format!("{min}")
            } else if max == usize::MAX {
                format!(">= {min}")
            } else {
                format!("{min}..={max}")
            }
        )));
    }
    Ok(())
}

fn check_unary_dtype(op: UnaryOp, dt: DataType) -> Result<()> {
    use UnaryOp::*;
    let ok = match op {
        Sqrt | Rsqrt | Exp | Log | Sin | Cos | Tanh | Erf | Floor | Ceil => dt.is_float(),
        Neg => dt.is_float() || matches!(dt, DataType::I64 | DataType::I32 | DataType::I8),
        Abs => dt.is_float() || (dt.is_int() && !dt.is_packed()),
        Not => dt == DataType::Bool,
    };
    if ok {
        Ok(())
    } else {
        Err(Error::DType(format!(
            "unary op does not support dtype {dt}"
        )))
    }
}

fn require_same_dtype(a: DataType, b: DataType, what: &str) -> Result<()> {
    if a != b {
        return Err(Error::DType(format!(
            "{what}: dtype mismatch {a} vs {b} (no implicit promotion; \
             lowering must insert casts)"
        )));
    }
    Ok(())
}

fn require_numeric(dt: DataType, what: &str) -> Result<()> {
    if dt == DataType::Bool || dt.is_packed() {
        return Err(Error::DType(format!("{what}: dtype {dt} not supported")));
    }
    Ok(())
}

fn require_unpacked(dt: DataType, what: &str) -> Result<()> {
    if dt.is_packed() {
        return Err(Error::DType(format!(
            "{what}: packed dtype {dt} must be cast/dequantized first"
        )));
    }
    Ok(())
}

fn require_bool(dt: DataType, what: &str) -> Result<()> {
    if dt != DataType::Bool {
        return Err(Error::DType(format!("{what}: expected bool, got {dt}")));
    }
    Ok(())
}

fn check_axes(axes: &[usize], rank: usize, what: &str) -> Result<()> {
    if axes.is_empty() {
        return Err(Error::Shape(format!(
            "{what}: axes must be non-empty (reduce-over-all lists every axis)"
        )));
    }
    let mut prev: Option<usize> = None;
    for &a in axes {
        if a >= rank {
            return Err(Error::Shape(format!(
                "{what}: axis {a} out of range for rank {rank}"
            )));
        }
        if prev.is_some_and(|p| p >= a) {
            return Err(Error::Shape(format!(
                "{what}: axes must be sorted and unique, got {axes:?}"
            )));
        }
        prev = Some(a);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dim::SymbolTable;
    use crate::prim::{CmpOp, ReduceOp, SliceSpec};

    fn t(dtype: DataType, dims: &[u64]) -> TensorType {
        TensorType::of(dtype, dims)
    }

    fn sym_shape(table: &mut SymbolTable, spec: &[&str]) -> SymbolicShape {
        SymbolicShape(
            spec.iter()
                .map(|s| match s.parse::<u64>() {
                    Ok(c) => DimExpr::constant(c),
                    Err(_) => DimExpr::sym(table.intern(s)),
                })
                .collect(),
        )
    }

    #[test]
    fn binary_broadcast_symbolic() {
        let mut tab = SymbolTable::new();
        // [1, S, 256] + [256] -> [1, S, 256]
        let a = TensorType::new(DataType::F32, sym_shape(&mut tab, &["1", "S", "256"]));
        let b = t(DataType::F32, &[256]);
        let out = infer_prim(&Prim::Binary(BinaryOp::Add), &[&a, &b]).unwrap();
        assert_eq!(out[0].shape, a.shape);

        // [S, 1] * [1, T] -> [S, T]
        let a = TensorType::new(DataType::F32, sym_shape(&mut tab, &["S", "1"]));
        let b = TensorType::new(DataType::F32, sym_shape(&mut tab, &["1", "T"]));
        let out = infer_prim(&Prim::Binary(BinaryOp::Mul), &[&a, &b]).unwrap();
        assert_eq!(out[0].shape, sym_shape(&mut tab, &["S", "T"]));

        // S vs T is ambiguous -> error.
        let a = TensorType::new(DataType::F32, sym_shape(&mut tab, &["S"]));
        let b = TensorType::new(DataType::F32, sym_shape(&mut tab, &["T"]));
        assert!(infer_prim(&Prim::Binary(BinaryOp::Add), &[&a, &b]).is_err());
    }

    #[test]
    fn no_implicit_promotion() {
        let a = t(DataType::F32, &[2]);
        let b = t(DataType::F16, &[2]);
        let err = infer_prim(&Prim::Binary(BinaryOp::Add), &[&a, &b]).unwrap_err();
        assert!(err.to_string().contains("no implicit promotion"));
    }

    #[test]
    fn compare_produces_bool() {
        let a = t(DataType::I64, &[4]);
        let out = infer_prim(&Prim::Compare(CmpOp::Lt), &[&a, &a]).unwrap();
        assert_eq!(out[0].dtype, DataType::Bool);
    }

    #[test]
    fn matmul_shapes() {
        let mut tab = SymbolTable::new();
        // [B, S, 64] x [64, 256] -> [B, S, 256] (batch broadcast)
        let a = TensorType::new(DataType::F32, sym_shape(&mut tab, &["B", "S", "64"]));
        let b = t(DataType::F32, &[64, 256]);
        let out = infer_prim(
            &Prim::MatMul {
                trans_a: false,
                trans_b: false,
            },
            &[&a, &b],
        )
        .unwrap();
        assert_eq!(out[0].shape, sym_shape(&mut tab, &["B", "S", "256"]));

        // trans_b: [S, 64] x [256, 64]^T -> [S, 256]
        let a = TensorType::new(DataType::F32, sym_shape(&mut tab, &["S", "64"]));
        let b = t(DataType::F32, &[256, 64]);
        let out = infer_prim(
            &Prim::MatMul {
                trans_a: false,
                trans_b: true,
            },
            &[&a, &b],
        )
        .unwrap();
        assert_eq!(out[0].shape, sym_shape(&mut tab, &["S", "256"]));

        // Contraction mismatch.
        let bad = t(DataType::F32, &[63, 256]);
        assert!(
            infer_prim(
                &Prim::MatMul {
                    trans_a: false,
                    trans_b: false
                },
                &[&a, &bad]
            )
            .is_err()
        );
    }

    #[test]
    fn reduce_keepdims() {
        let mut tab = SymbolTable::new();
        let x = TensorType::new(DataType::F32, sym_shape(&mut tab, &["B", "S", "256"]));
        let keep = infer_prim(
            &Prim::Reduce {
                op: ReduceOp::Mean,
                axes: vec![2],
                keepdims: true,
            },
            &[&x],
        )
        .unwrap();
        assert_eq!(keep[0].shape, sym_shape(&mut tab, &["B", "S", "1"]));
        let drop = infer_prim(
            &Prim::Reduce {
                op: ReduceOp::Sum,
                axes: vec![0, 2],
                keepdims: false,
            },
            &[&x],
        )
        .unwrap();
        assert_eq!(drop[0].shape, sym_shape(&mut tab, &["S"]));
        // Unsorted axes rejected.
        assert!(
            infer_prim(
                &Prim::Reduce {
                    op: ReduceOp::Sum,
                    axes: vec![2, 0],
                    keepdims: false
                },
                &[&x]
            )
            .is_err()
        );
    }

    #[test]
    fn reshape_numel_policy() {
        let mut tab = SymbolTable::new();
        // Const mismatch: hard error.
        let x = t(DataType::F32, &[2, 3]);
        let bad = Prim::Reshape {
            shape: vec![DimExpr::constant(7)],
        };
        assert!(infer_prim(&bad, &[&x]).is_err());

        // Provably equal symbolic reshape: ok.
        let s = DimExpr::sym(tab.intern("S"));
        let x = TensorType::new(
            DataType::F32,
            SymbolicShape(vec![s.clone(), DimExpr::constant(64)]),
        );
        let ok = Prim::Reshape {
            shape: vec![s.clone() * DimExpr::constant(64)],
        };
        assert_eq!(
            infer_prim(&ok, &[&x]).unwrap()[0].shape,
            SymbolicShape(vec![s.clone() * DimExpr::constant(64)])
        );

        // Unprovable (late-bound L): accepted, checked at binding.
        let l = DimExpr::sym(tab.intern("L"));
        let late = Prim::Reshape {
            shape: vec![l, DimExpr::constant(64)],
        };
        assert!(infer_prim(&late, &[&x]).is_ok());
    }

    #[test]
    fn transpose_and_bad_perm() {
        let x = t(DataType::F32, &[2, 3, 4]);
        let out = infer_prim(
            &Prim::Transpose {
                perm: vec![2, 0, 1],
            },
            &[&x],
        )
        .unwrap();
        assert_eq!(out[0].shape, SymbolicShape::fixed(&[4, 2, 3]));
        assert!(
            infer_prim(
                &Prim::Transpose {
                    perm: vec![0, 0, 1]
                },
                &[&x]
            )
            .is_err()
        );
        assert!(infer_prim(&Prim::Transpose { perm: vec![0, 1] }, &[&x]).is_err());
    }

    #[test]
    fn concat_symbolic_axis_sum() {
        let mut tab = SymbolTable::new();
        // concat([B, S, 64], [B, T, 64], axis=1) -> [B, S+T, 64]
        let a = TensorType::new(DataType::F32, sym_shape(&mut tab, &["B", "S", "64"]));
        let b = TensorType::new(DataType::F32, sym_shape(&mut tab, &["B", "T", "64"]));
        let out = infer_prim(&Prim::Concat { axis: 1 }, &[&a, &b]).unwrap();
        let s = DimExpr::sym(tab.get("S").unwrap());
        let t_ = DimExpr::sym(tab.get("T").unwrap());
        let b_ = DimExpr::sym(tab.get("B").unwrap());
        assert_eq!(
            out[0].shape,
            SymbolicShape(vec![b_, s + t_, DimExpr::constant(64)])
        );
        // Non-axis mismatch rejected.
        let c = TensorType::new(DataType::F32, sym_shape(&mut tab, &["B", "S", "65"]));
        assert!(infer_prim(&Prim::Concat { axis: 1 }, &[&a, &c]).is_err());
    }

    #[test]
    fn slice_dims() {
        let mut tab = SymbolTable::new();
        let s = DimExpr::sym(tab.intern("S"));
        let x = TensorType::new(
            DataType::F32,
            SymbolicShape(vec![s.clone(), DimExpr::constant(10)]),
        );
        // Symbolic step-1 slice: [2, S) -> S-2.
        let spec = SliceSpec {
            axis: 0,
            start: DimExpr::constant(2),
            end: s.clone(),
            step: 1,
        };
        let out = infer_prim(&Prim::Slice { specs: vec![spec] }, &[&x]).unwrap();
        assert_eq!(out[0].shape.dims()[0], s.clone() - DimExpr::constant(2));
        // step 2 with const bounds: [1, 10) step 2 -> 5 elements.
        let spec = SliceSpec {
            axis: 1,
            start: DimExpr::constant(1),
            end: DimExpr::constant(10),
            step: 2,
        };
        let out = infer_prim(&Prim::Slice { specs: vec![spec] }, &[&x]).unwrap();
        assert_eq!(out[0].shape.dims()[1], DimExpr::constant(5));
        // step 2 with symbolic bounds: rejected.
        let spec = SliceSpec {
            axis: 0,
            start: DimExpr::constant(0),
            end: s,
            step: 2,
        };
        assert!(infer_prim(&Prim::Slice { specs: vec![spec] }, &[&x]).is_err());
    }

    #[test]
    fn gather_scatter_shapes() {
        let mut tab = SymbolTable::new();
        // Embedding lookup: gather([262144, 640], [1, S], axis 0) -> [1, S, 640]
        let data = t(DataType::F32, &[262144, 640]);
        let idx = TensorType::new(DataType::I64, sym_shape(&mut tab, &["1", "S"]));
        let out = infer_prim(&Prim::Gather { axis: 0 }, &[&data, &idx]).unwrap();
        assert_eq!(out[0].shape, sym_shape(&mut tab, &["1", "S", "640"]));

        // ScatterND: data [4, 5], indices [3, 1], updates [3, 5] -> [4, 5]
        let data = t(DataType::F32, &[4, 5]);
        let idx = t(DataType::I64, &[3, 1]);
        let upd = t(DataType::F32, &[3, 5]);
        let out = infer_prim(&Prim::Scatter, &[&data, &idx, &upd]).unwrap();
        assert_eq!(out[0].shape, SymbolicShape::fixed(&[4, 5]));
        // Wrong updates shape.
        let bad = t(DataType::F32, &[3, 4]);
        assert!(infer_prim(&Prim::Scatter, &[&data, &idx, &bad]).is_err());
    }

    #[test]
    fn iota_and_cast() {
        let mut tab = SymbolTable::new();
        let s = DimExpr::sym(tab.intern("S"));
        let out = infer_prim(
            &Prim::Iota {
                len: s.clone(),
                dtype: DataType::I64,
            },
            &[],
        )
        .unwrap();
        assert_eq!(out[0].shape, SymbolicShape(vec![s]));
        let x = t(DataType::I64, &[3]);
        let out = infer_prim(&Prim::Cast { to: DataType::F32 }, &[&x]).unwrap();
        assert_eq!(out[0].dtype, DataType::F32);
    }

    #[test]
    fn dequantize_type() {
        // Logical element counts: 8 blocks of 32 u4 elements = 256 per row.
        let data = t(DataType::U4, &[64, 8, 32]);
        let scales = t(DataType::F32, &[64, 8]);
        let p = Prim::Dequantize {
            block_size: 32,
            out_shape: vec![DimExpr::constant(64), DimExpr::constant(256)],
        };
        let out = infer_prim(&p, &[&data, &scales]).unwrap();
        assert_eq!(out[0].dtype, DataType::F32);
        assert_eq!(out[0].shape, SymbolicShape::fixed(&[64, 256]));
        // Float data rejected.
        let bad = t(DataType::F32, &[64, 8, 16]);
        assert!(infer_prim(&p, &[&bad, &scales]).is_err());
    }
}
