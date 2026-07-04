//! Reference interpreter — the executable specification of the primitives.
//!
//! Written for obvious correctness, never for speed. Every backend kernel
//! must match this interpreter within the pinned tolerances.
//!
//! **Precision model:** floating-point operations are evaluated internally
//! in `f64` and rounded to the value's dtype on store. The specification a
//! kernel must meet is therefore "within tolerance of the high-precision
//! result", not "bit-identical to one particular f32 evaluation order".
//! Integer operations are evaluated in `i64`; overflow wraps (two's
//! complement truncation on store, matching Rust `as`). Integer division
//! truncates toward zero and errors on zero.
//!
//! **Symbol binding:** [`bind_inputs`] unifies the module's declared input
//! shapes with the concrete input tensors to produce [`Bindings`]. During
//! execution, late-bound symbols may additionally be bound when the tensor
//! carrying them materializes (e.g. an unresolvable `Reshape` target dim is
//! inferred from the element count, like ONNX `-1`).

use crate::dim::{Bindings, DimExpr};
use crate::graph::{Module, NodeKind, Origin};
use crate::prim::{BinaryOp, CmpOp, Prim, ReduceOp, SliceSpec, UnaryOp};
use crate::types::DataType;
use crate::{Error, Result};
use half::f16;

// ───────────────────────────── Tensor ──────────────────────────────────

/// A concrete CPU tensor: dtype, shape, and raw storage bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    dtype: DataType,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl Tensor {
    /// Create from raw bytes; `data` must match the dtype's storage size.
    pub fn new(dtype: DataType, shape: Vec<usize>, data: Vec<u8>) -> Result<Self> {
        let numel: usize = shape.iter().product();
        let expect = dtype.storage_bytes(numel);
        if data.len() != expect {
            return Err(Error::Interp(format!(
                "tensor data size mismatch: {} bytes for {dtype}{shape:?} (expected {expect})",
                data.len()
            )));
        }
        Ok(Self { dtype, shape, data })
    }

    /// f32 tensor from values.
    pub fn from_f32(values: &[f32], shape: &[usize]) -> Result<Self> {
        Self::new(
            DataType::F32,
            shape.to_vec(),
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    }

    /// i64 tensor from values.
    pub fn from_i64(values: &[i64], shape: &[usize]) -> Result<Self> {
        Self::new(
            DataType::I64,
            shape.to_vec(),
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    }

    /// bool tensor from values.
    pub fn from_bool(values: &[bool], shape: &[usize]) -> Result<Self> {
        Self::new(
            DataType::Bool,
            shape.to_vec(),
            values.iter().map(|&b| b as u8).collect(),
        )
    }

    /// Element dtype.
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Raw storage bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    /// Element count.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Read out as f32 (must be a float tensor).
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        // Direct byte read for f32 — hot on the generation path (the
        // full logits vector every token); the f64 detour below serves
        // the interpreter's remaining float dtypes.
        if self.dtype == DataType::F32 {
            return Ok(self
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect());
        }
        match self.decode()? {
            Values::F(v) => Ok(v.into_iter().map(|x| x as f32).collect()),
            _ => Err(Error::Interp(format!(
                "{} is not a float tensor",
                self.dtype
            ))),
        }
    }

    /// Read out as i64 (must be an integer tensor).
    pub fn to_i64(&self) -> Result<Vec<i64>> {
        match self.decode()? {
            Values::I(v) => Ok(v),
            _ => Err(Error::Interp(format!(
                "{} is not an int tensor",
                self.dtype
            ))),
        }
    }

    /// Read out as bool (must be a bool tensor).
    pub fn to_bool(&self) -> Result<Vec<bool>> {
        match self.decode()? {
            Values::B(v) => Ok(v),
            _ => Err(Error::Interp(format!(
                "{} is not a bool tensor",
                self.dtype
            ))),
        }
    }

    /// Decode storage into the interpreter's working domain.
    fn decode(&self) -> Result<Values> {
        let n = self.numel();
        let d = &self.data;
        Ok(match self.dtype {
            DataType::F32 => Values::F(
                d.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
                    .collect(),
            ),
            DataType::F16 => Values::F(
                d.chunks_exact(2)
                    .map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f64())
                    .collect(),
            ),
            DataType::I64 => Values::I(
                d.chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            ),
            DataType::I32 => Values::I(
                d.chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                    .collect(),
            ),
            DataType::U32 => Values::I(
                d.chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as i64)
                    .collect(),
            ),
            DataType::U8 => Values::I(d.iter().map(|&b| b as i64).collect()),
            DataType::I8 => Values::I(d.iter().map(|&b| b as i8 as i64).collect()),
            DataType::Bool => Values::B(d.iter().map(|&b| b != 0).collect()),
            // Little-endian nibble packing: low nibble is the earlier element.
            DataType::U4 => Values::I(
                (0..n)
                    .map(|i| ((d[i / 2] >> ((i % 2) * 4)) & 0xF) as i64)
                    .collect(),
            ),
            DataType::I4 => Values::I(
                (0..n)
                    .map(|i| {
                        let nib = (d[i / 2] >> ((i % 2) * 4)) & 0xF;
                        // Sign-extend from 4 bits.
                        ((nib as i8) << 4 >> 4) as i64
                    })
                    .collect(),
            ),
        })
    }
}

/// The interpreter's working domain: high-precision floats, wide ints, bools.
#[derive(Debug, Clone, PartialEq)]
enum Values {
    F(Vec<f64>),
    I(Vec<i64>),
    B(Vec<bool>),
}

impl Values {
    fn len(&self) -> usize {
        match self {
            Values::F(v) => v.len(),
            Values::I(v) => v.len(),
            Values::B(v) => v.len(),
        }
    }

    /// Encode into a tensor of `dtype`, converting domains as needed.
    ///
    /// Float→int saturates, int→int wraps (Rust `as`), anything→bool is
    /// `!= 0`, bool→number is `0/1`.
    fn encode(&self, dtype: DataType, shape: Vec<usize>) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        if self.len() != n {
            return Err(Error::Interp(format!(
                "encode length mismatch: {} values into shape {shape:?}",
                self.len()
            )));
        }
        // Convert domain first.
        let data: Vec<u8> = match dtype {
            DataType::F32 => self
                .iter_f()
                .flat_map(|v| (v as f32).to_le_bytes())
                .collect(),
            DataType::F16 => self
                .iter_f()
                .flat_map(|v| f16::from_f64(v).to_le_bytes())
                .collect(),
            DataType::I64 => self.iter_i().flat_map(|v| v.to_le_bytes()).collect(),
            DataType::I32 => self
                .iter_i()
                .flat_map(|v| (v as i32).to_le_bytes())
                .collect(),
            DataType::U32 => self
                .iter_i()
                .flat_map(|v| (v as u32).to_le_bytes())
                .collect(),
            DataType::U8 => self.iter_i().map(|v| v as u8).collect(),
            DataType::I8 => self.iter_i().map(|v| v as i8 as u8).collect(),
            DataType::Bool => match self {
                Values::B(v) => v.iter().map(|&b| b as u8).collect(),
                Values::F(v) => v.iter().map(|&x| (x != 0.0) as u8).collect(),
                Values::I(v) => v.iter().map(|&x| (x != 0) as u8).collect(),
            },
            DataType::U4 | DataType::I4 => {
                let vals: Vec<i64> = self.iter_i().collect();
                let mut out = vec![0u8; dtype.storage_bytes(n)];
                for (i, v) in vals.iter().enumerate() {
                    let nib = (*v as u8) & 0xF;
                    out[i / 2] |= nib << ((i % 2) * 4);
                }
                out
            }
        };
        Tensor::new(dtype, shape, data)
    }

    /// Iterate as f64 (bools as 0/1, saturating for out-of-range i64 is
    /// irrelevant — i64→f64 is always representable approximately).
    fn iter_f(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        match self {
            Values::F(v) => Box::new(v.iter().copied()),
            Values::I(v) => Box::new(v.iter().map(|&x| x as f64)),
            Values::B(v) => Box::new(v.iter().map(|&b| b as i64 as f64)),
        }
    }

    /// Iterate as i64 (floats saturate like Rust `as`).
    fn iter_i(&self) -> Box<dyn Iterator<Item = i64> + '_> {
        match self {
            Values::F(v) => Box::new(v.iter().map(|&x| x as i64)),
            Values::I(v) => Box::new(v.iter().copied()),
            Values::B(v) => Box::new(v.iter().map(|&b| b as i64)),
        }
    }
}

// ─────────────────────── Shape / index helpers ─────────────────────────

fn unravel(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        coords[i] = linear % shape[i];
        linear /= shape[i];
    }
    coords
}

fn ravel(coords: &[usize], shape: &[usize]) -> usize {
    coords
        .iter()
        .zip(shape)
        .fold(0, |acc, (&c, &d)| acc * d + c)
}

/// Linear index into `in_shape` for broadcast-consuming `out_coords`
/// (right-aligned; size-1 dims read index 0).
fn broadcast_index(out_coords: &[usize], in_shape: &[usize]) -> usize {
    let offset = out_coords.len() - in_shape.len();
    let coords: Vec<usize> = in_shape
        .iter()
        .enumerate()
        .map(|(i, &d)| if d == 1 { 0 } else { out_coords[i + offset] })
        .collect();
    ravel(&coords, in_shape)
}

/// Concrete multidirectional broadcast of two shapes.
fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = (i + a.len()).checked_sub(rank).map(|j| a[j]).unwrap_or(1);
        let db = (i + b.len()).checked_sub(rank).map(|j| b[j]).unwrap_or(1);
        if da == db || da == 1 || db == 1 {
            out.push(da.max(db));
        } else {
            return Err(Error::Interp(format!(
                "cannot broadcast shapes {a:?} and {b:?}"
            )));
        }
    }
    Ok(out)
}

// ───────────────────────────── Binding ─────────────────────────────────

/// Unify the module's declared input shapes with concrete tensors,
/// producing symbol bindings. See [`bind_shapes`].
pub fn bind_inputs(module: &Module, inputs: &[(&str, Tensor)]) -> Result<Bindings> {
    let described: Vec<(&str, DataType, &[usize])> = inputs
        .iter()
        .map(|(n, t)| (*n, t.dtype(), t.shape()))
        .collect();
    bind_shapes(module, &described)
}

/// Unify the module's declared input shapes with concrete dtypes/shapes,
/// producing symbol bindings. Shared by the interpreter and by device
/// backends (which hold device tensors, not host tensors).
///
/// Per input dim: a constant must match exactly; a bare symbol binds (and
/// must be consistent across all occurrences); a composite expression is
/// checked if all its symbols are already bound, otherwise it is an error
/// (module inputs should declare bare symbols).
pub fn bind_shapes(module: &Module, inputs: &[(&str, DataType, &[usize])]) -> Result<Bindings> {
    let mut bindings = Bindings::new();
    // Two passes: bind bare symbols first, then verify composite dims.
    for verify_only in [false, true] {
        for (name, declared_id) in &module.inputs {
            let Some(&(_, dtype, shape)) = inputs.iter().find(|(n, _, _)| n == name) else {
                return Err(Error::Binding(format!("missing input '{name}'")));
            };
            let declared = &module.value(*declared_id).ty;
            if declared.dtype != dtype {
                return Err(Error::Binding(format!(
                    "input '{name}': dtype {} declared, {dtype} provided",
                    declared.dtype,
                )));
            }
            if declared.shape.rank() != shape.len() {
                return Err(Error::Binding(format!(
                    "input '{name}': rank {} declared, shape {shape:?} provided",
                    declared.shape.rank(),
                )));
            }
            for (expr, &concrete) in declared.shape.dims().iter().zip(shape) {
                if let Some(sym) = expr.as_sym() {
                    if !verify_only {
                        bindings
                            .bind(sym, concrete as u64)
                            .map_err(|e| Error::Binding(format!("input '{name}': {e}")))?;
                    }
                } else if verify_only {
                    let expect = expr.eval(&bindings).map_err(|e| {
                        Error::Binding(format!(
                            "input '{name}': cannot resolve declared dim {expr}: {e}"
                        ))
                    })?;
                    if expect as usize != concrete {
                        return Err(Error::Binding(format!(
                            "input '{name}': dim {expr} = {expect} but got {concrete}"
                        )));
                    }
                }
            }
        }
    }
    Ok(bindings)
}

// ──────────────────────────── Execution ────────────────────────────────

/// Evaluate a module on concrete inputs, returning the named outputs.
///
/// The module must contain only primitive nodes — composites must be
/// inlined first (see [`crate::decomp::inline_composites`]).
pub fn eval(module: &Module, inputs: &[(&str, Tensor)]) -> Result<Vec<(String, Tensor)>> {
    let mut bindings = bind_inputs(module, inputs)?;
    let mut slots: Vec<Option<Tensor>> = vec![None; module.values.len()];

    // Materialize inputs and constants.
    for id in module.value_ids() {
        let def = module.value(id);
        match def.origin {
            Origin::Input => {
                let name = def.name.as_deref().unwrap_or_default();
                let (_, t) = inputs
                    .iter()
                    .find(|(n, _)| *n == name)
                    .ok_or_else(|| Error::Binding(format!("missing input '{name}'")))?;
                slots[id.index()] = Some(t.clone());
            }
            Origin::Const(cid) => {
                slots[id.index()] = Some(const_tensor(module, cid)?);
            }
            Origin::Node { .. } => {}
        }
    }

    // Execute.
    for node_id in module.topo_order()? {
        let node = module.node(node_id);
        let prim = match &node.kind {
            NodeKind::Prim(p) => p,
            NodeKind::Composite(c) => {
                return Err(Error::Interp(format!(
                    "composite '{}' must be inlined before interpretation",
                    c.name
                )));
            }
        };
        let ins: Vec<&Tensor> = node
            .inputs
            .iter()
            .map(|&v| {
                slots[v.index()]
                    .as_ref()
                    .ok_or_else(|| Error::Interp("input value not yet materialized".into()))
            })
            .collect::<Result<_>>()?;

        let out = eval_prim(prim, &ins, &mut bindings).map_err(|e| {
            let loc = node.loc.name.as_deref().unwrap_or("<unnamed>");
            Error::Interp(format!("{} (node '{loc}'): {e}", prim.name()))
        })?;

        // Opportunistically bind late-bound symbols from the produced shape.
        let out_id = node.outputs[0];
        let declared = &module.value(out_id).ty.shape;
        if declared.rank() == out.shape().len() {
            for (expr, &concrete) in declared.dims().iter().zip(out.shape()) {
                if let Some(sym) = expr.as_sym() {
                    if bindings.get(sym).is_none() {
                        bindings.bind(sym, concrete as u64)?;
                    }
                }
            }
        }
        slots[out_id.index()] = Some(out);
    }

    // Collect outputs.
    module
        .outputs
        .iter()
        .map(|(name, id)| {
            slots[id.index()]
                .clone()
                .map(|t| (name.clone(), t))
                .ok_or_else(|| Error::Interp(format!("output '{name}' was never produced")))
        })
        .collect()
}

/// Materialize a constant-pool entry as a tensor.
pub fn const_tensor(module: &Module, cid: crate::graph::ConstId) -> Result<Tensor> {
    let ty = module.consts.ty(cid);
    let dims: Vec<usize> = ty
        .shape
        .as_static()
        .expect("pool constants have static shapes")
        .iter()
        .map(|&d| d as usize)
        .collect();
    Tensor::new(ty.dtype, dims, module.consts.bytes(cid).to_vec())
}

/// Evaluate one primitive on concrete inputs.
pub(crate) fn eval_prim(prim: &Prim, ins: &[&Tensor], bindings: &mut Bindings) -> Result<Tensor> {
    match prim {
        Prim::Unary(op) => {
            let x = ins[0];
            let out = match x.decode()? {
                Values::F(v) => Values::F(v.into_iter().map(|a| unary_f(*op, a)).collect()),
                Values::I(v) => Values::I(
                    v.into_iter()
                        .map(|a| match op {
                            UnaryOp::Neg => a.wrapping_neg(),
                            UnaryOp::Abs => a.wrapping_abs(),
                            _ => unreachable!("dtype-checked by inference"),
                        })
                        .collect(),
                ),
                Values::B(v) => Values::B(v.into_iter().map(|b| !b).collect()),
            };
            out.encode(x.dtype(), x.shape().to_vec())
        }

        Prim::Binary(op) => {
            let (a, b) = (ins[0], ins[1]);
            let shape = broadcast_shape(a.shape(), b.shape())?;
            let (va, vb) = (a.decode()?, b.decode()?);
            let out = elementwise2(&shape, a.shape(), b.shape(), &va, &vb, |x, y| {
                binary_val(*op, x, y)
            })?;
            out.encode(a.dtype(), shape)
        }

        Prim::Compare(op) => {
            let (a, b) = (ins[0], ins[1]);
            let shape = broadcast_shape(a.shape(), b.shape())?;
            let (va, vb) = (a.decode()?, b.decode()?);
            let out = elementwise2(&shape, a.shape(), b.shape(), &va, &vb, |x, y| {
                Ok(Scalar::B(compare_val(*op, x, y)))
            })?;
            out.encode(DataType::Bool, shape)
        }

        Prim::Select => {
            let (c, a, b) = (ins[0], ins[1], ins[2]);
            let shape = broadcast_shape(&broadcast_shape(c.shape(), a.shape())?, b.shape())?;
            let (vc, va, vb) = (c.decode()?, a.decode()?, b.decode()?);
            let Values::B(cond) = &vc else {
                unreachable!("dtype-checked by inference");
            };
            let n: usize = shape.iter().product();
            let mut out: Vec<Scalar> = Vec::with_capacity(n);
            for i in 0..n {
                let coords = unravel(i, &shape);
                let pick_a = cond[broadcast_index(&coords, c.shape())];
                let src = if pick_a {
                    (&va, a.shape())
                } else {
                    (&vb, b.shape())
                };
                out.push(scalar_at(src.0, broadcast_index(&coords, src.1)));
            }
            collect_scalars(out)?.encode(a.dtype(), shape)
        }

        Prim::Cast { to } => ins[0].decode()?.encode(*to, ins[0].shape().to_vec()),

        Prim::MatMul { trans_a, trans_b } => matmul(ins[0], ins[1], *trans_a, *trans_b),

        Prim::Reduce { op, axes, keepdims } => reduce(ins[0], *op, axes, *keepdims),

        Prim::Reshape { shape } => {
            let x = ins[0];
            let dims = resolve_reshape(shape, x.numel(), bindings)?;
            Tensor::new(x.dtype(), dims, x.bytes().to_vec()).map_err(|_| {
                Error::Interp(format!(
                    "reshape element count mismatch: {:?} has {} elements",
                    x.shape(),
                    x.numel()
                ))
            })
        }

        Prim::Transpose { perm } => {
            let x = ins[0];
            let in_shape = x.shape();
            // Shape inference rejects non-permutations; re-check here so a
            // module that skipped inference errors instead of panicking.
            let mut inverse = vec![usize::MAX; in_shape.len()];
            if perm.len() != in_shape.len() {
                return Err(Error::Interp(format!(
                    "transpose perm {perm:?} does not match rank {}",
                    in_shape.len()
                )));
            }
            for (i, &p) in perm.iter().enumerate() {
                if p >= in_shape.len() || inverse[p] != usize::MAX {
                    return Err(Error::Interp(format!(
                        "transpose perm {perm:?} is not a permutation"
                    )));
                }
                inverse[p] = i;
            }
            let out_shape: Vec<usize> = perm.iter().map(|&p| in_shape[p]).collect();
            let v = x.decode()?;
            let n = x.numel();
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let oc = unravel(i, &out_shape);
                let ic: Vec<usize> = (0..in_shape.len()).map(|d| oc[inverse[d]]).collect();
                out.push(scalar_at(&v, ravel(&ic, in_shape)));
            }
            collect_scalars(out)?.encode(x.dtype(), out_shape)
        }

        Prim::Broadcast { shape } => {
            let x = ins[0];
            let target = eval_dims(shape, bindings)?;
            let out_shape = broadcast_shape(x.shape(), &target)?;
            let v = x.decode()?;
            let n: usize = out_shape.iter().product();
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let coords = unravel(i, &out_shape);
                out.push(scalar_at(&v, broadcast_index(&coords, x.shape())));
            }
            collect_scalars(out)?.encode(x.dtype(), out_shape)
        }

        Prim::Concat { axis } => {
            let first = ins[0];
            let mut out_shape = first.shape().to_vec();
            out_shape[*axis] = ins.iter().map(|t| t.shape()[*axis]).sum();
            let n: usize = out_shape.iter().product();
            let vals: Vec<Values> = ins.iter().map(|t| t.decode()).collect::<Result<_>>()?;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let mut coords = unravel(i, &out_shape);
                // Find which input this coordinate falls into.
                let mut k = coords[*axis];
                let mut src = 0;
                while k >= ins[src].shape()[*axis] {
                    k -= ins[src].shape()[*axis];
                    src += 1;
                }
                coords[*axis] = k;
                out.push(scalar_at(&vals[src], ravel(&coords, ins[src].shape())));
            }
            collect_scalars(out)?.encode(first.dtype(), out_shape)
        }

        Prim::Slice { specs } => {
            let x = ins[0];
            let mut out_shape = x.shape().to_vec();
            let mut starts = vec![0i64; x.shape().len()];
            let mut steps = vec![1i64; x.shape().len()];
            for spec in specs {
                let (len, start) = eval_slice_spec(spec, bindings)?;
                out_shape[spec.axis] = len;
                starts[spec.axis] = start;
                steps[spec.axis] = spec.step;
            }
            let v = x.decode()?;
            let n: usize = out_shape.iter().product();
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let oc = unravel(i, &out_shape);
                let ic: Vec<usize> = oc
                    .iter()
                    .enumerate()
                    .map(|(d, &c)| (starts[d] + c as i64 * steps[d]) as usize)
                    .collect();
                out.push(scalar_at(&v, ravel(&ic, x.shape())));
            }
            collect_scalars(out)?.encode(x.dtype(), out_shape)
        }

        Prim::Gather { axis } => {
            let (data, indices) = (ins[0], ins[1]);
            let idx = indices.to_i64()?;
            let dim = data.shape()[*axis] as i64;
            let mut out_shape: Vec<usize> = data.shape()[..*axis].to_vec();
            out_shape.extend_from_slice(indices.shape());
            out_shape.extend_from_slice(&data.shape()[*axis + 1..]);
            let v = data.decode()?;
            let n: usize = out_shape.iter().product();
            let ir = indices.shape().len();
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let oc = unravel(i, &out_shape);
                let mut raw = idx[ravel(&oc[*axis..*axis + ir], indices.shape())];
                if raw < 0 {
                    raw += dim;
                }
                if raw < 0 || raw >= dim {
                    return Err(Error::Interp(format!(
                        "gather index {raw} out of range for dim {dim}"
                    )));
                }
                let mut ic: Vec<usize> = oc[..*axis].to_vec();
                ic.push(raw as usize);
                ic.extend_from_slice(&oc[*axis + ir..]);
                out.push(scalar_at(&v, ravel(&ic, data.shape())));
            }
            collect_scalars(out)?.encode(data.dtype(), out_shape)
        }

        Prim::Scatter => {
            let (data, indices, updates) = (ins[0], ins[1], ins[2]);
            let idx = indices.to_i64()?;
            let ir = indices.shape().len();
            let k = indices.shape()[ir - 1];
            let num_updates: usize = indices.shape()[..ir - 1].iter().product();
            let slice_len: usize = data.shape()[k..].iter().product();
            let v_data = data.decode()?;
            let v_upd = updates.decode()?;
            let n = data.numel();
            let mut out: Vec<Scalar> = (0..n).map(|i| scalar_at(&v_data, i)).collect();
            // Duplicate indices: last write wins (deterministic; ONNX leaves
            // this unspecified).
            for u in 0..num_updates {
                let mut base = 0usize;
                for d in 0..k {
                    let mut raw = idx[u * k + d];
                    let dim = data.shape()[d] as i64;
                    if raw < 0 {
                        raw += dim;
                    }
                    if raw < 0 || raw >= dim {
                        return Err(Error::Interp(format!(
                            "scatter index {raw} out of range for dim {dim}"
                        )));
                    }
                    base = base * data.shape()[d] + raw as usize;
                }
                for j in 0..slice_len {
                    out[base * slice_len + j] = scalar_at(&v_upd, u * slice_len + j);
                }
            }
            collect_scalars(out)?.encode(data.dtype(), data.shape().to_vec())
        }

        Prim::Iota { len, dtype } => {
            let n = len.eval(bindings)? as usize;
            Values::I((0..n as i64).collect()).encode(*dtype, vec![n])
        }

        Prim::Dequantize {
            block_size,
            out_shape,
        } => {
            let (data, scales) = (ins[0], ins[1]);
            let q = match data.decode()? {
                Values::I(v) => v,
                _ => unreachable!("dtype-checked by inference"),
            };
            let s = match scales.decode()? {
                Values::F(v) => v,
                _ => unreachable!("dtype-checked by inference"),
            };
            let default_zp = if data.dtype() == DataType::I4 {
                0
            } else {
                1 << (data.dtype().bits() - 1)
            };
            let zp: Option<Vec<i64>> = match ins.get(2) {
                Some(t) => Some(t.to_i64()?),
                None => None,
            };
            let dims = eval_dims(out_shape, bindings)?;
            let out: Vec<f64> = q
                .iter()
                .enumerate()
                .map(|(i, &qv)| {
                    let block = i / block_size;
                    let z = zp.as_ref().map_or(default_zp, |v| v[block]);
                    (qv - z) as f64 * s[block]
                })
                .collect();
            Values::F(out).encode(scales.dtype(), dims)
        }
    }
}

// ─────────────────────── Scalar-level semantics ────────────────────────

/// One working-domain scalar.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Scalar {
    F(f64),
    I(i64),
    B(bool),
}

fn scalar_at(v: &Values, i: usize) -> Scalar {
    match v {
        Values::F(v) => Scalar::F(v[i]),
        Values::I(v) => Scalar::I(v[i]),
        Values::B(v) => Scalar::B(v[i]),
    }
}

fn collect_scalars(scalars: Vec<Scalar>) -> Result<Values> {
    match scalars.first() {
        None | Some(Scalar::F(_)) => Ok(Values::F(
            scalars
                .into_iter()
                .map(|s| if let Scalar::F(v) = s { v } else { f64::NAN })
                .collect(),
        )),
        Some(Scalar::I(_)) => Ok(Values::I(
            scalars
                .into_iter()
                .map(|s| if let Scalar::I(v) = s { v } else { 0 })
                .collect(),
        )),
        Some(Scalar::B(_)) => Ok(Values::B(
            scalars
                .into_iter()
                .map(|s| matches!(s, Scalar::B(true)))
                .collect(),
        )),
    }
}

fn unary_f(op: UnaryOp, a: f64) -> f64 {
    match op {
        UnaryOp::Neg => -a,
        UnaryOp::Abs => a.abs(),
        UnaryOp::Sqrt => a.sqrt(),
        UnaryOp::Rsqrt => 1.0 / a.sqrt(),
        UnaryOp::Exp => a.exp(),
        UnaryOp::Log => a.ln(),
        UnaryOp::Sin => a.sin(),
        UnaryOp::Cos => a.cos(),
        UnaryOp::Tanh => a.tanh(),
        UnaryOp::Erf => erf(a),
        UnaryOp::Floor => a.floor(),
        UnaryOp::Ceil => a.ceil(),
        UnaryOp::Not => unreachable!("bool-only, dtype-checked"),
    }
}

/// Abramowitz–Stegun 7.1.26 rational approximation of erf, max abs error
/// ~1.5e-7 — beyond f32 precision, adequate for the f32/f16 value domain.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

fn binary_val(op: BinaryOp, a: Scalar, b: Scalar) -> Result<Scalar> {
    Ok(match (a, b) {
        (Scalar::F(x), Scalar::F(y)) => Scalar::F(match op {
            BinaryOp::Add => x + y,
            BinaryOp::Sub => x - y,
            BinaryOp::Mul => x * y,
            BinaryOp::Div => x / y,
            BinaryOp::Pow => x.powf(y),
            BinaryOp::Max => x.max(y),
            BinaryOp::Min => x.min(y),
            _ => unreachable!("bool op on floats, dtype-checked"),
        }),
        (Scalar::I(x), Scalar::I(y)) => Scalar::I(match op {
            BinaryOp::Add => x.wrapping_add(y),
            BinaryOp::Sub => x.wrapping_sub(y),
            BinaryOp::Mul => x.wrapping_mul(y),
            BinaryOp::Div => {
                if y == 0 {
                    return Err(Error::Interp("integer division by zero".into()));
                }
                x.wrapping_div(y)
            }
            BinaryOp::Pow => {
                if y < 0 {
                    return Err(Error::Interp("integer pow with negative exponent".into()));
                }
                let mut acc: i64 = 1;
                for _ in 0..y {
                    acc = acc.wrapping_mul(x);
                }
                acc
            }
            BinaryOp::Max => x.max(y),
            BinaryOp::Min => x.min(y),
            _ => unreachable!("bool op on ints, dtype-checked"),
        }),
        (Scalar::B(x), Scalar::B(y)) => Scalar::B(match op {
            BinaryOp::And => x && y,
            BinaryOp::Or => x || y,
            BinaryOp::Xor => x != y,
            _ => unreachable!("numeric op on bools, dtype-checked"),
        }),
        _ => unreachable!("mixed domains, dtype-checked"),
    })
}

fn compare_val(op: CmpOp, a: Scalar, b: Scalar) -> bool {
    let ord = match (a, b) {
        (Scalar::F(x), Scalar::F(y)) => x.partial_cmp(&y),
        (Scalar::I(x), Scalar::I(y)) => Some(x.cmp(&y)),
        (Scalar::B(x), Scalar::B(y)) => Some(x.cmp(&y)),
        _ => unreachable!("mixed domains, dtype-checked"),
    };
    use std::cmp::Ordering::*;
    match (op, ord) {
        // NaN comparisons: only Ne is true, matching IEEE.
        (CmpOp::Ne, None) => true,
        (_, None) => false,
        (CmpOp::Eq, Some(o)) => o == Equal,
        (CmpOp::Ne, Some(o)) => o != Equal,
        (CmpOp::Lt, Some(o)) => o == Less,
        (CmpOp::Le, Some(o)) => o != Greater,
        (CmpOp::Gt, Some(o)) => o == Greater,
        (CmpOp::Ge, Some(o)) => o != Less,
    }
}

fn elementwise2(
    out_shape: &[usize],
    a_shape: &[usize],
    b_shape: &[usize],
    a: &Values,
    b: &Values,
    f: impl Fn(Scalar, Scalar) -> Result<Scalar>,
) -> Result<Values> {
    let n: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let coords = unravel(i, out_shape);
        let x = scalar_at(a, broadcast_index(&coords, a_shape));
        let y = scalar_at(b, broadcast_index(&coords, b_shape));
        out.push(f(x, y)?);
    }
    collect_scalars(out)
}

fn matmul(a: &Tensor, b: &Tensor, trans_a: bool, trans_b: bool) -> Result<Tensor> {
    let (ar, br) = (a.shape().len(), b.shape().len());
    let (m, ka) = {
        let (r, c) = (a.shape()[ar - 2], a.shape()[ar - 1]);
        if trans_a { (c, r) } else { (r, c) }
    };
    let (kb, n) = {
        let (r, c) = (b.shape()[br - 2], b.shape()[br - 1]);
        if trans_b { (c, r) } else { (r, c) }
    };
    if ka != kb {
        return Err(Error::Interp(format!(
            "matmul contraction mismatch: {ka} vs {kb}"
        )));
    }
    let batch = broadcast_shape(&a.shape()[..ar - 2], &b.shape()[..br - 2])?;
    let mut out_shape = batch.clone();
    out_shape.extend([m, n]);

    let (va, vb) = (a.decode()?, b.decode()?);
    let is_float = a.dtype().is_float();
    let a_mat = m * ka; // elements per a-matrix
    let b_mat = kb * n;
    let num_batches: usize = batch.iter().product();
    let mut out_f = Vec::new();
    let mut out_i = Vec::new();
    if is_float {
        out_f.reserve(num_batches * m * n);
    } else {
        out_i.reserve(num_batches * m * n);
    }

    for bi in 0..num_batches {
        let coords = unravel(bi, &batch);
        let a_base = broadcast_index(&coords, &a.shape()[..ar - 2]) * a_mat;
        let b_base = broadcast_index(&coords, &b.shape()[..br - 2]) * b_mat;
        for i in 0..m {
            for j in 0..n {
                // Row-major within a matrix; transpose flips (row, col).
                let a_at = |kk: usize| a_base + if trans_a { kk * m + i } else { i * ka + kk };
                let b_at = |kk: usize| b_base + if trans_b { j * kb + kk } else { kk * n + j };
                if is_float {
                    let (Values::F(fa), Values::F(fb)) = (&va, &vb) else {
                        unreachable!()
                    };
                    let mut acc = 0.0f64;
                    for kk in 0..ka {
                        acc += fa[a_at(kk)] * fb[b_at(kk)];
                    }
                    out_f.push(acc);
                } else {
                    let (Values::I(ia), Values::I(ib)) = (&va, &vb) else {
                        unreachable!()
                    };
                    let mut acc = 0i64;
                    for kk in 0..ka {
                        acc = acc.wrapping_add(ia[a_at(kk)].wrapping_mul(ib[b_at(kk)]));
                    }
                    out_i.push(acc);
                }
            }
        }
    }
    let vals = if is_float {
        Values::F(out_f)
    } else {
        Values::I(out_i)
    };
    vals.encode(a.dtype(), out_shape)
}

fn reduce(x: &Tensor, op: ReduceOp, axes: &[usize], keepdims: bool) -> Result<Tensor> {
    let in_shape = x.shape();
    let mut out_shape = Vec::new();
    for (i, &d) in in_shape.iter().enumerate() {
        if axes.contains(&i) {
            if keepdims {
                out_shape.push(1);
            }
        } else {
            out_shape.push(d);
        }
    }
    let count: usize = axes.iter().map(|&a| in_shape[a]).product();
    let out_n: usize = out_shape.iter().product();
    let v = x.decode()?;
    let is_float = x.dtype().is_float();

    // Accumulators per output element.
    let mut acc_f = vec![init_f(op); out_n];
    let mut acc_i = vec![init_i(op); out_n];

    for i in 0..x.numel() {
        let coords = unravel(i, in_shape);
        // Output coordinate: drop (or zero) reduced axes.
        let mut oc = Vec::with_capacity(out_shape.len());
        for (d, &c) in coords.iter().enumerate() {
            if axes.contains(&d) {
                if keepdims {
                    oc.push(0);
                }
            } else {
                oc.push(c);
            }
        }
        let oi = ravel(&oc, &out_shape);
        match scalar_at(&v, i) {
            Scalar::F(val) => acc_f[oi] = step_f(op, acc_f[oi], val),
            Scalar::I(val) => acc_i[oi] = step_i(op, acc_i[oi], val),
            Scalar::B(_) => unreachable!("dtype-checked"),
        }
    }

    let vals = if is_float {
        if op == ReduceOp::Mean {
            for a in &mut acc_f {
                *a /= count as f64;
            }
        }
        Values::F(acc_f)
    } else {
        if op == ReduceOp::Mean {
            for a in &mut acc_i {
                *a /= count as i64; // truncating integer mean
            }
        }
        Values::I(acc_i)
    };
    vals.encode(x.dtype(), out_shape)
}

fn init_f(op: ReduceOp) -> f64 {
    match op {
        ReduceOp::Sum | ReduceOp::Mean => 0.0,
        ReduceOp::Prod => 1.0,
        ReduceOp::Max => f64::NEG_INFINITY,
        ReduceOp::Min => f64::INFINITY,
    }
}

fn step_f(op: ReduceOp, acc: f64, v: f64) -> f64 {
    match op {
        ReduceOp::Sum | ReduceOp::Mean => acc + v,
        ReduceOp::Prod => acc * v,
        ReduceOp::Max => acc.max(v),
        ReduceOp::Min => acc.min(v),
    }
}

fn init_i(op: ReduceOp) -> i64 {
    match op {
        ReduceOp::Sum | ReduceOp::Mean => 0,
        ReduceOp::Prod => 1,
        ReduceOp::Max => i64::MIN,
        ReduceOp::Min => i64::MAX,
    }
}

fn step_i(op: ReduceOp, acc: i64, v: i64) -> i64 {
    match op {
        ReduceOp::Sum | ReduceOp::Mean => acc.wrapping_add(v),
        ReduceOp::Prod => acc.wrapping_mul(v),
        ReduceOp::Max => acc.max(v),
        ReduceOp::Min => acc.min(v),
    }
}

// ───────────────────── Parameter evaluation helpers ────────────────────

fn eval_dims(dims: &[DimExpr], bindings: &Bindings) -> Result<Vec<usize>> {
    dims.iter()
        .map(|d| d.eval(bindings).map(|v| v as usize))
        .collect()
}

/// Resolve a reshape target, inferring at most one unresolvable bare-symbol
/// dim from the element count (the runtime face of ONNX `-1`), binding the
/// symbol as a side effect.
fn resolve_reshape(shape: &[DimExpr], numel: usize, bindings: &mut Bindings) -> Result<Vec<usize>> {
    let mut dims = Vec::with_capacity(shape.len());
    let mut unknown: Option<(usize, crate::dim::SymId)> = None;
    for (i, d) in shape.iter().enumerate() {
        match d.eval(bindings) {
            Ok(v) => dims.push(v as usize),
            Err(e) => match d.as_sym() {
                Some(sym) if unknown.is_none() => {
                    unknown = Some((i, sym));
                    dims.push(0); // placeholder
                }
                _ => return Err(e),
            },
        }
    }
    if let Some((i, sym)) = unknown {
        let rest: usize = dims
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, &d)| d)
            .product();
        if rest == 0 || numel % rest != 0 {
            return Err(Error::Interp(format!(
                "cannot infer reshape dim: {numel} elements over {rest}"
            )));
        }
        dims[i] = numel / rest;
        bindings.bind(sym, dims[i] as u64)?;
    }
    Ok(dims)
}

/// Evaluate a slice spec to `(output_len, concrete_start)`.
fn eval_slice_spec(spec: &SliceSpec, bindings: &Bindings) -> Result<(usize, i64)> {
    let start = spec.start.eval(bindings)? as i64;
    let end = spec.end.eval(bindings)? as i64;
    let ceil_div = |num: i64, d: i64| (num + d - 1).div_euclid(d);
    let len = if spec.step > 0 {
        ceil_div(end - start, spec.step)
    } else {
        ceil_div(start - end, -spec.step)
    };
    Ok((len.max(0) as usize, start))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::dim::SymbolicShape;
    use crate::types::TensorType;

    fn f32s(t: &Tensor) -> Vec<f32> {
        t.to_f32().unwrap()
    }

    /// Run a single primitive on inputs via a throwaway module.
    fn run1(prim: Prim, inputs: Vec<Tensor>) -> Result<Tensor> {
        let mut bindings = Bindings::new();
        let refs: Vec<&Tensor> = inputs.iter().collect();
        eval_prim(&prim, &refs, &mut bindings)
    }

    #[test]
    fn binary_broadcast_golden() {
        // [2,3] + [3] with broadcasting.
        let a = Tensor::from_f32(&[1., 2., 3., 4., 5., 6.], &[2, 3]).unwrap();
        let b = Tensor::from_f32(&[10., 20., 30.], &[3]).unwrap();
        let out = run1(Prim::Binary(BinaryOp::Add), vec![a, b]).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(f32s(&out), vec![11., 22., 33., 14., 25., 36.]);
    }

    #[test]
    fn int_division_semantics() {
        let a = Tensor::from_i64(&[7, -7, 1], &[3]).unwrap();
        let b = Tensor::from_i64(&[2, 2, 1], &[3]).unwrap();
        let out = run1(Prim::Binary(BinaryOp::Div), vec![a.clone(), b]).unwrap();
        assert_eq!(out.to_i64().unwrap(), vec![3, -3, 1]); // trunc toward zero
        let zero = Tensor::from_i64(&[0, 1, 1], &[3]).unwrap();
        assert!(run1(Prim::Binary(BinaryOp::Div), vec![a, zero]).is_err());
    }

    #[test]
    fn compare_nan_semantics() {
        let a = Tensor::from_f32(&[f32::NAN, 1.0], &[2]).unwrap();
        let b = Tensor::from_f32(&[f32::NAN, 1.0], &[2]).unwrap();
        let eq = run1(Prim::Compare(CmpOp::Eq), vec![a.clone(), b.clone()]).unwrap();
        assert_eq!(eq.to_bool().unwrap(), vec![false, true]);
        let ne = run1(Prim::Compare(CmpOp::Ne), vec![a, b]).unwrap();
        assert_eq!(ne.to_bool().unwrap(), vec![true, false]);
    }

    #[test]
    fn select_broadcasts() {
        let c = Tensor::from_bool(&[true, false], &[2, 1]).unwrap();
        let a = Tensor::from_f32(&[1., 2.], &[2]).unwrap();
        let b = Tensor::from_f32(&[9., 8.], &[2]).unwrap();
        let out = run1(Prim::Select, vec![c, a, b]).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(f32s(&out), vec![1., 2., 9., 8.]);
    }

    #[test]
    fn matmul_golden() {
        // [2,3] x [3,2] = [2,2]
        let a = Tensor::from_f32(&[1., 2., 3., 4., 5., 6.], &[2, 3]).unwrap();
        let b = Tensor::from_f32(&[7., 8., 9., 10., 11., 12.], &[3, 2]).unwrap();
        let out = run1(
            Prim::MatMul {
                trans_a: false,
                trans_b: false,
            },
            vec![a.clone(), b],
        )
        .unwrap();
        assert_eq!(f32s(&out), vec![58., 64., 139., 154.]);

        // trans_b: [2,3] x ([2,3])^T = [2,2]
        let bt = Tensor::from_f32(&[7., 9., 11., 8., 10., 12.], &[2, 3]).unwrap();
        let out2 = run1(
            Prim::MatMul {
                trans_a: false,
                trans_b: true,
            },
            vec![a, bt],
        )
        .unwrap();
        assert_eq!(f32s(&out2), vec![58., 64., 139., 154.]);
    }

    #[test]
    fn matmul_batch_broadcast() {
        // [2,1,2,2] x [2,2] -> [2,1,2,2]
        let a = Tensor::from_f32(&[1., 0., 0., 1., 2., 0., 0., 2.], &[2, 1, 2, 2]).unwrap();
        let b = Tensor::from_f32(&[1., 2., 3., 4.], &[2, 2]).unwrap();
        let out = run1(
            Prim::MatMul {
                trans_a: false,
                trans_b: false,
            },
            vec![a, b],
        )
        .unwrap();
        assert_eq!(out.shape(), &[2, 1, 2, 2]);
        assert_eq!(f32s(&out), vec![1., 2., 3., 4., 2., 4., 6., 8.]);
    }

    #[test]
    fn reduce_golden() {
        let x = Tensor::from_f32(&[1., 2., 3., 4., 5., 6.], &[2, 3]).unwrap();
        let sum = run1(
            Prim::Reduce {
                op: ReduceOp::Sum,
                axes: vec![1],
                keepdims: false,
            },
            vec![x.clone()],
        )
        .unwrap();
        assert_eq!(f32s(&sum), vec![6., 15.]);
        let mean = run1(
            Prim::Reduce {
                op: ReduceOp::Mean,
                axes: vec![0],
                keepdims: true,
            },
            vec![x.clone()],
        )
        .unwrap();
        assert_eq!(mean.shape(), &[1, 3]);
        assert_eq!(f32s(&mean), vec![2.5, 3.5, 4.5]);
        let max = run1(
            Prim::Reduce {
                op: ReduceOp::Max,
                axes: vec![0, 1],
                keepdims: false,
            },
            vec![x],
        )
        .unwrap();
        assert_eq!(max.shape(), &[] as &[usize]);
        assert_eq!(f32s(&max), vec![6.]);
    }

    #[test]
    fn transpose_round_trip() {
        let x = Tensor::from_f32(&[1., 2., 3., 4., 5., 6.], &[2, 3]).unwrap();
        let t = run1(Prim::Transpose { perm: vec![1, 0] }, vec![x.clone()]).unwrap();
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(f32s(&t), vec![1., 4., 2., 5., 3., 6.]);
        let back = run1(Prim::Transpose { perm: vec![1, 0] }, vec![t]).unwrap();
        assert_eq!(f32s(&back), f32s(&x)); // transpose ∘ transpose = id
    }

    #[test]
    fn concat_and_slice() {
        let a = Tensor::from_f32(&[1., 2.], &[1, 2]).unwrap();
        let b = Tensor::from_f32(&[3., 4., 5., 6.], &[2, 2]).unwrap();
        let cat = run1(Prim::Concat { axis: 0 }, vec![a, b]).unwrap();
        assert_eq!(cat.shape(), &[3, 2]);
        assert_eq!(f32s(&cat), vec![1., 2., 3., 4., 5., 6.]);

        // Slice rows [1,3) of the concat.
        let sl = run1(
            Prim::Slice {
                specs: vec![SliceSpec {
                    axis: 0,
                    start: DimExpr::constant(1),
                    end: DimExpr::constant(3),
                    step: 1,
                }],
            },
            vec![cat.clone()],
        )
        .unwrap();
        assert_eq!(f32s(&sl), vec![3., 4., 5., 6.]);

        // Negative step: reverse rows, start=2 (inclusive), end sentinel
        // normalized by lowering; here [2 .. 0) step -1 -> rows 2, 1.
        let rev = run1(
            Prim::Slice {
                specs: vec![SliceSpec {
                    axis: 0,
                    start: DimExpr::constant(2),
                    end: DimExpr::constant(0),
                    step: -1,
                }],
            },
            vec![cat],
        )
        .unwrap();
        assert_eq!(f32s(&rev), vec![5., 6., 3., 4.]);
    }

    #[test]
    fn gather_embedding_and_negative_index() {
        let table = Tensor::from_f32(&[0., 0., 1., 1., 2., 2., 3., 3.], &[4, 2]).unwrap();
        let idx = Tensor::from_i64(&[3, 0, -1], &[3]).unwrap();
        let out = run1(Prim::Gather { axis: 0 }, vec![table, idx]).unwrap();
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(f32s(&out), vec![3., 3., 0., 0., 3., 3.]);
    }

    #[test]
    fn scatter_last_write_wins() {
        let data = Tensor::from_f32(&[0.; 4], &[4]).unwrap();
        let idx = Tensor::from_i64(&[1, 1], &[2, 1]).unwrap();
        let upd = Tensor::from_f32(&[5., 7.], &[2]).unwrap();
        let out = run1(Prim::Scatter, vec![data, idx, upd]).unwrap();
        assert_eq!(f32s(&out), vec![0., 7., 0., 0.]);
    }

    #[test]
    fn cast_semantics() {
        let x = Tensor::from_f32(&[1.9, -1.9, 3.0e9], &[3]).unwrap();
        let out = run1(Prim::Cast { to: DataType::I32 }, vec![x]).unwrap();
        // Truncation toward zero; overflow saturates then wraps to i32 via
        // the working domain: 3e9 saturates i64 fine, i32 wraps.
        let vals = out.to_i64().unwrap();
        assert_eq!(vals[0], 1);
        assert_eq!(vals[1], -1);
        let b = Tensor::from_i64(&[0, 2], &[2]).unwrap();
        let as_bool = run1(Prim::Cast { to: DataType::Bool }, vec![b]).unwrap();
        assert_eq!(as_bool.to_bool().unwrap(), vec![false, true]);
    }

    #[test]
    fn f16_round_trip() {
        let x = Tensor::from_f32(&[1.5, -2.25], &[2]).unwrap();
        let h = run1(Prim::Cast { to: DataType::F16 }, vec![x]).unwrap();
        assert_eq!(h.dtype(), DataType::F16);
        let back = run1(Prim::Cast { to: DataType::F32 }, vec![h]).unwrap();
        assert_eq!(f32s(&back), vec![1.5, -2.25]); // exactly representable
    }

    #[test]
    fn iota_and_dequantize() {
        let mut b = Bindings::new();
        let out = eval_prim(
            &Prim::Iota {
                len: DimExpr::constant(4),
                dtype: DataType::I64,
            },
            &[],
            &mut b,
        )
        .unwrap();
        assert_eq!(out.to_i64().unwrap(), vec![0, 1, 2, 3]);

        // u4 block dequant: one block of 4 elements, scale 0.5, default zp 8.
        // Elements 8, 9, 10, 15 -> (q-8)*0.5 = 0.0, 0.5, 1.0, 3.5
        let data = Tensor::new(DataType::U4, vec![1, 4], vec![0x98, 0xFA]).unwrap();
        let scales = Tensor::from_f32(&[0.5], &[1]).unwrap();
        let out = run1(
            Prim::Dequantize {
                block_size: 4,
                out_shape: vec![DimExpr::constant(4)],
            },
            vec![data, scales],
        )
        .unwrap();
        assert_eq!(f32s(&out), vec![0.0, 0.5, 1.0, 3.5]);
    }

    #[test]
    fn end_to_end_rms_norm_with_symbols() {
        // The builder doc-test graph, evaluated with S bound from the input.
        let mut b = GraphBuilder::new();
        let s = b.sym("S");
        let x = b.input(
            "x",
            TensorType::new(DataType::F32, SymbolicShape(vec![s, DimExpr::constant(4)])),
        );
        let sq = b.mul(x, x).unwrap();
        let ms = b.reduce(ReduceOp::Mean, sq, &[1], true).unwrap();
        let eps = b.const_f32(&[1e-6], &[1]).unwrap();
        let denom = b.add(ms, eps).unwrap();
        let inv = b.unary(UnaryOp::Rsqrt, denom).unwrap();
        let out = b.mul(x, inv).unwrap();
        b.output("out", out);
        let module = b.finish().unwrap();

        let input = Tensor::from_f32(&[1., 1., 1., 1., 2., 2., 2., 2.], &[2, 4]).unwrap();
        let outputs = eval(&module, &[("x", input)]).unwrap();
        let (name, t) = &outputs[0];
        assert_eq!(name, "out");
        assert_eq!(t.shape(), &[2, 4]);
        for v in f32s(t) {
            assert!((v - 1.0).abs() < 1e-4, "rms-normalized value ~1, got {v}");
        }
    }

    #[test]
    fn binding_validation() {
        let mut b = GraphBuilder::new();
        let s = b.sym("S");
        // Two inputs share S — must agree.
        let x = b.input(
            "x",
            TensorType::new(DataType::F32, SymbolicShape(vec![s.clone()])),
        );
        let y = b.input("y", TensorType::new(DataType::F32, SymbolicShape(vec![s])));
        let sum = b.add(x, y).unwrap();
        b.output("sum", sum);
        let m = b.finish().unwrap();

        let t3 = Tensor::from_f32(&[1., 2., 3.], &[3]).unwrap();
        let t4 = Tensor::from_f32(&[1., 2., 3., 4.], &[4]).unwrap();
        assert!(eval(&m, &[("x", t3.clone()), ("y", t3.clone())]).is_ok());
        let err = eval(&m, &[("x", t3), ("y", t4)]).unwrap_err().to_string();
        assert!(err.contains("conflicting values"), "got: {err}");
    }

    #[test]
    fn late_bound_reshape_infers_dim() {
        // Reshape [2, 6] -> [L, 3] where L is a fresh symbol: interpreter
        // infers L = 4 from the element count.
        let mut b = GraphBuilder::new();
        let x = b.input("x", TensorType::of(DataType::F32, &[2, 6]));
        let l = b.sym("L");
        let out = b.reshape(x, vec![l, DimExpr::constant(3)]).unwrap();
        b.output("out", out);
        let m = b.finish().unwrap();

        let input =
            Tensor::from_f32(&(0..12).map(|i| i as f32).collect::<Vec<_>>(), &[2, 6]).unwrap();
        let outputs = eval(&m, &[("x", input)]).unwrap();
        assert_eq!(outputs[0].1.shape(), &[4, 3]);
    }
}
