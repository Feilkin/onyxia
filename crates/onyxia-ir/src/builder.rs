//! Checked graph construction.
//!
//! [`GraphBuilder`] is the API decomposition authors and lowering rules use.
//! Types are inferred **eagerly**: each `prim` call runs shape inference for
//! the new node, so authors can chain results and mistakes surface at the
//! call site, not at some later validation pass.
//!
//! ```
//! use onyxia_ir::{DataType, GraphBuilder, TensorType};
//!
//! // out = x * rsqrt(mean(x*x) + eps)   — the RMS-norm core.
//! let mut b = GraphBuilder::new();
//! let s = b.sym("S");
//! let x = b.input("x", TensorType::new(
//!     DataType::F32,
//!     onyxia_ir::SymbolicShape(vec![s.into(), 640u64.into()]),
//! ));
//! let sq = b.mul(x, x).unwrap();
//! let ms = b.reduce(onyxia_ir::ReduceOp::Mean, sq, &[1], true).unwrap();
//! let eps = b.const_f32(&[1e-6], &[1]).unwrap();
//! let denom = b.add(ms, eps).unwrap();
//! let inv = b.unary(onyxia_ir::UnaryOp::Rsqrt, denom).unwrap();
//! let out = b.mul(x, inv).unwrap();
//! b.output("out", out);
//! let module = b.finish().unwrap();
//! assert_eq!(module.nodes.len(), 5);
//! ```

use crate::attrs::Attrs;
use crate::dim::{DimExpr, SymId};
use crate::graph::{Composite, Module, Node, NodeKind, Origin, SourceInfo, ValueDef, ValueId};
use crate::infer::infer_prim;
use crate::prim::{BinaryOp, CmpOp, Prim, ReduceOp, SliceSpec, UnaryOp};
use crate::types::{DataType, TensorType};
use crate::{Error, Result};

/// Builds a [`Module`] with eager type inference and final validation.
#[derive(Debug, Default)]
pub struct GraphBuilder {
    module: Module,
    /// Provenance to stamp on nodes created next (used by lowering).
    loc: SourceInfo,
}

impl GraphBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Wrap an existing module for further construction (used by lowering
    /// and composite inlining). The module is returned by
    /// [`into_module`](Self::into_module) or, validated, by
    /// [`finish`](Self::finish).
    pub fn from_module(module: Module) -> Self {
        Self {
            module,
            loc: SourceInfo::default(),
        }
    }

    /// Return the module **without** validating. Prefer
    /// [`finish`](Self::finish) at natural completion points.
    pub fn into_module(self) -> Module {
        self.module
    }

    /// Intern a dimension symbol by name.
    pub fn sym(&mut self, name: &str) -> DimExpr {
        DimExpr::sym(self.module.symbols.intern(name))
    }

    /// Allocate a fresh (late-bound) dimension symbol.
    pub fn fresh_sym(&mut self, hint: &str) -> SymId {
        self.module.symbols.fresh(hint)
    }

    /// Set the provenance stamped on subsequently created nodes.
    pub fn set_loc(&mut self, loc: SourceInfo) {
        self.loc = loc;
    }

    /// The type of an existing value.
    pub fn ty(&self, v: ValueId) -> &TensorType {
        &self.module.value(v).ty
    }

    /// Read access to the module under construction.
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// Mutable access to the module under construction. Intended for
    /// metadata fix-ups (value naming, const-pool reinterpretation) by
    /// lowering — structural edits should go through builder methods.
    pub fn module_mut(&mut self) -> &mut Module {
        &mut self.module
    }

    /// Declare a graph input.
    pub fn input(&mut self, name: &str, ty: TensorType) -> ValueId {
        let id = self.module.add_value(ValueDef {
            name: Some(name.to_string()),
            ty,
            origin: Origin::Input,
            content: None,
        });
        self.module.inputs.push((name.to_string(), id));
        id
    }

    /// Add a constant from raw bytes.
    pub fn constant(&mut self, ty: TensorType, data: Vec<u8>) -> Result<ValueId> {
        let cid = self.module.consts.add(ty.clone(), data)?;
        Ok(self.module.add_value(ValueDef {
            name: None,
            ty,
            origin: Origin::Const(cid),
            content: None,
        }))
    }

    /// Add an f32 constant.
    pub fn const_f32(&mut self, values: &[f32], dims: &[u64]) -> Result<ValueId> {
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.constant(TensorType::of(DataType::F32, dims), bytes)
    }

    /// Add an i64 constant.
    pub fn const_i64(&mut self, values: &[i64], dims: &[u64]) -> Result<ValueId> {
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.constant(TensorType::of(DataType::I64, dims), bytes)
    }

    /// Add a primitive node (single output — all current primitives).
    ///
    /// Runs shape inference immediately; errors point at this call.
    pub fn prim(&mut self, prim: Prim, inputs: &[ValueId]) -> Result<ValueId> {
        let input_tys: Vec<&TensorType> =
            inputs.iter().map(|&v| &self.module.value(v).ty).collect();
        let out_tys = infer_prim(&prim, &input_tys).map_err(|e| {
            Error::Shape(format!(
                "{} (while building '{}' node{})",
                e,
                prim.name(),
                self.loc
                    .name
                    .as_deref()
                    .map(|n| format!(" for {n}"))
                    .unwrap_or_default()
            ))
        })?;
        let node_index = self.module.nodes.len() as u32;
        let outputs: Vec<ValueId> = out_tys
            .into_iter()
            .enumerate()
            .map(|(i, ty)| {
                self.module.add_value(ValueDef {
                    name: None,
                    ty,
                    origin: Origin::Node {
                        node: crate::graph::NodeId::from_index(node_index),
                        output: i,
                    },
                    content: None,
                })
            })
            .collect();
        let out = outputs[0];
        self.module.add_node(Node {
            kind: NodeKind::Prim(prim),
            inputs: inputs.to_vec(),
            outputs,
            loc: self.loc.clone(),
        });
        Ok(out)
    }

    /// Add a composite node. Output types must be declared by the caller —
    /// they are not derived through the decomposition here (inlining
    /// cross-checks them later).
    pub fn composite(
        &mut self,
        name: &str,
        attrs: Attrs,
        inputs: &[ValueId],
        output_tys: Vec<TensorType>,
    ) -> Result<Vec<ValueId>> {
        if output_tys.is_empty() {
            return Err(Error::InvalidGraph(format!(
                "composite '{name}' must declare at least one output"
            )));
        }
        let node_index = self.module.nodes.len() as u32;
        let outputs: Vec<ValueId> = output_tys
            .into_iter()
            .enumerate()
            .map(|(i, ty)| {
                self.module.add_value(ValueDef {
                    name: None,
                    ty,
                    origin: Origin::Node {
                        node: crate::graph::NodeId::from_index(node_index),
                        output: i,
                    },
                    content: None,
                })
            })
            .collect();
        self.module.add_node(Node {
            kind: NodeKind::Composite(Composite {
                name: name.to_string(),
                attrs,
            }),
            inputs: inputs.to_vec(),
            outputs: outputs.clone(),
            loc: self.loc.clone(),
        });
        Ok(outputs)
    }

    /// Mark a value as a named graph output.
    pub fn output(&mut self, name: &str, v: ValueId) {
        if self.module.value(v).name.is_none() {
            self.module.value_mut(v).name = Some(name.to_string());
        }
        self.module.outputs.push((name.to_string(), v));
    }

    /// Finish, validate, and return the module.
    pub fn finish(self) -> Result<Module> {
        crate::validate::validate(&self.module)?;
        Ok(self.module)
    }

    // ── Sugar for common primitives ─────────────────────────────────────

    /// Element-wise unary op.
    pub fn unary(&mut self, op: UnaryOp, x: ValueId) -> Result<ValueId> {
        self.prim(Prim::Unary(op), &[x])
    }

    /// `a + b`.
    pub fn add(&mut self, a: ValueId, b: ValueId) -> Result<ValueId> {
        self.prim(Prim::Binary(BinaryOp::Add), &[a, b])
    }

    /// `a - b`.
    pub fn sub(&mut self, a: ValueId, b: ValueId) -> Result<ValueId> {
        self.prim(Prim::Binary(BinaryOp::Sub), &[a, b])
    }

    /// `a * b`.
    pub fn mul(&mut self, a: ValueId, b: ValueId) -> Result<ValueId> {
        self.prim(Prim::Binary(BinaryOp::Mul), &[a, b])
    }

    /// `a / b`.
    pub fn div(&mut self, a: ValueId, b: ValueId) -> Result<ValueId> {
        self.prim(Prim::Binary(BinaryOp::Div), &[a, b])
    }

    /// Element-wise comparison.
    pub fn cmp(&mut self, op: CmpOp, a: ValueId, b: ValueId) -> Result<ValueId> {
        self.prim(Prim::Compare(op), &[a, b])
    }

    /// `cond ? a : b`.
    pub fn select(&mut self, cond: ValueId, a: ValueId, b: ValueId) -> Result<ValueId> {
        self.prim(Prim::Select, &[cond, a, b])
    }

    /// Dtype conversion.
    pub fn cast(&mut self, x: ValueId, to: DataType) -> Result<ValueId> {
        self.prim(Prim::Cast { to }, &[x])
    }

    /// Batched matrix multiply.
    pub fn matmul(&mut self, a: ValueId, b: ValueId) -> Result<ValueId> {
        self.prim(
            Prim::MatMul {
                trans_a: false,
                trans_b: false,
            },
            &[a, b],
        )
    }

    /// Reduction over `axes`.
    pub fn reduce(
        &mut self,
        op: ReduceOp,
        x: ValueId,
        axes: &[usize],
        keepdims: bool,
    ) -> Result<ValueId> {
        self.prim(
            Prim::Reduce {
                op,
                axes: axes.to_vec(),
                keepdims,
            },
            &[x],
        )
    }

    /// Reshape to a symbolic target.
    pub fn reshape(&mut self, x: ValueId, shape: Vec<DimExpr>) -> Result<ValueId> {
        self.prim(Prim::Reshape { shape }, &[x])
    }

    /// Permute dimensions.
    pub fn transpose(&mut self, x: ValueId, perm: &[usize]) -> Result<ValueId> {
        self.prim(
            Prim::Transpose {
                perm: perm.to_vec(),
            },
            &[x],
        )
    }

    /// Broadcast (ONNX `Expand`) to a target shape.
    pub fn broadcast(&mut self, x: ValueId, shape: Vec<DimExpr>) -> Result<ValueId> {
        self.prim(Prim::Broadcast { shape }, &[x])
    }

    /// Concatenate along `axis`.
    pub fn concat(&mut self, inputs: &[ValueId], axis: usize) -> Result<ValueId> {
        self.prim(Prim::Concat { axis }, inputs)
    }

    /// Slice by specs.
    pub fn slice(&mut self, x: ValueId, specs: Vec<SliceSpec>) -> Result<ValueId> {
        self.prim(Prim::Slice { specs }, &[x])
    }

    /// Gather along `axis`.
    pub fn gather(&mut self, data: ValueId, indices: ValueId, axis: usize) -> Result<ValueId> {
        self.prim(Prim::Gather { axis }, &[data, indices])
    }

    /// The ramp `0..len` as `dtype`.
    pub fn iota(&mut self, len: DimExpr, dtype: DataType) -> Result<ValueId> {
        self.prim(Prim::Iota { len, dtype }, &[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_and_validates() {
        let mut b = GraphBuilder::new();
        let x = b.input("x", TensorType::of(DataType::F32, &[2, 3]));
        let y = b.input("y", TensorType::of(DataType::F32, &[3]));
        let sum = b.add(x, y).unwrap();
        let out = b.unary(UnaryOp::Neg, sum).unwrap();
        b.output("out", out);
        let m = b.finish().unwrap();
        assert_eq!(m.nodes.len(), 2);
        assert_eq!(m.value(out).ty, TensorType::of(DataType::F32, &[2, 3]));
        assert_eq!(m.inputs.len(), 2);
        assert_eq!(m.outputs.len(), 1);
    }

    #[test]
    fn eager_inference_errors_at_call_site() {
        let mut b = GraphBuilder::new();
        let x = b.input("x", TensorType::of(DataType::F32, &[2]));
        let y = b.input("y", TensorType::of(DataType::I64, &[2]));
        // Mismatched dtypes surface immediately.
        assert!(b.add(x, y).is_err());
    }

    #[test]
    fn composite_requires_outputs() {
        let mut b = GraphBuilder::new();
        let x = b.input("x", TensorType::of(DataType::F32, &[4]));
        assert!(b.composite("Softmax", Attrs::new(), &[x], vec![]).is_err());
        let outs = b
            .composite(
                "Softmax",
                Attrs::new(),
                &[x],
                vec![TensorType::of(DataType::F32, &[4])],
            )
            .unwrap();
        b.output("out", outs[0]);
        assert!(b.finish().is_ok());
    }

    #[test]
    fn constants_round_trip() {
        let mut b = GraphBuilder::new();
        let c = b.const_f32(&[1.0, 2.0], &[2]).unwrap();
        let i = b.const_i64(&[5], &[1]).unwrap();
        assert_eq!(b.ty(c).dtype, DataType::F32);
        assert_eq!(b.ty(i).dtype, DataType::I64);
        let m = b.module();
        assert_eq!(m.consts.len(), 2);
        assert_eq!(
            m.consts.bytes(crate::graph::ConstId::from_index(0)).len(),
            8
        );
    }
}
