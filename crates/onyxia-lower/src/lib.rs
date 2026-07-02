//! ONNX frontend: lowers a parsed [`onyxia_onnx::Graph`] into an
//! [`onyxia_ir::Module`].
//!
//! Lowering is where all ONNX-isms die (`doc/ir-design.md` §4):
//!
//! - Every op enters the IR through the [`LoweringRegistry`] — built-ins and
//!   custom ops through the same door. A rule either emits primitives or a
//!   composite node (whose decomposition lives in
//!   [`onyxia_ir::decomp`]).
//! - Attributes are normalized here (defaults resolved, negative axes
//!   fixed up), so decompositions and backend kernels see one clean form.
//! - **Shape arithmetic evaluates symbolically**: `Shape` produces a
//!   compile-time value ([`SymbolicContent`]), and Gather/Concat/arith over
//!   such values fold via [`onyxia_ir::fold::eval_content`] instead of
//!   emitting nodes. Reshape targets, Slice bounds, and Expand shapes are
//!   resolved to [`DimExpr`]s. A chain that escapes the symbolic fragment
//!   becomes a late-bound symbol; one that is consumed by an actual runtime
//!   tensor operation is materialized when constant, and is an error
//!   otherwise (transformer graphs don't do this).
//! - Weight initializers move (not copy) into the module's
//!   [`ConstPool`](onyxia_ir::ConstPool).
//!
//! Version handling: `onyxia-onnx` does not currently expose opset imports,
//! so rules detect old-vs-new op forms *structurally* (e.g. Reduce axes as
//! attribute vs as input), which covers every model we target. Opset-range
//! keying can be added to the registry when a real conflict appears.

mod rules;

use onyxia_ir::graph::{Origin, SourceInfo, SymbolicContent};
use onyxia_ir::{
    AttrValue, Attrs, DataType, DimExpr, Error, GraphBuilder, Module, Prim, Result, SymbolicShape,
    TensorType, ValueId, fold,
};
use std::collections::HashMap;

/// A lowering rule: consume one ONNX node, emit IR through the context.
pub type Rule = fn(&mut LowerCtx) -> Result<()>;

/// Registry of lowering rules, keyed by `(domain, op_type)`.
///
/// The empty domain and `"ai.onnx"` are treated as the same (standard)
/// domain.
#[derive(Default)]
pub struct LoweringRegistry {
    rules: HashMap<(String, String), Rule>,
}

impl LoweringRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a rule for `(domain, op_type)`, replacing any existing one.
    pub fn register(&mut self, domain: &str, op_type: &str, rule: Rule) {
        self.rules
            .insert((normalize_domain(domain), op_type.to_string()), rule);
    }

    /// Look up a rule.
    pub fn get(&self, domain: &str, op_type: &str) -> Option<Rule> {
        self.rules
            .get(&(normalize_domain(domain), op_type.to_string()))
            .copied()
    }
}

fn normalize_domain(domain: &str) -> String {
    if domain == "ai.onnx" {
        String::new()
    } else {
        domain.to_string()
    }
}

/// The rules shipped with onyxia: standard ONNX ops plus the
/// `com.microsoft` contrib ops the Gemma models use.
pub fn standard_registry() -> LoweringRegistry {
    let mut r = LoweringRegistry::new();
    rules::register_all(&mut r);
    r
}

/// What an ONNX tensor name currently lowers to.
#[derive(Debug, Clone)]
pub enum Lowered {
    /// A runtime IR value.
    Value(ValueId),
    /// A compile-time shape-domain value (i64, rank ≤ 1) that has no
    /// runtime representation — consumed symbolically by rules.
    Content(SymbolicContent),
}

/// Per-node context handed to lowering rules.
pub struct LowerCtx<'a> {
    node: &'a onyxia_onnx::Node,
    builder: &'a mut GraphBuilder,
    values: &'a mut HashMap<String, Lowered>,
}

impl LowerCtx<'_> {
    /// Number of declared inputs (including absent optionals).
    pub fn num_inputs(&self) -> usize {
        self.node.inputs.len()
    }

    /// Whether input `i` is present (ONNX marks absent optionals with "").
    pub fn has_input(&self, i: usize) -> bool {
        self.node.inputs.get(i).is_some_and(|n| !n.is_empty())
    }

    fn lowered(&self, i: usize) -> Result<&Lowered> {
        let name = &self.node.inputs[i];
        self.values.get(name).ok_or_else(|| {
            Error::InvalidGraph(format!(
                "node '{}': input '{name}' has not been lowered (graph not topological?)",
                self.node.name
            ))
        })
    }

    /// Input `i` as currently lowered, without materializing anything.
    pub fn peek(&self, i: usize) -> Result<&Lowered> {
        self.lowered(i)
    }

    /// Record output `i` from an existing lowered form (Identity-style).
    pub fn set_lowered(&mut self, i: usize, lowered: Lowered) {
        match lowered {
            Lowered::Value(v) => self.set_value(i, v),
            Lowered::Content(c) => self.set_content(i, c),
        }
    }

    /// Input `i` as a runtime value, materializing constant shape-domain
    /// content when necessary.
    pub fn value(&mut self, i: usize) -> Result<ValueId> {
        match self.lowered(i)?.clone() {
            Lowered::Value(v) => Ok(v),
            Lowered::Content(c) => {
                let consts: Option<Vec<i64>> = c.elems.iter().map(signed_const_of).collect();
                let Some(vals) = consts else {
                    return Err(Error::Unsupported(format!(
                        "node '{}': a symbolic shape value is consumed by a runtime \
                         tensor operation — cannot materialize",
                        self.node.name
                    )));
                };
                let dims: Vec<u64> = c.shape.iter().map(|&d| d as u64).collect();
                self.builder.const_i64(&vals, &dims)
            }
        }
    }

    /// Input `i` as compile-time content, if known: either an explicit
    /// shape-domain value, or a small i64 constant.
    pub fn content(&self, i: usize) -> Option<SymbolicContent> {
        match self.lowered(i).ok()? {
            Lowered::Content(c) => Some(c.clone()),
            Lowered::Value(v) => content_of_value(self.builder, *v),
        }
    }

    /// Content of input `i` as concrete i64s, when fully constant.
    pub fn const_ints(&self, i: usize) -> Option<Vec<i64>> {
        self.content(i)?.elems.iter().map(signed_const_of).collect()
    }

    /// The type of a lowered value.
    pub fn ty(&self, v: ValueId) -> &TensorType {
        self.builder.ty(v)
    }

    /// Builder access for rules that need more than [`emit`](Self::emit).
    pub fn builder(&mut self) -> &mut GraphBuilder {
        self.builder
    }

    /// Emit a primitive over runtime values.
    pub fn emit(&mut self, prim: Prim, inputs: &[ValueId]) -> Result<ValueId> {
        self.builder.prim(prim, inputs)
    }

    /// Try to evaluate `prim` over the inputs' compile-time contents.
    /// On success the node produces no runtime IR.
    pub fn try_content(&mut self, prim: &Prim) -> Result<bool> {
        let contents: Vec<Option<SymbolicContent>> =
            (0..self.num_inputs()).map(|i| self.content(i)).collect();
        let refs: Vec<Option<&SymbolicContent>> = contents.iter().map(|c| c.as_ref()).collect();
        if let Some(out) = fold::eval_content(prim, &refs) {
            self.set_content(0, out);
            return Ok(true);
        }
        Ok(false)
    }

    /// Record output `i` as a runtime value.
    pub fn set_value(&mut self, i: usize, v: ValueId) {
        let name = self.node.outputs[i].clone();
        if self.builder.module().value(v).name.is_none() {
            // Keep the ONNX tensor name for I/O mapping and debugging.
            let module = self.builder_module_mut();
            module.value_mut(v).name = Some(name.clone());
        }
        self.values.insert(name, Lowered::Value(v));
    }

    /// Record output `i` as compile-time content (no runtime value).
    pub fn set_content(&mut self, i: usize, c: SymbolicContent) {
        self.values
            .insert(self.node.outputs[i].clone(), Lowered::Content(c));
    }

    // Attribute access (ONNX-typed) ---------------------------------------

    /// Optional i64 attribute.
    pub fn attr_i(&self, name: &str) -> Option<i64> {
        match self.node.attributes.get(name) {
            Some(onyxia_onnx::AttributeValue::Int(v)) => Some(*v),
            _ => None,
        }
    }

    /// Optional f32 attribute.
    pub fn attr_f(&self, name: &str) -> Option<f32> {
        match self.node.attributes.get(name) {
            Some(onyxia_onnx::AttributeValue::Float(v)) => Some(*v),
            _ => None,
        }
    }

    /// Optional i64-list attribute.
    pub fn attr_is(&self, name: &str) -> Option<Vec<i64>> {
        match self.node.attributes.get(name) {
            Some(onyxia_onnx::AttributeValue::Ints(v)) => Some(v.clone()),
            _ => None,
        }
    }

    /// Optional string attribute.
    pub fn attr_s(&self, name: &str) -> Option<&str> {
        match self.node.attributes.get(name) {
            Some(onyxia_onnx::AttributeValue::String(v)) => Some(v),
            _ => None,
        }
    }

    /// Optional tensor attribute (dtype, dims, and raw bytes).
    pub fn attr_tensor(&self, name: &str) -> Option<&onyxia_onnx::AttrTensor> {
        match self.node.attributes.get(name) {
            Some(onyxia_onnx::AttributeValue::Tensor(t)) => Some(t),
            _ => None,
        }
    }

    /// Required attribute error helper.
    pub fn missing_attr(&self, name: &str) -> Error {
        Error::Attribute(format!(
            "node '{}' ({}): missing required attribute '{name}'",
            self.node.name, self.node.op_type
        ))
    }

    /// Normalize a possibly-negative axis against `rank`.
    pub fn norm_axis(&self, axis: i64, rank: usize) -> Result<usize> {
        let a = if axis < 0 { axis + rank as i64 } else { axis };
        if a < 0 || a as usize >= rank {
            return Err(Error::Shape(format!(
                "node '{}': axis {axis} out of range for rank {rank}",
                self.node.name
            )));
        }
        Ok(a as usize)
    }

    /// The node's error-message identity.
    pub fn node_name(&self) -> &str {
        &self.node.name
    }

    fn builder_module_mut(&mut self) -> &mut Module {
        // GraphBuilder doesn't expose &mut Module; go through a scoped
        // rebuild-free accessor.
        self.builder.module_mut()
    }
}

/// If a value is a small i64 constant (or carries content), view it as
/// compile-time content.
fn content_of_value(builder: &GraphBuilder, v: ValueId) -> Option<SymbolicContent> {
    const MAX_ELEMS: usize = 64;
    let module = builder.module();
    let def = module.value(v);
    if let Some(c) = &def.content {
        return Some(c.clone());
    }
    let Origin::Const(cid) = def.origin else {
        return None;
    };
    if def.ty.dtype != DataType::I64 || def.ty.shape.rank() > 1 {
        return None;
    }
    let dims = def.ty.shape.as_static()?;
    let n: u64 = dims.iter().product();
    if n as usize > MAX_ELEMS {
        return None;
    }
    let bytes = module.consts.bytes(cid);
    let vals: Vec<i64> = bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let elems: Vec<DimExpr> = vals.iter().map(|&v| signed_dim(v)).collect();
    Some(if dims.is_empty() {
        SymbolicContent::scalar(elems.into_iter().next()?)
    } else {
        SymbolicContent::vector(elems)
    })
}

/// A possibly-negative constant as a `DimExpr`.
pub(crate) fn signed_dim(v: i64) -> DimExpr {
    if v >= 0 {
        DimExpr::constant(v as u64)
    } else {
        DimExpr::constant(0) - DimExpr::constant(v.unsigned_abs())
    }
}

/// Signed constant value of an expression, if constant.
pub(crate) fn signed_const_of(e: &DimExpr) -> Option<i64> {
    if !e.is_const() {
        return None;
    }
    let b = onyxia_ir::Bindings::new();
    e.eval(&b).map(|v| v as i64).ok().or_else(|| {
        (DimExpr::constant(0) - e.clone())
            .eval(&b)
            .ok()
            .map(|v| -(v as i64))
    })
}

/// Map an ONNX dtype to the IR dtype.
pub(crate) fn convert_dtype(dt: onyxia_onnx::DataType) -> DataType {
    use onyxia_onnx::DataType as D;
    match dt {
        D::F32 => DataType::F32,
        D::F16 => DataType::F16,
        D::I32 => DataType::I32,
        D::I64 => DataType::I64,
        D::U8 => DataType::U8,
        D::U32 => DataType::U32,
        D::Bool => DataType::Bool,
        // onyxia-onnx's quantized markers; storage is byte-identical.
        D::Q4 => DataType::U4,
        D::Q8 => DataType::U8,
    }
}

/// Map an ONNX `TensorProto.DataType` integer (Cast's `to` attribute).
pub(crate) fn convert_proto_dtype(v: i64) -> Result<DataType> {
    Ok(match v {
        1 => DataType::F32,
        2 => DataType::U8,
        3 => DataType::I8,
        6 => DataType::I32,
        7 => DataType::I64,
        9 => DataType::Bool,
        10 => DataType::F16,
        12 => DataType::U32,
        21 => DataType::U4,
        22 => DataType::I4,
        other => {
            return Err(Error::Unsupported(format!("ONNX tensor data type {other}")));
        }
    })
}

/// Convert an ONNX shape to a symbolic shape, interning named dims.
fn convert_shape(
    shape: &onyxia_onnx::TensorShape,
    builder: &mut GraphBuilder,
    what: &str,
) -> Result<SymbolicShape> {
    use onyxia_onnx::{Dimension, TensorShape};
    match shape {
        TensorShape::Static(dims) => Ok(SymbolicShape(
            dims.iter().map(|&d| DimExpr::constant(d as u64)).collect(),
        )),
        TensorShape::Dynamic(dims) => Ok(SymbolicShape(
            dims.iter()
                .map(|d| match d {
                    Dimension::Static(v) => DimExpr::constant(*v as u64),
                    Dimension::Named(n) => builder.sym(n),
                })
                .collect(),
        )),
        TensorShape::Unknown | TensorShape::Absent => Err(Error::Shape(format!(
            "{what}: rank must be known (shape is {shape:?})"
        ))),
    }
}

/// Statistics reported by [`lower`], for the whole-model gate.
#[derive(Debug, Default, Clone)]
pub struct LowerStats {
    /// ONNX nodes consumed.
    pub onnx_nodes: usize,
    /// IR nodes produced (after folding).
    pub ir_nodes: usize,
    /// IR nodes that are composites (after folding).
    pub composite_nodes: usize,
    /// Constants pooled (count, total bytes).
    pub consts: (usize, usize),
    /// Dimension symbols created.
    pub symbols: usize,
}

/// Lower a parsed ONNX graph into a validated IR module.
///
/// Consumes the graph so weight initializers *move* into the constant pool.
/// Runs constant folding + DCE and validates before returning.
pub fn lower(graph: onyxia_onnx::Graph, registry: &LoweringRegistry) -> Result<Module> {
    lower_with_stats(graph, registry).map(|(m, _)| m)
}

/// [`lower`], also returning statistics.
pub fn lower_with_stats(
    mut graph: onyxia_onnx::Graph,
    registry: &LoweringRegistry,
) -> Result<(Module, LowerStats)> {
    let mut stats = LowerStats {
        onnx_nodes: graph.nodes.len(),
        ..Default::default()
    };
    let mut builder = GraphBuilder::new();
    let mut values: HashMap<String, Lowered> = HashMap::new();

    // Graph inputs.
    for name in graph.inputs.clone() {
        let info = graph.tensor_by_name(&name).map_err(onnx_err)?;
        let dtype = convert_dtype(info.dtype);
        let shape = convert_shape(&info.shape, &mut builder, &format!("input '{name}'"))?;
        let v = builder.input(&name, TensorType::new(dtype, shape));
        values.insert(name, Lowered::Value(v));
    }

    // Initializers (weights and Constant-node payloads) move into the pool.
    for info in &mut graph.tensor_info {
        let Some(data) = info.initializer.take() else {
            continue;
        };
        if values.contains_key(&info.name) {
            continue; // an input with a default initializer: input wins
        }
        let dtype = convert_dtype(info.dtype);
        let Some(dims) = info.shape.as_static() else {
            return Err(Error::Shape(format!(
                "initializer '{}' must have a static shape",
                info.name
            )));
        };
        let ty = TensorType::of(dtype, &dims.iter().map(|&d| d as u64).collect::<Vec<_>>());
        let v = builder.constant(ty, data)?;
        builder.module_mut().value_mut(v).name = Some(info.name.clone());
        values.insert(info.name.clone(), Lowered::Value(v));
    }

    // Nodes, in graph order (ONNX requires topological order).
    for node in &graph.nodes {
        if node.op_type == "Constant" {
            continue; // payload extracted by the parser into an initializer
        }
        let rule = registry.get(&node.domain, &node.op_type).ok_or_else(|| {
            Error::Unsupported(format!(
                "no lowering rule for '{}{}{}' (node '{}')",
                node.domain,
                if node.domain.is_empty() { "" } else { "." },
                node.op_type,
                node.name
            ))
        })?;
        builder.set_loc(SourceInfo {
            name: Some(node.name.clone()),
            op_type: Some(node.op_type.clone()),
        });
        let mut ctx = LowerCtx {
            node,
            builder: &mut builder,
            values: &mut values,
        };
        rule(&mut ctx)?;
    }

    // Graph outputs.
    for name in &graph.outputs {
        match values.get(name) {
            Some(Lowered::Value(v)) => {
                let v = *v;
                builder.output(name, v);
            }
            Some(Lowered::Content(c)) => {
                let vals: Option<Vec<i64>> = c.elems.iter().map(signed_const_of).collect();
                let Some(vals) = vals else {
                    return Err(Error::Unsupported(format!(
                        "graph output '{name}' is a symbolic shape value"
                    )));
                };
                let dims: Vec<u64> = c.shape.iter().map(|&d| d as u64).collect();
                let v = builder.const_i64(&vals, &dims)?;
                builder.output(name, v);
            }
            None => {
                return Err(Error::InvalidGraph(format!(
                    "graph output '{name}' was never produced"
                )));
            }
        }
    }

    let mut module = builder.into_module();
    fold::fold(&mut module, &fold::FoldOptions::default())?;
    onyxia_ir::validate::validate(&module)?;

    stats.ir_nodes = module.nodes.len();
    stats.composite_nodes = module
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, onyxia_ir::NodeKind::Composite(_)))
        .count();
    stats.consts = (module.consts.len(), module.consts.total_bytes());
    stats.symbols = module.symbols.len();
    Ok((module, stats))
}

fn onnx_err(e: onyxia_onnx::OnnxError) -> Error {
    Error::InvalidGraph(e.to_string())
}

/// Convert normalized attrs into IR [`Attrs`] entries (used by composite
/// rules).
pub(crate) fn attrs(pairs: Vec<(&str, AttrValue)>) -> Attrs {
    let mut a = Attrs::new();
    for (k, v) in pairs {
        a.set(k, v);
    }
    a
}
