//! The IR graph: an SSA value graph.
//!
//! A [`Module`] holds a table of *values* (tensors, each defined exactly
//! once) and a list of *nodes* (operations producing values). There is no
//! aliasing and no mutation: a node consumes values and produces new ones.
//! Buffer reuse is a backend concern derived from liveness, never
//! represented here (`doc/ir-design.md` §3).
//!
//! Weights live out-of-line in the [`ConstPool`], referenced by
//! [`ConstId`] — values born from the pool have [`Origin::Const`]. There is
//! deliberately no "Constant" node.

use crate::attrs::Attrs;
use crate::dim::{DimExpr, SymbolTable};
use crate::types::TensorType;
use crate::{Error, Result};

/// Identifier of a value in a [`Module`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(u32);

impl ValueId {
    /// Index into [`Module::values`].
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Identifier of a node in a [`Module`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(u32);

impl NodeId {
    /// Index into [`Module::nodes`].
    pub fn index(self) -> usize {
        self.0 as usize
    }

    pub(crate) fn from_index(index: u32) -> Self {
        Self(index)
    }
}

/// Identifier of an entry in the [`ConstPool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstId(u32);

impl ConstId {
    /// Index into the pool.
    pub fn index(self) -> usize {
        self.0 as usize
    }

    #[cfg(test)]
    pub(crate) fn from_index(index: u32) -> Self {
        Self(index)
    }
}

/// Where a value comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// A graph input, provided at run time.
    Input,
    /// Constant data in the module's pool (weights, folded results).
    Const(ConstId),
    /// Output `output` of node `node`.
    Node { node: NodeId, output: usize },
}

/// Compile-time known content of a small integer value.
///
/// This is the *compile-time value domain* (`doc/ir-design.md` §3): shape
/// arithmetic (`Shape → Gather → Concat → …`) evaluates at lowering into
/// tensors of [`DimExpr`]s instead of runtime nodes. Only rank-0 and rank-1
/// i64 values participate — exactly what ONNX shape plumbing produces.
#[derive(Debug, Clone, PartialEq)]
pub struct SymbolicContent {
    /// Concrete shape of the content tensor: `[]` (scalar) or `[n]`.
    pub shape: Vec<usize>,
    /// Element expressions, row-major (`shape.iter().product()` of them).
    pub elems: Vec<DimExpr>,
}

impl SymbolicContent {
    /// Scalar content.
    pub fn scalar(e: DimExpr) -> Self {
        Self {
            shape: vec![],
            elems: vec![e],
        }
    }

    /// 1-D content.
    pub fn vector(elems: Vec<DimExpr>) -> Self {
        Self {
            shape: vec![elems.len()],
            elems,
        }
    }
}

/// Definition of one value.
#[derive(Debug, Clone)]
pub struct ValueDef {
    /// ONNX tensor name, kept for I/O mapping and debugging.
    pub name: Option<String>,
    /// The value's type. Shape may be symbolic.
    pub ty: TensorType,
    /// Where the value comes from.
    pub origin: Origin,
    /// Compile-time known content, if any (see [`SymbolicContent`]).
    pub content: Option<SymbolicContent>,
}

/// Operation kind of a node.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    /// A primitive — closed set, fully specified semantics.
    Prim(crate::prim::Prim),
    /// A composite — open set; its decomposition lives in the lowering
    /// registry, keyed by `name` (see `doc/ir-design.md` §2).
    Composite(Composite),
}

/// A composite operation: domain-qualified name plus normalized attributes.
#[derive(Debug, Clone, PartialEq)]
pub struct Composite {
    /// Domain-qualified name, e.g. `"com.microsoft.GroupQueryAttention"`.
    pub name: String,
    /// Normalized attributes (ONNX defaults resolved at lowering).
    pub attrs: Attrs,
}

/// Provenance of a node, for errors, tracing, and dot output.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SourceInfo {
    /// Original ONNX node name (e.g. `"/model/layers.0/attn/MatMul"`).
    pub name: Option<String>,
    /// Original ONNX op type, when the node came from lowering.
    pub op_type: Option<String>,
}

/// One operation.
#[derive(Debug, Clone)]
pub struct Node {
    /// What the node does.
    pub kind: NodeKind,
    /// Consumed values, in operator-defined order.
    pub inputs: Vec<ValueId>,
    /// Produced values, in operator-defined order.
    pub outputs: Vec<ValueId>,
    /// Provenance.
    pub loc: SourceInfo,
}

/// One constant-pool entry.
#[derive(Debug, Clone)]
pub struct ConstEntry {
    /// Type; the shape must be fully static.
    pub ty: TensorType,
    /// Raw bytes, in the dtype's storage layout.
    data: Vec<u8>,
}

/// Out-of-line storage for constant tensor data (weights, folded results).
///
/// Kept behind accessors so the storage representation can later become
/// zero-copy (`Arc<[u8]>` slices or mmap views) without touching callers
/// (plan, pinned decision 7).
#[derive(Debug, Clone, Default)]
pub struct ConstPool {
    entries: Vec<ConstEntry>,
}

impl ConstPool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entry. The type's shape must be fully static and `data` must
    /// have exactly the storage size implied by the type.
    pub fn add(&mut self, ty: TensorType, data: Vec<u8>) -> Result<ConstId> {
        let Some(dims) = ty.shape.as_static() else {
            return Err(Error::InvalidGraph(format!(
                "constant must have a static shape, got {}",
                ty.shape
            )));
        };
        let numel: u64 = dims.iter().product();
        let expect = ty.dtype.storage_bytes(numel as usize);
        if data.len() != expect {
            return Err(Error::InvalidGraph(format!(
                "constant data size mismatch: {} bytes for {} (expected {expect})",
                data.len(),
                ty
            )));
        }
        let id = ConstId(self.entries.len() as u32);
        self.entries.push(ConstEntry { ty, data });
        Ok(id)
    }

    /// The entry's type.
    pub fn ty(&self, id: ConstId) -> &TensorType {
        &self.entries[id.index()].ty
    }

    /// The entry's raw bytes.
    pub fn bytes(&self, id: ConstId) -> &[u8] {
        &self.entries[id.index()].data
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total bytes held.
    pub fn total_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.data.len()).sum()
    }

    /// Reinterpret an entry's type without touching its bytes (e.g. viewing
    /// a packed `U8 [N, nb, blob]` weight blob as `U4 [N, nb, block_size]`).
    /// The new type must describe exactly the same storage size.
    pub fn reinterpret(&mut self, id: ConstId, ty: TensorType) -> Result<()> {
        let Some(dims) = ty.shape.as_static() else {
            return Err(Error::InvalidGraph(format!(
                "reinterpret requires a static shape, got {}",
                ty.shape
            )));
        };
        let numel: u64 = dims.iter().product();
        let expect = ty.dtype.storage_bytes(numel as usize);
        let actual = self.entries[id.index()].data.len();
        if actual != expect {
            return Err(Error::InvalidGraph(format!(
                "reinterpret size mismatch: entry holds {actual} bytes, {ty} needs {expect}"
            )));
        }
        self.entries[id.index()].ty = ty;
        Ok(())
    }
}

/// A complete IR graph.
#[derive(Debug, Clone, Default)]
pub struct Module {
    /// All values; index = [`ValueId`].
    pub values: Vec<ValueDef>,
    /// All nodes; index = [`NodeId`].
    pub nodes: Vec<Node>,
    /// Constant data.
    pub consts: ConstPool,
    /// Dimension symbols.
    pub symbols: SymbolTable,
    /// Named graph inputs, in signature order.
    pub inputs: Vec<(String, ValueId)>,
    /// Named graph outputs, in signature order.
    pub outputs: Vec<(String, ValueId)>,
}

impl Module {
    /// Create an empty module.
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a value definition.
    pub fn value(&self, id: ValueId) -> &ValueDef {
        &self.values[id.index()]
    }

    /// Mutable value lookup.
    pub fn value_mut(&mut self, id: ValueId) -> &mut ValueDef {
        &mut self.values[id.index()]
    }

    /// Look up a node.
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.index()]
    }

    /// Append a value definition, returning its id.
    pub fn add_value(&mut self, def: ValueDef) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(def);
        id
    }

    /// Append a node, returning its id. Does **not** fix up value origins —
    /// use [`GraphBuilder`](crate::builder::GraphBuilder) for checked
    /// construction.
    pub fn add_node(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    /// Iterate node ids.
    pub fn node_ids(&self) -> impl Iterator<Item = NodeId> + use<> {
        (0..self.nodes.len() as u32).map(NodeId)
    }

    /// Iterate value ids.
    pub fn value_ids(&self) -> impl Iterator<Item = ValueId> + use<> {
        (0..self.values.len() as u32).map(ValueId)
    }

    /// Nodes in topological (dependency) order.
    ///
    /// Errors if the graph contains a cycle.
    pub fn topo_order(&self) -> Result<Vec<NodeId>> {
        #[derive(Clone, Copy, PartialEq)]
        enum State {
            Unvisited,
            Visiting,
            Done,
        }
        let mut state = vec![State::Unvisited; self.nodes.len()];
        let mut order = Vec::with_capacity(self.nodes.len());

        for start in self.node_ids() {
            if state[start.index()] != State::Unvisited {
                continue;
            }
            // Iterative DFS: (node, next-input-index-to-visit).
            let mut stack: Vec<(NodeId, usize)> = vec![(start, 0)];
            state[start.index()] = State::Visiting;
            while let Some(&(node, next)) = stack.last() {
                let inputs = &self.nodes[node.index()].inputs;
                if next < inputs.len() {
                    stack.last_mut().expect("stack is non-empty").1 += 1;
                    if let Origin::Node { node: producer, .. } = self.value(inputs[next]).origin {
                        match state[producer.index()] {
                            State::Unvisited => {
                                state[producer.index()] = State::Visiting;
                                stack.push((producer, 0));
                            }
                            State::Visiting => {
                                return Err(Error::InvalidGraph(
                                    "graph contains a cycle".to_string(),
                                ));
                            }
                            State::Done => {}
                        }
                    }
                } else {
                    state[node.index()] = State::Done;
                    order.push(node);
                    stack.pop();
                }
            }
        }
        Ok(order)
    }

    /// Replace every use of `old` (node inputs and module outputs) with
    /// `new`. Does not touch producers — `old`'s definition stays intact
    /// until DCE removes it. Carries `old`'s name over to `new` when `new`
    /// is unnamed, so output naming survives composite inlining.
    pub fn replace_uses(&mut self, old: ValueId, new: ValueId) {
        for node in &mut self.nodes {
            for v in node.inputs.iter_mut() {
                if *v == old {
                    *v = new;
                }
            }
        }
        for (_, id) in self.outputs.iter_mut() {
            if *id == old {
                *id = new;
            }
        }
        if self.values[new.index()].name.is_none() {
            self.values[new.index()].name = self.values[old.index()].name.clone();
        }
    }

    /// Remove the given nodes and prune values that become unreferenced,
    /// remapping all node and value ids.
    ///
    /// A value survives if it is a module input/output or is referenced by a
    /// surviving node. Callers must have rewritten the origins of any
    /// removed node's outputs that should *survive* (e.g. folding rewrites
    /// them to [`Origin::Const`] first); outputs still originating from a
    /// removed node are pruned with it.
    pub fn remove_nodes(&mut self, dead: &std::collections::HashSet<NodeId>) {
        // Remap nodes.
        let mut node_remap: Vec<Option<NodeId>> = Vec::with_capacity(self.nodes.len());
        let mut kept_nodes = Vec::with_capacity(self.nodes.len().saturating_sub(dead.len()));
        for (i, node) in std::mem::take(&mut self.nodes).into_iter().enumerate() {
            if dead.contains(&NodeId(i as u32)) {
                node_remap.push(None);
            } else {
                node_remap.push(Some(NodeId(kept_nodes.len() as u32)));
                kept_nodes.push(node);
            }
        }

        // A value survives if referenced by module I/O or a kept node, and
        // its origin doesn't dangle into a removed node.
        let mut live = vec![false; self.values.len()];
        for (_, id) in self.inputs.iter().chain(&self.outputs) {
            live[id.index()] = true;
        }
        for node in &kept_nodes {
            for &v in node.inputs.iter().chain(&node.outputs) {
                live[v.index()] = true;
            }
        }
        for (i, def) in self.values.iter().enumerate() {
            if let Origin::Node { node, .. } = def.origin {
                if node_remap[node.index()].is_none() {
                    live[i] = false; // origin dangles: prune
                }
            }
        }

        // Remap values.
        let mut value_remap: Vec<Option<ValueId>> = Vec::with_capacity(self.values.len());
        let mut kept_values = Vec::new();
        for (i, def) in std::mem::take(&mut self.values).into_iter().enumerate() {
            if live[i] {
                value_remap.push(Some(ValueId(kept_values.len() as u32)));
                kept_values.push(def);
            } else {
                value_remap.push(None);
            }
        }

        // Apply remaps.
        for def in &mut kept_values {
            if let Origin::Node { node, .. } = &mut def.origin {
                *node = node_remap[node.index()].expect("live value's producer survives");
            }
        }
        for node in &mut kept_nodes {
            for v in node.inputs.iter_mut().chain(node.outputs.iter_mut()) {
                *v = value_remap[v.index()].expect("kept node references live values");
            }
        }
        for (_, id) in self.inputs.iter_mut().chain(self.outputs.iter_mut()) {
            *id = value_remap[id.index()].expect("module I/O values survive");
        }

        self.nodes = kept_nodes;
        self.values = kept_values;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dim::SymbolicShape;
    use crate::prim::{BinaryOp, Prim};
    use crate::types::DataType;

    fn f32_val(module: &mut Module, origin: Origin) -> ValueId {
        module.add_value(ValueDef {
            name: None,
            ty: TensorType::of(DataType::F32, &[2]),
            origin,
            content: None,
        })
    }

    #[test]
    fn const_pool_checks() {
        let mut pool = ConstPool::new();
        let ty = TensorType::of(DataType::F32, &[3]);
        // Wrong byte count.
        assert!(pool.add(ty.clone(), vec![0; 11]).is_err());
        // Symbolic shape rejected.
        let mut symbols = SymbolTable::new();
        let s = symbols.intern("S");
        let sym_ty = TensorType::new(DataType::F32, SymbolicShape(vec![DimExpr::sym(s)]));
        assert!(pool.add(sym_ty, vec![]).is_err());
        // Good entry round-trips.
        let id = pool.add(ty.clone(), vec![0; 12]).unwrap();
        assert_eq!(pool.ty(id), &ty);
        assert_eq!(pool.bytes(id).len(), 12);
        assert_eq!(pool.total_bytes(), 12);
        // Packed dtype sizing.
        let q = TensorType::of(DataType::U4, &[5]);
        assert!(pool.add(q.clone(), vec![0; 2]).is_err());
        pool.add(q, vec![0; 3]).unwrap();
    }

    #[test]
    fn topo_order_diamond() {
        // a = input; b = a+a; c = a+b; d = b+c  — must order b before c
        // before d.
        let mut m = Module::new();
        let a = f32_val(&mut m, Origin::Input);
        let b_out = f32_val(
            &mut m,
            Origin::Node {
                node: NodeId(0),
                output: 0,
            },
        );
        let c_out = f32_val(
            &mut m,
            Origin::Node {
                node: NodeId(1),
                output: 0,
            },
        );
        let d_out = f32_val(
            &mut m,
            Origin::Node {
                node: NodeId(2),
                output: 0,
            },
        );
        // Insert nodes intentionally out of order: d first.
        m.nodes = vec![
            Node {
                kind: NodeKind::Prim(Prim::Binary(BinaryOp::Add)),
                inputs: vec![a, a],
                outputs: vec![b_out],
                loc: SourceInfo::default(),
            },
            Node {
                kind: NodeKind::Prim(Prim::Binary(BinaryOp::Add)),
                inputs: vec![a, b_out],
                outputs: vec![c_out],
                loc: SourceInfo::default(),
            },
            Node {
                kind: NodeKind::Prim(Prim::Binary(BinaryOp::Add)),
                inputs: vec![b_out, c_out],
                outputs: vec![d_out],
                loc: SourceInfo::default(),
            },
        ];
        let order = m.topo_order().unwrap();
        let pos = |id: NodeId| order.iter().position(|&n| n == id).unwrap();
        assert!(pos(NodeId(0)) < pos(NodeId(1)));
        assert!(pos(NodeId(1)) < pos(NodeId(2)));
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn topo_order_detects_cycle() {
        // x = f(y); y = f(x)
        let mut m = Module::new();
        let x = f32_val(
            &mut m,
            Origin::Node {
                node: NodeId(0),
                output: 0,
            },
        );
        let y = f32_val(
            &mut m,
            Origin::Node {
                node: NodeId(1),
                output: 0,
            },
        );
        m.nodes = vec![
            Node {
                kind: NodeKind::Prim(Prim::Binary(BinaryOp::Add)),
                inputs: vec![y, y],
                outputs: vec![x],
                loc: SourceInfo::default(),
            },
            Node {
                kind: NodeKind::Prim(Prim::Binary(BinaryOp::Add)),
                inputs: vec![x, x],
                outputs: vec![y],
                loc: SourceInfo::default(),
            },
        ];
        assert!(m.topo_order().is_err());
    }

    #[test]
    fn remove_nodes_remaps_origins() {
        let mut m = Module::new();
        let a = f32_val(&mut m, Origin::Input);
        m.inputs.push(("a".into(), a)); // keep `a` alive through pruning
        let b = f32_val(
            &mut m,
            Origin::Node {
                node: NodeId(0),
                output: 0,
            },
        );
        let c = f32_val(
            &mut m,
            Origin::Node {
                node: NodeId(1),
                output: 0,
            },
        );
        m.outputs.push(("c".into(), c));
        m.nodes = vec![
            Node {
                kind: NodeKind::Prim(Prim::Binary(BinaryOp::Add)),
                inputs: vec![a, a],
                outputs: vec![b],
                loc: SourceInfo::default(),
            },
            Node {
                kind: NodeKind::Prim(Prim::Binary(BinaryOp::Add)),
                inputs: vec![b, b],
                outputs: vec![c],
                loc: SourceInfo::default(),
            },
        ];
        // Pretend node 0 was folded: b becomes a pool constant.
        let cid = m
            .consts
            .add(TensorType::of(DataType::F32, &[2]), vec![0; 8])
            .unwrap();
        m.value_mut(b).origin = Origin::Const(cid);
        let dead = std::collections::HashSet::from([NodeId(0)]);
        m.remove_nodes(&dead);
        assert_eq!(m.nodes.len(), 1);
        // c's producer (was node 1) is now node 0.
        assert_eq!(
            m.value(c).origin,
            Origin::Node {
                node: NodeId(0),
                output: 0
            }
        );
    }
}
