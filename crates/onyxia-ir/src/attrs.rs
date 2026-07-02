//! Attributes for composite nodes.
//!
//! Primitives carry their parameters inside the [`Prim`](crate::prim::Prim)
//! variants; attributes exist only on composites, where they hold the
//! *normalized* form of the original ONNX attributes (defaults resolved,
//! opset quirks flattened) so kernels and decompositions see one clean view.

use crate::graph::ConstId;
use crate::{Error, Result};

/// A single attribute value.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    Int(i64),
    Ints(Vec<i64>),
    Float(f64),
    Floats(Vec<f64>),
    Str(String),
    Bool(bool),
    /// Reference to tensor data in the module's constant pool.
    Tensor(ConstId),
}

/// An ordered map of attribute name → value.
///
/// Order is preserved for deterministic display/serialization; lookups are
/// linear (attribute counts are tiny).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Attrs(Vec<(String, AttrValue)>);

impl Attrs {
    /// Create an empty attribute map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set an attribute, replacing any existing value of the same name.
    pub fn set(&mut self, name: impl Into<String>, value: AttrValue) -> &mut Self {
        let name = name.into();
        match self.0.iter_mut().find(|(n, _)| *n == name) {
            Some((_, v)) => *v = value,
            None => self.0.push((name, value)),
        }
        self
    }

    /// Builder-style [`set`](Self::set).
    pub fn with(mut self, name: impl Into<String>, value: AttrValue) -> Self {
        self.set(name, value);
        self
    }

    /// Raw lookup.
    pub fn get(&self, name: &str) -> Option<&AttrValue> {
        self.0.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }

    /// Iterate in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &AttrValue)> {
        self.0.iter().map(|(n, v)| (n.as_str(), v))
    }

    /// Whether no attributes are set.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Required i64 attribute.
    pub fn int(&self, name: &str) -> Result<i64> {
        match self.get(name) {
            Some(AttrValue::Int(v)) => Ok(*v),
            Some(other) => Err(type_err(name, "int", other)),
            None => Err(missing(name)),
        }
    }

    /// Optional i64 attribute with a default.
    pub fn int_or(&self, name: &str, default: i64) -> Result<i64> {
        match self.get(name) {
            Some(AttrValue::Int(v)) => Ok(*v),
            Some(other) => Err(type_err(name, "int", other)),
            None => Ok(default),
        }
    }

    /// Required i64-list attribute.
    pub fn ints(&self, name: &str) -> Result<&[i64]> {
        match self.get(name) {
            Some(AttrValue::Ints(v)) => Ok(v),
            Some(other) => Err(type_err(name, "ints", other)),
            None => Err(missing(name)),
        }
    }

    /// Required f64 attribute.
    pub fn float(&self, name: &str) -> Result<f64> {
        match self.get(name) {
            Some(AttrValue::Float(v)) => Ok(*v),
            Some(other) => Err(type_err(name, "float", other)),
            None => Err(missing(name)),
        }
    }

    /// Optional f64 attribute with a default.
    pub fn float_or(&self, name: &str, default: f64) -> Result<f64> {
        match self.get(name) {
            Some(AttrValue::Float(v)) => Ok(*v),
            Some(other) => Err(type_err(name, "float", other)),
            None => Ok(default),
        }
    }

    /// Required string attribute.
    pub fn str(&self, name: &str) -> Result<&str> {
        match self.get(name) {
            Some(AttrValue::Str(v)) => Ok(v),
            Some(other) => Err(type_err(name, "str", other)),
            None => Err(missing(name)),
        }
    }

    /// Optional bool attribute with a default.
    pub fn bool_or(&self, name: &str, default: bool) -> Result<bool> {
        match self.get(name) {
            Some(AttrValue::Bool(v)) => Ok(*v),
            Some(other) => Err(type_err(name, "bool", other)),
            None => Ok(default),
        }
    }

    /// Required constant-tensor attribute.
    pub fn tensor(&self, name: &str) -> Result<ConstId> {
        match self.get(name) {
            Some(AttrValue::Tensor(v)) => Ok(*v),
            Some(other) => Err(type_err(name, "tensor", other)),
            None => Err(missing(name)),
        }
    }
}

fn missing(name: &str) -> Error {
    Error::Attribute(format!("missing attribute '{name}'"))
}

fn type_err(name: &str, expected: &str, got: &AttrValue) -> Error {
    let got = match got {
        AttrValue::Int(_) => "int",
        AttrValue::Ints(_) => "ints",
        AttrValue::Float(_) => "float",
        AttrValue::Floats(_) => "floats",
        AttrValue::Str(_) => "str",
        AttrValue::Bool(_) => "bool",
        AttrValue::Tensor(_) => "tensor",
    };
    Error::Attribute(format!(
        "attribute '{name}': expected {expected}, found {got}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get_and_defaults() {
        let attrs = Attrs::new()
            .with("axis", AttrValue::Int(2))
            .with("epsilon", AttrValue::Float(1e-5))
            .with("mode", AttrValue::Str("linear".into()));

        assert_eq!(attrs.int("axis").unwrap(), 2);
        assert_eq!(attrs.int_or("axis", 0).unwrap(), 2);
        assert_eq!(attrs.int_or("missing", 7).unwrap(), 7);
        assert_eq!(attrs.float("epsilon").unwrap(), 1e-5);
        assert_eq!(attrs.str("mode").unwrap(), "linear");
        assert!(attrs.bool_or("flag", true).unwrap());
    }

    #[test]
    fn errors_are_precise() {
        let attrs = Attrs::new().with("axis", AttrValue::Str("oops".into()));
        let err = attrs.int("axis").unwrap_err().to_string();
        assert!(err.contains("expected int"), "got: {err}");
        assert!(err.contains("found str"), "got: {err}");
        let err = attrs.int("nope").unwrap_err().to_string();
        assert!(err.contains("missing attribute 'nope'"), "got: {err}");
    }

    #[test]
    fn set_replaces() {
        let mut attrs = Attrs::new();
        attrs.set("k", AttrValue::Int(1));
        attrs.set("k", AttrValue::Int(2));
        assert_eq!(attrs.int("k").unwrap(), 2);
        assert_eq!(attrs.iter().count(), 1);
    }
}
