//! ONNX model parser for Onyxia.
//!
//! This crate parses ONNX protobuf models and converts them into an internal
//! intermediate representation (IR) that can be consumed by `onyxia-codegen`.

/// Generated protobuf types from ONNX schema.
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub use onnx::ModelProto;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_proto_exists() {
        // Verify generated types are accessible
        let _model = ModelProto::default();
    }
}
