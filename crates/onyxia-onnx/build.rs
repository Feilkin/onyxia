//! Build script for onyxia-onnx.
//!
//! Generates Rust types from ONNX protobuf definitions using prost-build.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Rerun if the proto file changes
    println!("cargo::rerun-if-changed=proto/onnx.proto");

    prost_build::compile_protos(&["proto/onnx.proto"], &["proto/"])?;

    Ok(())
}
