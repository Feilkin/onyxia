//! Build script for onyxia-onnx.
//!
//! Generates Rust types from ONNX protobuf definitions. The proto is
//! compiled with `protox` (a pure-Rust protobuf compiler), so building this
//! crate needs no `protoc` binary — that keeps `docs.rs` and fresh clones
//! working with nothing but cargo.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Rerun if the proto file changes
    println!("cargo::rerun-if-changed=proto/onnx.proto");

    let descriptors = protox::compile(["proto/onnx.proto"], ["proto/"])?;
    prost_build::compile_fds(descriptors)?;

    Ok(())
}
