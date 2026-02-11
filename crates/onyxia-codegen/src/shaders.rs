//! WGSL shader loading and compilation using naga_oil.
//!
//! This module provides shader sources for ONNX operators. Shader compilation
//! with runtime shader defs happens in the runtime crate using naga_oil's Composer.

// Re-export naga_oil types for runtime use
pub use naga_oil::compose::ShaderDefValue;
use std::collections::HashMap;

/// Collection of shader definitions (HashMap of shader def names to values)
pub type ShaderDefs = HashMap<String, ShaderDefValue>;

/// Get the raw WGSL source for a shader.
/// 
/// Runtime will use naga_oil::compose::Composer to compile these sources
/// with dynamic shader definitions (e.g., actual tensor dimensions).
///
/// # Example
/// ```no_run
/// use naga_oil::compose::{Composer, NagaModuleDescriptor};
/// use onyxia_codegen::shaders::{get_shader_source, ShaderDefs, ShaderDefValue};
/// 
/// let source = get_shader_source("add").unwrap();
/// let mut defs = ShaderDefs::new();
/// defs.insert("WORKGROUP_SIZE".to_string(), ShaderDefValue::UInt(256));
/// 
/// let mut composer = Composer::default();
/// let module = composer.make_naga_module(NagaModuleDescriptor {
///     source,
///     file_path: "add.wgsl",
///     shader_defs: defs,
///     ..Default::default()
/// });
/// ```
pub fn get_shader_source(name: &str) -> Option<&'static str> {
    match name {
        "add" => Some(include_str!("../shaders/elementwise/add.wgsl")),
        "mul" => Some(include_str!("../shaders/elementwise/mul.wgsl")),
        "gelu" => Some(include_str!("../shaders/activation/gelu.wgsl")),
        "rmsnorm" => Some(include_str!("../shaders/normalization/rmsnorm.wgsl")),
        "matmul_f32" => Some(include_str!("../shaders/matmul/matmul_f32.wgsl")),
        _ => None,
    }
}

/// Create default shader definitions for workgroup size.
pub fn default_workgroup_defs() -> ShaderDefs {
    [("WORKGROUP_SIZE".to_string(), ShaderDefValue::UInt(256))].into()
}

/// Create default shader definitions for tiled matrix multiplication.
pub fn default_tile_defs() -> ShaderDefs {
    [
        ("TILE_M".to_string(), ShaderDefValue::UInt(16)),
        ("TILE_N".to_string(), ShaderDefValue::UInt(16)),
        ("TILE_K".to_string(), ShaderDefValue::UInt(16)),
    ]
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shader_sources_available() {
        // Test that all shader sources are embedded and accessible
        assert!(get_shader_source("add").is_some());
        assert!(get_shader_source("mul").is_some());
        assert!(get_shader_source("gelu").is_some());
        assert!(get_shader_source("rmsnorm").is_some());
        assert!(get_shader_source("matmul_f32").is_some());
        
        // Verify content
        let add = get_shader_source("add").unwrap();
        assert!(add.contains("@compute"));
        assert!(add.contains("Elementwise addition"));
        
        let gelu = get_shader_source("gelu").unwrap();
        assert!(gelu.contains("GELU"));
    }
    
    #[test]
    fn test_shader_def_functions() {
        // Test that shader def functions return expected values
        let wg_defs = default_workgroup_defs();
        assert_eq!(wg_defs.get("WORKGROUP_SIZE"), Some(&ShaderDefValue::UInt(256)));
        
        let tile_defs = default_tile_defs();
        assert_eq!(tile_defs.get("TILE_M"), Some(&ShaderDefValue::UInt(16)));
        assert_eq!(tile_defs.get("TILE_N"), Some(&ShaderDefValue::UInt(16)));
        assert_eq!(tile_defs.get("TILE_K"), Some(&ShaderDefValue::UInt(16)));
    }
}