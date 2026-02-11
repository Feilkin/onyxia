---
name: search-rustdoc
description: Search dependency documentation by building rustdoc locally and grepping HTML files. Use when you need to find functions, types, method signatures, or usage patterns from Rust dependencies like wgpu, prost, naga_oil, or any crate in the project.
---

# Search Rust Dependency Documentation

Use this skill to find API details, function signatures, and usage patterns from project dependencies when docs.rs is insufficient or you need to search across multiple crates.

## Step 1: Build Documentation

Build docs for the workspace and all dependencies:

```bash
cargo doc --workspace
```

This generates HTML documentation in `target/doc/`. Each crate gets its own subdirectory.

## Step 2: Search the Generated Docs

The docs are HTML files. Use grep to search:

```bash
# Search for a function name
grep -r "fn create_buffer" target/doc/wgpu/

# Search for a struct
grep -r "struct Device" target/doc/wgpu/

# Search for a pattern across all deps
grep -rh "compile_protos" target/doc/ --include="*.html" | head -50

# Find methods on a type
grep -r "impl.*Queue" target/doc/wgpu/
```

## Step 3: Read Specific Documentation

Documentation files follow predictable paths:

| What | Path |
|------|------|
| Crate root | `target/doc/<crate>/index.html` |
| Module | `target/doc/<crate>/<module>/index.html` |
| Struct | `target/doc/<crate>/struct.<Name>.html` |
| Enum | `target/doc/<crate>/enum.<Name>.html` |
| Function | `target/doc/<crate>/fn.<name>.html` |
| Trait | `target/doc/<crate>/trait.<Name>.html` |
| All items | `target/doc/<crate>/all.html` |

## Tips

- HTML contains full docstrings — search for text you'd see on docs.rs
- Use `--include="*.html"` to skip source maps and JS files
- Crate names use underscores: `wgpu_core`, `prost_build`
- The `all.html` file lists every public item in a crate
