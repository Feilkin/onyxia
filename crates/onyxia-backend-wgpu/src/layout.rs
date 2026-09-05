//! Physical tensor layouts: how each logical dtype is stored on the device
//! and how generated kernels load and store it.
//!
//! WGSL storage buffers hold 32-bit scalars, optionally `f16` (with the
//! `SHADER_F16` feature) and 64-bit integers (`SHADER_INT64`). Everything
//! else is packed into `u32` words: four 8-bit values or two `f16`s per
//! word, in memory order, so the device bytes are exactly the host bytes.
//! Kernels always *compute* in a 32-bit type (or `i64` when native);
//! [`Layout::load_fn`] and [`Layout::store_block`] generate the WGSL that
//! converts at the buffer boundary, and every kernel runs one thread per
//! output **word**, so packed lanes are never written by two threads.
//!
//! Without `SHADER_INT64`, `I64` narrows to `i32` (range-checked at
//! upload) — the index domain, which is all a model without 64-bit data
//! needs.

use onyxia_ir::{DataType, Error, Result};

/// Device features that change layouts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Caps {
    /// `wgpu::Features::SHADER_F16`: `f16` storage and arithmetic.
    pub f16: bool,
    /// `wgpu::Features::SHADER_INT64`: `i64` storage and arithmetic.
    pub int64: bool,
}

/// How a dtype lives in a storage buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Repr {
    /// One 32-bit scalar per element (`f32`, `i32`, `u32`; `Bool` as u32).
    Plain,
    /// One native `f16` per element.
    F16Native,
    /// Two `f16` per `u32` word (`pack2x16float`).
    F16Packed,
    /// One native `i64` per element.
    I64Native,
    /// `I64` narrowed to `i32` (range-checked at upload).
    I64Narrow,
    /// Four `u8` per `u32` word, memory order.
    U8Packed,
    /// Four `i8` per `u32` word, memory order.
    I8Packed,
}

/// A logical dtype with its physical representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Layout {
    pub logical: DataType,
    pub repr: Repr,
}

/// Plain f32 — the layout the fused kernels and fast paths assume.
pub const F32: Layout = Layout {
    logical: DataType::F32,
    repr: Repr::Plain,
};

impl Layout {
    /// The layout for `dt` on a device with `caps`.
    pub fn of(dt: DataType, caps: Caps) -> Result<Layout> {
        let repr = match dt {
            DataType::F32 | DataType::I32 | DataType::U32 | DataType::Bool => Repr::Plain,
            DataType::F16 => {
                if caps.f16 {
                    Repr::F16Native
                } else {
                    Repr::F16Packed
                }
            }
            DataType::I64 => {
                if caps.int64 {
                    Repr::I64Native
                } else {
                    Repr::I64Narrow
                }
            }
            DataType::U8 => Repr::U8Packed,
            DataType::I8 => Repr::I8Packed,
            DataType::U4 | DataType::I4 => {
                return Err(Error::Unsupported(format!(
                    "dtype {dt} on the wgpu backend (only Dequantize reads packed 4-bit data)"
                )));
            }
        };
        Ok(Layout { logical: dt, repr })
    }

    /// WGSL element type of the storage array.
    pub fn store(&self) -> &'static str {
        match self.repr {
            Repr::Plain => match self.logical {
                DataType::F32 => "f32",
                DataType::I32 => "i32",
                _ => "u32",
            },
            Repr::F16Native => "f16",
            Repr::I64Native => "i64",
            Repr::I64Narrow => "i32",
            Repr::F16Packed | Repr::U8Packed | Repr::I8Packed => "u32",
        }
    }

    /// WGSL type kernels compute in after a load.
    pub fn compute(&self) -> &'static str {
        match self.repr {
            Repr::Plain => self.store(),
            Repr::F16Native | Repr::F16Packed => "f32",
            Repr::I64Native => "i64",
            Repr::I64Narrow => "i32",
            Repr::U8Packed => "u32",
            Repr::I8Packed => "i32",
        }
    }

    /// Elements per storage word.
    pub fn lanes(&self) -> u32 {
        match self.repr {
            Repr::F16Packed => 2,
            Repr::U8Packed | Repr::I8Packed => 4,
            _ => 1,
        }
    }

    /// Bytes per storage element.
    fn store_bytes(&self) -> u64 {
        match self.repr {
            Repr::F16Native => 2,
            Repr::I64Native => 8,
            _ => 4,
        }
    }

    /// Storage words (or native elements) needed for `numel` elements.
    pub fn words(&self, numel: usize) -> u64 {
        (numel as u64).div_ceil(self.lanes() as u64)
    }

    /// Buffer size in bytes for `numel` elements: whole storage elements,
    /// at least 4 bytes, rounded up to a multiple of 4 (a WebGPU
    /// requirement, and what odd-length native f16 needs).
    pub fn buffer_bytes(&self, numel: usize) -> u64 {
        let raw = self.words(numel).max(1) * self.store_bytes();
        raw.div_ceil(4) * 4
    }

    /// Whether the device bytes are the host bytes (padding aside).
    pub fn is_host_identical(&self) -> bool {
        !matches!(self.repr, Repr::I64Narrow) && self.logical != DataType::Bool
    }

    /// Plain 32-bit float storage — the fast paths' precondition.
    pub fn is_plain_f32(&self) -> bool {
        self.repr == Repr::Plain && self.logical == DataType::F32
    }

    /// Whether kernels over this layout need `enable f16;`.
    pub fn needs_f16(&self) -> bool {
        self.repr == Repr::F16Native
    }

    /// Short tag for pipeline-cache labels.
    pub fn tag(&self) -> &'static str {
        match (self.repr, self.logical) {
            (Repr::Plain, DataType::F32) => "f32",
            (Repr::Plain, DataType::I32) => "i32",
            (Repr::Plain, DataType::U32) => "u32",
            (Repr::Plain, _) => "b",
            (Repr::F16Native, _) => "f16",
            (Repr::F16Packed, _) => "f16p",
            (Repr::I64Native, _) => "i64",
            (Repr::I64Narrow, _) => "i64n",
            (Repr::U8Packed, _) => "u8p",
            (Repr::I8Packed, _) => "i8p",
        }
    }

    /// The WGSL zero of the compute type.
    pub fn zero(&self) -> &'static str {
        match self.compute() {
            "f32" => "0.0",
            "u32" => "0u",
            "i64" => "i64(0)",
            _ => "0",
        }
    }

    /// `array<T>` element type of the binding.
    pub fn binding(&self) -> String {
        format!("array<{}>", self.store())
    }

    /// WGSL `fn {name}(i: u32) -> compute` reading element `i` of `buf`.
    pub fn load_fn(&self, name: &str, buf: &str) -> String {
        let c = self.compute();
        let body = match self.repr {
            Repr::Plain | Repr::I64Native | Repr::I64Narrow => format!("return {buf}[i];"),
            Repr::F16Native => format!("return f32({buf}[i]);"),
            Repr::F16Packed => {
                format!("let v = unpack2x16float({buf}[i >> 1u]); return v[i & 1u];")
            }
            Repr::U8Packed => format!("return ({buf}[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu;"),
            Repr::I8Packed => {
                format!("let w = {buf}[i >> 2u]; return i32(w << (24u - (i & 3u) * 8u)) >> 24u;")
            }
        };
        format!("fn {name}(i: u32) -> {c} {{ {body} }}\n")
    }

    /// The `main` body: one thread per output word of `out`, calling
    /// `compute(e)` for each element `e` it owns. `p.size` is the element
    /// count; `p.x_stride` the dispatch stride (see `linear_idx`).
    pub fn store_block(&self, out: &str) -> String {
        match self.repr {
            Repr::Plain | Repr::I64Native | Repr::I64Narrow => format!(
                "    let w = linear_idx(gid, p.x_stride);
    if (w >= p.size) {{ return; }}
    {out}[w] = compute(w);"
            ),
            Repr::F16Native => format!(
                "    let w = linear_idx(gid, p.x_stride);
    if (w >= p.size) {{ return; }}
    {out}[w] = f16(compute(w));"
            ),
            Repr::F16Packed => format!(
                "    let w = linear_idx(gid, p.x_stride);
    if (w >= (p.size + 1u) >> 1u) {{ return; }}
    var v = vec2<f32>(0.0, 0.0);
    for (var l = 0u; l < 2u; l = l + 1u) {{
        let e = w * 2u + l;
        if (e < p.size) {{ v[l] = compute(e); }}
    }}
    {out}[w] = pack2x16float(v);"
            ),
            Repr::U8Packed => format!(
                "    let w = linear_idx(gid, p.x_stride);
    if (w >= (p.size + 3u) >> 2u) {{ return; }}
    var acc = 0u;
    for (var l = 0u; l < 4u; l = l + 1u) {{
        let e = w * 4u + l;
        if (e < p.size) {{ acc = acc | ((compute(e) & 0xffu) << (l * 8u)); }}
    }}
    {out}[w] = acc;"
            ),
            Repr::I8Packed => format!(
                "    let w = linear_idx(gid, p.x_stride);
    if (w >= (p.size + 3u) >> 2u) {{ return; }}
    var acc = 0u;
    for (var l = 0u; l < 4u; l = l + 1u) {{
        let e = w * 4u + l;
        if (e < p.size) {{ acc = acc | ((u32(compute(e)) & 0xffu) << (l * 8u)); }}
    }}
    {out}[w] = acc;"
            ),
        }
    }

    /// Extract lane `lane` of a 32-bit word `old` as the compute type
    /// (for atomic read-modify-write kernels; 32-bit words only).
    pub fn lane_extract(&self, old: &str, lane: &str) -> Result<String> {
        Ok(match (self.repr, self.compute()) {
            (Repr::Plain, "f32") => format!("bitcast<f32>({old})"),
            (Repr::Plain, "i32") | (Repr::I64Narrow, _) => format!("bitcast<i32>({old})"),
            (Repr::Plain, _) => old.to_string(),
            (Repr::F16Packed, _) => format!("unpack2x16float({old})[{lane}]"),
            (Repr::U8Packed, _) => format!("(({old} >> ({lane} * 8u)) & 0xffu)"),
            (Repr::I8Packed, _) => {
                format!("(i32({old} << (24u - {lane} * 8u)) >> 24u)")
            }
            _ => {
                return Err(Error::Unsupported(format!(
                    "atomic lane access on {} storage",
                    self.tag()
                )));
            }
        })
    }

    /// Insert compute value `v` into lane `lane` of word `old`.
    pub fn lane_insert(&self, old: &str, lane: &str, v: &str) -> Result<String> {
        Ok(match (self.repr, self.compute()) {
            (Repr::Plain, "f32") => format!("bitcast<u32>({v})"),
            (Repr::Plain, "i32") | (Repr::I64Narrow, _) => format!("bitcast<u32>({v})"),
            (Repr::Plain, _) => v.to_string(),
            (Repr::F16Packed, _) => format!(
                "pack2x16float(select(vec2<f32>({v}, unpack2x16float({old}).y), \
                 vec2<f32>(unpack2x16float({old}).x, {v}), {lane} == 1u))"
            ),
            (Repr::U8Packed, _) => {
                format!("(({old} & ~(0xffu << ({lane} * 8u))) | (({v} & 0xffu) << ({lane} * 8u)))")
            }
            (Repr::I8Packed, _) => format!(
                "(({old} & ~(0xffu << ({lane} * 8u))) | ((u32({v}) & 0xffu) << ({lane} * 8u)))"
            ),
            _ => {
                return Err(Error::Unsupported(format!(
                    "atomic lane access on {} storage",
                    self.tag()
                )));
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        let caps = Caps::default();
        let u8 = Layout::of(DataType::U8, caps).unwrap();
        assert_eq!(u8.lanes(), 4);
        assert_eq!(u8.buffer_bytes(5), 8);
        assert_eq!(u8.buffer_bytes(0), 4);
        let f16 = Layout::of(DataType::F16, caps).unwrap();
        assert_eq!(f16.repr, Repr::F16Packed);
        assert_eq!(f16.buffer_bytes(3), 8);
        let f16n = Layout::of(
            DataType::F16,
            Caps {
                f16: true,
                int64: false,
            },
        )
        .unwrap();
        assert_eq!(f16n.buffer_bytes(3), 8);
        assert_eq!(f16n.buffer_bytes(4), 8);
        let i64 = Layout::of(
            DataType::I64,
            Caps {
                f16: false,
                int64: true,
            },
        )
        .unwrap();
        assert_eq!(i64.buffer_bytes(1), 8);
        assert_eq!(Layout::of(DataType::I64, caps).unwrap().buffer_bytes(1), 4);
    }
}
