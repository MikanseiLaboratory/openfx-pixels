//! SIMD OFX row conversion to packed 8-bit BGRA or RGBA.
//!
//! Runtime dispatch matches vmx-rs: one binary per architecture, best path at
//! startup (`avx2` / `ssse3` / `sse2` / `neon` / `scalar`).

use std::sync::OnceLock;

use openfx::image::{PixelComponents, PixelDepth};

/// Packed 8-bit output order after OFX→u8 conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedOrder {
    Bgra,
    Rgba,
}

impl PackedOrder {
    fn swizzle_bgra(self) -> bool {
        matches!(self, Self::Bgra)
    }
}

/// Selected OFX→packed conversion path for this process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvertSimdPath {
    Scalar,
    Sse2,
    Ssse3,
    Avx2,
    Neon,
}

impl ConvertSimdPath {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
            if is_x86_feature_detected!("ssse3") {
                return Self::Ssse3;
            }
            if is_x86_feature_detected!("sse2") {
                return Self::Sse2;
            }
            Self::Scalar
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self::Neon
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::Scalar
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Sse2 => "sse2",
            Self::Ssse3 => "ssse3",
            Self::Avx2 => "avx2",
            Self::Neon => "neon",
        }
    }
}

impl std::fmt::Display for ConvertSimdPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

pub fn convert_simd_path() -> ConvertSimdPath {
    static PATH: OnceLock<ConvertSimdPath> = OnceLock::new();
    *PATH.get_or_init(ConvertSimdPath::detect)
}

/// Convert `count` OFX pixels into packed 8-bit BGRA or RGBA.
///
/// # Safety
/// `src` must be valid for `count` pixels of `depth`/`components`.
pub unsafe fn write_packed_row(
    order: PackedOrder,
    depth: PixelDepth,
    components: PixelComponents,
    src: *const u8,
    dst: &mut [u8],
    count: usize,
) -> bool {
    debug_assert!(dst.len() >= count.saturating_mul(4));
    let swizzle = order.swizzle_bgra();
    match (convert_simd_path(), depth, components) {
        #[cfg(target_arch = "x86_64")]
        (ConvertSimdPath::Avx2, PixelDepth::Float, PixelComponents::Rgba) => unsafe {
            x86::write_float_rgba_avx2(src, dst, count, swizzle)
        },
        #[cfg(target_arch = "x86_64")]
        (ConvertSimdPath::Avx2, PixelDepth::Byte, PixelComponents::Rgba) => unsafe {
            x86::write_byte_rgba_avx2(src, dst, count, swizzle)
        },
        #[cfg(target_arch = "x86_64")]
        (ConvertSimdPath::Avx2, PixelDepth::Short, PixelComponents::Rgba) => unsafe {
            x86::write_short_rgba_avx2(src, dst, count, swizzle)
        },
        #[cfg(target_arch = "x86_64")]
        (ConvertSimdPath::Ssse3, PixelDepth::Byte, PixelComponents::Rgba) => unsafe {
            x86::write_byte_rgba_ssse3(src, dst, count, swizzle)
        },
        #[cfg(target_arch = "x86_64")]
        (ConvertSimdPath::Ssse3, PixelDepth::Short, PixelComponents::Rgba) => unsafe {
            x86::write_short_rgba_ssse3(src, dst, count, swizzle)
        },
        #[cfg(target_arch = "x86_64")]
        (
            ConvertSimdPath::Sse2 | ConvertSimdPath::Ssse3,
            PixelDepth::Float,
            PixelComponents::Rgba,
        ) => unsafe { x86::write_float_rgba_sse2(src, dst, count, swizzle) },
        #[cfg(target_arch = "aarch64")]
        (ConvertSimdPath::Neon, PixelDepth::Float, PixelComponents::Rgba) => unsafe {
            neon::write_float_rgba(src, dst, count, swizzle)
        },
        #[cfg(target_arch = "aarch64")]
        (ConvertSimdPath::Neon, PixelDepth::Byte, PixelComponents::Rgba) => unsafe {
            neon::write_byte_rgba(src, dst, count, swizzle)
        },
        #[cfg(target_arch = "aarch64")]
        (ConvertSimdPath::Neon, PixelDepth::Short, PixelComponents::Rgba) => unsafe {
            neon::write_short_rgba(src, dst, count, swizzle)
        },
        _ => unsafe { write_packed_row_scalar(order, depth, components, src, dst, count) },
    }
}

pub(crate) unsafe fn write_packed_row_scalar(
    order: PackedOrder,
    depth: PixelDepth,
    components: PixelComponents,
    src: *const u8,
    dst: &mut [u8],
    count: usize,
) -> bool {
    let ch = depth.bytes_per_channel();
    let src_bpp = ch * components.count();
    let swizzle = order.swizzle_bgra();
    let mut has_alpha = false;
    for i in 0..count {
        let px = unsafe { src.add(i * src_bpp) };
        let r = unsafe { sample_u8_ptr(depth, px) };
        let g = unsafe { sample_u8_ptr(depth, px.add(ch)) };
        let b = unsafe { sample_u8_ptr(depth, px.add(ch * 2)) };
        let a = if components == PixelComponents::Rgba {
            unsafe { sample_u8_ptr(depth, px.add(ch * 3)) }
        } else {
            255
        };
        if a != 255 {
            has_alpha = true;
        }
        let o = i * 4;
        if swizzle {
            dst[o] = b;
            dst[o + 1] = g;
            dst[o + 2] = r;
            dst[o + 3] = a;
        } else {
            dst[o] = r;
            dst[o + 1] = g;
            dst[o + 2] = b;
            dst[o + 3] = a;
        }
    }
    has_alpha
}

pub fn packed_row_to_pixel(
    order: PackedOrder,
    depth: PixelDepth,
    components: PixelComponents,
    bytes: &[u8],
) -> [u8; 4] {
    let ch = depth.bytes_per_channel();
    let r = sample_u8(depth, bytes);
    let g = sample_u8(depth, &bytes[ch..]);
    let b = sample_u8(depth, &bytes[ch * 2..]);
    let a = if components == PixelComponents::Rgba {
        sample_u8(depth, &bytes[ch * 3..])
    } else {
        255
    };
    match order {
        PackedOrder::Bgra => [b, g, r, a],
        PackedOrder::Rgba => [r, g, b, a],
    }
}

fn sample_u8(depth: PixelDepth, bytes: &[u8]) -> u8 {
    match depth {
        PixelDepth::Byte => bytes[0],
        PixelDepth::Short => {
            let value = u16::from_le_bytes([bytes[0], bytes[1]]);
            (value >> 8) as u8
        }
        PixelDepth::Float => {
            let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            (value.clamp(0.0, 1.0) * 255.0).round() as u8
        }
    }
}

unsafe fn sample_u8_ptr(depth: PixelDepth, ptr: *const u8) -> u8 {
    match depth {
        PixelDepth::Byte => unsafe { *ptr },
        PixelDepth::Short => {
            let bytes = unsafe { [*ptr, *ptr.add(1)] };
            (u16::from_le_bytes(bytes) >> 8) as u8
        }
        PixelDepth::Float => {
            let bytes = unsafe { [*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)] };
            (f32::from_le_bytes(bytes).clamp(0.0, 1.0) * 255.0).round() as u8
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{PackedOrder, PixelComponents, PixelDepth, write_packed_row_scalar};
    use std::arch::x86_64::*;

    fn remainder_order(swizzle: bool) -> PackedOrder {
        if swizzle {
            PackedOrder::Bgra
        } else {
            PackedOrder::Rgba
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn write_float_rgba_avx2(
        src: *const u8,
        dst: &mut [u8],
        count: usize,
        swizzle: bool,
    ) -> bool {
        unsafe {
            let zero = _mm256_setzero_ps();
            let one = _mm256_set1_ps(1.0);
            let scale = _mm256_set1_ps(255.0);
            let alpha_mask = _mm_set1_epi32(0xFF00_0000u32 as i32);
            let mut has_alpha = false;
            let mut i = 0;
            while i + 8 <= count {
                let base = src.add(i * 16) as *const f32;
                let a = load2_i32(base, zero, one, scale, swizzle);
                let b = load2_i32(base.add(8), zero, one, scale, swizzle);
                let c = load2_i32(base.add(16), zero, one, scale, swizzle);
                let d = load2_i32(base.add(24), zero, one, scale, swizzle);
                let p0 = pack4(a, b);
                let p1 = pack4(c, d);
                has_alpha |=
                    alpha_mismatch_128(p0, alpha_mask) || alpha_mismatch_128(p1, alpha_mask);
                _mm_storeu_si128(dst.as_mut_ptr().add(i * 4) as *mut __m128i, p0);
                _mm_storeu_si128(dst.as_mut_ptr().add(i * 4 + 16) as *mut __m128i, p1);
                i += 8;
            }
            if i < count {
                has_alpha |=
                    write_float_rgba_sse2(src.add(i * 16), &mut dst[i * 4..], count - i, swizzle);
            }
            has_alpha
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load2_i32(
        ptr: *const f32,
        zero: __m256,
        one: __m256,
        scale: __m256,
        swizzle: bool,
    ) -> __m256i {
        unsafe {
            let px = _mm256_loadu_ps(ptr);
            let ordered = if swizzle {
                _mm256_shuffle_ps(px, px, 0b11_00_01_10)
            } else {
                px
            };
            let clamped = _mm256_min_ps(_mm256_max_ps(ordered, zero), one);
            _mm256_cvtps_epi32(_mm256_mul_ps(clamped, scale))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn pack4(i0: __m256i, i1: __m256i) -> __m128i {
        let p0 = _mm256_castsi256_si128(i0);
        let p1 = _mm256_extracti128_si256(i0, 1);
        let p2 = _mm256_castsi256_si128(i1);
        let p3 = _mm256_extracti128_si256(i1, 1);
        let pk01 = _mm_packs_epi32(p0, p1);
        let pk23 = _mm_packs_epi32(p2, p3);
        _mm_packus_epi16(pk01, pk23)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn alpha_mismatch_128(packed: __m128i, alpha_mask: __m128i) -> bool {
        let eq = _mm_cmpeq_epi8(_mm_and_si128(packed, alpha_mask), alpha_mask);
        _mm_movemask_epi8(eq) & 0x8888 != 0x8888
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn short2_alpha_mismatch(packed: __m128i) -> bool {
        let alpha_mask = _mm_set1_epi32(0xFF00_0000u32 as i32);
        _mm_movemask_epi8(_mm_cmpeq_epi8(
            _mm_and_si128(packed, alpha_mask),
            alpha_mask,
        )) & 0x0088
            != 0x0088
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn write_byte_rgba_avx2(
        src: *const u8,
        dst: &mut [u8],
        count: usize,
        swizzle: bool,
    ) -> bool {
        unsafe {
            let shuf128 = if swizzle {
                _mm_set_epi8(15, 12, 13, 14, 11, 8, 9, 10, 7, 4, 5, 6, 3, 0, 1, 2)
            } else {
                _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
            };
            let shuf = _mm256_broadcastsi128_si256(shuf128);
            let alpha_mask = _mm256_set1_epi32(0xFF00_0000u32 as i32);
            let mut has_alpha = false;
            let mut i = 0;
            while i + 8 <= count {
                let v = _mm256_loadu_si256(src.add(i * 4) as *const __m256i);
                let packed = _mm256_shuffle_epi8(v, shuf);
                let eq = _mm256_cmpeq_epi8(_mm256_and_si256(packed, alpha_mask), alpha_mask);
                if (_mm256_movemask_epi8(eq) as u32) & 0x8888_8888 != 0x8888_8888 {
                    has_alpha = true;
                }
                _mm256_storeu_si256(dst.as_mut_ptr().add(i * 4) as *mut __m256i, packed);
                i += 8;
            }
            if i < count {
                has_alpha |=
                    write_byte_rgba_ssse3(src.add(i * 4), &mut dst[i * 4..], count - i, swizzle);
            }
            has_alpha
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn write_short_rgba_avx2(
        src: *const u8,
        dst: &mut [u8],
        count: usize,
        swizzle: bool,
    ) -> bool {
        unsafe {
            let shuf = if swizzle {
                _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 7, 4, 5, 6, 3, 0, 1, 2)
            } else {
                _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 7, 6, 5, 4, 3, 2, 1, 0)
            };
            let mut has_alpha = false;
            let mut i = 0;
            while i + 4 <= count {
                let v = _mm256_loadu_si256(src.add(i * 8) as *const __m256i);
                let hi = _mm256_srli_epi16(v, 8);
                let packed8 = _mm256_packus_epi16(hi, _mm256_setzero_si256());
                let lo = _mm256_castsi256_si128(packed8);
                let hi128 = _mm256_extracti128_si256(packed8, 1);
                let p01 = _mm_shuffle_epi8(lo, shuf);
                let p23 = _mm_shuffle_epi8(hi128, shuf);
                if short2_alpha_mismatch(p01) || short2_alpha_mismatch(p23) {
                    has_alpha = true;
                }
                _mm_storel_epi64(dst.as_mut_ptr().add(i * 4) as *mut __m128i, p01);
                _mm_storel_epi64(dst.as_mut_ptr().add(i * 4 + 8) as *mut __m128i, p23);
                i += 4;
            }
            if i < count {
                has_alpha |=
                    write_short_rgba_ssse3(src.add(i * 8), &mut dst[i * 4..], count - i, swizzle);
            }
            has_alpha
        }
    }

    #[target_feature(enable = "sse2")]
    pub unsafe fn write_float_rgba_sse2(
        src: *const u8,
        dst: &mut [u8],
        count: usize,
        swizzle: bool,
    ) -> bool {
        unsafe {
            let zero = _mm_setzero_ps();
            let one = _mm_set1_ps(1.0);
            let scale = _mm_set1_ps(255.0);
            let alpha_mask = _mm_set1_epi32(0xFF00_0000u32 as i32);
            let mut has_alpha = false;
            let mut i = 0;
            while i + 4 <= count {
                let base = src.add(i * 16) as *const f32;
                let t0 = cvt_pixel(_mm_loadu_ps(base), zero, one, scale, swizzle);
                let t1 = cvt_pixel(_mm_loadu_ps(base.add(4)), zero, one, scale, swizzle);
                let t2 = cvt_pixel(_mm_loadu_ps(base.add(8)), zero, one, scale, swizzle);
                let t3 = cvt_pixel(_mm_loadu_ps(base.add(12)), zero, one, scale, swizzle);
                let packed = _mm_packus_epi16(_mm_packs_epi32(t0, t1), _mm_packs_epi32(t2, t3));
                has_alpha |= alpha_mismatch_128(packed, alpha_mask);
                _mm_storeu_si128(dst.as_mut_ptr().add(i * 4) as *mut __m128i, packed);
                i += 4;
            }
            if i < count {
                has_alpha |= write_packed_row_scalar(
                    remainder_order(swizzle),
                    PixelDepth::Float,
                    PixelComponents::Rgba,
                    src.add(i * 16),
                    &mut dst[i * 4..],
                    count - i,
                );
            }
            has_alpha
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn cvt_pixel(
        px: __m128,
        zero: __m128,
        one: __m128,
        scale: __m128,
        swizzle: bool,
    ) -> __m128i {
        let ordered = if swizzle {
            _mm_shuffle_ps(px, px, 0b11_00_01_10)
        } else {
            px
        };
        let clamped = _mm_min_ps(_mm_max_ps(ordered, zero), one);
        _mm_cvtps_epi32(_mm_mul_ps(clamped, scale))
    }

    #[target_feature(enable = "ssse3")]
    pub unsafe fn write_byte_rgba_ssse3(
        src: *const u8,
        dst: &mut [u8],
        count: usize,
        swizzle: bool,
    ) -> bool {
        unsafe {
            let shuf = if swizzle {
                _mm_set_epi8(15, 12, 13, 14, 11, 8, 9, 10, 7, 4, 5, 6, 3, 0, 1, 2)
            } else {
                _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
            };
            let alpha_mask = _mm_set1_epi32(0xFF00_0000u32 as i32);
            let mut has_alpha = false;
            let mut i = 0;
            while i + 4 <= count {
                let packed =
                    _mm_shuffle_epi8(_mm_loadu_si128(src.add(i * 4) as *const __m128i), shuf);
                has_alpha |= alpha_mismatch_128(packed, alpha_mask);
                _mm_storeu_si128(dst.as_mut_ptr().add(i * 4) as *mut __m128i, packed);
                i += 4;
            }
            if i < count {
                has_alpha |= write_packed_row_scalar(
                    remainder_order(swizzle),
                    PixelDepth::Byte,
                    PixelComponents::Rgba,
                    src.add(i * 4),
                    &mut dst[i * 4..],
                    count - i,
                );
            }
            has_alpha
        }
    }

    #[target_feature(enable = "ssse3")]
    pub unsafe fn write_short_rgba_ssse3(
        src: *const u8,
        dst: &mut [u8],
        count: usize,
        swizzle: bool,
    ) -> bool {
        unsafe {
            let shuf = if swizzle {
                _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 7, 4, 5, 6, 3, 0, 1, 2)
            } else {
                _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 7, 6, 5, 4, 3, 2, 1, 0)
            };
            let alpha_mask = _mm_set1_epi32(0xFF00_0000u32 as i32);
            let mut has_alpha = false;
            let mut i = 0;
            while i + 2 <= count {
                let v = _mm_loadu_si128(src.add(i * 8) as *const __m128i);
                let packed8 = _mm_packus_epi16(_mm_srli_epi16(v, 8), _mm_setzero_si128());
                let packed = _mm_shuffle_epi8(packed8, shuf);
                if _mm_movemask_epi8(_mm_cmpeq_epi8(
                    _mm_and_si128(packed, alpha_mask),
                    alpha_mask,
                )) & 0x0088
                    != 0x0088
                {
                    has_alpha = true;
                }
                let px0 = _mm_cvtsi128_si32(packed) as u32;
                let px1 = _mm_cvtsi128_si32(_mm_srli_si128(packed, 4)) as u32;
                dst[i * 4..i * 4 + 4].copy_from_slice(&px0.to_le_bytes());
                dst[i * 4 + 4..i * 4 + 8].copy_from_slice(&px1.to_le_bytes());
                i += 2;
            }
            if i < count {
                has_alpha |= write_packed_row_scalar(
                    remainder_order(swizzle),
                    PixelDepth::Short,
                    PixelComponents::Rgba,
                    src.add(i * 8),
                    &mut dst[i * 4..],
                    count - i,
                );
            }
            has_alpha
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{PackedOrder, PixelComponents, PixelDepth, write_packed_row_scalar};
    use std::arch::aarch64::*;

    fn remainder_order(swizzle: bool) -> PackedOrder {
        if swizzle {
            PackedOrder::Bgra
        } else {
            PackedOrder::Rgba
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn write_float_rgba(
        src: *const u8,
        dst: &mut [u8],
        count: usize,
        swizzle: bool,
    ) -> bool {
        unsafe {
            let zero = vdupq_n_f32(0.0);
            let one = vdupq_n_f32(1.0);
            let scale = vdupq_n_f32(255.0);
            let swizzle_bytes = if swizzle {
                [8u8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15]
            } else {
                [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            };
            let tbl = vld1q_u8(swizzle_bytes.as_ptr());
            let mut has_alpha = false;
            let mut i = 0;
            while i + 4 <= count {
                let base = src.add(i * 16) as *const f32;
                let t0 = cvt_pixel(vld1q_f32(base), zero, one, scale, tbl);
                let t1 = cvt_pixel(vld1q_f32(base.add(4)), zero, one, scale, tbl);
                let t2 = cvt_pixel(vld1q_f32(base.add(8)), zero, one, scale, tbl);
                let t3 = cvt_pixel(vld1q_f32(base.add(12)), zero, one, scale, tbl);
                let n01 = vcombine_s16(vqmovn_s32(t0), vqmovn_s32(t1));
                let n23 = vcombine_s16(vqmovn_s32(t2), vqmovn_s32(t3));
                let packed = vcombine_u8(vqmovun_s16(n01), vqmovun_s16(n23));
                has_alpha |= neon_alpha_mismatch(packed);
                vst1q_u8(dst.as_mut_ptr().add(i * 4), packed);
                i += 4;
            }
            if i < count {
                has_alpha |= write_packed_row_scalar(
                    remainder_order(swizzle),
                    PixelDepth::Float,
                    PixelComponents::Rgba,
                    src.add(i * 16),
                    &mut dst[i * 4..],
                    count - i,
                );
            }
            has_alpha
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn cvt_pixel(
        px: float32x4_t,
        zero: float32x4_t,
        one: float32x4_t,
        scale: float32x4_t,
        swizzle: uint8x16_t,
    ) -> int32x4_t {
        let bgra = vreinterpretq_f32_u8(vqtbl1q_u8(vreinterpretq_u8_f32(px), swizzle));
        let clamped = vminq_f32(vmaxq_f32(bgra, zero), one);
        vcvtnq_s32_f32(vmulq_f32(clamped, scale))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn neon_alpha_mismatch(packed: uint8x16_t) -> bool {
        let px = vreinterpretq_u32_u8(packed);
        let a = vandq_u32(px, vdupq_n_u32(0xFF00_0000));
        vminvq_u32(vceqq_u32(a, vdupq_n_u32(0xFF00_0000))) != 0xFFFF_FFFF
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn write_byte_rgba(
        src: *const u8,
        dst: &mut [u8],
        count: usize,
        swizzle: bool,
    ) -> bool {
        unsafe {
            let idx_bytes = if swizzle {
                [2u8, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15]
            } else {
                [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            };
            let idx = vld1q_u8(idx_bytes.as_ptr());
            let mut has_alpha = false;
            let mut i = 0;
            while i + 4 <= count {
                let packed = vqtbl1q_u8(vld1q_u8(src.add(i * 4)), idx);
                has_alpha |= neon_alpha_mismatch(packed);
                vst1q_u8(dst.as_mut_ptr().add(i * 4), packed);
                i += 4;
            }
            if i < count {
                has_alpha |= write_packed_row_scalar(
                    remainder_order(swizzle),
                    PixelDepth::Byte,
                    PixelComponents::Rgba,
                    src.add(i * 4),
                    &mut dst[i * 4..],
                    count - i,
                );
            }
            has_alpha
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn write_short_rgba(
        src: *const u8,
        dst: &mut [u8],
        count: usize,
        swizzle: bool,
    ) -> bool {
        unsafe {
            let idx_bytes = if swizzle {
                [2u8, 1, 0, 3, 6, 5, 4, 7]
            } else {
                [0u8, 1, 2, 3, 4, 5, 6, 7]
            };
            let idx = vld1_u8(idx_bytes.as_ptr());
            let mut has_alpha = false;
            let mut i = 0;
            while i + 2 <= count {
                let v = vld1q_u16(src.add(i * 8) as *const u16);
                let packed8 = vtbl1_u8(vshrn_n_u16(v, 8), idx);
                has_alpha |= vget_lane_u8(packed8, 3) != 255 || vget_lane_u8(packed8, 7) != 255;
                vst1_u8(dst.as_mut_ptr().add(i * 4), packed8);
                i += 2;
            }
            if i < count {
                has_alpha |= write_packed_row_scalar(
                    remainder_order(swizzle),
                    PixelDepth::Short,
                    PixelComponents::Rgba,
                    src.add(i * 8),
                    &mut dst[i * 4..],
                    count - i,
                );
            }
            has_alpha
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill_row_both(
        order: PackedOrder,
        depth: PixelDepth,
        components: PixelComponents,
        src: &[u8],
        count: usize,
    ) -> (Vec<u8>, Vec<u8>, bool, bool) {
        let mut simd = vec![0u8; count * 4];
        let mut scalar = vec![0u8; count * 4];
        let has_simd =
            unsafe { write_packed_row(order, depth, components, src.as_ptr(), &mut simd, count) };
        let has_scalar = unsafe {
            write_packed_row_scalar(order, depth, components, src.as_ptr(), &mut scalar, count)
        };
        (simd, scalar, has_simd, has_scalar)
    }

    #[test]
    fn detects_a_named_path() {
        let path = convert_simd_path();
        assert!(
            matches!(
                path,
                ConvertSimdPath::Scalar
                    | ConvertSimdPath::Sse2
                    | ConvertSimdPath::Ssse3
                    | ConvertSimdPath::Avx2
                    | ConvertSimdPath::Neon
            ),
            "unexpected path {path}"
        );
        #[cfg(target_arch = "x86_64")]
        {
            assert!(matches!(
                path,
                ConvertSimdPath::Avx2 | ConvertSimdPath::Ssse3 | ConvertSimdPath::Sse2
            ));
        }
        #[cfg(target_arch = "aarch64")]
        {
            assert_eq!(path, ConvertSimdPath::Neon);
        }
    }

    #[test]
    fn byte_rgba_matches_scalar() {
        let mut src = vec![0u8; 1920 * 4];
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i * 13 + 7) as u8;
        }
        src[7] = 200;
        let (simd, scalar, has_simd, has_scalar) = fill_row_both(
            PackedOrder::Bgra,
            PixelDepth::Byte,
            PixelComponents::Rgba,
            &src,
            1920,
        );
        assert_eq!(simd, scalar);
        assert_eq!(has_simd, has_scalar);
        assert!(has_scalar);
    }

    #[test]
    fn float_rgba_matches_scalar_on_exact_values() {
        let mut src = vec![0u8; 64 * 16];
        for i in 0..64 {
            let r = (i as f32) / 63.0;
            let bytes = [r, 0.25, 0.5, 1.0];
            for (c, v) in bytes.into_iter().enumerate() {
                src[i * 16 + c * 4..i * 16 + c * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let (simd, scalar, has_simd, has_scalar) = fill_row_both(
            PackedOrder::Bgra,
            PixelDepth::Float,
            PixelComponents::Rgba,
            &src,
            64,
        );
        assert_eq!(simd, scalar);
        assert_eq!(has_simd, has_scalar);
        assert!(!has_scalar);
    }

    #[test]
    fn short_rgba_matches_scalar() {
        let mut src = vec![0u8; 32 * 8];
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i * 9) as u8;
        }
        src[7] = 0x80;
        let (simd, scalar, has_simd, has_scalar) = fill_row_both(
            PackedOrder::Bgra,
            PixelDepth::Short,
            PixelComponents::Rgba,
            &src,
            32,
        );
        assert_eq!(simd, scalar);
        assert_eq!(has_simd, has_scalar);
    }

    #[test]
    fn remainders_match_scalar() {
        let mut src = vec![0u8; 9 * 4];
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i * 3) as u8;
        }
        src[3] = 200;
        let (simd, scalar, has_simd, has_scalar) = fill_row_both(
            PackedOrder::Bgra,
            PixelDepth::Byte,
            PixelComponents::Rgba,
            &src,
            9,
        );
        assert_eq!(simd, scalar);
        assert_eq!(has_simd, has_scalar);

        let mut srcf = vec![0u8; 5 * 16];
        for i in 0..5 {
            let v = [0.0f32, 0.25, 1.0, 1.0];
            for (c, f) in v.into_iter().enumerate() {
                srcf[i * 16 + c * 4..i * 16 + c * 4 + 4].copy_from_slice(&f.to_le_bytes());
            }
        }
        let (simd, scalar, has_simd, has_scalar) = fill_row_both(
            PackedOrder::Bgra,
            PixelDepth::Float,
            PixelComponents::Rgba,
            &srcf,
            5,
        );
        assert_eq!(simd, scalar);
        assert_eq!(has_simd, has_scalar);
    }

    #[test]
    fn rgb_byte_stays_opaque() {
        let src = [10u8, 20, 30].repeat(17);
        let mut dst = vec![0u8; 17 * 4];
        let has_alpha = unsafe {
            write_packed_row(
                PackedOrder::Bgra,
                PixelDepth::Byte,
                PixelComponents::Rgb,
                src.as_ptr(),
                &mut dst,
                17,
            )
        };
        assert!(!has_alpha);
        assert_eq!(&dst[0..4], &[30, 20, 10, 255]);
    }

    #[test]
    fn byte_rgba_identity_matches_scalar() {
        let mut src = vec![0u8; 1920 * 4];
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i * 13 + 7) as u8;
        }
        src[7] = 200;
        let (simd, scalar, has_simd, has_scalar) = fill_row_both(
            PackedOrder::Rgba,
            PixelDepth::Byte,
            PixelComponents::Rgba,
            &src,
            1920,
        );
        assert_eq!(simd, scalar);
        assert_eq!(has_simd, has_scalar);
        assert!(has_scalar);
        assert_eq!(&simd[0..4], &src[0..4]);
    }

    #[test]
    fn float_rgba_identity_matches_scalar() {
        let mut src = vec![0u8; 64 * 16];
        for i in 0..64 {
            let r = (i as f32) / 63.0;
            let bytes = [r, 0.25, 0.5, 1.0];
            for (c, v) in bytes.into_iter().enumerate() {
                src[i * 16 + c * 4..i * 16 + c * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let (simd, scalar, has_simd, has_scalar) = fill_row_both(
            PackedOrder::Rgba,
            PixelDepth::Float,
            PixelComponents::Rgba,
            &src,
            64,
        );
        assert_eq!(simd, scalar);
        assert_eq!(has_simd, has_scalar);
        assert!(!has_scalar);
    }

    #[test]
    fn packed_pixel_helper_orders() {
        assert_eq!(
            packed_row_to_pixel(
                PackedOrder::Bgra,
                PixelDepth::Byte,
                PixelComponents::Rgb,
                &[1, 2, 3]
            ),
            [3, 2, 1, 255]
        );
        assert_eq!(
            packed_row_to_pixel(
                PackedOrder::Rgba,
                PixelDepth::Byte,
                PixelComponents::Rgb,
                &[1, 2, 3]
            ),
            [1, 2, 3, 255]
        );
    }
}
