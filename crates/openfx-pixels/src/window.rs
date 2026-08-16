use openfx::image::{PixelComponents, PixelDepth, RectI, pixel_byte_offset};

use crate::convert::{PackedOrder, write_packed_row};

pub const DEFAULT_MIN_VIDEO_DIM: u32 = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvertSpec {
    pub order: PackedOrder,
    pub min_dim: u32,
    pub even_align: bool,
}

impl ConvertSpec {
    pub const BGRA_VMX: Self = Self {
        order: PackedOrder::Bgra,
        min_dim: DEFAULT_MIN_VIDEO_DIM,
        even_align: true,
    };

    pub const RGBA: Self = Self {
        order: PackedOrder::Rgba,
        min_dim: DEFAULT_MIN_VIDEO_DIM,
        even_align: false,
    };
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaError {
    TooSmall {
        width: u32,
        height: u32,
        min_dim: u32,
    },
    EmptyWindow,
}

impl std::fmt::Display for MediaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooSmall {
                width,
                height,
                min_dim,
            } => write!(
                f,
                "video {width}x{height} is smaller than {min_dim}x{min_dim}"
            ),
            Self::EmptyWindow => write!(f, "render window is empty"),
        }
    }
}

impl std::error::Error for MediaError {}

#[derive(Debug, Clone)]
pub struct ConvertedVideo {
    pub width: u32,
    pub height: u32,
    pub stride: i32,
    pub data: Vec<u8>,
    pub has_alpha: bool,
    pub order: PackedOrder,
}

#[derive(Debug, Clone, Copy)]
pub struct ConvertSource {
    pub window: RectI,
    pub bounds: RectI,
    pub row_bytes: i32,
    pub data: *const u8,
    pub depth: PixelDepth,
    pub components: PixelComponents,
}

/// Convert an OFX image window to tightly packed top-down BGRA8.
///
/// # Safety
/// `data` must remain valid for `bounds` / `row_bytes` for the duration of the call.
pub unsafe fn convert_window_to_bgra(
    window: RectI,
    bounds: RectI,
    row_bytes: i32,
    data: *const u8,
    depth: PixelDepth,
    components: PixelComponents,
) -> Result<ConvertedVideo, MediaError> {
    unsafe {
        convert_window_into(
            Vec::new(),
            ConvertSource {
                window,
                bounds,
                row_bytes,
                data,
                depth,
                components,
            },
            ConvertSpec::BGRA_VMX,
        )
    }
}

/// Convert an OFX image window to tightly packed top-down RGBA8.
///
/// # Safety
/// `data` must remain valid for `bounds` / `row_bytes` for the duration of the call.
pub unsafe fn convert_window_to_rgba(
    window: RectI,
    bounds: RectI,
    row_bytes: i32,
    data: *const u8,
    depth: PixelDepth,
    components: PixelComponents,
) -> Result<ConvertedVideo, MediaError> {
    unsafe {
        convert_window_into(
            Vec::new(),
            ConvertSource {
                window,
                bounds,
                row_bytes,
                data,
                depth,
                components,
            },
            ConvertSpec::RGBA,
        )
    }
}

/// Same as the window converters, reusing `packed`'s allocation when possible.
///
/// # Safety
/// `source.data` must remain valid for `source.bounds` / `source.row_bytes`.
pub unsafe fn convert_window_into(
    mut packed: Vec<u8>,
    source: ConvertSource,
    spec: ConvertSpec,
) -> Result<ConvertedVideo, MediaError> {
    let mut window = source.window;
    if spec.even_align {
        if window.width() % 2 != 0 {
            window.x2 -= 1;
        }
        if window.height() % 2 != 0 {
            window.y2 -= 1;
        }
    }
    let width = window.width();
    let height = window.height();
    if width <= 0 || height <= 0 {
        return Err(MediaError::EmptyWindow);
    }
    let width = width as u32;
    let height = height as u32;
    if width < spec.min_dim || height < spec.min_dim {
        return Err(MediaError::TooSmall {
            width,
            height,
            min_dim: spec.min_dim,
        });
    }

    let ConvertSource {
        bounds,
        row_bytes,
        data,
        depth,
        components,
        ..
    } = source;
    let bpp = depth.bytes_per_channel() * components.count();
    let stride = (width as usize).saturating_mul(4);
    let needed = stride.saturating_mul(height as usize);
    packed.clear();
    packed.resize(needed, 0);
    let x1 = window.x1.max(bounds.x1);
    let x2 = window.x2.min(bounds.x2);
    if x2 <= x1 {
        return Err(MediaError::EmptyWindow);
    }
    let count = (x2 - x1) as usize;
    let dst_x0 = (x1 - window.x1) as usize;

    let mut has_alpha = false;
    for out_y in 0..height as i32 {
        let src_y = window.y2 - 1 - out_y;
        if src_y < bounds.y1 || src_y >= bounds.y2 {
            continue;
        }
        let Ok(offset) = pixel_byte_offset(bounds, row_bytes, bpp, x1, src_y) else {
            continue;
        };
        let src = unsafe { data.offset(offset) };
        let dst_row = out_y as usize * stride + dst_x0 * 4;
        has_alpha |= unsafe {
            write_packed_row(
                spec.order,
                depth,
                components,
                src,
                &mut packed[dst_row..dst_row + count * 4],
                count,
            )
        };
    }

    Ok(ConvertedVideo {
        width,
        height,
        stride: stride as i32,
        data: packed,
        has_alpha,
        order: spec.order,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_small_frames() {
        let window = RectI {
            x1: 0,
            y1: 0,
            x2: 8,
            y2: 8,
        };
        let err = unsafe {
            convert_window_to_bgra(
                window,
                window,
                32,
                [0u8; 8 * 8 * 4].as_ptr(),
                PixelDepth::Byte,
                PixelComponents::Rgba,
            )
        }
        .unwrap_err();
        assert!(matches!(err, MediaError::TooSmall { .. }));
    }

    #[test]
    fn even_aligns_odd_bgra_windows() {
        let window = RectI {
            x1: 0,
            y1: 0,
            x2: 17,
            y2: 17,
        };
        let converted = unsafe {
            convert_window_to_bgra(
                window,
                window,
                17 * 4,
                [0u8; 17 * 17 * 4].as_ptr(),
                PixelDepth::Byte,
                PixelComponents::Rgba,
            )
        }
        .expect("odd window should crop to even");
        assert_eq!(converted.width, 16);
        assert_eq!(converted.height, 16);
        assert_eq!(converted.order, PackedOrder::Bgra);
    }

    #[test]
    fn rgba_keeps_odd_windows() {
        let window = RectI {
            x1: 0,
            y1: 0,
            x2: 16,
            y2: 16,
        };
        let converted = unsafe {
            convert_window_to_rgba(
                window,
                window,
                16 * 4,
                [0u8; 16 * 16 * 4].as_ptr(),
                PixelDepth::Byte,
                PixelComponents::Rgba,
            )
        }
        .expect("rgba convert");
        assert_eq!(converted.order, PackedOrder::Rgba);
        assert_eq!(converted.width, 16);
    }
}
