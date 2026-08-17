use std::sync::atomic::{AtomicBool, Ordering};

use openfx::MultiThread;
use openfx::image::{PixelComponents, PixelDepth, RectI, pixel_byte_offset};

use crate::convert::{PackedOrder, RowWriter};

pub const DEFAULT_MIN_VIDEO_DIM: u32 = 16;

/// Host services required for parallel conversion.
#[derive(Clone, Copy)]
pub struct ConvertHost<'a> {
    /// `OfxMultiThreadSuite` from the host. Required when [`ConvertSpec::parallel_rows`] is true.
    pub multithread: &'a MultiThread,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvertSpec {
    pub order: PackedOrder,
    pub min_dim: u32,
    pub even_align: bool,
    /// Process scanlines on multiple CPU cores. Enabled by default.
    pub parallel_rows: bool,
    /// Scan each row for non-opaque alpha. Disable when callers do not need `has_alpha`.
    pub track_alpha: bool,
}

impl ConvertSpec {
    pub const BGRA_VMX: Self = Self {
        order: PackedOrder::Bgra,
        min_dim: DEFAULT_MIN_VIDEO_DIM,
        even_align: true,
        parallel_rows: true,
        track_alpha: true,
    };

    pub const RGBA: Self = Self {
        order: PackedOrder::Rgba,
        min_dim: DEFAULT_MIN_VIDEO_DIM,
        even_align: false,
        parallel_rows: true,
        track_alpha: true,
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
    /// [`ConvertSpec::parallel_rows`] was set without a [`ConvertHost::multithread`] handle.
    MissingMultiThread,
    /// `OfxMultiThreadSuite::multiThread` failed (e.g. recursive call).
    ParallelFailed,
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
            Self::MissingMultiThread => {
                write!(f, "parallel_rows requires ConvertHost::multithread")
            }
            Self::ParallelFailed => write!(f, "OfxMultiThreadSuite::multiThread failed"),
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
            ConvertSpec {
                parallel_rows: false,
                ..ConvertSpec::BGRA_VMX
            },
            None,
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
            ConvertSpec {
                parallel_rows: false,
                ..ConvertSpec::RGBA
            },
            None,
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
    host: Option<ConvertHost<'_>>,
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
    if packed.capacity() < needed {
        packed.reserve(needed);
    }
    // SAFETY: every pixel in the output rectangle is written below.
    unsafe { packed.set_len(needed) };
    let x1 = window.x1.max(bounds.x1);
    let x2 = window.x2.min(bounds.x2);
    if x2 <= x1 {
        return Err(MediaError::EmptyWindow);
    }
    let count = (x2 - x1) as usize;
    let dst_x0 = (x1 - window.x1) as usize;
    let writer = RowWriter::resolve(spec.order, depth, components);
    let ctx = RowConvertCtx {
        window,
        bounds,
        row_bytes,
        data: data as usize,
        bpp,
        stride,
        count,
        dst_x0,
        writer,
        track_alpha: spec.track_alpha,
    };

    let has_alpha = if spec.parallel_rows && height > 1 {
        let multithread = host.ok_or(MediaError::MissingMultiThread)?.multithread;
        unsafe { convert_rows_ofx(multithread, &packed, height, ctx) }
            .map_err(|_| MediaError::ParallelFailed)?
    } else {
        unsafe { convert_rows_serial(&mut packed, height, ctx) }
    };

    Ok(ConvertedVideo {
        width,
        height,
        stride: stride as i32,
        data: packed,
        has_alpha,
        order: spec.order,
    })
}

#[derive(Clone, Copy)]
struct RowConvertCtx {
    window: RectI,
    bounds: RectI,
    row_bytes: i32,
    data: usize,
    bpp: usize,
    stride: usize,
    count: usize,
    dst_x0: usize,
    writer: RowWriter,
    track_alpha: bool,
}

#[inline]
unsafe fn convert_one_row(packed: *mut u8, out_y: i32, ctx: &RowConvertCtx) -> bool {
    let RowConvertCtx {
        window,
        bounds,
        row_bytes,
        data: _,
        bpp,
        stride,
        count,
        dst_x0,
        writer,
        track_alpha,
    } = *ctx;
    let src_y = window.y2 - 1 - out_y;
    if src_y < bounds.y1 || src_y >= bounds.y2 {
        return false;
    }
    let Ok(offset) = pixel_byte_offset(bounds, row_bytes, bpp, window.x1.max(bounds.x1), src_y)
    else {
        return false;
    };
    let src = unsafe { (ctx.data as *const u8).offset(offset) };
    let dst_row = out_y as usize * stride + dst_x0 * 4;
    unsafe {
        writer.write_row(
            src,
            std::slice::from_raw_parts_mut(packed.add(dst_row), count * 4),
            count,
            track_alpha,
        )
    }
}

unsafe fn convert_rows_serial(packed: &mut [u8], height: u32, ctx: RowConvertCtx) -> bool {
    let base = packed.as_mut_ptr();
    let mut has_alpha = false;
    for out_y in 0..height as i32 {
        has_alpha |= unsafe { convert_one_row(base, out_y, &ctx) };
    }
    has_alpha
}

struct RowParallelWork {
    base: usize,
    height: u32,
    ctx: RowConvertCtx,
    has_alpha: AtomicBool,
}

unsafe extern "C" fn convert_rows_worker(
    thread_index: u32,
    thread_max: u32,
    custom_arg: *mut std::ffi::c_void,
) {
    let work = unsafe { &*(custom_arg as *const RowParallelWork) };
    let base = work.base as *mut u8;
    let mut local_alpha = false;
    let mut out_y = thread_index as i32;
    while out_y < work.height as i32 {
        local_alpha |= unsafe { convert_one_row(base, out_y, &work.ctx) };
        out_y += thread_max as i32;
    }
    if local_alpha {
        work.has_alpha.store(true, Ordering::Relaxed);
    }
}

unsafe fn convert_rows_ofx(
    multithread: &MultiThread,
    packed: &[u8],
    height: u32,
    ctx: RowConvertCtx,
) -> Result<bool, openfx::OfxStatus> {
    let n_threads = multithread.num_cpus()?.min(height).max(1);
    let work = RowParallelWork {
        base: packed.as_ptr() as usize,
        height,
        ctx,
        has_alpha: AtomicBool::new(false),
    };
    multithread.parallel(
        n_threads,
        Some(convert_rows_worker),
        &work as *const RowParallelWork as *mut std::ffi::c_void,
    )?;
    Ok(work.has_alpha.load(Ordering::Relaxed))
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

    #[test]
    fn parallel_matches_serial() {
        let window = RectI {
            x1: 0,
            y1: 0,
            x2: 64,
            y2: 64,
        };
        let mut src = vec![0u8; 64 * 64 * 4];
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i * 17 + 3) as u8;
        }
        src[7] = 128;

        let serial = unsafe {
            convert_window_into(
                Vec::new(),
                ConvertSource {
                    window,
                    bounds: window,
                    row_bytes: 64 * 4,
                    data: src.as_ptr(),
                    depth: PixelDepth::Byte,
                    components: PixelComponents::Rgba,
                },
                ConvertSpec {
                    parallel_rows: false,
                    ..ConvertSpec::RGBA
                },
                None,
            )
        }
        .expect("serial");

        let parallel = unsafe {
            convert_window_into(
                Vec::new(),
                ConvertSource {
                    window,
                    bounds: window,
                    row_bytes: 64 * 4,
                    data: src.as_ptr(),
                    depth: PixelDepth::Byte,
                    components: PixelComponents::Rgba,
                },
                ConvertSpec::RGBA,
                Some(ConvertHost {
                    multithread: test_multithread(),
                }),
            )
        }
        .expect("parallel");

        assert_eq!(serial.data, parallel.data);
        assert_eq!(serial.has_alpha, parallel.has_alpha);
    }

    #[test]
    fn ofx_parallel_matches_serial() {
        let window = RectI {
            x1: 0,
            y1: 0,
            x2: 64,
            y2: 64,
        };
        let mut src = vec![0u8; 64 * 64 * 4];
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i * 17 + 3) as u8;
        }
        src[7] = 128;

        let serial = unsafe {
            convert_window_into(
                Vec::new(),
                ConvertSource {
                    window,
                    bounds: window,
                    row_bytes: 64 * 4,
                    data: src.as_ptr(),
                    depth: PixelDepth::Byte,
                    components: PixelComponents::Rgba,
                },
                ConvertSpec {
                    parallel_rows: false,
                    ..ConvertSpec::RGBA
                },
                None,
            )
        }
        .expect("serial");

        let ofx_parallel = unsafe {
            convert_window_into(
                Vec::new(),
                ConvertSource {
                    window,
                    bounds: window,
                    row_bytes: 64 * 4,
                    data: src.as_ptr(),
                    depth: PixelDepth::Byte,
                    components: PixelComponents::Rgba,
                },
                ConvertSpec::RGBA,
                Some(ConvertHost {
                    multithread: test_multithread(),
                }),
            )
        }
        .expect("ofx parallel");

        assert_eq!(serial.data, ofx_parallel.data);
        assert_eq!(serial.has_alpha, ofx_parallel.has_alpha);
    }

    #[test]
    fn parallel_without_host_errors() {
        let window = RectI {
            x1: 0,
            y1: 0,
            x2: 64,
            y2: 64,
        };
        let src = vec![0u8; 64 * 64 * 4];
        let err = unsafe {
            convert_window_into(
                Vec::new(),
                ConvertSource {
                    window,
                    bounds: window,
                    row_bytes: 64 * 4,
                    data: src.as_ptr(),
                    depth: PixelDepth::Byte,
                    components: PixelComponents::Rgba,
                },
                ConvertSpec::RGBA,
                None,
            )
        }
        .unwrap_err();
        assert!(matches!(err, MediaError::MissingMultiThread));
    }

    fn test_multithread() -> &'static MultiThread {
        use openfx::bindings::OfxMultiThreadSuiteV1;
        use openfx::status::kOfxStat;

        unsafe extern "C" fn mock_multi_thread(
            func: openfx::bindings::OfxThreadFunctionV1,
            n_threads: u32,
            custom_arg: *mut std::ffi::c_void,
        ) -> openfx::OfxStatus {
            let Some(func) = func else {
                return kOfxStat::Failed;
            };
            for thread_index in 0..n_threads {
                unsafe { func(thread_index, n_threads, custom_arg) };
            }
            kOfxStat::OK
        }

        unsafe extern "C" fn mock_num_cpus(n_cpus: *mut u32) -> openfx::OfxStatus {
            if n_cpus.is_null() {
                return kOfxStat::Failed;
            }
            unsafe { *n_cpus = 4 };
            kOfxStat::OK
        }

        static SUITE: OfxMultiThreadSuiteV1 = OfxMultiThreadSuiteV1 {
            multiThread: Some(mock_multi_thread),
            multiThreadNumCPUs: Some(mock_num_cpus),
            multiThreadIndex: None,
            multiThreadIsSpawnedThread: None,
            mutexCreate: None,
            mutexDestroy: None,
            mutexLock: None,
            mutexUnLock: None,
            mutexTryLock: None,
        };
        static MT: MultiThread = MultiThread::from_suite(&SUITE);
        &MT
    }
}
