//! Shared OFX pixel conversion for live send plugins.
//!
//! Runtime SIMD dispatch (`avx2` / `ssse3` / `sse2` / `neon` / `scalar`) converts
//! host windows to packed 8-bit BGRA or RGBA.

mod clock;
mod convert;
mod copy;
mod hash;
mod pool;
mod window;

pub use clock::{SessionClock, TICKS_PER_SECOND, video_interval_ticks};
pub use convert::{
    ConvertSimdPath, PackedOrder, RowWriter, convert_simd_path, packed_row_to_pixel,
    write_packed_row,
};
pub use copy::copy_image_window;
pub use hash::packed_frame_hash;
pub use pool::PixelPool;
pub use window::{
    ConvertSource, ConvertSpec, ConvertedVideo, DEFAULT_MIN_VIDEO_DIM, MediaError,
    convert_window_into, convert_window_to_bgra, convert_window_to_rgba,
};
