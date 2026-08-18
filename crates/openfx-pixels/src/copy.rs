use openfx::image::{ClipImage, RectI};
use openfx::status::{OfxResult, kOfxStat};

pub fn copy_image_window(src: &ClipImage<'_>, dst: &ClipImage<'_>, window: RectI) -> OfxResult<()> {
    if src.depth != dst.depth || src.components != dst.components {
        return Err(kOfxStat::ErrUnsupported);
    }
    let bpp = src.bytes_per_pixel();
    let x1 = window.x1.max(src.bounds.x1).max(dst.bounds.x1);
    let x2 = window.x2.min(src.bounds.x2).min(dst.bounds.x2);
    let y1 = window.y1.max(src.bounds.y1).max(dst.bounds.y1);
    let y2 = window.y2.min(src.bounds.y2).min(dst.bounds.y2);
    if x2 <= x1 || y2 <= y1 {
        return Ok(());
    }
    let width_bytes = (x2 - x1) as usize * bpp;
    for y in y1..y2 {
        unsafe {
            let src_ptr = src.pixel_ptr(x1, y)?;
            let dst_ptr = dst.pixel_ptr(x1, y)?;
            copy_row(src_ptr, dst_ptr, width_bytes);
        }
    }
    Ok(())
}

#[inline]
unsafe fn copy_row(src: *const u8, dst: *mut u8, len: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if len >= 32 && is_x86_feature_detected!("avx2") {
            unsafe { copy_row_avx2(src, dst, len) };
            return;
        }
    }
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, len);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy_row_avx2(src: *const u8, dst: *mut u8, len: usize) {
    use std::arch::x86_64::*;

    unsafe {
        let mut i = 0;
        while i + 32 <= len {
            let v = _mm256_loadu_si256(src.add(i) as *const __m256i);
            _mm256_storeu_si256(dst.add(i) as *mut __m256i, v);
            i += 32;
        }
        if i < len {
            std::ptr::copy_nonoverlapping(src.add(i), dst.add(i), len - i);
        }
    }
}
