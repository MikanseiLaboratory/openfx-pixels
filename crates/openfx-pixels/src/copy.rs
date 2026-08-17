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
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, width_bytes);
        }
    }
    Ok(())
}
