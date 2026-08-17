use criterion::{Criterion, black_box, criterion_group, criterion_main};
use openfx::image::{PixelComponents, PixelDepth, RectI};
use openfx_pixels::{
    ConvertSource, ConvertSpec, PackedOrder, packed_frame_hash, write_packed_row,
};

fn bench_row(c: &mut Criterion) {
    let mut src = vec![0u8; 1920 * 4];
    for (i, b) in src.iter_mut().enumerate() {
        *b = (i * 13 + 7) as u8;
    }
    let mut dst = vec![0u8; 1920 * 4];
    c.bench_function("write_packed_row_byte_1920", |b| {
        b.iter(|| unsafe {
            write_packed_row(
                PackedOrder::Bgra,
                PixelDepth::Byte,
                PixelComponents::Rgba,
                src.as_ptr(),
                black_box(&mut dst),
                1920,
            )
        });
    });
}

fn bench_window(c: &mut Criterion) {
    let window = RectI {
        x1: 0,
        y1: 0,
        x2: 1920,
        y2: 1080,
    };
    let mut src = vec![0u8; 1920 * 1080 * 4];
    for (i, b) in src.iter_mut().enumerate() {
        *b = (i * 11 + 5) as u8;
    }
    let source = ConvertSource {
        window,
        bounds: window,
        row_bytes: 1920 * 4,
        data: src.as_ptr(),
        depth: PixelDepth::Byte,
        components: PixelComponents::Rgba,
    };

    c.bench_function("convert_window_1080p_serial", |b| {
        b.iter(|| unsafe {
            openfx_pixels::convert_window_into(
                Vec::new(),
                source,
                ConvertSpec {
                    parallel_rows: false,
                    track_alpha: false,
                    ..ConvertSpec::BGRA_VMX
                },
            )
        });
    });

    c.bench_function("convert_window_1080p_parallel", |b| {
        b.iter(|| unsafe {
            openfx_pixels::convert_window_into(
                Vec::new(),
                source,
                ConvertSpec {
                    track_alpha: false,
                    ..ConvertSpec::BGRA_VMX
                },
            )
        });
    });
}

fn bench_hash(c: &mut Criterion) {
    let data = vec![0u8; 1920 * 1080 * 4];
    c.bench_function("packed_frame_hash_1080p", |b| {
        b.iter(|| packed_frame_hash(black_box(1920), black_box(1080), black_box(&data)));
    });
}

criterion_group!(benches, bench_row, bench_window, bench_hash);
criterion_main!(benches);
