use criterion::{Criterion, criterion_group, criterion_main};
use openfx::MultiThread;
use openfx::bindings::OfxMultiThreadSuiteV1;
use openfx::image::{PixelComponents, PixelDepth, RectI};
use openfx::status::kOfxStat;
use openfx_pixels::{
    ConvertHost, ConvertSource, ConvertSpec, PackedOrder, packed_frame_hash, write_packed_row,
};

static BENCH_SUITE: OfxMultiThreadSuiteV1 = OfxMultiThreadSuiteV1 {
    multiThread: Some(bench_multi_thread),
    multiThreadNumCPUs: Some(bench_num_cpus),
    multiThreadIndex: None,
    multiThreadIsSpawnedThread: None,
    mutexCreate: None,
    mutexDestroy: None,
    mutexLock: None,
    mutexUnLock: None,
    mutexTryLock: None,
};

static BENCH_MT: MultiThread = MultiThread::from_suite(&BENCH_SUITE);

unsafe extern "C" fn bench_multi_thread(
    func: openfx::bindings::OfxThreadFunctionV1,
    n_threads: u32,
    custom_arg: *mut std::ffi::c_void,
) -> openfx::OfxStatus {
    let Some(func) = func else {
        return kOfxStat::Failed;
    };
    let arg = custom_arg as usize;
    std::thread::scope(|scope| {
        for thread_index in 0..n_threads {
            scope.spawn(move || unsafe {
                func(thread_index, n_threads, arg as *mut std::ffi::c_void);
            });
        }
    });
    kOfxStat::OK
}

unsafe extern "C" fn bench_num_cpus(n_cpus: *mut u32) -> openfx::OfxStatus {
    if n_cpus.is_null() {
        return kOfxStat::Failed;
    }
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1);
    unsafe { *n_cpus = cpus.max(1) };
    kOfxStat::OK
}

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
                &mut dst,
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
    let host = Some(ConvertHost {
        multithread: &BENCH_MT,
    });

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
                None,
            )
        });
    });

    c.bench_function("convert_window_1080p_ofx_parallel", |b| {
        b.iter(|| unsafe {
            openfx_pixels::convert_window_into(
                Vec::new(),
                source,
                ConvertSpec {
                    track_alpha: false,
                    ..ConvertSpec::BGRA_VMX
                },
                host,
            )
        });
    });
}

fn bench_hash(c: &mut Criterion) {
    let data = vec![0u8; 1920 * 1080 * 4];
    c.bench_function("packed_frame_hash_1080p", |b| {
        b.iter(|| packed_frame_hash(1920, 1080, &data));
    });
}

criterion_group!(benches, bench_row, bench_window, bench_hash);
criterion_main!(benches);
