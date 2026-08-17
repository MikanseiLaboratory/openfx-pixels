//! Compares serial vs OFX `multiThread` row scheduling.
//!
//! The mock host spawns real OS threads per `multiThread` call. A historical
//! `std::thread::scope` chunk scheduler was ~10% faster in this environment
//! (192 µs vs 213 µs at 1080p) but has been removed in favour of the OFX API.

use criterion::{Criterion, criterion_group, criterion_main};
use openfx::bindings::OfxMultiThreadSuiteV1;
use openfx::image::{PixelComponents, PixelDepth, RectI};
use openfx::status::kOfxStat;
use openfx::MultiThread;
use openfx_pixels::{ConvertHost, ConvertSource, ConvertSpec};

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

fn bench_1080p(c: &mut Criterion) {
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
    let spec = ConvertSpec {
        track_alpha: false,
        ..ConvertSpec::BGRA_VMX
    };
    let host = Some(ConvertHost {
        multithread: &BENCH_MT,
    });

    let mut group = c.benchmark_group("convert_1080p_threading");
    group.bench_function("serial", |b| {
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
    group.bench_function("ofx_multithread_stride", |b| {
        b.iter(|| unsafe {
            openfx_pixels::convert_window_into(Vec::new(), source, spec, host)
        });
    });
    group.finish();
}

criterion_group!(threading_benches, bench_1080p);
criterion_main!(threading_benches);
