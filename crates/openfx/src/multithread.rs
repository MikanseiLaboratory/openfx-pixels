use std::ffi::c_void;

use crate::bindings::{OfxMultiThreadSuiteV1, OfxThreadFunctionV1, kOfxMultiThreadSuite};
use crate::status::OfxResult;
use crate::suites::Host;

/// Host-managed SMP via `OfxMultiThreadSuiteV1`.
#[derive(Clone, Copy)]
pub struct MultiThread {
    suite: &'static OfxMultiThreadSuiteV1,
}

unsafe impl Send for MultiThread {}
unsafe impl Sync for MultiThread {}

impl MultiThread {
    pub unsafe fn fetch(host: Host) -> OfxResult<Self> {
        let suite = unsafe { host.fetch(kOfxMultiThreadSuite, 1) } as *const OfxMultiThreadSuiteV1;
        Ok(Self {
            suite: unsafe { suite.as_ref() }
                .ok_or(crate::status::kOfxStat::ErrMissingHostFeature)?,
        })
    }

    /// Wraps a static suite table (unit tests only).
    #[doc(hidden)]
    pub const fn from_suite(suite: &'static OfxMultiThreadSuiteV1) -> Self {
        Self { suite }
    }

    /// CPUs the host allows plugins to use for SMP.
    pub fn num_cpus(&self) -> OfxResult<u32> {
        let get = self
            .suite
            .multiThreadNumCPUs
            .ok_or(crate::status::kOfxStat::ErrMissingHostFeature)?;
        let mut cpus = 0u32;
        unsafe { get(&mut cpus) }.ofx_ok()?;
        Ok(cpus.max(1))
    }

    /// Spawn `n_threads` workers via the host thread pool.
    ///
    /// `multiThread` blocks until every worker returns. It must not be called
    /// recursively from inside `func`.
    pub fn parallel(
        &self,
        n_threads: u32,
        func: OfxThreadFunctionV1,
        custom_arg: *mut c_void,
    ) -> OfxResult<()> {
        let spawn = self
            .suite
            .multiThread
            .ok_or(crate::status::kOfxStat::ErrMissingHostFeature)?;
        let n_threads = n_threads.max(1);
        unsafe { spawn(func, n_threads, custom_arg) }.ofx_ok()
    }

    pub fn is_spawned_thread(&self) -> bool {
        self.suite
            .multiThreadIsSpawnedThread
            .is_some_and(|f| unsafe { f() != 0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OfxStatus;
    use crate::status::kOfxStat;
    use std::ptr;
    use std::sync::atomic::{AtomicU32, Ordering};

    unsafe extern "C" fn count_worker(thread_index: u32, thread_max: u32, custom_arg: *mut c_void) {
        let seen = unsafe { &*(custom_arg as *mut AtomicU32) };
        seen.fetch_add(1 << thread_index, Ordering::SeqCst);
        assert!(thread_index < thread_max);
    }

    unsafe extern "C" fn mock_multi_thread(
        func: OfxThreadFunctionV1,
        n_threads: u32,
        custom_arg: *mut c_void,
    ) -> OfxStatus {
        let Some(func) = func else {
            return kOfxStat::Failed;
        };
        for thread_index in 0..n_threads {
            unsafe { func(thread_index, n_threads, custom_arg) };
        }
        kOfxStat::OK
    }

    unsafe extern "C" fn mock_num_cpus(n_cpus: *mut u32) -> OfxStatus {
        if n_cpus.is_null() {
            return kOfxStat::Failed;
        }
        unsafe { *n_cpus = 4 };
        kOfxStat::OK
    }

    #[test]
    fn parallel_invokes_each_worker() {
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
        let mt = MultiThread::from_suite(&SUITE);
        let seen = AtomicU32::new(0);
        mt.parallel(4, Some(count_worker), &seen as *const _ as *mut c_void)
            .expect("parallel");
        assert_eq!(seen.load(Ordering::SeqCst), 0b1111);
        assert_eq!(mt.num_cpus().expect("cpus"), 4);
    }

    #[test]
    fn fetch_missing_suite_fails() {
        unsafe extern "C" fn fetch_null(
            _host: *mut crate::bindings::OfxPropertySetStruct,
            _name: *const i8,
            _version: i32,
        ) -> *const std::ffi::c_void {
            ptr::null()
        }
        let host = Host {
            host: ptr::null_mut(),
            fetch_suite: fetch_null,
        };
        assert!(unsafe { MultiThread::fetch(host) }.is_err());
    }
}
