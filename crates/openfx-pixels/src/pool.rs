use std::sync::Mutex;

/// Recycles packed-frame scratch buffers so 1080p convert does not allocate every frame.
#[derive(Debug)]
pub struct PixelPool {
    inner: Mutex<Vec<Vec<u8>>>,
}

impl PixelPool {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Vec::new()),
        }
    }

    pub fn take(&self) -> Vec<u8> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop()
            .unwrap_or_default()
    }

    pub fn release(&self, mut buf: Vec<u8>) {
        buf.clear();
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.len() < 4 {
            inner.push(buf);
        }
    }
}

impl Default for PixelPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuses_capacity() {
        let pool = PixelPool::new();
        let mut buf = pool.take();
        buf.resize(64, 1);
        pool.release(buf);
        let again = pool.take();
        assert!(again.capacity() >= 64);
        assert!(again.is_empty());
    }
}
