use std::time::Instant;

pub const TICKS_PER_SECOND: i64 = 10_000_000;

#[derive(Debug)]
pub struct SessionClock {
    start: Instant,
    last: i64,
}

impl SessionClock {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            last: -1,
        }
    }

    pub fn next_monotonic(&mut self) -> i64 {
        let nanos = self.start.elapsed().as_nanos();
        let candidate = (nanos / 100) as i64;
        let next = if candidate <= self.last {
            self.last.saturating_add(1)
        } else {
            candidate
        };
        self.last = next;
        next
    }
}

impl Default for SessionClock {
    fn default() -> Self {
        Self::new()
    }
}

pub fn video_interval_ticks(fps_n: i32, fps_d: i32) -> i64 {
    let n = fps_n.max(1) as i64;
    let d = fps_d.max(1) as i64;
    (TICKS_PER_SECOND * d) / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_is_monotonic() {
        let mut clock = SessionClock::new();
        let a = clock.next_monotonic();
        let b = clock.next_monotonic();
        assert!(b > a);
        assert!(a >= 0);
    }

    #[test]
    fn video_interval_matches_100ns_ticks() {
        assert_eq!(video_interval_ticks(60, 1), TICKS_PER_SECOND / 60);
        assert_eq!(
            video_interval_ticks(30_000, 1_001),
            (TICKS_PER_SECOND * 1_001) / 30_000
        );
    }
}
