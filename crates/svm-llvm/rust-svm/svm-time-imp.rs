//! svm time: `Instant`/`SystemTime` read the host clock through the POSIX personality's `clock` op
//! (svm-posix `OP_CLOCK`) — monotonic (`clock_id 1`) for `Instant`, realtime (`clock_id 0`) for
//! `SystemTime`. Without a granted posix cap the clock reads as the zero epoch (a program that needs
//! time must run via `run_with_caps` + a posix cap). Arithmetic mirrors the shared `unsupported` shape.
use crate::sys::pal::host;
use crate::time::Duration;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant(Duration);

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SystemTime(Duration);

pub const UNIX_EPOCH: SystemTime = SystemTime(Duration::from_secs(0));

fn now(clock_id: i64) -> Duration {
    if !host::have_posix() {
        return Duration::ZERO;
    }
    Duration::from_nanos(host::clock(clock_id).max(0) as u64)
}

impl Instant {
    pub fn now() -> Instant {
        Instant(now(1)) // monotonic
    }

    pub fn checked_sub_instant(&self, other: &Instant) -> Option<Duration> {
        self.0.checked_sub(other.0)
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant(self.0.checked_add(*other)?))
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant(self.0.checked_sub(*other)?))
    }
}

impl SystemTime {
    pub const MAX: SystemTime = SystemTime(Duration::MAX);

    pub const MIN: SystemTime = SystemTime(Duration::ZERO);

    pub fn now() -> SystemTime {
        SystemTime(now(0)) // realtime
    }

    pub fn sub_time(&self, other: &SystemTime) -> Result<Duration, Duration> {
        self.0.checked_sub(other.0).ok_or_else(|| other.0 - self.0)
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime(self.0.checked_add(*other)?))
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime(self.0.checked_sub(*other)?))
    }
}
