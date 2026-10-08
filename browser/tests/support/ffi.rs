//! The cdylib's exports keep their state in process-wide statics: the result stashes, the resident
//! libraries and what the engine keeps for them, the compiler's program cache. The page calls them one
//! at a time, and a test binary's threads must too, so every test that calls them holds [`lock`].

/// Hold the exports for the calling test. A test that panicked holding it leaves nothing the next one
/// needs undone, so a poisoned lock is taken as is.
pub fn lock() -> std::sync::MutexGuard<'static, ()> {
    static EXPORTS: std::sync::Mutex<()> = std::sync::Mutex::new(());
    EXPORTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
