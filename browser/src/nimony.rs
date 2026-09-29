//! **nimony's own driver, in the browser** (#958). `nimony t` builds a program as it does on a host:
//! it parses the program's dependency graph, writes a build plan and runs nifmake over it, which
//! forks and execs every step through `/bin/sh` — nifler2 and nimsem per module, hexer, then
//! temen-link — and nimsem's compile-time evaluation builds and runs programs of its own. Every one
//! of those processes runs on the browser's interpreter tier, over one POSIX personality and its
//! memfs, as the self-hosted lane runs them natively (`scripts/ci/nim-selfhost-lane.sh`).

use temen_interp::bytecode::{CoopEvent, CoopRun, Footprint};
use temen_interp::Trap;
use temen_ir::Module;

use crate::{
    blob_entries, posix_host_build, stash, PosixRun, ERR, EXIT_CODE, LAST_STATUS, OUT,
    STATUS_DECODE_ERR, STATUS_EXIT, STATUS_OK, STATUS_TRAP, STATUS_UNSUPPORTED,
};

/// What a build left: how the driver ended, what it printed, what the run held at its end, and the
/// personality whose memfs holds what it built.
pub struct NimBuild {
    pub status: i32,
    pub exit_code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub footprint: Footprint,
    pub posix: temen_posix::Posix,
}

/// Run `driver` (nimony) with `argv`, in `cwd`, over a memfs holding `files`. `commands` are what its
/// processes can exec, each module at its paths: for nimony's toolchain, `<tree>/bin/<tool>`, where
/// the driver looks for a tool, and `/bin/<tool>`, where the shell's `PATH` walk does. They may also
/// run the programs they build. `None` when the driver is not a program this tier runs.
pub fn nim_build(
    driver: &Module,
    commands: &[(&Module, Vec<&str>)],
    files: &[(&str, &[u8])],
    argv: &[&[u8]],
    cwd: &str,
) -> Option<NimBuild> {
    let env = [("PATH", "/bin")];
    let run = PosixRun {
        argv,
        env: &env,
        stdin: &[],
        commands,
        interactive: false,
        loader: true,
    };
    let (host, posix, init_mem) = posix_host_build(driver, &run)?;
    posix.set_cwd(cwd);
    // In the order given: the memfs stamps write order into `st_mtim`, which the freshness checks
    // of nimony's `deps.nim` and of nifmake read, so a tree's sources go in before anything made
    // from them.
    for (path, bytes) in files {
        posix.write_file(path, bytes);
    }
    let mut run = CoopRun::new_reserved(
        driver,
        0,
        &[],
        u64::MAX,
        host,
        None,
        &init_mem,
        temen_ir::DEFAULT_RESERVED_LOG2,
    )?
    .ok()?;
    let (status, exit_code) = match run.run() {
        CoopEvent::Done(_) => (STATUS_OK, 0),
        CoopEvent::Trapped(Trap::Exit(code)) => (STATUS_EXIT, code),
        CoopEvent::Trapped(_) => (STATUS_TRAP, 0),
        _ => (STATUS_UNSUPPORTED, 0),
    };
    Some(NimBuild {
        status,
        exit_code,
        stdout: posix.stdout(),
        stderr: posix.stderr(),
        footprint: run.footprint(),
        posix,
    })
}

/// The memfs of the most recent [`temen_nim_build`], which [`temen_nim_file`] reads.
static mut LAST_BUILD: Option<temen_posix::Posix> = None;
/// The file the most recent [`temen_nim_file`] read ([`temen_nim_file_ptr`]).
static mut FILE: (*mut u8, usize) = (core::ptr::null_mut(), 0);

/// Run nimony's driver ([`nim_build`]). `[driver)` is nimony's module; `[cmds)` the commands, a
/// registry blob ([`blob_entries`]) whose entry names list a module's paths, one per line; `[files)`
/// the tree it builds in, a blob of `path → bytes`; `[argv)` its arguments, each NUL-terminated;
/// `[cwd)` the directory it runs in. Returns the status ([`crate::temen_status`]); the exit code,
/// stdout and stderr read back as after any run, and [`temen_nim_file`] reads what the build wrote.
///
/// # Safety
/// Each `(ptr, len)` must be a live [`crate::temen_alloc`]ation the host filled, or `(null, 0)`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn temen_nim_build(
    driver_ptr: *const u8,
    driver_len: usize,
    cmds_ptr: *const u8,
    cmds_len: usize,
    files_ptr: *const u8,
    files_len: usize,
    argv_ptr: *const u8,
    argv_len: usize,
    cwd_ptr: *const u8,
    cwd_len: usize,
) -> i32 {
    // SAFETY: the caller's contract.
    let (driver, cmds, files, argv, cwd) = unsafe {
        (
            crate::host_slice(driver_ptr, driver_len),
            crate::host_slice(cmds_ptr, cmds_len),
            crate::host_slice(files_ptr, files_len),
            crate::host_slice(argv_ptr, argv_len),
            crate::host_slice(cwd_ptr, cwd_len),
        )
    };
    let status = (|| {
        let driver = temen_encode::decode_module(driver).map_err(|_| STATUS_DECODE_ERR)?;
        let decoded: Vec<(Vec<&str>, Module)> = blob_entries(cmds)
            .into_iter()
            .map(|(paths, b)| {
                let m = temen_encode::decode_module(b).map_err(|_| STATUS_DECODE_ERR)?;
                Ok((paths.lines().collect(), m))
            })
            .collect::<Result<_, i32>>()?;
        let commands: Vec<(&Module, Vec<&str>)> = decoded
            .iter()
            .map(|(paths, m)| (m, paths.clone()))
            .collect();
        let files = blob_entries(files);
        // Each argument ends at its NUL; what follows the last one is not an argument.
        let mut argv: Vec<&[u8]> = argv.split(|&b| b == 0).collect();
        argv.pop();
        let cwd = core::str::from_utf8(cwd).map_err(|_| STATUS_DECODE_ERR)?;
        let b = nim_build(&driver, &commands, &files, &argv, cwd).ok_or(STATUS_UNSUPPORTED)?;
        // SAFETY: single-threaded wasm; the slots are read back only through the accessors.
        unsafe {
            stash(&mut *core::ptr::addr_of_mut!(OUT), b.stdout);
            stash(&mut *core::ptr::addr_of_mut!(ERR), b.stderr);
            EXIT_CODE = b.exit_code;
            *core::ptr::addr_of_mut!(LAST_BUILD) = Some(b.posix);
        }
        Ok(b.status)
    })()
    .unwrap_or_else(|s| s);
    // SAFETY: as above.
    unsafe { LAST_STATUS = status };
    status
}

/// Read `[path)` from the memfs of the most recent [`temen_nim_build`]: its length, or `-1` when
/// there is no such file. The bytes are at [`temen_nim_file_ptr`] until the next read.
///
/// # Safety
/// `(path_ptr, path_len)` must be a live [`crate::temen_alloc`]ation the host filled.
#[no_mangle]
pub unsafe extern "C" fn temen_nim_file(path_ptr: *const u8, path_len: usize) -> i64 {
    // SAFETY: the caller's contract.
    let path = unsafe { crate::host_slice(path_ptr, path_len) };
    // SAFETY: single-threaded wasm; the build's memfs and the slot are touched only here.
    let build = unsafe { (*core::ptr::addr_of!(LAST_BUILD)).as_ref() };
    let bytes = core::str::from_utf8(path)
        .ok()
        .zip(build)
        .and_then(|(path, posix)| posix.read_file(path));
    let len = bytes.as_ref().map_or(-1, |b| b.len() as i64);
    // SAFETY: as above.
    unsafe {
        stash(
            &mut *core::ptr::addr_of_mut!(FILE),
            bytes.unwrap_or_default(),
        )
    };
    len
}

/// Pointer to the bytes the most recent [`temen_nim_file`] read.
#[no_mangle]
pub extern "C" fn temen_nim_file_ptr() -> *const u8 {
    // SAFETY: single-threaded wasm; a plain read of the slot.
    unsafe { (*core::ptr::addr_of!(FILE)).0 }
}
