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
///
/// `temen-link` is served natively at both of its paths ([`native_link`]), so `commands` need not
/// carry it.
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
    for path in [
        format!("{cwd}/bin/temen-link"),
        "/bin/temen-link".to_string(),
    ] {
        posix.register_host_command(&path, std::sync::Arc::new(native_link));
    }
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

/// The **library pack** of a build run in `at`: every file it wrote under `<at>/nimcache/` for a
/// library module, in the order it wrote them, and nimony's memo of the options the cache was built
/// with. A library module is one whose `.p.nif` records a source under `lib/`; its files include
/// what a build of it as a program wrote (`<stem>.temen/…`), such as the helper compile-time
/// evaluation runs. Seeded into a later build in the same directory, after the library's sources
/// and in this order, they are newer than everything they were made from, so that build compiles
/// only its own modules (#958).
pub fn library_pack(posix: &temen_posix::Posix, at: &str) -> Vec<(String, Vec<u8>)> {
    let cache = format!("{at}/nimcache/");
    let mut library = std::collections::HashMap::new();
    let mut pack = Vec::new();
    for name in posix.file_names_by_write() {
        let Some(rest) = name.strip_prefix(&cache) else {
            continue;
        };
        let stem = rest.split(['.', '/']).next().unwrap_or(rest).to_string();
        let keep = rest == OPTIONS_MEMO
            || *library.entry(stem).or_insert_with_key(|stem| {
                posix
                    .read_file(&format!("{cache}{stem}.p.nif"))
                    .is_some_and(|nif| nif_source(&nif).is_some_and(|src| src.starts_with("lib/")))
            });
        if let (true, Some(bytes)) = (keep, posix.read_file(&name)) {
            pack.push((name, bytes));
        }
    }
    pack
}

/// nimony's memo of the options its cache was built with (`deps.nim`'s `cachedConfigFile`), compared
/// by content, not by time. A build that finds it missing or different re-runs every step, so a pack
/// without it would be rebuilt from its sources.
const OPTIONS_MEMO: &str = "cachedconfigfile.txt";

/// The source file a `.p.nif` was parsed from: the file of its top `stmts` node's line info
/// (`(stmts@<col>,<line>,<file>`).
fn nif_source(nif: &[u8]) -> Option<&str> {
    const TOP: &[u8] = b"(stmts@";
    let at = nif.windows(TOP.len()).position(|w| w == TOP)? + TOP.len();
    let info = nif[at..]
        .split(|b| b.is_ascii_whitespace() || *b == b')')
        .next()?;
    let file = info.splitn(3, |&b| b == b',').nth(2)?;
    core::str::from_utf8(file).ok()
}

/// The toolchain's `temen-link`, run natively: [`temen_leng::link_command`], the function the in-guest
/// `temen-link` is built from, over the files of the process that exec'd it. Interpreted, the same
/// link of a 7-module program took 30–40 s on this tier; natively it takes a fraction of a second, and
/// the module it writes is the same (#958).
fn native_link(argv: &[String], files: &mut dyn temen_posix::CommandFiles) -> i32 {
    let (names, sigs) = temen_posix::cap_vtable();
    let args: Vec<&str> = argv.iter().skip(1).map(String::as_str).collect();
    let files = core::cell::RefCell::new(files);
    temen_leng::link_command(
        &args,
        (&names, &sigs),
        &mut |path| files.borrow_mut().read(path),
        &mut |path, bytes| files.borrow_mut().write(path, bytes),
    )
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
