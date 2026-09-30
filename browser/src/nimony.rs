//! **nimony's own driver, in the browser** (#958). `nimony t` builds a program as it does on a host:
//! it parses the program's dependency graph, writes a build plan and runs nifmake over it, which
//! forks and execs every step through `/bin/sh` — nifler2 and nimsem per module, hexer, then
//! temen-link — and nimsem's compile-time evaluation builds and runs programs of its own. The process
//! tree runs on the browser's interpreter tier, over one POSIX personality and its memfs, as the
//! self-hosted lane runs it natively (`scripts/ci/nim-selfhost-lane.sh`); a **leaf** process — one
//! that cannot park, such as hexer and nifler2, or one that parks only on its pipes where the host can
//! suspend its emitted frames — runs whole on the emitted tier (#1896).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use temen_interp::bytecode::{CoopEvent, CoopRun, Footprint, LeafEmitter, LeafOffer, TierUpConfig};
use temen_interp::Trap;
use temen_ir::Module;

use crate::{
    blob_entries, posix_host_build, stash, PosixRun, LAST_STATUS, STATUS_DECODE_ERR, STATUS_EXIT,
    STATUS_OK, STATUS_TRAP, STATUS_UNSUPPORTED,
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
    /// How many processes ran whole as leaves, tiered up at their entry (#1896).
    pub leaves: usize,
    /// How many calls parked in a leaf and were resumed.
    pub resumes: usize,
}

/// Open `driver` (nimony) with `argv`, in `cwd`, over a memfs holding `files`. `commands` are what
/// its processes can exec, each module at its paths: for nimony's toolchain, `<tree>/bin/<tool>`,
/// where the driver looks for a tool, and `/bin/<tool>`, where the shell's `PATH` walk does. They may
/// also run the programs they build. The run pauses to tier up each leaf process `leaf` emits
/// ([`CoopEvent::TierUp`]); without `leaf` every process interprets. `None` when the driver is not a
/// program this tier runs.
///
/// `temen-link` is served natively at both of its paths ([`native_link`]), so `commands` need not
/// carry it.
pub fn nim_open(
    driver: &Module,
    commands: &[(&Module, Vec<&str>)],
    files: &[(&str, &[u8])],
    argv: &[&[u8]],
    cwd: &str,
    leaf: Option<LeafEmitter>,
) -> Option<(CoopRun, temen_posix::Posix)> {
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
        posix.register_host_command(&path, Arc::new(native_link));
    }
    posix.set_cwd(cwd);
    // In the order given: the memfs stamps write order into `st_mtim`, which the freshness checks
    // of nimony's `deps.nim` and of nifmake read, so a tree's sources go in before anything made
    // from them.
    for (path, bytes) in files {
        posix.write_file(path, bytes);
    }
    // No function of the driver tiers up: it forks and waits, and a tiered-up function cannot park.
    let tierup = leaf.map(|leaf| TierUpConfig {
        eligible: Arc::from([]),
        page_checked: false,
        leaf: Some(leaf),
    });
    let run = CoopRun::new_reserved(
        driver,
        0,
        &[],
        u64::MAX,
        host,
        tierup,
        &init_mem,
        temen_ir::DEFAULT_RESERVED_LOG2,
    )?
    .ok()?;
    Some((run, posix))
}

/// Run nimony's driver to its end ([`nim_open`]). With `leaves`, each leaf process tiers up at its
/// entry, and this serves it by bouncing the entry: the nested interpretation an emitted image's own
/// ops bounce into, so the native stand-in for running it emitted. When that bounce parks, its frames
/// are what the stand-in holds, suspended, until the resume hands it the entry's results.
pub fn nim_build(
    driver: &Module,
    commands: &[(&Module, Vec<&str>)],
    files: &[(&str, &[u8])],
    argv: &[&[u8]],
    cwd: &str,
    leaves: bool,
) -> Option<NimBuild> {
    let leaf: Option<LeafEmitter> = match leaves {
        true => Some(Arc::new(|_: &LeafOffer| true)),
        false => None,
    };
    let (mut run, posix) = nim_open(driver, commands, files, argv, cwd, leaf)?;
    let (mut ran, mut resumes) = (0, 0);
    let end = loop {
        match run.run() {
            CoopEvent::TierUp { func, argv, .. } => {
                ran += 1;
                let mut io = argv.to_vec();
                io.resize(io.len().max(1), 0);
                match run.bounce(func, &mut io, None) {
                    Ok(Some(n)) => run.deliver_tierup(&io[..n]),
                    Ok(None) => {}
                    Err(t) => run.deliver_tierup_trap(t),
                }
            }
            CoopEvent::Resume { results } => {
                resumes += 1;
                run.deliver_tierup(&results);
            }
            end => break end,
        }
    };
    let (status, exit_code) = match end {
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
        leaves: ran,
        resumes,
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

/// #1896 — a nimony build open as a cooperative tier-up session ([`temen_nim_open`]): the
/// personality whose memfs holds the tree, and the leaf images the build emitted.
pub(crate) struct NimSession {
    posix: temen_posix::Posix,
    leaves: Leaves,
}

/// The emitted wasm of each leaf image a build offered, by program index, and whether it carries the
/// page check; `None` for an image the emitter declined (it runs interpreted).
type Leaves = Arc<Mutex<HashMap<u32, Option<(Arc<[u8]>, bool)>>>>;

impl NimSession {
    /// Leaf image `module`'s emitted wasm and whether it is paged.
    pub(crate) fn leaf(&self, module: u32) -> Option<(Arc<[u8]>, bool)> {
        self.leaves.lock().ok()?.get(&module).cloned().flatten()
    }
}

/// The session's leaf emitter: emit an image whole, wasm-driven from its entry — page-checked when
/// the engine says its page state can change — once per program. An image that is not wasm-drivable
/// (it could suspend a frame) runs interpreted, and so does one that can park when the host cannot
/// suspend its frames (`suspends`).
fn leaf_emitter(leaves: Leaves, suspends: bool) -> LeafEmitter {
    Arc::new(move |o: &LeafOffer| {
        if o.parks && !suspends {
            return false;
        }
        let Ok(mut leaves) = leaves.lock() else {
            return false;
        };
        let leaf = leaves.entry(o.module as u32).or_insert_with(|| {
            let shape = temen_wasm_jit::Shape::Batch { entry: o.entry };
            let a = match o.paged {
                true => {
                    let page_log2 = temen_interp::host_page_size().trailing_zeros() as u8;
                    temen_wasm_jit::compile_jit_page_checked(o.image, shape, true, page_log2)
                }
                false => temen_wasm_jit::compile_jit(o.image, shape, true),
            }
            .ok()?;
            let temen_wasm_jit::DriveMode::WasmDriven { .. } = a.drive else {
                return None;
            };
            Some((a.wasm.into(), o.paged))
        });
        leaf.is_some()
    })
}

/// The memfs of the most recent nimony build, which [`temen_nim_file`] reads.
static mut LAST_BUILD: Option<temen_posix::Posix> = None;
/// The file the most recent [`temen_nim_file`] read ([`temen_nim_file_ptr`]).
static mut FILE: (*mut u8, usize) = (core::ptr::null_mut(), 0);

/// A nim session's run ended: keep its memfs for [`temen_nim_file`] and hand back what it printed.
pub(crate) fn finish(nim: NimSession) -> (Vec<u8>, Vec<u8>) {
    let out = (nim.posix.stdout(), nim.posix.stderr());
    // SAFETY: single-threaded wasm; the build's memfs is read only through `temen_nim_file`.
    unsafe { *core::ptr::addr_of_mut!(LAST_BUILD) = Some(nim.posix) };
    out
}

/// Open nimony's driver ([`nim_open`]) as the cooperative tier-up session the `temen_coop_*` exports
/// drive (`driveCoopTierupRun`): the process tree runs on the interpreter, and each leaf process
/// pauses the run as a `COOP_RUN_TIERUP` of its own program ([`crate::temen_coop_module`]) to run whole
/// on the emitted tier (#1896). `[driver)` is nimony's module; `[cmds)` the commands, a registry blob
/// ([`blob_entries`]) whose entry names list a module's paths, one per line; `[files)` the tree it
/// builds in, a blob of `path → bytes`; `[argv)` its arguments, each NUL-terminated; `[cwd)` the
/// directory it runs in. `suspend` is non-zero when the driver can suspend a leaf's emitted frames
/// where a call parks (JSPI): a leaf process that parks only on its pipes then runs emitted too, and
/// a parked call surfaces as [`crate::COOP_RUN_RESUME`] once it returns. Returns `0`, or a negative
/// status. When the run is done the exit code, stdout and stderr read back as after any run, and
/// [`temen_nim_file`] reads what the build wrote.
///
/// # Safety
/// Each `(ptr, len)` must be a live [`crate::temen_alloc`]ation the host filled, or `(null, 0)`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn temen_nim_open(
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
    suspend: i32,
) -> i32 {
    crate::temen_coop_close();
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
        let leaves = Leaves::default();
        let emit = leaf_emitter(Arc::clone(&leaves), suspend != 0);
        let (run, posix) = nim_open(&driver, &commands, &files, &argv, cwd, Some(emit))
            .ok_or(STATUS_UNSUPPORTED)?;
        // SAFETY: single-threaded wasm; the session is read back only via the coop exports.
        unsafe {
            *core::ptr::addr_of_mut!(crate::COOP_RUN) =
                Some(crate::CoopTierupRun::nim(run, NimSession { posix, leaves }));
        }
        Ok(STATUS_OK)
    })()
    .unwrap_or_else(|s| s);
    // SAFETY: as above.
    unsafe { LAST_STATUS = status };
    match status {
        STATUS_OK => 0,
        s => -s,
    }
}

/// Read `[path)` from the memfs of the most recent nimony build: its length, or `-1` when
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
