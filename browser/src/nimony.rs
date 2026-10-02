//! **nimony's own driver, in the browser** (#958). `nimony t` builds a program as it does on a host:
//! it parses the program's dependency graph, writes a build plan and runs nifmake over it, which
//! spawns every step through `/bin/sh` — nifler2 and nimsem per module, hexer, then
//! temen-link — and nimsem's compile-time evaluation builds and runs programs of its own. The process
//! tree runs on the browser's interpreter tier, over one POSIX personality and its memfs, as the
//! self-hosted lane runs it natively (`scripts/ci/nim-selfhost-lane.sh`); a **leaf** process — one
//! that cannot park, such as hexer and nifler2, or one that parks only on its pipes and its children
//! where the host can suspend its emitted frames, such as nimsem — runs whole on the emitted tier
//! (#1896).

use std::sync::Arc;

use temen_interp::bytecode::{CoopEvent, CoopRun, Footprint, LeafEmitter, LeafOffer, TierUpConfig};
use temen_interp::{PreparedModule, Trap};
use temen_ir::Module;

use crate::{
    blob_entries, posix_host_build, prepare, stash, PosixRun, LAST_STATUS, STATUS_DECODE_ERR,
    STATUS_EXIT, STATUS_OK, STATUS_TRAP, STATUS_UNSUPPORTED,
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
/// With `dir`, the personality of an earlier build, this one continues in its directory (#2099):
/// its files move here, and of `files` only those that differ from what it holds are written. A
/// file left as it was keeps its write stamp, so nimony's freshness checks find what that build made
/// from it up to date, as they would in a directory on a host that it builds in again.
///
/// `temen-link` is served natively at both of its paths ([`native_link`]), so `commands` need not
/// carry it.
pub fn nim_open(
    driver: &Module,
    commands: &[(PreparedModule, Vec<&str>)],
    dir: Option<&temen_posix::Posix>,
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
    if let Some(dir) = dir {
        posix.adopt_files(dir);
    }
    for path in [
        format!("{cwd}/bin/temen-link"),
        "/bin/temen-link".to_string(),
    ] {
        posix.register_host_command(&path, Arc::new(native_link));
    }
    posix.set_cwd(cwd);
    // In the order given: the memfs stamps write order into `st_mtim`, which the freshness checks
    // of nimony's `deps.nim` and of nifmake read, so a tree's sources go in before anything made
    // from them. Over `dir`, a file it already holds keeps its stamp.
    for (path, bytes) in files {
        posix.write_file_if_changed(path, bytes);
    }
    // No function of the driver tiers up: it spawns and waits, and a tiered-up function cannot park.
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
    dir: Option<&temen_posix::Posix>,
    files: &[(&str, &[u8])],
    argv: &[&[u8]],
    cwd: &str,
    leaves: bool,
) -> Option<NimBuild> {
    let leaf: Option<LeafEmitter> = match leaves {
        true => Some(Arc::new(|_: &LeafOffer| true)),
        false => None,
    };
    let commands: Vec<(PreparedModule, Vec<&str>)> = commands
        .iter()
        .map(|(m, paths)| (prepare(m), paths.clone()))
        .collect();
    let (mut run, posix) = nim_open(driver, &commands, dir, files, argv, cwd, leaf)?;
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
/// with. A library module is one whose `.p.nif` records a source in nimony's tree: under `lib/`,
/// or under `src/`, which the library imports (#2033). Its files include what a build of it as a
/// program wrote (`<stem>.temen/…`), such as the helper compile-time evaluation runs. So do the
/// plugins the library declares (#2049): their executables, which nimony links into the cache's
/// root (`deps.nim`'s `pluginExe`). Seeded into a later build in the same directory, after the
/// library's sources and in this order, they are newer than everything they were made from, so that
/// build compiles only its own modules (#958).
pub fn library_pack(posix: &temen_posix::Posix, at: &str) -> Vec<(String, Vec<u8>)> {
    let cache = format!("{at}/nimcache/");
    let mut library = std::collections::HashMap::new();
    let mut pack = Vec::new();
    for name in posix.file_names_by_write() {
        let Some(rest) = name.strip_prefix(&cache) else {
            continue;
        };
        let Some(bytes) = posix.read_file(&name) else {
            continue;
        };
        let stem = rest.split(['.', '/']).next().unwrap_or(rest).to_string();
        let keep = rest == OPTIONS_MEMO
            || (!rest.contains('/') && temen_encode::wire::is_module_blob(&bytes))
            || *library.entry(stem).or_insert_with_key(|stem| {
                posix
                    .read_file(&format!("{cache}{stem}.p.nif"))
                    .is_some_and(|nif| {
                        nif_source(&nif).is_some_and(|src| {
                            src.starts_with("lib/") || src.starts_with("src/")
                        })
                    })
            });
        if keep {
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
/// personality whose memfs holds the tree. The leaf images the build emitted are the session's
/// ([`crate::Leaves`]).
pub(crate) struct NimSession {
    posix: temen_posix::Posix,
}

// ---- nimony's module-stem hash (gear2/modnames.nim + lib/tinyhashes.nim), reproduced exactly -------

fn uhash(s: &str) -> u32 {
    let mut h: u32 = 0;
    for c in s.bytes() {
        h = h.wrapping_add(c as u32);
        h = h.wrapping_add(h << 10);
        h ^= h >> 6;
    }
    h = h.wrapping_add(h << 3);
    h ^= h >> 11;
    h = h.wrapping_add(h << 15);
    h
}

fn base36(mut id: u32) -> String {
    const B36: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut r = String::new();
    while id > 0 {
        r.push(B36[(id % 36) as usize] as char);
        id /= 36;
    }
    r
}

fn relative_path(path: &str, base: &str) -> String {
    let p: Vec<&str> = path
        .trim_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();
    let b: Vec<&str> = base
        .trim_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();
    let mut i = 0;
    while i < p.len() && i < b.len() && p[i] == b[i] {
        i += 1;
    }
    let mut out: Vec<&str> = vec![".."; b.len() - i];
    out.extend_from_slice(&p[i..]);
    out.join("/")
}

/// `gear2/modnames.moduleSuffix` — `name[0..3]` + base36(`uhash`) of the shortest of the file's path
/// relative to the cwd (`/`) and to each search path (`/lib`).
fn module_suffix(file: &str) -> String {
    let mut rel = relative_path(file, "/");
    let c = relative_path(file, "/lib");
    if c.len() < rel.len() {
        rel = c;
    }
    let name = rel.rsplit('/').next().unwrap_or(&rel);
    let name = name.strip_suffix(".nim").unwrap_or(name);
    let mut stem: String = name.chars().take(3).collect();
    stem.push_str(&base36(uhash(&rel)));
    stem
}

/// nimony's module-stem hash for `[path)`, a path relative to the build's directory ([`module_suffix`]),
/// onto the stdout slot; returns its length. The card finds the module a build of `prog.nim` linked by
/// it: `nimcache/<stem>.temen/prog.temen`.
///
/// # Safety
/// `(path_ptr, path_len)` must be a live `temen_alloc`ation the host just filled.
#[no_mangle]
pub unsafe extern "C" fn temen_nim_module_suffix(path_ptr: *const u8, path_len: usize) -> usize {
    let path = String::from_utf8_lossy(unsafe { core::slice::from_raw_parts(path_ptr, path_len) })
        .into_owned();
    let bytes = module_suffix(&path).into_bytes();
    let len = bytes.len();
    unsafe { stash(&mut *core::ptr::addr_of_mut!(crate::OUT), bytes) };
    len
}

/// #2087 — a build's toolchain, decoded and prepared: nimony, and its commands with their paths.
struct Toolchain {
    driver_bytes: Vec<u8>,
    cmds_bytes: Vec<u8>,
    driver: Arc<Module>,
    commands: Vec<(Vec<String>, PreparedModule)>,
    /// #2099 — the directory its builds share: the personality of the last one to end, whose memfs
    /// [`temen_nim_file`] reads and the next build continues in. A new toolchain starts from the
    /// files it is given.
    dir: Option<temen_posix::Posix>,
}

/// The toolchain the last build opened ([`toolchain`]).
static mut TOOLCHAIN: Option<Toolchain> = None;

/// The toolchain of `driver` and `cmds` (a [`temen_nim_open`]'s), kept in [`TOOLCHAIN`].
#[allow(static_mut_refs)]
fn toolchain(driver: &[u8], cmds: &[u8]) -> Result<&'static mut Toolchain, i32> {
    // SAFETY: single-threaded wasm; the cache is touched only here, and a session holds what it
    // granted by `Arc`, never by borrow.
    toolchain_in(
        unsafe { &mut *core::ptr::addr_of_mut!(TOOLCHAIN) },
        driver,
        cmds,
    )
}

/// The toolchain of `driver` and `cmds`: `slot`'s when its bytes are these, else decoded and prepared
/// now and kept in `slot` for the next build. Every build of the nim card passes the same ones, so
/// only its first decodes and prepares them (#2087).
fn toolchain_in<'s>(
    slot: &'s mut Option<Toolchain>,
    driver: &[u8],
    cmds: &[u8],
) -> Result<&'s mut Toolchain, i32> {
    if !slot
        .as_ref()
        .is_some_and(|t| t.driver_bytes == driver && t.cmds_bytes == cmds)
    {
        let decode = |b| temen_encode::decode_module(b).map_err(|_| STATUS_DECODE_ERR);
        let entries = blob_entries(cmds);
        let commands: Vec<(Vec<String>, PreparedModule)> = entries
            .iter()
            .map(|(paths, b)| {
                let m = prepare(&decode(b)?);
                Ok((paths.lines().map(String::from).collect(), m))
            })
            .collect::<Result<_, i32>>()?;
        // nimony is one of its own commands: a driver that is one runs that command's module.
        let module = match entries.iter().position(|(_, b)| *b == driver) {
            Some(i) => Arc::clone(commands[i].1.module()),
            None => Arc::new(decode(driver)?),
        };
        *slot = Some(Toolchain {
            driver_bytes: driver.to_vec(),
            cmds_bytes: cmds.to_vec(),
            driver: module,
            commands,
            dir: None,
        });
    }
    slot.as_mut().ok_or(STATUS_DECODE_ERR)
}
/// The file the most recent [`temen_nim_file`] read ([`temen_nim_file_ptr`]).
static mut FILE: (*mut u8, usize) = (core::ptr::null_mut(), 0);

/// A nim session's run ended: keep its memfs as its toolchain's directory, for [`temen_nim_file`]
/// and the next build, and hand back what it printed.
pub(crate) fn finish(nim: NimSession) -> (Vec<u8>, Vec<u8>) {
    let out = (nim.posix.stdout(), nim.posix.stderr());
    // SAFETY: single-threaded wasm; the toolchain is touched only by the nim exports.
    if let Some(tc) = unsafe { (*core::ptr::addr_of_mut!(TOOLCHAIN)).as_mut() } {
        tc.dir = Some(nim.posix);
    }
    out
}

/// Open nimony's driver ([`nim_open`]) as the cooperative tier-up session the `temen_coop_*` exports
/// drive (`driveCoopTierupRun`): the process tree runs on the interpreter, and each leaf process
/// pauses the run as a `COOP_RUN_TIERUP` of its own program ([`crate::temen_coop_module`]) to run whole
/// on the emitted tier (#1896). `[driver)` is nimony's module; `[cmds)` the commands, a registry blob
/// ([`blob_entries`]) whose entry names list a module's paths, one per line; `[files)` the tree it
/// builds in, a blob of `path → bytes`, written over the directory the toolchain's last build left
/// ([`nim_open`]); `[argv)` its arguments, each NUL-terminated; `[cwd)` the
/// directory it runs in. `suspend` is non-zero when the driver can suspend a leaf's emitted frames
/// where a call parks (JSPI): a leaf process that parks only on its pipes and its children then runs
/// emitted too, and a parked call surfaces as [`crate::COOP_RUN_RESUME`] once it returns. Returns
/// `0`, or a negative status. When the run is done the exit code, stdout and stderr read back as
/// after any run, and [`temen_nim_file`] reads what the build wrote.
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
        let tc = toolchain(driver, cmds)?;
        let dir = tc.dir.take();
        let commands: Vec<(PreparedModule, Vec<&str>)> = tc
            .commands
            .iter()
            .map(|(paths, m)| (m.clone(), paths.iter().map(String::as_str).collect()))
            .collect();
        let files = blob_entries(files);
        // Each argument ends at its NUL; what follows the last one is not an argument.
        let mut argv: Vec<&[u8]> = argv.split(|&b| b == 0).collect();
        argv.pop();
        let cwd = core::str::from_utf8(cwd).map_err(|_| STATUS_DECODE_ERR)?;
        let leaves = crate::Leaves::default();
        // A nim build runs on the threads cdylib, over shared memory.
        let emit = crate::leaf_emitter(Arc::clone(&leaves), suspend != 0, true);
        let (run, posix) = nim_open(
            &tc.driver,
            &commands,
            dir.as_ref(),
            &files,
            &argv,
            cwd,
            Some(emit),
        )
        .ok_or(STATUS_UNSUPPORTED)?;
        // SAFETY: single-threaded wasm; the session is read back only via the coop exports.
        unsafe {
            *core::ptr::addr_of_mut!(crate::COOP_RUN) =
                Some(crate::CoopTierupRun::nim(run, NimSession { posix }, leaves));
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
    // SAFETY: single-threaded wasm; the build's memfs and the slot are touched only by the nim
    // exports.
    let build = unsafe { (*core::ptr::addr_of!(TOOLCHAIN)).as_ref() }.and_then(|t| t.dir.as_ref());
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

/// Remove `[path)` from the directory the toolchain's next build continues in ([`Toolchain::dir`]):
/// `1` when it was there. A host that tells whether a build linked its program by the program's file
/// removes it first, since the build otherwise finds the last build's in its place (#2099).
///
/// # Safety
/// `(path_ptr, path_len)` must be a live [`crate::temen_alloc`]ation the host filled.
#[no_mangle]
pub unsafe extern "C" fn temen_nim_remove(path_ptr: *const u8, path_len: usize) -> i32 {
    // SAFETY: the caller's contract.
    let path = unsafe { crate::host_slice(path_ptr, path_len) };
    // SAFETY: single-threaded wasm; the build's memfs is touched only by the nim exports.
    let dir = unsafe { (*core::ptr::addr_of!(TOOLCHAIN)).as_ref() }.and_then(|t| t.dir.as_ref());
    core::str::from_utf8(path)
        .ok()
        .zip(dir)
        .is_some_and(|(path, posix)| posix.remove_file(path)) as i32
}

/// Pointer to the bytes the most recent [`temen_nim_file`] read.
#[no_mangle]
pub extern "C" fn temen_nim_file_ptr() -> *const u8 {
    // SAFETY: single-threaded wasm; a plain read of the slot.
    unsafe { (*core::ptr::addr_of!(FILE)).0 }
}

#[cfg(test)]
mod toolchain_tests {
    use super::*;

    fn encoded(value: i64) -> Vec<u8> {
        let text = format!(
            "func () -> (i64) {{\nblock 0 () {{\n  vr = i64.const {value}\n  return vr\n  }}\n}}\n"
        );
        temen_encode::encode_module(&temen_text::parse_module(&text).expect("parse"))
    }

    /// #2087 — a build whose toolchain bytes are the last build's keeps its decoded, prepared
    /// commands; other bytes are decoded and prepared afresh, and bytes that do not decode fail.
    #[test]
    fn a_build_with_the_last_builds_toolchain_keeps_it() {
        let (one, two) = (encoded(1), encoded(2));
        let cmds = crate::registry_blob(&[("bin/one\n/bin/one", one.as_slice())]);
        let other = crate::registry_blob(&[("bin/one\n/bin/one", two.as_slice())]);
        let mut slot = None;
        let first = toolchain_in(&mut slot, &one, &cmds).expect("decodes");
        assert_eq!(first.commands[0].0, ["bin/one", "/bin/one"]);
        let kept = Arc::clone(first.commands[0].1.module());
        let again = toolchain_in(&mut slot, &one, &cmds).expect("decodes");
        assert!(
            Arc::ptr_eq(again.commands[0].1.module(), &kept),
            "the same bytes keep it"
        );
        assert!(
            Arc::ptr_eq(&again.driver, &kept),
            "a driver that is a command runs its module"
        );
        let changed = toolchain_in(&mut slot, &one, &other).expect("decodes");
        assert!(
            !Arc::ptr_eq(changed.commands[0].1.module(), &kept),
            "other bytes decode afresh"
        );
        assert_eq!(
            *changed.driver, *kept,
            "a driver that is no command decodes alone"
        );
        assert_eq!(
            toolchain_in(&mut slot, b"not a module", &cmds).err(),
            Some(STATUS_DECODE_ERR)
        );
    }
}
