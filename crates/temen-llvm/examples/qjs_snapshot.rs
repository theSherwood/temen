//! **Warm-runtime-snapshot prototype for QuickJS** (WASM_AOT.md follow-on / the "do-nothing program
//! takes >1s" finding). The playground rebuilds the whole QuickJS runtime on every Run; a trivial
//! program's wall-clock is ~all that fixed init, not the program. This measures the alternative:
//! init once, snapshot the guest's whole memory, then **restore the snapshot per Run and only eval** —
//! fresh-per-Run isolation (each Run restores the identical warm state) at a fraction of the cost.
//!
//! Uses the two-phase driver `crates/temen-run/demos/quickjs/qjs_snapshot.c`:
//!   - `main`     — the original cold path (read → init → eval → print), the BASELINE.
//!   - `warmup`   — init runtime+context+bindings into statics; the host snapshots memory after.
//!   - `eval_run` — read stdin, eval over the warm (restored) context, print.
//!
//! **Memory model.** These three are ordinary exports (`params = [i64 sp]`), not the synthesized
//! `_start` (func 0). So the harness must reproduce what `_start` does for the on-ramp: (1) grant the
//! §3e powerbox and bind the module's manifest imports (`grant_powerbox_prefix` + `default_cap_resolver`
//! in temen-run), (2) seed the guest heap bump-pointer words (`POWERBOX_HEAP_BRK`/`_TOP`) to the window's
//! mapped boundary, and (3) pass `sp = powerbox_entry_sp(..)` as the entry arg. The on-ramp malloc grows
//! the heap **above** the mapped window (`heap_base = 1 << size_log2`), so the shared `back` region must
//! be sized to cover window + heap headroom (the browser does the same — a large `winSize`); the whole
//! `back` is what we snapshot and restore.
//!
//! Build the module first (fetched-not-vendored, like the qjs demo):
//!   see the top of this repo's `browser/build-onramp-assets.mjs` buildQuickJS(), but link
//!   `qjs_snapshot.c` instead of `qjs_repl.c` and translate to `/tmp/qjs_snapshot.temen`.
//!
//! Usage: `cargo run --release --example qjs_snapshot -- /tmp/qjs_snapshot.temen`

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::sync::Arc;
use std::time::Instant;

use temen_interp::bytecode::SharedProgram;
use temen_interp::{cap_id, BoundImport, Host, Region, StreamRole, Trap, Value};
use temen_ir::{Module, POWERBOX_HEAP_BRK, POWERBOX_HEAP_TOP};

const FUEL: u64 = 100_000_000_000;
/// The `Jit` install-table log2 (`temen-run`'s `CLI_JIT_TABLE_LOG2`); inert here (QuickJS never JITs).
const JIT_TABLE_LOG2: u8 = 10;
/// The **mapped**-window log2 we run the on-ramp guest under, overriding the module's declared size. The
/// on-ramp heap grows above the declared window (`heap_base = 1 << declared_log2`); by mapping a much
/// larger window we keep the whole heap **inside the mapped region**, so (a) no `vm_map` growth is needed
/// (its mapped-width state lives in the `Mem`, *not* the window bytes, and so would not survive a byte
/// snapshot), and (b) the whole guest image is contiguous in `back` for a plain memcpy snapshot/restore.
/// 2^26 = 64 MiB: a few MiB of globals/stack + generous heap headroom for QuickJS. The browser's large
/// `winSize` plays the same role.
const MAPPED_LOG2: u8 = 26;

/// The fixed per-module facts the run helpers need: the compiled program, the module (for imports), the
/// window sizing, and the entry data-stack pointer.
struct Ctx<'a> {
    prog: &'a SharedProgram,
    module: &'a Module,
    /// Bytes of the mapped window we run under, `1 << MAPPED_LOG2` (also the shared `back` size).
    win: u64,
    /// The on-ramp heap base — the declared window top (`1 << declared_size_log2`), where the guest's
    /// bump allocator starts; seeded into `POWERBOX_HEAP_BRK`/`_TOP`. Heap grows from here into the
    /// (larger) mapped window.
    heap_base: u64,
    /// The powerbox data-stack base (`powerbox_entry_sp`), passed as every entry's `sp` arg.
    entry_sp: u64,
}

/// An 8-aligned zeroed buffer of `size` bytes + a `Region::shared` over it. Caller frees via the layout.
fn window(size: usize) -> (Arc<Region>, *mut u8, Layout) {
    let layout = Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero size; `size` valid 8-aligned bytes owned here, used only as this window.
    let base = unsafe { alloc_zeroed(layout) };
    assert!(!base.is_null(), "window alloc failed");
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this window's, freed after use.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });
    (back, base, layout)
}

/// Seed the on-ramp guest-heap bump-pointer words the synthesized `_start` writes with `seed_heap`: both
/// `POWERBOX_HEAP_BRK` and `_TOP` start at the window's mapped boundary (`ctx.win`), from which malloc
/// grows the heap up (via `vm_map`) into the reserved tail — here, into `back` above the window. Only for
/// a **fresh** window (cold / warmup); a restored snapshot already carries the warm allocator state.
fn seed_heap(base: *mut u8, ctx: &Ctx) {
    // #964: a `__null_guard`-marked module's heap words sit one guard up.
    let scratch = temen_ir::module_null_guard(ctx.module).unwrap_or(0);
    for off in [POWERBOX_HEAP_BRK, POWERBOX_HEAP_TOP] {
        // SAFETY: `scratch + off + 8 <= heap_base <= win`; `base` owns `win` bytes.
        unsafe {
            let p = base.add((scratch + off) as usize) as *mut i64;
            p.write_unaligned(ctx.heap_base as i64);
        }
    }
}

/// Grant the fixed §3e powerbox on a fresh host and bind `module`'s manifest import slots to it — the
/// same setup `temen-run`'s instantiation does for an on-ramp guest (`grant_powerbox_prefix` + the
/// `default_cap_resolver` name→handle map). Without this a `warmup`/`eval_run` that reaches libc
/// (`write`/`read`/`exit`, or `malloc`'s `vm_map`) traps `CapFault`: those are manifest imports the host
/// must bind, not `_start` args. Grants are deterministic, so a fresh host per Run sees identical
/// handles — the snapshot's window-relative state stays valid across restores. `win` is the mapped
/// window size (the whole-window `AddressSpace` grant).
fn powerbox_host(module: &Module, win: u64, stdin: &[u8]) -> Host {
    let mut h = Host::new();
    h.stdin = stdin.to_vec();
    // Grant order is canonical (stdout, stdin, exit, memory, addrspace, jit, stderr) so handle values
    // match temen-run's; QuickJS only reaches write/read/exit + malloc's vm_map, but grant the full set.
    let stdout = h.grant_stream(StreamRole::Out);
    let stdin_h = h.grant_stream(StreamRole::In);
    let exit = h.grant_exit();
    let memory = h.grant_memory();
    let mem_log2 = (win != 0).then(|| win.trailing_zeros() as u8);
    let addrspace = h.grant_address_space(0, win);
    let jit = h.grant_jit_with_table(mem_log2, JIT_TABLE_LOG2);
    let stderr = h.grant_stream(StreamRole::Err);
    // Bind each import slot by name (mirrors `default_cap_resolver` + the `None`-arm handle picking).
    let bindings: Vec<BoundImport> = module
        .imports
        .iter()
        .map(|im| match im.name.as_str() {
            "write" => BoundImport::required(cap_id::STREAM, 1, stdout),
            "read" => BoundImport::required(cap_id::STREAM, 0, stdin_h),
            "stderr" => BoundImport::required(cap_id::STREAM, 1, stderr),
            "exit" => BoundImport::required(cap_id::EXIT, 0, exit),
            "vm_map" => BoundImport::required(cap_id::ADDRESS_SPACE, 0, memory),
            "vm_unmap" => BoundImport::required(cap_id::ADDRESS_SPACE, 1, memory),
            "vm_protect" => BoundImport::required(cap_id::ADDRESS_SPACE, 2, memory),
            "vm_page_size" => BoundImport::required(cap_id::ADDRESS_SPACE, 3, memory),
            "vm_region_create" => BoundImport::required(cap_id::ADDRESS_SPACE, 5, addrspace),
            "vm_jit_compile" => BoundImport::required(cap_id::JIT, 0, jit),
            "vm_jit_invoke2" => BoundImport::required(cap_id::JIT, 1, jit),
            "vm_jit_release" => BoundImport::required(cap_id::JIT, 2, jit),
            "vm_jit_install" => BoundImport::required(cap_id::JIT, 3, jit),
            "vm_jit_uninstall" => BoundImport::required(cap_id::JIT, 4, jit),
            "vm_jit_compile_linked" => BoundImport::required(cap_id::JIT, 5, jit),
            // Unknown name: declared but unbound — a dispatch through it is a fail-closed CapFault.
            _ => BoundImport::rebindable(0, 0, None),
        })
        .collect();
    if !module.imports.is_empty() {
        h.set_import_bindings(bindings);
    }
    h
}

/// Run `func` over `back` (a fresh seeded window, or a restored snapshot), returning captured stdout.
/// `func` is called with `sp = ctx.entry_sp` (the on-ramp entry ABI). `seed_data` lays the module's
/// data segments (pass `true` for a fresh window). `Exit`/normal return both count as success (the guest
/// returns 0); a real trap panics (the prototype wants to know).
fn run_capture(ctx: &Ctx, func: u32, back: Arc<Region>, stdin: &[u8], seed_data: bool) -> Vec<u8> {
    let mut host = powerbox_host(ctx.module, ctx.win, stdin);
    let mut fuel = FUEL;
    let args = [Value::I64(ctx.entry_sp as i64)];
    match ctx
        .prog
        .run_over(func, &args, &mut fuel, back, &mut host, seed_data)
    {
        Ok(_) | Err(Trap::Exit(_)) => {}
        Err(t) => panic!("func {func} trapped: {t:?}"),
    }
    host.stdout
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: qjs_snapshot <temen>");
    let bytes = std::fs::read(&path).expect("read temen");
    let mut m = temen_encode::decode_module(&bytes).expect("decode module");
    let declared_log2 = m.memory.expect("module declares memory").size_log2;
    let heap_base = 1u64 << declared_log2;
    // Run under a larger *mapped* window so the on-ramp heap (which starts at `heap_base` and grows up)
    // stays inside the mapped region — no `vm_map` growth, a contiguous image to snapshot. See MAPPED_LOG2.
    assert!(
        MAPPED_LOG2 > declared_log2,
        "mapped window must exceed the declared window"
    );
    m.memory = Some(temen_ir::Memory {
        size_log2: MAPPED_LOG2,
    });
    let win = 1u64 << MAPPED_LOG2;
    let back_size = win as usize;
    println!(
        "module: {} funcs, declared window {} MiB (2^{}), mapped/back {} MiB (2^{}), heap base {} MiB",
        m.funcs.len(),
        heap_base >> 20,
        declared_log2,
        back_size >> 20,
        MAPPED_LOG2,
        heap_base >> 20,
    );

    // Resolve the three entries by exported name; `main` is the on-ramp `_start` (func 0).
    let find = |needle: &str| -> Option<u32> {
        m.exports
            .iter()
            .find(|e| e.name == needle || e.name.ends_with(&format!("::{needle}")))
            .map(|e| e.func)
    };
    let warmup = find("warmup").expect("no `warmup` export");
    let eval_run = find("eval_run").expect("no `eval_run` export");
    let cold = find("main").expect("no `main` export");
    let entry_sp = temen_ir::powerbox_entry_sp(&m);
    println!("entries: main=f{cold}  warmup=f{warmup}  eval_run=f{eval_run}  sp=0x{entry_sp:x}\n");

    let prog = SharedProgram::compile(&m).expect("module in bytecode subset");
    let ctx = Ctx {
        prog: &prog,
        module: &m,
        win,
        heap_base,
        entry_sp,
    };

    // ---- Snapshot the warm runtime once: run `warmup` over a fresh, seeded window, then keep the
    // **live prefix** — `[0, brk)`, where `brk` (the guest's bump-allocator high-water, at
    // POWERBOX_HEAP_BRK) is how far the heap has grown. Everything above `brk` is still zero, so a
    // fresh `alloc_zeroed` restore target needs only this prefix copied in. This is the
    // program-independent post-init state. ----
    let t = Instant::now();
    let (warm_back, warm_base, warm_layout) = window(back_size);
    seed_heap(warm_base, &ctx);
    let _ = run_capture(&ctx, warmup, warm_back, b"", true);
    let warmup_ms = t.elapsed().as_secs_f64() * 1e3;
    // SAFETY: warmup finished; read the post-warmup bump pointer, then the live prefix, before free.
    let live = unsafe {
        // #964: the brk word rides the (possibly guard-shifted) low scratch.
        let brk_off = temen_ir::module_null_guard(ctx.module).unwrap_or(0) + POWERBOX_HEAP_BRK;
        let brk = (warm_base.add(brk_off as usize) as *const i64).read_unaligned() as usize;
        // `brk` is the heap high-water; clamp defensively to the buffer.
        brk.min(back_size)
    };
    // SAFETY: `warm_base` owns `back_size >= live` bytes, live here before free.
    let snapshot: Vec<u8> = unsafe { std::slice::from_raw_parts(warm_base, live).to_vec() };
    // SAFETY: the Region (and thus its alias) is dropped after run_capture returned; free the buffer.
    unsafe { dealloc(warm_base, warm_layout) };
    println!(
        "warmup (once): {warmup_ms:.0} ms — live snapshot {:.1} MiB of the {} MiB window\n",
        live as f64 / (1 << 20) as f64,
        back_size >> 20
    );

    let programs: &[(&str, &str)] = &[
        ("trivial", "1;\n"),
        (
            "user's program",
            "function fib(n){return n<2?n:fib(n-1)+fib(n-2);}\n\
             console.log(\"fib\", Array.from({length:11},(_,i)=>fib(i)).join(\" \"));\n\
             const xs=[5,3,8,1,9,2,7];\n\
             console.log(\"sorted\", xs.slice().sort((a,b)=>a-b).join(\",\"));\n\
             console.log(\"json\", JSON.stringify({ok:true,nums:[1,2,3],nested:{pi:Math.PI}}));\n\
             \"0.1 + 0.2 = \" + (0.1 + 0.2);\n",
        ),
        ("100k loop", "let s=0;for(let i=0;i<100000;i++)s+=i;s;\n"),
    ];

    // `warm ms` = restore (memcpy the live snapshot in) + eval; the restore is a real per-Run cost, so
    // it's counted. `restore` breaks that out. Parity asserts cold and warm produce identical output.
    println!(
        "{:<16}{:>10}{:>10}{:>10}{:>9}  parity",
        "program", "cold ms", "warm ms", "restore", "speedup"
    );
    for (name, js) in programs {
        let stdin = js.as_bytes();

        // COLD: the original full path (read → init → eval → print) over a fresh seeded window.
        let (cb, cbase, cl) = window(back_size);
        seed_heap(cbase, &ctx);
        let t = Instant::now();
        let cold_out = run_capture(&ctx, cold, cb, stdin, true);
        let cold_ms = t.elapsed().as_secs_f64() * 1e3;
        // SAFETY: run_capture returned, so the Region (and its Mem alias) is dropped; free the buffer.
        unsafe { dealloc(cbase, cl) };

        // WARM: restore the live snapshot into a fresh (zeroed) window — no re-seed, the snapshot carries
        // the warm allocator state and heap — then eval only. Time the restore + eval together.
        let (wb, wbase, wl) = window(back_size);
        let t = Instant::now();
        // SAFETY: fresh buffer of `back_size >= live`; write the live prefix, rest stays zero.
        unsafe { std::slice::from_raw_parts_mut(wbase, snapshot.len()).copy_from_slice(&snapshot) };
        let restore_ms = t.elapsed().as_secs_f64() * 1e3;
        let warm_out = run_capture(&ctx, eval_run, wb, stdin, false);
        let warm_ms = t.elapsed().as_secs_f64() * 1e3;
        // SAFETY: both runs returned (Regions dropped); free the warm buffer.
        unsafe { dealloc(wbase, wl) };

        let parity = cold_out == warm_out;
        println!(
            "{:<16}{:>10.0}{:>10.0}{:>10.1}{:>8.1}x  {}",
            name,
            cold_ms,
            warm_ms,
            restore_ms,
            cold_ms / warm_ms,
            if parity { "OK" } else { "MISMATCH!" }
        );
        if !parity || std::env::var("QJS_SHOW").is_ok() {
            println!("  cold: {:?}", String::from_utf8_lossy(&cold_out));
            println!("  warm: {:?}", String::from_utf8_lossy(&warm_out));
        }
    }
}
