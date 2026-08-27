//! Scratch probe for the bash bring-up (#802): translate the linked module, run it on the
//! interpreter with the temen-posix personality granted as the named "posix" capability (the
//! `posix_cap.rs` lane — bash's shim band 0 resolves it via `__vm_cap_resolve` and drives the op
//! ABI through `__vm_host_call`), and print stdout/stderr — the gap-walk driver for slices 3-4.
//! Like `try_translate`, a probe, not a test.
fn main() {
    let ll = std::env::args()
        .nth(1)
        .expect("usage: bash_probe <bash_linked.ll> [args...]");
    let extra: Vec<String> = std::env::args().skip(2).collect();
    let opts = temen_llvm::TranslateOptions {
        stub_unresolved_externs: true,
        ..Default::default()
    };
    let t = temen_llvm::translate_ll_path_with_options(&ll, opts).expect("translate");
    temen_verify::verify_module(&t.module).expect("verify");
    if let Ok(list) = std::env::var("BASH_PROBE_NAMES") {
        for tok in list.split(',') {
            if let Ok(i) = tok.trim().parse::<u32>() {
                // Defined functions resolve through the exports; stubs through the §6 name waist.
                let export = t.exports.iter().find(|(_, idx)| *idx == i).map(|(n, _)| n);
                match export {
                    Some(n) => println!("f{i} = export {n:?}"),
                    None => println!("f{i} = {:?}", temen_interp::func_name(&t.module, i)),
                }
            }
        }
        return;
    }
    let mut argv: Vec<&[u8]> = vec![b"bash"];
    argv.extend(extra.iter().map(|s| s.as_bytes()));
    let env: Vec<&[u8]> = vec![b"PATH=/bin", b"HOME=/", b"PS1=$ "];
    // bash runs on the INTERPRETER: setjmp/longjmp (its whole error model) and fork are
    // interp-only tiers — the JIT declines them.
    let inst = temen_run::instantiate(t.module.clone()).expect("instantiate");
    let config = temen_run::RunConfig {
        stdin: Vec::new(),
        args: argv.iter().map(|s| s.to_vec()).collect(),
        env: env.iter().map(|s| s.to_vec()).collect(),
        ..Default::default()
    };
    // Heap 0,0: bash brings the waist malloc (powerbox `vm_map`); the personality serves only the
    // process/fd/signal ops. The personality captures ITS OWN stdio (bash's fd 1/2 live in its fd
    // table), separate from the run's powerbox streams.
    //
    // Optional /bin staging (slice 4 — external commands): BASH_PROBE_BIN names a directory of
    // `.temt` text-IR command modules (see demos/bash/stage_bin.sh); each is granted as a `Module`
    // and registered as the filesystem executable `/bin/<stem>` inside the grant closure (the
    // c_posix.rs `stage_executable` shape), so `fork → execve("/bin/echo")` resolves.
    let bins: Vec<(String, temen_ir::Module, u8)> = std::env::var("BASH_PROBE_BIN")
        .ok()
        .map(|dir| {
            let mut v = Vec::new();
            for e in std::fs::read_dir(&dir).expect("read BASH_PROBE_BIN") {
                let p = e.expect("dir entry").path();
                if p.extension().is_none_or(|x| x != "temen") {
                    continue;
                }
                let name = p.file_stem().unwrap().to_string_lossy().into_owned();
                let ir = std::fs::read_to_string(&p).expect("read command IR");
                let m = temen_text::parse_module(&ir).expect("parse command");
                temen_verify::verify_module(&m).expect("verify command");
                let wl = m.memory.expect("command window").size_log2;
                v.push((format!("/bin/{name}"), m, wl));
            }
            v
        })
        .unwrap_or_default();
    // Interactive mode (#797): BASH_PROBE_TERM holds `;`-separated lines to type at the terminal
    // (each fed with a newline after a short delay — the `run_interp_terminal` witness shape).
    // The terminal is enabled at grant time; the feeder thread types while bash runs.
    let term_feed: Option<Vec<String>> = std::env::var("BASH_PROBE_TERM")
        .ok()
        .map(|s| s.split(';').map(|x| x.to_string()).collect());
    let with_term = term_feed.is_some();
    // `posix_cap` plus the /bin registration and terminal enable, which must happen inside the
    // grant (module handles and the terminal input pipe live in the run's Host).
    let (posix, make) = temen_posix::cap(0, 0, Vec::new());
    let fork = temen_posix::cap_fork_factory(&posix);
    let p = posix.clone();
    let cap = temen_run::HostCap::custom(temen_interp::cap_id::HOST_PROC, 0, move |h, _win| {
        let handle = h.grant_host_proc_forkable(make(), std::sync::Arc::clone(&fork));
        let (door, armed) = temen_posix::cap_signal_source(&p);
        h.set_signal_source(door, armed);
        h.push_exec_remap_hook(temen_posix::cap_exec_remap_hook(&p));
        let (names, sigs) = temen_posix::cap_vtable();
        h.set_host_proc_vtable(handle, names, sigs);
        for (path, m, wl) in &bins {
            let mh = h.grant_module(m);
            p.register_executable(path, mh, *wl);
        }
        if with_term {
            p.enable_terminal(h);
            // #1122 — let the cooperative bytecode driver BLOCK for the feeder's keystrokes at
            // its all-parked point instead of faulting as a deadlock. Inert on the other tiers
            // (the tree-walker wires its own scheduler doors; the parallel driver polls).
            h.arm_external_wake();
        }
        handle
    });
    let feeder = term_feed.map(|lines| {
        let px = posix.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(400)); // let bash reach the prompt
            for l in lines {
                // Control tokens: a line that is exactly `^C`/`^D`/`^Z` feeds the raw control
                // byte (no newline) — the VINTR/VEOF/VSUSP keystrokes.
                match l.as_str() {
                    "^C" => px.feed_terminal(&[0x03]),
                    "^D" => px.feed_terminal(&[0x04]),
                    "^Z" => px.feed_terminal(&[0x1a]),
                    _ => px.feed_terminal(format!("{l}\n").as_bytes()),
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        })
    });
    run_and_print(&inst, &config, cap, posix);
    if let Some(f) = feeder {
        f.join().expect("feeder thread");
    }
}

fn run_and_print(
    inst: &temen_run::Instance,
    config: &temen_run::RunConfig,
    cap: temen_run::HostCap,
    posix: temen_posix::Posix,
) {
    // BASH_PROBE_BACKEND=bytecode runs on the wasm-safe bytecode tier (the browser engine);
    // =parallel runs the same bytecode over `drive_parallel` (#748 — every fork twin a real OS
    // thread, blocking waitpid/pipes real condvar/poll blocks); default is the tree-walk interp.
    let run = match std::env::var("BASH_PROBE_BACKEND").as_deref() {
        Ok("parallel") => inst.run_with_caps_parallel(config, &[("posix", cap)]),
        Ok("bytecode") => {
            inst.run_with_caps(temen_run::Backend::Bytecode, config, &[("posix", cap)])
        }
        _ => inst.run_with_caps(temen_run::Backend::TreeWalk, config, &[("posix", cap)]),
    };
    match run {
        Ok(r) => {
            println!("outcome: {:?}", r.outcome);
            println!(
                "--- posix stdout ---\n{}",
                String::from_utf8_lossy(&posix.stdout())
            );
            println!(
                "--- posix stderr ---\n{}",
                String::from_utf8_lossy(&posix.stderr())
            );
            println!("--- run stdout ---\n{}", String::from_utf8_lossy(&r.stdout));
            println!("--- run stderr ---\n{}", String::from_utf8_lossy(&r.stderr));
        }
        Err(e) => {
            println!("RUN ERR: {e}");
            println!(
                "--- posix stdout ---\n{}",
                String::from_utf8_lossy(&posix.stdout())
            );
        }
    }
}
