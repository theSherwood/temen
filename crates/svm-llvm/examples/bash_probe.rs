//! Scratch probe for the bash bring-up (#802): translate the linked module, run it on the
//! interpreter with the svm-posix personality granted as the named "posix" capability (the
//! `posix_cap.rs` lane — bash's shim band 0 resolves it via `__vm_cap_resolve` and drives the op
//! ABI through `__vm_host_call`), and print stdout/stderr — the gap-walk driver for slices 3-4.
//! Like `try_translate`, a probe, not a test.
fn main() {
    let ll = std::env::args()
        .nth(1)
        .expect("usage: bash_probe <bash_linked.ll> [args...]");
    let extra: Vec<String> = std::env::args().skip(2).collect();
    let opts = svm_llvm::TranslateOptions {
        stub_unresolved_externs: true,
        ..Default::default()
    };
    let t = svm_llvm::translate_ll_path_with_options(&ll, opts).expect("translate");
    svm_verify::verify_module(&t.module).expect("verify");
    if let Ok(list) = std::env::var("BASH_PROBE_NAMES") {
        for tok in list.split(',') {
            if let Ok(i) = tok.trim().parse::<u32>() {
                // Defined functions resolve through the exports; stubs through the §6 name waist.
                let export = t.exports.iter().find(|(_, idx)| *idx == i).map(|(n, _)| n);
                match export {
                    Some(n) => println!("f{i} = export {n:?}"),
                    None => println!("f{i} = {:?}", svm_interp::func_name(&t.module, i)),
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
    let inst = svm_run::instantiate(t.module.clone()).expect("instantiate");
    let config = svm_run::RunConfig {
        stdin: Vec::new(),
        args: argv.iter().map(|s| s.to_vec()).collect(),
        env: env.iter().map(|s| s.to_vec()).collect(),
        ..Default::default()
    };
    // Heap 0,0: bash brings the waist malloc (powerbox `vm_map`); the personality serves only the
    // process/fd/signal ops. The personality captures ITS OWN stdio (bash's fd 1/2 live in its fd
    // table), separate from the run's powerbox streams.
    let (cap, posix) = svm_run::posix::posix_cap(0, 0, Vec::new());
    match inst.run_with_caps(svm_run::Backend::TreeWalk, &config, &[("posix", cap)]) {
        Ok(r) => {
            println!("outcome: {:?}", r.outcome);
            println!(
                "--- posix stdout ---\n{}",
                String::from_utf8_lossy(&posix.stdout())
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
