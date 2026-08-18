//! `svmb-strip` — the `.svmb` packaging step (#907): demote non-entry-point exports to diagnostic
//! names, the native `strip`/`objcopy`-with-a-keep-list analogue. Translate exports **every** defined
//! function by its symbol name, welding the semantic table (`.dynsym`) and the diagnostic names
//! (`.symtab`) together; this tool splits them for a shipped artifact:
//!
//! ```text
//! svmb-strip <in.svmb> -o <out.svmb> [--keep name,name,...] [--strip-names]
//! ```
//!
//! Exports named in `--keep` (plus `_start`, always — the powerbox-entry marker) stay semantic; every
//! other export moves into `debug_info.func_names`, the strippable §6 name waist that backtraces and
//! the wasm-JIT emitter's name section already read — so demotion changes **no** diagnostic output,
//! only what `resolve_export` can reach. `--strip-names` additionally drops the diagnostic table for
//! a size-critical artifact (traces then fall back to `gfN`); with the split, that strip is
//! guaranteed inert. Only sound per-artifact for a known-single-module `.svmb` — a parent guest can
//! resolve any child export by name through the Module capability, so keep everything a linking or
//! child-module artifact publishes.

use std::process;

fn main() {
    if let Err(e) = try_main() {
        eprintln!("svmb-strip: {e}");
        process::exit(1);
    }
}

fn try_main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "usage: svmb-strip <in.svmb> -o <out.svmb> [--keep name,name,...] [--strip-names]\n\
             \n  Demotes exports not named in --keep (plus `_start`, always kept) from the semantic\n\
             \n  exports table into the strippable debug_info.func_names diagnostic table (#907).\n\
             \n  --strip-names additionally drops the diagnostic name table (traces fall back to gfN).\n\
             \n  Per-artifact only: a module resolved by name from a parent guest or the linker must\n\
             \n  keep everything it publishes."
        );
        return Err("no input file".into());
    }

    let mut input: Option<String> = None;
    let mut out: Option<String> = None;
    let mut keep: Vec<String> = Vec::new();
    let mut strip_names = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-o" | "--output" => out = Some(it.next().ok_or("-o needs a file argument")?.clone()),
            "--keep" => keep.extend(
                it.next()
                    .ok_or("--keep needs a comma-separated name list")?
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(str::to_string),
            ),
            "--strip-names" => strip_names = true,
            _ if a.starts_with('-') => return Err(format!("unknown flag `{a}`")),
            _ => {
                if input.replace(a.clone()).is_some() {
                    return Err("more than one input file given".into());
                }
            }
        }
    }
    let input = input.ok_or("no input file")?;
    let out = out.ok_or("no output file (-o <out>)")?;

    let bytes = std::fs::read(&input).map_err(|e| format!("read `{input}`: {e}"))?;
    let mut m =
        svm_encode::decode_module(&bytes).map_err(|e| format!("decode `{input}`: {e:?}"))?;

    let before = m.exports.len();
    let keep_refs: Vec<&str> = keep.iter().map(String::as_str).collect();
    let demoted = svm_run::demote_exports(&mut m, &keep_refs);
    if strip_names {
        if let Some(di) = &mut m.debug_info {
            di.func_names.clear();
            if *di == Default::default() {
                m.debug_info = None;
            }
        }
    }

    let out_bytes = svm_encode::encode_module(&m);
    std::fs::write(&out, &out_bytes).map_err(|e| format!("write `{out}`: {e}"))?;
    eprintln!(
        "svmb-strip: exports {before} -> {} ({demoted} demoted{}), {} -> {} bytes",
        m.exports.len(),
        if strip_names {
            ", diagnostic names stripped"
        } else {
            " to diagnostic names"
        },
        bytes.len(),
        out_bytes.len(),
    );
    Ok(())
}
