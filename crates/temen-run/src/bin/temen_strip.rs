//! `temen-strip` — the `.temen` packaging step (#907, #2790): keep a shipped image's entry points and
//! the code they reach, the native `strip`/`objcopy`-with-a-keep-list analogue. Translate exports
//! **every** defined function by its symbol name, welding the semantic table (`.dynsym`) and the
//! diagnostic names (`.symtab`) together, so every function is a root and no code is ever dead; this
//! tool splits them for a shipped artifact:
//!
//! ```text
//! temen-strip <in.temen> -o <out.temen> [--keep name,name,...] [--strip-names]
//! ```
//!
//! Exports named in `--keep` (plus `_start` and `_child`, always) stay semantic; every other export
//! moves into `debug_info.func_names`, the strippable §6 name waist that backtraces and the wasm-JIT
//! emitter's name section already read. Then every function nothing kept reaches has its body
//! emptied to a trap, indices unmoved ([`temen_run::keep_entries`]); function 0, which a host that
//! names no entry runs, is always kept. `--strip-names` additionally drops the diagnostic table for
//! a size-critical artifact (traces then fall back to `gfN`); with the split, that strip is
//! guaranteed inert. `temen-llvm-translate --keep` runs the same step as it translates. Only sound
//! per artifact: name every export a host looks up, and keep everything a linking artifact
//! publishes.

use std::process;

fn main() {
    if let Err(e) = try_main() {
        eprintln!("temen-strip: {e}");
        process::exit(1);
    }
}

fn try_main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "usage: temen-strip <in.temen> -o <out.temen> [--keep name,name,...] [--strip-names]\n\
             \n  Demotes exports not named in --keep (plus `_start` and `_child`, always kept) from the\n\
             \n  semantic exports table into the strippable debug_info.func_names diagnostic table\n\
             \n  (#907), then empties every function nothing kept reaches (#2790; function 0 is kept).\n\
             \n  --strip-names additionally drops the diagnostic name table (traces fall back to gfN).\n\
             \n  Per-artifact only: name every export a host looks up, and keep everything a linking\n\
             \n  artifact publishes."
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
        temen_encode::decode_module(&bytes).map_err(|e| format!("decode `{input}`: {e:?}"))?;

    let before = m.exports.len();
    let keep_refs: Vec<&str> = keep.iter().map(String::as_str).collect();
    let kept =
        temen_run::keep_entries(&mut m, &keep_refs).map_err(|e| format!("`{input}`: {e:?}"))?;
    if strip_names {
        if let Some(di) = &mut m.debug_info {
            di.func_names.clear();
            if *di == Default::default() {
                m.debug_info = None;
            }
        }
    }

    let out_bytes = temen_encode::encode_module(&m);
    std::fs::write(&out, &out_bytes).map_err(|e| format!("write `{out}`: {e}"))?;
    eprintln!(
        "temen-strip: exports {before} -> {} ({} demoted{}), {} functions emptied, {} -> {} bytes",
        m.exports.len(),
        kept.demoted,
        if strip_names {
            ", diagnostic names stripped"
        } else {
            " to diagnostic names"
        },
        kept.stubbed,
        bytes.len(),
        out_bytes.len(),
    );
    Ok(())
}
