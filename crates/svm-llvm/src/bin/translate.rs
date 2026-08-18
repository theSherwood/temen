//! `svm-llvm-translate` — translate legalized LLVM IR to an SVM-IR module. Input is textual `.ll`
//! (the in-house version-tolerant reader — the preferred path) or bitcode `*.bc` (disassembled via
//! `llvm-dis`); the reader is chosen by the input's file extension.
//!
//! ```text
//! svm-llvm-translate <input.ll|input.bc> -o <out> [--binary]
//! ```
//!
//! This is the **separate-artifact** on-ramp (the scriptable companion to the [`svm_llvm`] library):
//! a frontend like JACL compiles its runtime once to bitcode and translates it here to a reusable
//! module — compile the runtime once, link many programs against it (`svm-run --link`, or
//! [`svm_ir::link`] over [`svm_ir::LinkUnit`]s built from the module's first-class export tables).
//!
//! Output format: text (`svm_text::print_module`) by default, binary (`svm_encode::encode_module`)
//! when `-o` ends in `.svmb` or `--binary` is given, or a binary **object** / link unit
//! (`svm_encode::encode_unit`, the v9 object dialect) when `-o` ends in `.svmo`. Exports ride
//! in-band in every form (the retired `.syms` sidecar is gone) — `.svmo` is the artifact to
//! produce for anything that links.

use std::path::Path;
use std::{env, fs, process};

fn main() {
    if let Err(e) = try_main() {
        eprintln!("svm-llvm-translate: {e}");
        process::exit(1);
    }
}

fn try_main() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "usage: svm-llvm-translate <input.ll|input.bc> -o <out> [--binary] [--host-page <bytes>] [--stub-externs] [--null-guard]\n\
             \n  Translates legalized LLVM IR (textual .ll, or .bc via llvm-dis) to an SVM-IR module written to <out>:\n\
             \n    text (.svm) by default, binary (.svmb) when -o ends in .svmb or --binary,\n\
             \n    or a binary object/link unit (.svmo, v9 object dialect). Exports ride in-band\n\
             \n    in every form; emit .svmo for anything that links (svm-run --link).\n\
             \n  --host-page <bytes> sets the powerbox RO/writable page-isolation granularity\n\
             \n  (default 16384). Pass 65536 when the .svmb targets a wasm host (64 KiB pages) —\n\
             \n  e.g. the browser interpreter — so read-only globals never share a host page with\n\
             \n  the writable data stack (which would fault under D40).\n\
             \n  --stub-externs lowers undefined externals to trap-if-called stubs instead of\n\
             \n  failing translation (large-program bring-up, e.g. Postgres).\n\
             \n  --null-guard (#964) lays the powerbox low scratch out one 16 KiB guard above zero\n\
             \n  and marks the module (`__null_guard` export) so a host seeds [0, 16384) unmapped\n\
             \n  and NULL dereferences trap. Off by default (legacy layout, byte-identical)."
        );
        return Err("no input file".into());
    }

    let mut input: Option<String> = None;
    let mut out: Option<String> = None;
    let mut binary = false;
    let mut host_page: u64 = svm_ir::POWERBOX_STACK_PAGE;
    let mut stub_externs = false;
    let mut null_guard = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-o" | "--output" => out = Some(it.next().ok_or("-o needs a file argument")?.clone()),
            "--binary" => binary = true,
            "--host-page" => {
                host_page = it
                    .next()
                    .ok_or("--host-page needs a byte-count argument")?
                    .parse()
                    .map_err(|e| format!("--host-page: {e}"))?
            }
            // Lower undefined externals to trap-if-called stubs instead of failing translation — for a
            // large-program bring-up (Postgres) where most externals are dead on the exercised path.
            "--stub-externs" => stub_externs = true,
            // #964 trap-on-NULL: the guarded layout (low scratch shifted one guard up + the
            // `__null_guard` marker export), so a host may seed `[0, 16384)` unmapped.
            "--null-guard" => null_guard = true,
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
    // Binary if asked explicitly or the output names a `.svmb` file; text otherwise. A `.svmo`
    // output writes the v9 **object dialect** (`encode_unit`): a self-contained binary link unit
    // whose first-class export tables are its link symbols.
    let object = Path::new(&out).extension().is_some_and(|e| e == "svmo");
    let binary = binary || Path::new(&out).extension().is_some_and(|e| e == "svmb");

    // Translate the input. A `.ll` extension takes the in-house **textual** reader (no `llvm-dis`,
    // version-tolerant — the direction the on-ramp is developed on); anything else is treated as
    // bitcode and disassembled via `llvm-dis`. `Error` is `Debug`-only (no `Display`), render it so.
    let opts = svm_llvm::TranslateOptions {
        stub_unresolved_externs: stub_externs,
        stack_page: host_page,
        null_guard,
    };
    let is_ll = Path::new(&input).extension().is_some_and(|e| e == "ll");
    let translated = if is_ll {
        svm_llvm::translate_ll_path_with_options(&input, opts)
    } else {
        svm_llvm::translate_bc_path_with_options(&input, opts)
    }
    .map_err(|e| format!("translate `{input}`: {e:?}"))?;

    let module_bytes = if object {
        svm_encode::encode_unit(&translated.module)
    } else if binary {
        svm_encode::encode_module(&translated.module)
    } else {
        svm_text::print_module(&translated.module).into_bytes()
    };
    fs::write(&out, &module_bytes).map_err(|e| format!("write `{out}`: {e}"))?;

    Ok(())
}
