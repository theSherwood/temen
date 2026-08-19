# bash — GNU bash on the LLVM on-ramp (#802)

The bring-up of the umbrella's (#794) target program: **the literal GNU bash source**, compiled
through the on-ramp as one whole-program module, hosted by the svm-posix personality this tree
built for it (fork #863, signals #796, job control #798, pipes #972, exec + `/bin` #801, the
controlling terminal #797, `longjmp`-from-a-handler #802-slice-1).

## Slice 2 (DONE) — translate + verify

`./build_bitcode.sh`: fetch bash 5.2.21 (fetched-not-vendored, GPLv3) → configure the bring-up
config → **native oracle build** (also generates y.tab.c and the `.def`-built builtins) → per-TU
bitcode with each Makefile's own flags (152 TUs: the link-line objects + libbuiltins/libglob/
libsh/libhistory(hist* only — the rest are readline standalone shims that duplicate bash's own)/
libtilde) → `llvm-link` + `bash_shim.c` + the reused waist → **translate: 1716 funcs (~0.7 s),
verify: clean**. Gate: `demo_bash_translates_and_verifies` (svm-llvm `translate.rs`, `#[ignore]`d
for wall-clock).

The bring-up config (`configure` flags, each with a reason in the script): `--without-bash-malloc`
(the waist malloc, not sbrk), `--disable-readline` (non-interactive first; interactive rides the
#797 terminal in slice 4), `--disable-nls`, `--disable-net-redirections` (no sockets), and
`ac_cv_type_long_double=no` (the printf builtin's `%Lf` would need x86_fp80 — denying the type
keeps `floatmax_t = double` in guest AND oracle). **Job control stays on.**

## The gap-walk (the Tcl discipline: every gap gets a pinned unit test)

1. **`align 4294967296`** — clang stamps the max alignment (2^32, one past `u32`) on
   deliberately-trapping null stores (bash's `programming_error`). The `.ll` parser now saturates
   an alignment literal instead of refusing the module. Pin: `align_u32_max_saturates`.
2. **Old-C call-site drift** — bash's empty-parens prototypes (`extern void f ();`) let call
   sites invent their own function types: `add_unwind_protect(fn, 0)` is typed
   `(ptr, i32, ...)` at the site against a plain `(ptr, ptr)` definition. The native ABI hides
   the drift; the lowering now follows the **definition** for direct calls — arity split,
   va-area deposit only for a genuinely variadic callee, and integer args coerced to the
   definition's widths. Pin: `old_c_call_site_drift_follows_the_definition`.

## What remains (the slice ladder from the #802 sketch)

- **Slice 3 — first run**: wire the stdio `FILE*` surface (a trial run already reaches deep into
  `shell.c` startup and stops at `xtrace_set: NULL file pointer` — stderr is a real `FILE*` in
  bash) + the OS shim over the svm-posix ops; `bash -c 'echo hi'` byte-differential vs the
  oracle. `bash_shim.c` grows from the stub report, the same walk every capstone did.
- **Slice 4 — the differential suite**: pipelines over the #801 `/bin`, subshells, redirections,
  traps; then interactive on the #797 terminal.

| File | Role |
|---|---|
| `build_bitcode.sh` | the faithful fetch→configure→oracle→bitcode→link→translate pipeline |
| `bash_shim.c` | the bash-specific libc/OS surface (grows per slice; see its header) |
