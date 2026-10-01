//! Larger-libc coverage for the playground C compiler (SELFHOST_C.md §7, "larger libc surface"
//! residual). The seeded `browser/playground-include/*.h` grew a broad guest-C libc — `<math.h>`,
//! `<assert.h>`, `<limits.h>`, `<stddef.h>`, `<errno.h>`, plus additions to `<string.h>`/`<stdlib.h>`/
//! `<ctype.h>`. These tests compile real, non-trivial programs that lean on that surface (a stats
//! pipeline over `strtok`/`strtod`/`qsort`/`sqrt`, an algebraic-math sweep, and the extended string/
//! stdlib functions) in the sandbox and assert their output — so the libc is exercised end to end,
//! exactly as the card runs it. Fail-soft: SKIPs if `chibicc.temen` isn't built.

use temen_browser::{
    onramp_exec, onramp_fs_exec, playground_include_files, STATUS_EXIT, STATUS_OK,
};

fn chibicc_temen() -> Option<temen_ir::Module> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/web/assets/chibicc.temen");
    let bytes = std::fs::read(p).ok()?;
    Some(temen_encode::decode_module(&bytes).expect("decode chibicc.temen"))
}

/// Compile `src` with the seeded playground headers, run the result, return its captured stdout.
fn compile_and_run(chibicc: &temen_ir::Module, src: &str) -> (i32, String) {
    let run = onramp_exec(&compile(chibicc, src), b"");
    (
        run.status,
        String::from_utf8_lossy(&run.stdout).into_owned(),
    )
}

/// Compile `src` with the seeded playground headers into a runnable module.
fn compile(chibicc: &temen_ir::Module, src: &str) -> temen_ir::Module {
    let mut files: Vec<(String, Vec<u8>)> = playground_include_files();
    files.push(("in.c".to_string(), src.as_bytes().to_vec()));
    let dirs = vec!["include".to_string()];
    let image = temen_fs::encode_image(&files, &dirs);

    let compiled = onramp_fs_exec(
        chibicc,
        &image,
        &[b"chibicc", b"--data-page", b"65536", b"/in.c"],
        b"",
    );
    assert!(
        compiled.status == STATUS_OK || compiled.status == STATUS_EXIT,
        "compile status {} — stderr: {}",
        compiled.status,
        String::from_utf8_lossy(&compiled.stderr)
    );
    let ir = String::from_utf8(compiled.stdout).expect("IR is utf8");
    assert!(ir.contains("func"), "expected Temen IR, got: {ir:.200}");

    temen_text::parse_module(&ir).unwrap_or_else(|e| panic!("parse IR: {e:?}"))
}

/// **A release run pumped in slices** — the tier-up session with no regions (`COOP_NO_REGIONS`,
/// `temen_coop_run_for`): each slice hands back the output it produced, a program that never ends
/// keeps reporting `COOP_RUN_PAUSED` (so the embedder can stream it and stop pumping on Pause), and a
/// finite program pumped in small slices ends with exactly the one-shot run's output and exit code.
/// A trap ends it with the one-shot run's trap name and fault address, and the files the program
/// wrote stay readable (`temen_coop_fs_image`) after the session closes.
#[test]
fn a_release_run_pumps_in_slices() {
    use temen_browser::{
        temen_alloc, temen_coop_close, temen_coop_fs_image, temen_coop_fs_ptr, temen_coop_open,
        temen_coop_run_for, temen_exit_code, temen_fault_addr, temen_status, temen_stdout_len,
        temen_stdout_ptr, temen_trap_len, temen_trap_ptr, COOP_NO_REGIONS, COOP_RUN_DONE,
        COOP_RUN_PAUSED, COOP_RUN_TRAP,
    };
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let open = |m: &temen_ir::Module| {
        let bytes = temen_encode::encode_module(m);
        let p = temen_alloc(bytes.len());
        // SAFETY: `temen_alloc` returned a live allocation of that length.
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len()) };
        let opened = temen_coop_open(
            p,
            bytes.len(),
            core::ptr::null(),
            0,
            0,
            core::ptr::null(),
            0,
            COOP_NO_REGIONS,
        );
        assert_eq!(opened, STATUS_OK);
    };
    let slice_out = || {
        let (p, n) = (temen_stdout_ptr(), temen_stdout_len());
        if p.is_null() || n == 0 {
            return Vec::new(); // a slice that printed nothing
        }
        // SAFETY: the stash stays live until the next call that replaces it.
        unsafe { core::slice::from_raw_parts(p, n) }.to_vec()
    };
    // Pump to the end in `budget`-op slices: the output, and how many slices it took.
    let drain = |budget: u64| {
        let (mut out, mut slices) = (Vec::new(), 0);
        loop {
            let r = temen_coop_run_for(budget);
            out.extend(slice_out());
            slices += 1;
            if r == COOP_RUN_DONE || r == COOP_RUN_TRAP {
                return (out, slices);
            }
            assert_eq!(r, COOP_RUN_PAUSED);
        }
    };

    let spin = compile(
        &chibicc,
        "#include <stdio.h>\nint main(void) {\n  for (int i = 0; i < 3; i++) printf(\"line %d\\n\", i);\n  for (;;) {}\n}\n",
    );
    open(&spin);
    let mut out = Vec::new();
    for _ in 0..50 {
        assert_eq!(
            temen_coop_run_for(100_000),
            COOP_RUN_PAUSED,
            "an endless loop keeps running"
        );
        out.extend(slice_out());
    }
    assert_eq!(
        String::from_utf8_lossy(&out),
        "line 0\nline 1\nline 2\n",
        "streamed as it ran"
    );
    temen_coop_close();

    let finite = compile(
        &chibicc,
        "#include <stdio.h>\nint main(void) {\n  long s = 0;\n  for (int i = 0; i < 20000; i++) { s += i; if (i % 5000 == 0) printf(\"%d\\n\", i); }\n  printf(\"sum %ld\\n\", s);\n  return 3;\n}\n",
    );
    let one_shot = onramp_exec(&finite, b"");
    open(&finite);
    let (out, slices) = drain(1_000);
    temen_coop_close();
    assert!(slices > 10, "it took many slices ({slices})");
    assert_eq!(out, one_shot.stdout, "the same output as the one-shot run");
    assert_eq!(
        (temen_status(), temen_exit_code()),
        (one_shot.status, one_shot.exit_code),
        "and the same ending"
    );

    let faults = compile(
        &chibicc,
        "#include <stdio.h>\nint main(void) {\n  int *p = (int *)8;\n  printf(\"before\\n\");\n  return *p;\n}\n",
    );
    let one_shot = onramp_exec(&faults, b"");
    open(&faults);
    let (out, _) = drain(1_000);
    temen_coop_close();
    // SAFETY: the trap name is a static string.
    let trap = unsafe { core::slice::from_raw_parts(temen_trap_ptr(), temen_trap_len()) };
    assert_eq!(out, b"before\n");
    assert_eq!(
        (temen_status(), trap, temen_fault_addr()),
        (
            one_shot.status,
            one_shot.trap.map_or("", |t| t.name()).as_bytes(),
            one_shot.fault_addr.map_or(-1, |a| a as i64),
        ),
        "a trap ends it as the one-shot run's does"
    );
    assert!(!trap.is_empty(), "it trapped");

    let writes = compile(
        &chibicc,
        "#include <stdio.h>\nint main(void) {\n  FILE *f = fopen(\"notes.txt\", \"w\");\n  fputs(\"kept\\n\", f);\n  fclose(f);\n  return 0;\n}\n",
    );
    open(&writes);
    drain(1_000);
    temen_coop_close();
    let n = temen_coop_fs_image();
    // SAFETY: the blob stays live until the next call.
    let image = unsafe { core::slice::from_raw_parts(temen_coop_fs_ptr(), n) };
    let has = |needle: &[u8]| image.windows(needle.len()).any(|w| w == needle);
    assert!(
        has(b"notes.txt") && has(b"kept\n"),
        "the files the run ended with outlive its session"
    );
}

/// **The scanf family** (`sscanf`/`scanf`), which the playground libc lacked: c_interpret's `scanf`
/// lessons failed to link. One scanner serves both, with C's return rules — the count of
/// assignments, or EOF for an input failure before the first — and one byte of `ungetc` lookahead,
/// so a number read by one call leaves the byte that ended it for the next.
///
/// (Read from a file: this harness's `onramp_exec` delivers no stdin to the program — even `getchar`
/// reads EOF there, with or without this change.)
#[test]
fn scanf_family_converts_like_c() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let run = |body: &str, stdin: &[u8]| {
        let src = format!("#include <stdio.h>\nint main(void) {{\n{body}\n  return 0;\n}}\n");
        let out = onramp_exec(&compile(&chibicc, &src), stdin);
        String::from_utf8_lossy(&out.stdout).into_owned()
    };

    let out = run(
        r#"  int a, n; unsigned u; long l; short h; char c, w[8], line[32]; double d; float f;
  int r = sscanf(" -42 7 0x1F 3.25 1e3 xyz! hello world", "%d %u %li %lf %f %c%3s%*c %[^\n]%n",
                 &a, &u, &l, &d, &f, &c, w, line, &n);
  printf("%d|%d %u %ld %.2f %.1f %c %s [%s] %d\n", r, a, u, l, d, f, c, w, line, n);
  printf("%d\n", sscanf("12abc", "%d%hd", &a, &h));
  printf("%d\n", sscanf("", "%d", &a));
  printf("%d\n", sscanf("x", "%d", &a));
  printf("%d\n", sscanf("5,6", "%d,%d", &a, &n));
  printf("%d %d\n", a, n);
  printf("%d\n", sscanf("077 10", "%i %o", &a, &n));
  printf("%d %d\n", a, n);"#,
        b"",
    );
    assert_eq!(
        out, "8|-42 7 31 3.25 1000.0 x yz! [hello world] 37\n1\n-1\n0\n2\n5 6\n2\n63 8\n",
        "sscanf"
    );

    // `fscanf` from a stream, three times: the newline after `17` is left for the next call to skip,
    // and the third call reaches the end — EOF. (A stream exercises `fgetc`/`ungetc`, the same path
    // `scanf` takes on stdin.)
    let out = run(
        r#"  FILE *w = fopen("nums.txt", "w");
  fputs("17\n  25\n", w);
  fclose(w);
  FILE *f = fopen("nums.txt", "r");
  int a = 0, b = 0;
  int r1 = fscanf(f, "%d", &a);
  int r2 = fscanf(f, "%d", &b);
  int r3 = fscanf(f, "%d", &b);
  int c = fgetc(f);
  printf("%d %d %d %d %d %d\n", r1, r2, r3, a, b, c);"#,
        b"",
    );
    assert_eq!(out, "1 1 -1 17 25 -1\n", "fscanf over a stream");
}

/// **`free` refuses what it must not release**, the way glibc does: a double free, or a pointer
/// `malloc` never returned, names the misuse on **stderr** — its own stream, not stdout — and aborts
/// (exit 134). `free` used to be a
/// no-op, so c_interpret's double-free lesson ran straight past the bug it teaches.
#[test]
fn free_detects_a_double_free_and_an_invalid_pointer() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let run = |body: &str| {
        let src = format!(
            "#include <stdio.h>\n#include <stdlib.h>\nint main(void) {{\n{body}\n  return 0;\n}}\n"
        );
        let out = onramp_exec(&compile(&chibicc, &src), b"");
        (
            out.exit_code,
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };

    // Ordinary use is untouched: free(NULL), a free per block, and a realloc that moves.
    let (code, out, err) = run("  free(NULL);\n\
           int *a = malloc(8), *b = malloc(8);\n\
           a[0] = 7;\n\
           a = realloc(a, 64);\n\
           printf(\"%d\\n\", a[0]);\n\
           free(a);\n\
           free(b);");
    assert_eq!(
        (code, out.as_str(), err.as_str()),
        (0, "7\n", ""),
        "ordinary use"
    );

    // A double free names itself on stderr and aborts right there: `after` never prints.
    let (code, out, err) = run("  int *p = malloc(sizeof(int));\n\
           printf(\"before\\n\");\n\
           free(p);\n\
           free(p);\n\
           printf(\"after\\n\");");
    assert_eq!(
        (code, out.as_str(), err.as_str()),
        (134, "before\n", "free(): double free detected\n"),
        "a double free"
    );

    let (code, out, err) = run("  int x;\n  free(&x);");
    assert_eq!(
        (code, out.as_str(), err.as_str()),
        (134, "", "free(): invalid pointer\n"),
        "a stack pointer"
    );

    // realloc releases the old block, so freeing it afterwards is a double free too.
    let (code, out, err) =
        run("  char *p = malloc(8);\n  char *q = realloc(p, 4096);\n  free(q);\n  free(p);");
    assert_eq!(
        (code, out.as_str(), err.as_str()),
        (134, "", "free(): double free detected\n"),
        "a realloc'd-away block"
    );
}

/// A real ~90-line program: parse a delimited list of numbers, compute summary statistics (mean,
/// variance→stddev via `sqrt`, min/max, median via `qsort`), asserting invariants along the way. It
/// leans on `<string.h>` (`strtok`), `<stdlib.h>` (`strtod`, `qsort`), `<math.h>` (`sqrt`/`fabs`),
/// `<assert.h>`, and `<stdio.h>` (`printf`) together. The dataset is the textbook {2,4,4,4,5,5,7,9}
/// whose mean (5) and stddev (2) are exact, so the output is deterministic.
#[test]
fn stats_pipeline_over_the_expanded_libc() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let (status, out) = compile_and_run(
        &chibicc,
        r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

int main(void) {
  char line[] = "2, 4, 4, 4, 5, 5, 7, 9";
  double v[32];
  int n = 0;
  for (char *tok = strtok(line, ", "); tok; tok = strtok(NULL, ", ")) {
    assert(n < 32);
    v[n++] = strtod(tok, NULL);
  }
  assert(n == 8);

  double sum = 0, mn = v[0], mx = v[0];
  for (int i = 0; i < n; i++) {
    sum += v[i];
    if (v[i] < mn) mn = v[i];
    if (v[i] > mx) mx = v[i];
  }
  double mean = sum / n;

  double var = 0;
  for (int i = 0; i < n; i++) { double d = v[i] - mean; var += d * d; }
  var /= n;
  double sd = sqrt(var);

  qsort(v, n, sizeof(double), cmp_double);
  double median = (n % 2) ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;

  assert(fabs(mean - 5.0) < 1e-9);
  assert(fabs(sd - 2.0) < 1e-9);
  printf("count=%d sum=%.2f mean=%.2f min=%.2f max=%.2f median=%.2f stddev=%.2f\n",
         n, sum, mean, mn, mx, median, sd);
  return 0;
}
"#,
    );
    assert_eq!(status, STATUS_OK, "run status");
    assert_eq!(
        out,
        "count=8 sum=40.00 mean=5.00 min=2.00 max=9.00 median=4.50 stddev=2.00\n"
    );
}

/// The algebraic `<math.h>` surface on inputs whose results are exact (or clean to 6 `%g` sig-figs):
/// `floor`/`ceil`/`round`/`trunc`/`fmod`/`pow`/`sqrt`/`fabs`/`hypot`/`cbrt`.
#[test]
fn math_h_algebraic_functions() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let (status, out) = compile_and_run(
        &chibicc,
        r#"
#include <stdio.h>
#include <math.h>
int main(void) {
  printf("%g %g %g %g %g\n", floor(3.7), ceil(3.2), round(2.5), trunc(-3.7), fmod(10.0, 3.0));
  printf("%g %g %g %g %g\n", pow(2.0, 10.0), sqrt(144.0), fabs(-5.5), hypot(3.0, 4.0), cbrt(27.0));
  printf("%g %g %g\n", pow(3.0, -2.0), log2(8.0), floor(-0.5));
  return 0;
}
"#,
    );
    assert_eq!(status, STATUS_OK, "run status");
    assert_eq!(
        out,
        "3 4 3 -3 1\n\
         1024 12 5.5 5 3\n\
         0.111111 3 -1\n"
    );
}

/// **`*` width and precision from the argument list** (#1422). `printf("%.*f", 3, x)` takes the
/// precision from an `int` argument rather than the format string; nim's `formatBiggestFloat` emits
/// exactly `%#.*g`/`%#.*e`/`%#.*f`, so the nim card cannot format a float without it. A negative `*`
/// width means left-justify; a negative `*` precision means "omitted". Also covers the hyperbolics and
/// the `float` (`…f`) overloads added for nim's `std/math`.
#[test]
fn printf_star_width_precision_and_new_math() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let (status, out) = compile_and_run(
        &chibicc,
        r#"
#include <stdio.h>
#include <math.h>
int main(void) {
  printf("[%.*f]\n", 3, 3.14159);
  printf("[%*d]\n", 5, 42);
  printf("[%-*d]\n", 5, 42);
  printf("[%*d]\n", -5, 42);
  printf("[%.*e]\n", 2, 2.5);
  printf("%.4f %.4f %.4f\n", sinh(1.0), cosh(1.0), tanh(1.0));
  printf("%.4f %.4f %.4f\n", asinh(1.0), acosh(2.0), atanh(0.5));
  printf("%.4f %.4f\n", (double)sinf(1.0f), (double)tanhf(1.0f));
  return 0;
}
"#,
    );
    assert_eq!(status, STATUS_OK, "run status");
    assert_eq!(
        out,
        "[3.142]\n\
         [   42]\n\
         [42   ]\n\
         [42   ]\n\
         [2.50e+00]\n\
         1.1752 1.5431 0.7616\n\
         0.8814 1.3170 0.5493\n\
         0.8415 0.7616\n"
    );
}

/// The extended `<string.h>` / `<stdlib.h>` / `<ctype.h>` surface: `strdup`, `strncat`, `strspn`/
/// `strcspn`, `strcasecmp`, `bsearch`, `strtoul` (hex).
#[test]
fn extended_string_and_stdlib() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let (status, out) = compile_and_run(
        &chibicc,
        r#"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int cmp_int(const void *a, const void *b) {
  return *(const int *)a - *(const int *)b;
}

int main(void) {
  char *d = strdup("hello");
  char buf[16];
  strcpy(buf, "foo");
  strncat(buf, "barbaz", 3);
  printf("%s len=%lu %s\n", d, (unsigned long)strlen(d), buf);

  printf("spn=%lu cspn=%lu case=%d\n",
         (unsigned long)strspn("   abc", " "),
         (unsigned long)strcspn("abc,def", ","),
         strcasecmp("Hello", "hELLO"));

  int arr[] = {1, 3, 5, 7, 9, 11};
  int key = 7;
  int *hit = bsearch(&key, arr, 6, sizeof(int), cmp_int);
  printf("found=%d hex=%lu\n", hit ? *hit : -1, strtoul("ff", NULL, 16));
  return 0;
}
"#,
    );
    assert_eq!(status, STATUS_OK, "run status");
    assert_eq!(
        out,
        "hello len=5 foobar\n\
         spn=3 cspn=3 case=0\n\
         found=7 hex=255\n"
    );
}

/// #1323 slice 1 — a scratch-file round-trip over the `fs` cap (c_interpret #16 bucket 1, file I/O):
/// `fopen("w")` → `fwrite` → `fclose` → `fopen("r")` → `fread` must return the bytes written. Today
/// the plain run on-ramp (`onramp_exec` → `grant_onramp_caps(_, _, None)`) mounts no memfs and the
/// playground libc's file ops don't reach the fs cap, so this fails; slice 1 mounts an empty RW memfs
/// and routes the playground libc's file fds (>= 2) to it, keeping stdout/stdin on the Stream cap.
#[test]
fn scratch_file_write_then_read_roundtrips() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let (status, out) = compile_and_run(
        &chibicc,
        r#"#include <stdio.h>
int main(void) {
  FILE *w = fopen("scratch.dat", "w");
  if (!w) { printf("open-w failed\n"); return 1; }
  const char *msg = "hello, memfs";
  fwrite(msg, 1, 12, w);
  fclose(w);
  FILE *r = fopen("scratch.dat", "r");
  if (!r) { printf("open-r failed\n"); return 2; }
  char buf[32] = {0};
  size_t n = fread(buf, 1, sizeof(buf) - 1, r);
  fclose(r);
  printf("n=%d buf=%s\n", (int)n, buf);

  // A *second* file is isolated from the first (multi-file store).
  FILE *w2 = fopen("other.dat", "w");
  fwrite("XY", 1, 2, w2);
  fclose(w2);

  // O_APPEND: reopening "scratch.dat" with "a" extends rather than truncates.
  FILE *a = fopen("scratch.dat", "a");
  fwrite("!", 1, 1, a);
  fclose(a);
  FILE *r2 = fopen("scratch.dat", "r");
  char buf2[32] = {0};
  size_t n2 = fread(buf2, 1, sizeof(buf2) - 1, r2);
  fclose(r2);
  printf("n2=%d buf2=%s\n", (int)n2, buf2);
  return 0;
}
"#,
    );
    assert_eq!(status, STATUS_OK, "run status");
    assert_eq!(out, "n=12 buf=hello, memfs\nn2=13 buf2=hello, memfs!\n");
}

/// `<sys/mman.h>` — anonymous `mmap`/`munmap` over the Memory capability (c_interpret #16 bucket 1):
/// map two pages, write one byte in each, sum them, unmap. The mapping must be page-aligned and
/// writable across both pages (offset 0 and 4096), returning 141 (= 42 + 99). A file-backed mapping
/// (fd >= 0) is refused with `MAP_FAILED`.
#[test]
fn anonymous_mmap_maps_two_writable_pages() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let (status, out) = compile_and_run(
        &chibicc,
        r#"#include <sys/mman.h>
#include <stdio.h>
int main() {
  char *mem = mmap((void *)0, 8192, PROT_READ | PROT_WRITE,
                   MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  if (mem == MAP_FAILED) { printf("map failed\n"); return 1; }
  mem[0] = 42;
  mem[4096] = 99;
  int sum = mem[0] + mem[4096];
  munmap(mem, 8192);
  // A file-backed request (fd >= 0) is unsupported and must fail cleanly.
  void *bad = mmap((void *)0, 4096, PROT_READ, MAP_PRIVATE, 3, 0);
  printf("sum=%d aligned=%d file=%d\n", sum, ((long)mem & 4095) == 0, bad == MAP_FAILED);
  return 0;
}
"#,
    );
    assert_eq!(status, STATUS_OK, "run status");
    assert_eq!(out, "sum=141 aligned=1 file=1\n");
}
