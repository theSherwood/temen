//! Larger-libc coverage for the playground C compiler (SELFHOST_C.md §7, "larger libc surface"
//! residual). The seeded `browser/playground-include/*.h` grew a broad guest-C libc — `<math.h>`,
//! `<assert.h>`, `<limits.h>`, `<stddef.h>`, `<errno.h>`, plus additions to `<string.h>`/`<stdlib.h>`/
//! `<ctype.h>`. These tests compile real, non-trivial programs that lean on that surface (a stats
//! pipeline over `strtok`/`strtod`/`qsort`/`sqrt`, an algebraic-math sweep, and the extended string/
//! stdlib functions) in the sandbox and assert their output — so the libc is exercised end to end,
//! exactly as the card runs it. Fail-soft: SKIPs if `chibicc.svmb` isn't built.

use svm_browser::{onramp_exec, onramp_fs_exec, playground_include_files, STATUS_EXIT, STATUS_OK};

fn chibicc_svmb() -> Option<svm_ir::Module> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/web/assets/chibicc.svmb");
    let bytes = std::fs::read(p).ok()?;
    Some(svm_encode::decode_module(&bytes).expect("decode chibicc.svmb"))
}

/// Compile `src` with the seeded playground headers, run the result, return its captured stdout.
fn compile_and_run(chibicc: &svm_ir::Module, src: &str) -> (i32, String) {
    let mut files: Vec<(String, Vec<u8>)> = playground_include_files();
    files.push(("in.c".to_string(), src.as_bytes().to_vec()));
    let dirs = vec!["include".to_string()];
    let image = svm_fs::encode_image(&files, &dirs);

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
    assert!(ir.contains("func"), "expected SVM IR, got: {ir:.200}");

    let m = svm_text::parse_module(&ir).unwrap_or_else(|e| panic!("parse IR: {e:?}"));
    let run = onramp_exec(&m, b"");
    (
        run.status,
        String::from_utf8_lossy(&run.stdout).into_owned(),
    )
}

/// A real ~90-line program: parse a delimited list of numbers, compute summary statistics (mean,
/// variance→stddev via `sqrt`, min/max, median via `qsort`), asserting invariants along the way. It
/// leans on `<string.h>` (`strtok`), `<stdlib.h>` (`strtod`, `qsort`), `<math.h>` (`sqrt`/`fabs`),
/// `<assert.h>`, and `<stdio.h>` (`printf`) together. The dataset is the textbook {2,4,4,4,5,5,7,9}
/// whose mean (5) and stddev (2) are exact, so the output is deterministic.
#[test]
fn stats_pipeline_over_the_expanded_libc() {
    let Some(chibicc) = chibicc_svmb() else {
        eprintln!("SKIP: chibicc.svmb absent");
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
    let Some(chibicc) = chibicc_svmb() else {
        eprintln!("SKIP: chibicc.svmb absent");
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

/// The extended `<string.h>` / `<stdlib.h>` / `<ctype.h>` surface: `strdup`, `strncat`, `strspn`/
/// `strcspn`, `strcasecmp`, `bsearch`, `strtoul` (hex).
#[test]
fn extended_string_and_stdlib() {
    let Some(chibicc) = chibicc_svmb() else {
        eprintln!("SKIP: chibicc.svmb absent");
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
