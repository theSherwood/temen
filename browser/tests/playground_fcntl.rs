//! `<fcntl.h>` in the seeded playground include set (c_interpret#28 cluster F).
//!
//! `unistd.h` has always declared `open`/`read`/`write`/`close` over the `vm_fs` cap, but the flag
//! names lived nowhere, so an ordinary program —
//!
//! ```c
//! #include <fcntl.h>
//! int fd = open("f.txt", O_CREAT | O_WRONLY | O_TRUNC, 0644);
//! ```
//!
//! — died at the *preprocessor* (`fcntl.h: cannot open file`) before reaching anything interesting.
//! That read as a capability gap for a year and was a missing file: 10 spec failures across
//! `mmap-file`, `posix-fd` and `framebuffer`.
//!
//! **Why this test runs the program instead of just compiling it.** The flag values are the fs cap's,
//! not Linux's, so a header that merely *exists* is not enough — one with glibc's numbers would compile
//! and then misbehave silently. The sharp case: on Linux `O_RDONLY` is `0`, a no-op bit pattern, while
//! `temen-fs` gates `readable: flags & O_READ != 0`. A zero `O_RDONLY` therefore yields a fd that is
//! open but unreadable, and the guest's first `read` comes back empty with no error. Only a round trip
//! — write bytes, reopen read-only, read them back — can tell the two apart, so that is what this does.

use temen_browser::{onramp_fs_exec, playground_include_files, STATUS_EXIT, STATUS_OK};

/// The committed in-sandbox compiler. `None` when the asset is absent, which SKIPs — the same
/// convention as `chibicc_link_libc`.
fn chibicc_temen() -> Option<temen_ir::Module> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/web/assets/chibicc.temen");
    let bytes = std::fs::read(p).ok()?;
    Some(temen_encode::decode_module(&bytes).expect("decode chibicc.temen"))
}

/// Compile one TU to a linkable object against the seeded playground headers plus `extra`.
fn emit_object(
    chibicc: &temen_ir::Module,
    path: &str,
    extra: &[(&str, &str)],
    flags: &[&[u8]],
) -> String {
    let mut files: Vec<(String, Vec<u8>)> = playground_include_files();
    for (name, body) in extra {
        files.push((name.to_string(), body.as_bytes().to_vec()));
    }
    let dirs = vec!["include".to_string()];
    let image = temen_fs::encode_image(&files, &dirs);
    let tu = format!("/{path}");
    let mut argv: Vec<&[u8]> = vec![b"chibicc", b"--emit-object", b"-Iinclude"];
    argv.extend_from_slice(flags);
    argv.push(tu.as_bytes());
    let out = onramp_fs_exec(chibicc, &image, &argv, b"");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status == STATUS_OK || out.status == STATUS_EXIT,
        "emit-object {path}: status {} — {}{}",
        out.status,
        stdout,
        String::from_utf8_lossy(&out.stderr),
    );
    // chibicc reports a preprocessor error on *stdout* and exits 1, so a payload that is not IR is the
    // diagnostic itself. Surface it here rather than deep inside the parser — this is the exact shape
    // the missing `fcntl.h` produced (`fcntl.h: cannot open file`).
    assert!(
        !stdout.contains("cannot open file"),
        "{path} did not compile — chibicc said:\n{stdout}"
    );
    stdout
}

/// Compile `src` as a **program unit** against libc declarations only, link it against the prebuilt
/// libc unit, and run it — the same shape the chibicc card and c_interpret use, and the one that
/// resolves the `call.sym`s a whole-program compile would leave dangling. Returns stdout.
fn compile_and_run(src: &str) -> String {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen not built");
        return String::new();
    };
    let lib_ir = emit_object(
        &chibicc,
        "__pg_libc.c",
        &[("__pg_libc.c", temen_browser::playground_libc_tu())],
        &[],
    );
    let prog_ir = emit_object(
        &chibicc,
        "in.c",
        &[("in.c", src)],
        temen_browser::PG_DECLS_ONLY_ARGV,
    );
    let lib = temen_text::parse_module(&lib_ir).expect("libc unit parses");
    let prog = temen_text::parse_module(&prog_ir).expect("user unit parses");
    let out = temen_browser::link_run_units(&lib, &prog, "main", b"");
    assert!(
        out.status == STATUS_OK || out.status == STATUS_EXIT,
        "link+run status {} — stderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr),
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn fcntl_h_is_in_the_seeded_include_set() {
    // The cheap half: the header resolves at all. This is the exact failure c_interpret saw.
    let names: Vec<String> = playground_include_files()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert!(
        names.iter().any(|n| n == "include/fcntl.h"),
        "no include/fcntl.h among {} seeded headers — `#include <fcntl.h>` cannot resolve",
        names.len()
    );
}

#[test]
fn the_open_flags_are_the_fs_caps_values_not_linuxs() {
    // Pinned against `temen-fs`'s constants. A copy of glibc's numbers would break every one of these,
    // and `O_RDONLY` is the one that fails *silently* at runtime rather than loudly at compile time.
    let src = r#"
#include <fcntl.h>
int printf(const char *fmt, ...);
int main(void) {
  printf("%d %d %d %d %d %d\n", O_RDONLY, O_WRONLY, O_RDWR, O_APPEND, O_TRUNC, O_CREAT);
  return 0;
}
"#;
    let out = compile_and_run(src);
    if out.is_empty() {
        return; // SKIP: no chibicc asset
    }
    assert!(
        out.contains("1 2 3 4 8 16"),
        "flags must be temen-fs's (O_READ 1, O_WRITE 2, APPEND 4, TRUNC 8, CREATE 16), got: {out:?}"
    );
}

#[test]
fn a_file_written_through_the_fs_cap_reads_back_through_a_read_only_open() {
    // The round trip the values actually have to satisfy. `O_RDONLY` must *request* read access — if it
    // were Linux's 0 the reopen would yield an unreadable fd and `n` would come back 0 with the buffer
    // untouched, which is exactly the silent failure this guards.
    let src = r#"
#include <fcntl.h>
int printf(const char *fmt, ...);
int main(void) {
  int fd = open("rt.txt", O_CREAT | O_WRONLY | O_TRUNC, 0644);
  if (fd < 0) { printf("open-for-write failed: %d\n", fd); return 1; }
  write(fd, "HELLO", 5);
  close(fd);

  fd = open("rt.txt", O_RDONLY);
  if (fd < 0) { printf("reopen-read-only failed: %d\n", fd); return 2; }
  char buf[6];
  int n = read(fd, buf, 5);
  close(fd);
  buf[n < 0 ? 0 : n] = 0;
  printf("n=%d buf=%s\n", n, buf);
  return 0;
}
"#;
    let out = compile_and_run(src);
    if out.is_empty() {
        return; // SKIP: no chibicc asset
    }
    assert!(
        out.contains("n=5 buf=HELLO"),
        "a write/reopen/read round trip through the fs cap must return the bytes; got {out:?}"
    );
}

#[test]
fn o_trunc_empties_an_existing_file() {
    // `O_TRUNC` is 8 here and 512 on Linux, so a wrong value shows up as a file that keeps its old
    // contents — again silent. Write "LONGER", truncate, write "HI", and the tail must be gone.
    let src = r#"
#include <fcntl.h>
int printf(const char *fmt, ...);
int main(void) {
  int fd = open("tr.txt", O_CREAT | O_WRONLY | O_TRUNC, 0644);
  write(fd, "LONGER", 6);
  close(fd);

  fd = open("tr.txt", O_WRONLY | O_TRUNC);
  write(fd, "HI", 2);
  close(fd);

  fd = open("tr.txt", O_RDONLY);
  char buf[16];
  int n = read(fd, buf, 15);
  close(fd);
  buf[n < 0 ? 0 : n] = 0;
  printf("n=%d buf=%s\n", n, buf);
  return 0;
}
"#;
    let out = compile_and_run(src);
    if out.is_empty() {
        return; // SKIP: no chibicc asset
    }
    assert!(
        out.contains("n=2 buf=HI"),
        "O_TRUNC must empty the file before the second write; got {out:?} (a stale tail means the value is wrong)"
    );
}

#[test]
fn lseek_overwrites_in_place_and_write_past_two_is_not_stdout() {
    // Two things at once, because they are the same bug. `lseek` has to reach the cap's `SEEK_SET`
    // (the second write must land *inside* the file, not append), and `write(fd, …)` for fd >= 3 has
    // to reach the memfs rather than the ambient Stream. The frontend's `write`/`read` builtins drop
    // the fd and always use the stdout/stdin handle, so before the fd-dispatching definitions in
    // `unistd.h` the bytes went to the console and the file stayed empty — a failure that *looks*
    // like output and leaves nothing behind to read back. Asserting "ABXYEF" on stdout and nothing
    // stray pins both halves.
    let src = r#"
#include <fcntl.h>
#include <unistd.h>
int printf(const char *fmt, ...);
int main(void) {
  int fd = open("sk.txt", O_CREAT | O_WRONLY | O_TRUNC, 0644);
  write(fd, "ABCDEF", 6);
  close(fd);

  fd = open("sk.txt", O_RDWR);
  lseek(fd, 2, SEEK_SET);
  write(fd, "XY", 2);
  close(fd);

  fd = open("sk.txt", O_RDONLY);
  char buf[8];
  int n = read(fd, buf, 6);
  close(fd);
  buf[n < 0 ? 0 : n] = 0;
  printf("[%s]\n", buf);
  return 0;
}
"#;
    let out = compile_and_run(src);
    if out.is_empty() {
        return; // SKIP: no chibicc asset
    }
    assert!(
        out.contains("[ABXYEF]"),
        "lseek(SEEK_SET) + write must overwrite in place; got {out:?}"
    );
    // The guest wrote 8 bytes to file fds. Had any of them taken the fd-less Stream builtin they
    // would be sitting in stdout beside the printf, so stdout must be *only* the printf.
    assert_eq!(
        out, "[ABXYEF]\n",
        "stdout must carry the printf alone \u{2014} anything else is file-fd bytes on the Stream cap"
    );
}
