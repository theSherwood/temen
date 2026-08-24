//! The playground's **threading headers** (INTERACTIVE_EMBEDDING.md slice 9): `<pthread.h>` +
//! `<semaphore.h>` are seeded into the C-compiler card's `/include` set (from the frontend's
//! bundled copies — one source of truth), so a real pthreads/semaphore program compiles in the
//! sandboxed chibicc exactly as it does under the native frontend. Native tests can't spin the
//! cross-Worker vCPU host, so this proves the *compile* pipeline end to end — the headers parse,
//! the `__vm_*` builtins lower to the thread/atomic/futex ops, and the emitted IR parses back;
//! *execution* parity (interp == JIT) for the same fixtures is `temen/tests/c_frontend.rs`.
//! Fail-soft: SKIPs if `chibicc.temen` isn't built.

use temen_browser::{onramp_fs_exec, playground_include_files, STATUS_EXIT, STATUS_OK};

fn chibicc_temen() -> Option<temen_ir::Module> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/web/assets/chibicc.temen");
    let bytes = std::fs::read(p).ok()?;
    Some(temen_encode::decode_module(&bytes).expect("decode chibicc.temen"))
}

/// Compile `src` with the seeded playground headers and return the emitted IR text.
fn compile(chibicc: &temen_ir::Module, src: &str) -> String {
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
    String::from_utf8(compiled.stdout).expect("IR is utf8")
}

/// The bounded producer/consumer + a barrier phase — the same shapes the native-frontend fixtures
/// run — compiled by chibicc-the-guest against the seeded `/include` set. The IR must carry the
/// real primitives (`thread.spawn`, the futex `memory.wait`/`notify` lowering, the atomic RMW),
/// and parse back as a module.
#[test]
fn pthread_and_semaphore_programs_compile_in_the_playground() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let ir = compile(
        &chibicc,
        r#"
#include <pthread.h>
#include <semaphore.h>

static int buf[4];
static int head = 0, tail = 0;
static sem_t items, slots;
static pthread_barrier_t bar;

static void *producer(void *arg) {
  (void)arg;
  for (int i = 1; i <= 8; i++) {
    sem_wait(&slots);
    buf[head & 3] = i;
    head++;
    sem_post(&items);
  }
  pthread_barrier_wait(&bar);
  return 0;
}

int main(void) {
  sem_init(&items, 0, 0);
  sem_init(&slots, 0, 4);
  pthread_barrier_init(&bar, 0, 2);
  pthread_t t;
  pthread_create(&t, 0, producer, 0);
  int sum = 0;
  for (int n = 0; n < 8; n++) {
    sem_wait(&items);
    sum += buf[tail & 3];
    tail++;
    sem_post(&slots);
  }
  pthread_barrier_wait(&bar);
  pthread_join(t, 0);
  return sum;
}
"#,
    );
    for op in ["thread.spawn", "thread.join", "atomic"] {
        assert!(ir.contains(op), "the emitted IR carries `{op}`:\n{ir:.400}");
    }
    let m = temen_text::parse_module(&ir).unwrap_or_else(|e| panic!("parse IR: {e:?}"));
    assert!(!m.funcs.is_empty(), "a non-empty module");
}

/// The playground's **real** `<stdatomic.h>` (INTERACTIVE_EMBEDDING.md): `atomic_fetch_add` /
/// `atomic_load` / `atomic_store` lower to the VM's genuine atomic RMW ops (not plain `*p += v`),
/// so a lock-free counter is correct under any interleaving. Compiled through chibicc-the-guest
/// against the seeded header, the emitted IR must carry the atomic ops.
#[test]
fn stdatomic_lowers_to_real_atomics_in_the_playground() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let ir = compile(
        &chibicc,
        r#"
#include <stdatomic.h>
atomic_int counter;
int main(void) {
  atomic_store(&counter, 0);
  for (int i = 0; i < 20; i++) atomic_fetch_add(&counter, 1);
  return atomic_load(&counter);
}
"#,
    );
    assert!(
        ir.contains("atomic.rmw.add"),
        "atomic_fetch_add lowered to a real atomic RMW (not plain +=):\n{ir:.400}"
    );
    let m = temen_text::parse_module(&ir).unwrap_or_else(|e| panic!("parse IR: {e:?}"));
    assert!(!m.funcs.is_empty(), "a non-empty module");
}
