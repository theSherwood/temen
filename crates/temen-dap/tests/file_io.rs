//! #1323 (c_interpret #16, file I/O) slice 2 — the **debug** on-ramp mounts the same private,
//! in-memory read-write memfs the browser Run path does, so a debugged program that does file I/O
//! (chibicc's `__vm_fs` builtin → `call.sym "vm_fs"`, op-in-arg0) can open/write/seek/read a scratch
//! file under the DAP bytecode powerbox. Without the mount + `vm_fs` binding in `grant_io_powerbox`,
//! `call.sym "vm_fs"` is an unbound import and the program traps at the first file op.

use temen_dap::{DapServer, Json};

mod support;
use support::{req, response};

/// Open "f" (READ|WRITE|CREATE = 1|2|16 = 19, so the one fd both writes and reads) → fd, write the
/// byte 'A' from mem[16440], seek to 0, read one
/// byte back into mem[16460], then store the read byte at mem[16392] (cell 8, above the #1094 NULL
/// guard) so `readMemory` can witness the round-trip. Every file op rides one `call.sym "vm_fs"` with
/// the fs op in arg0 (FS_OPEN=0, FS_READ=1, FS_WRITE=2, FS_SEEK=3), exactly as the compiled-C libc emits.
const FILE_IO: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vzero = i64.const 0
  vone = i64.const 1
  vpath = i64.const 16400
  vcf = i32.const 102
  i32.store8 vpath vcf
  vnp = i64.const 16420
  vcv = i32.const 118
  i32.store8 vnp vcv
  vn1 = i64.const 16421
  vcm = i32.const 109
  i32.store8 vn1 vcm
  vn2 = i64.const 16422
  vcu = i32.const 95
  i32.store8 vn2 vcu
  vn3 = i64.const 16423
  vcf2 = i32.const 102
  i32.store8 vn3 vcf2
  vn4 = i64.const 16424
  vcs = i32.const 115
  i32.store8 vn4 vcs
  vwbuf = i64.const 16440
  vbA = i32.const 65
  i32.store8 vwbuf vbA
  vnl = i64.const 5
  vh = self.resolve vnp vnl
  vopen = i64.const 0
  vplen = i64.const 1
  vflags = i64.const 19
  vfd = call.sym "vm_fs" (i64, i64, i64, i64, i64) -> (i64) vh (vopen, vpath, vplen, vflags, vzero)
  vwrite = i64.const 2
  vwn = call.sym "vm_fs" (i64, i64, i64, i64, i64) -> (i64) vh (vwrite, vfd, vwbuf, vone, vzero)
  vseek = i64.const 3
  vsk = call.sym "vm_fs" (i64, i64, i64, i64, i64) -> (i64) vh (vseek, vfd, vzero, vzero, vzero)
  vread = i64.const 1
  vrbuf = i64.const 16460
  vrn = call.sym "vm_fs" (i64, i64, i64, i64, i64) -> (i64) vh (vread, vfd, vrbuf, vone, vzero)
  vbyte = i64.load vrbuf
  vcell = i64.const 16392
  i64.store vcell vbyte
  return vbyte
  }
}
"#;

fn launch(s: &mut DapServer, src: &str) {
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let out = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(src)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("engine", Json::s("bytecode")),
            ("powerbox", Json::s("onramp")),
        ]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "launch ok"
    );
}

/// A scratch file written then read back through the `vm_fs` cap the debug on-ramp now grants: the
/// byte 'A' (0x41 = 65) round-trips, landing at mem[16392]. Had `grant_io_powerbox` not mounted the
/// memfs and bound `vm_fs`, the first `call.sym "vm_fs"` would trap (unbound import) and the run would
/// terminate without a stored byte.
#[test]
fn debug_onramp_serves_file_io_over_vm_fs() {
    let mut s = DapServer::new();
    launch(&mut s, FILE_IO);
    let out = s.handle(&req(3, "continue", Json::obj(vec![])));
    assert!(
        out.iter()
            .any(|m| m.get("event").and_then(|e| e.as_str()) == Some("terminated")),
        "the file-I/O program ran to completion (no trap on vm_fs)"
    );
    let out = s.handle(&req(
        4,
        "readMemory",
        Json::obj(vec![
            ("memoryReference", Json::s("16392")),
            ("count", Json::i(8)),
        ]),
    ));
    // 65 (0x41) little-endian over 8 bytes: the read byte 'A' followed by zeros.
    assert_eq!(
        response(&out)
            .get("body")
            .and_then(|b| b.get("data"))
            .cloned(),
        Some(Json::s("QQAAAAAAAAA=")),
        "the byte written to the scratch file reads back through the memfs"
    );
}

/// #1323 slice 3 — a program reads a file the launch **pre-seeded** into the memfs via `fsImage`.
/// Open "d" (READ), read one byte into mem[16460], store it at mem[16392]. The seeded byte 'Z'
/// (0x5A = 90) reads back. Without the `fsImage` seed the memfs would be empty, `open` would return
/// ENOENT, and the stored byte would be 0.
const READ_SEEDED: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vzero = i64.const 0
  vone = i64.const 1
  vpath = i64.const 16400
  vcd = i32.const 100
  i32.store8 vpath vcd
  vnp = i64.const 16420
  vcv = i32.const 118
  i32.store8 vnp vcv
  vn1 = i64.const 16421
  vcm = i32.const 109
  i32.store8 vn1 vcm
  vn2 = i64.const 16422
  vcu = i32.const 95
  i32.store8 vn2 vcu
  vn3 = i64.const 16423
  vcf = i32.const 102
  i32.store8 vn3 vcf
  vn4 = i64.const 16424
  vcs = i32.const 115
  i32.store8 vn4 vcs
  vnl = i64.const 5
  vh = self.resolve vnp vnl
  vopen = i64.const 0
  vplen = i64.const 1
  vread_flag = i64.const 1
  vfd = call.sym "vm_fs" (i64, i64, i64, i64, i64) -> (i64) vh (vopen, vpath, vplen, vread_flag, vzero)
  vread = i64.const 1
  vrbuf = i64.const 16460
  vrn = call.sym "vm_fs" (i64, i64, i64, i64, i64) -> (i64) vh (vread, vfd, vrbuf, vone, vzero)
  vbyte = i64.load vrbuf
  vcell = i64.const 16392
  i64.store vcell vbyte
  return vbyte
  }
}
"#;

/// Minimal RFC 4648 base64 (standard alphabet, padded) — the `fsImage` launch-arg encoding.
fn b64(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        out.push(A[(n >> 18 & 63) as usize] as char);
        out.push(A[(n >> 12 & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            A[(n >> 6 & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            A[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

#[test]
fn debug_onramp_seeds_the_memfs_from_the_launch_fs_image() {
    // Seed the memfs with a file "d" = "Z" (0x5A) via the fsImage launch arg.
    let image = temen_fs::encode_image(&[("d".to_string(), vec![0x5A])], &[]);
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let out = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(READ_SEEDED)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("engine", Json::s("bytecode")),
            ("powerbox", Json::s("onramp")),
            ("fsImage", Json::s(b64(&image))),
        ]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "launch ok"
    );
    let out = s.handle(&req(3, "continue", Json::obj(vec![])));
    assert!(
        out.iter()
            .any(|m| m.get("event").and_then(|e| e.as_str()) == Some("terminated")),
        "ran to completion"
    );
    let out = s.handle(&req(
        4,
        "readMemory",
        Json::obj(vec![
            ("memoryReference", Json::s("16392")),
            ("count", Json::i(8)),
        ]),
    ));
    // 90 (0x5A) little-endian over 8 bytes.
    assert_eq!(
        response(&out)
            .get("body")
            .and_then(|b| b.get("data"))
            .cloned(),
        Some(Json::s("WgAAAAAAAAA=")),
        "the seeded file's byte reads back through the memfs"
    );
}
