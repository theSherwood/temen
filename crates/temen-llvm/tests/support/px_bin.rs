//! The `/bin` a spawning test runs its children from: small text-IR programs, each a command's
//! `(i64) -> (i64)` entry returning its exit status. Their stdio goes through the POSIX personality's
//! `write`/`read`, and a core pipe end's tag is re-issued on the end itself, as a libc shim does
//! ([`PX_IO`]). A test grants each module on its run's host and registers it at its path.

/// Func 1 writes `(fd, buf, len)` and func 2 reads, each through the personality, re-issuing a tag
/// (`<= -(1 << 20)`, a core pipe end's handle) on the end.
const PX_IO: &str = "func (i64, i64, i64) -> (i64) {
block 0 (vfd: i64, vbuf: i64, vlen: i64) {
  vr = call.import 0 (vfd, vbuf, vlen)
  vtagb = i64.const -1048576
  vtag = i64.le_s vr vtagb
  br_if vtag 1(vr, vbuf, vlen) 2(vr)
  }
block 1 (vr: i64, vbuf: i64, vlen: i64) {
  vtagb = i64.const -1048576
  vh = i64.sub vtagb vr
  vh32 = i32.wrap_i64 vh
  vw = call.cap 0 1 (i64, i64) -> (i64) vh32 (vbuf, vlen)
  return vw
  }
block 2 (vr: i64) {
  return vr
  }
}
func (i64, i64, i64) -> (i64) {
block 0 (vfd: i64, vbuf: i64, vlen: i64) {
  vr = call.import 1 (vfd, vbuf, vlen)
  vtagb = i64.const -1048576
  vtag = i64.le_s vr vtagb
  br_if vtag 1(vr, vbuf, vlen) 2(vr)
  }
block 1 (vr: i64, vbuf: i64, vlen: i64) {
  vtagb = i64.const -1048576
  vh = i64.sub vtagb vr
  vh32 = i32.wrap_i64 vh
  vn = call.cap 0 0 (i64, i64) -> (i64) vh32 (vbuf, vlen)
  return vn
  }
block 2 (vr: i64) {
  return vr
  }
}
";

/// The window and the imports every program declares: [`PX_IO`]'s, then `argc`/`argv`.
const PX_IMPORTS: &str = "memory 17
import 0 \"__px_write\" (i64, i64, i64) -> (i64)
import 1 \"__px_read\" (i64, i64, i64) -> (i64)
import 2 \"__px_argc\" () -> (i64)
import 3 \"__px_argv\" (i64, i64, i64) -> (i64)
";

/// `echo`: its arguments, space-separated, then a newline.
const ECHO: &str = "data 40000 \" \\n\"
func (i64) -> (i64) {
block 0 (v0: i64) {
  vn = call.import 2 ()
  vone = i64.const 1
  br 1(vone, vn)
  }
block 1 (vi: i64, vn: i64) {
  vmore = i64.lt_s vi vn
  br_if vmore 2(vi, vn) 5()
  }
block 2 (vi: i64, vn: i64) {
  vone = i64.const 1
  vfirst = i64.eq vi vone
  br_if vfirst 4(vi, vn) 3(vi, vn)
  }
block 3 (vi: i64, vn: i64) {
  vfd = i64.const 1
  vsp = i64.const 40000
  vl = i64.const 1
  vw = call 1 (vfd, vsp, vl)
  br 4(vi, vn)
  }
block 4 (vi: i64, vn: i64) {
  vbuf = i64.const 41000
  vcap = i64.const 4096
  vlen = call.import 3 (vi, vbuf, vcap)
  vfd = i64.const 1
  vw = call 1 (vfd, vbuf, vlen)
  vone = i64.const 1
  vi2 = i64.add vi vone
  br 1(vi2, vn)
  }
block 5 () {
  vfd = i64.const 1
  vnl = i64.const 40001
  vl = i64.const 1
  vw = call 1 (vfd, vnl, vl)
  vz = i64.const 0
  return vz
  }
}
";

/// `cat`: its stdin to its stdout, until EOF.
const CAT: &str = "func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
  }
block 1 () {
  vfd = i64.const 0
  vbuf = i64.const 41000
  vcap = i64.const 4096
  vn = call 2 (vfd, vbuf, vcap)
  vz = i64.const 0
  vmore = i64.gt_s vn vz
  br_if vmore 2(vn) 3()
  }
block 2 (vn: i64) {
  vfd = i64.const 1
  vbuf = i64.const 41000
  vw = call 1 (vfd, vbuf, vn)
  br 1()
  }
block 3 () {
  vz = i64.const 0
  return vz
  }
}
";

/// `up`: its stdin to its stdout uppercased, until EOF; exits 42, a status its reaper must see.
const UP: &str = "func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
  }
block 1 () {
  vfd = i64.const 0
  vbuf = i64.const 41000
  vcap = i64.const 4096
  vn = call 2 (vfd, vbuf, vcap)
  vz = i64.const 0
  vmore = i64.gt_s vn vz
  br_if vmore 2(vn, vz) 4()
  }
block 2 (vn: i64, vi: i64) {
  vlt = i64.lt_s vi vn
  br_if vlt 3(vn, vi) 5(vn)
  }
block 3 (vn: i64, vi: i64) {
  vbase = i64.const 41000
  vaddr = i64.add vbase vi
  vc = i32.load8_u vaddr
  va = i32.const 97
  vzz = i32.const 122
  vge = i32.ge_u vc va
  vle = i32.le_u vc vzz
  vlow = i32.and vge vle
  vd = i32.const 32
  vup = i32.sub vc vd
  vout = select vlow vup vc
  i32.store8 vaddr vout
  vone = i64.const 1
  vi2 = i64.add vi vone
  br 2(vn, vi2)
  }
block 4 () {
  v42 = i64.const 42
  return v42
  }
block 5 (vn: i64) {
  vfd = i64.const 1
  vbuf = i64.const 41000
  vw = call 1 (vfd, vbuf, vn)
  br 1()
  }
}
";

/// `gen`: `hello` on its stdout.
const GEN: &str = "data 40000 \"hello\"
func (i64) -> (i64) {
block 0 (v0: i64) {
  vfd = i64.const 1
  vbuf = i64.const 40000
  vl = i64.const 5
  vw = call 1 (vfd, vbuf, vl)
  vz = i64.const 0
  return vz
  }
}
";

/// `noisy`: a line on stdout and a line on stderr.
const NOISY: &str = "data 40000 \"on stdout\\n\"
data 40016 \"on stderr\\n\"
func (i64) -> (i64) {
block 0 (v0: i64) {
  vout = i64.const 1
  vob = i64.const 40000
  vl = i64.const 10
  vw = call 1 (vout, vob, vl)
  verr = i64.const 2
  veb = i64.const 40016
  vw2 = call 1 (verr, veb, vl)
  vz = i64.const 0
  return vz
  }
}
";

/// A program that only exits with `code`: `true` and `false`.
fn exits(code: u8) -> String {
    format!(
        "func (i64) -> (i64) {{\nblock 0 (v0: i64) {{\n  vc = i64.const {code}\n  return vc\n  }}\n}}\n"
    )
}

/// The `/bin`, as `(path, text IR)`.
pub fn programs() -> Vec<(&'static str, String)> {
    let prog = |body: &str| format!("{PX_IMPORTS}{body}{PX_IO}");
    vec![
        ("/bin/echo", prog(ECHO)),
        ("/bin/cat", prog(CAT)),
        ("/bin/up", prog(UP)),
        ("/bin/gen", prog(GEN)),
        ("/bin/noisy", prog(NOISY)),
        ("/bin/true", prog(&exits(0))),
        ("/bin/false", prog(&exits(1))),
    ]
}

/// Grant each of [`programs`] on `host` and register it at its path in `posix`.
pub fn stage(host: &mut temen_interp::Host, posix: &temen_posix::Posix) {
    for (path, ir) in programs() {
        let m = temen_text::parse_module(&ir).expect("parse a /bin program");
        temen_verify::verify_module(&m).expect("a /bin program verifies");
        let wl = m
            .memory
            .expect("a /bin program declares its window")
            .size_log2;
        let h = host.grant_module(&m);
        posix.register_executable(path, h, wl);
    }
}
