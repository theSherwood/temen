//! **`temen-link` on the host**: the linker of nimony's Temen backend (`nimony t`,
//! `patches/nimony/temen-backend.patch`), as `scripts/nim-toolchain.sh` builds nimony's own tools
//! with it. The driver finds it beside itself in `bin/` and plans it as a build's last step:
//! `temen-link -o:<prog.temen> <main.c.nif> <dep.c.nif>...`.
//!
//! It is [`temen_leng::link_command`] over the host's files: the function the in-guest `temen-link`
//! is built from (`demos/temen_link`) and that the browser serves natively (`browser/src/nimony.rs`),
//! so every one of them writes the same module from the same `.c.nif`s.
//!
//! The guest libc it links a program against is `--libc:<path>`, else `$TEMEN_PG_LIBC`, else the
//! path the in-guest command reads (`/lib/temen/libc.temeno`).

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if !args.iter().any(|a| a.starts_with("--libc:")) {
        if let Ok(libc) = std::env::var("TEMEN_PG_LIBC") {
            args.push(format!("--libc:{libc}"));
        }
    }
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let (names, sigs) = temen_posix::cap_vtable();
    std::process::exit(temen_leng::link_command(
        &args,
        (&names, &sigs),
        &mut |path| std::fs::read(path).ok(),
        &mut |path, bytes| std::fs::write(path, bytes).is_ok(),
    ));
}
