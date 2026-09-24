//! The POSIX personality's **import vocabulary** (#1668): what a program imports to reach
//! `temen_posix` — each op's `__px_<name>` and signature, in op order. `temen_posix` serves it (its
//! `resolve` maps the same names to the same op numbers, pinned by its tests); a linker that binds a
//! program to the personality links against it. One table, so the two cannot drift.

use temen_ir::{FuncType, ValType::I64};

/// `(bare name, i64 argument count)`, one per op. **Order is the op number.**
pub const OPS: &[(&str, usize)] = &[
    ("write", 3),        // 0
    ("read", 3),         // 1
    ("malloc", 1),       // 2
    ("free", 1),         // 3
    ("exit", 0),         // 4 (special-cased in `vtable`: `(i64) -> ()`)
    ("open", 3),         // 5
    ("close", 1),        // 6
    ("lseek", 3),        // 7
    ("unlink", 2),       // 8
    ("getcwd", 2),       // 9
    ("chdir", 2),        // 10
    ("getenv", 2),       // 11
    ("setenv", 5),       // 12
    ("stat", 3),         // 13
    ("opendir", 2),      // 14
    ("readdir", 3),      // 15
    ("closedir", 1),     // 16
    ("argc", 0),         // 17
    ("argv", 3),         // 18
    ("exec_lookup", 2),  // 19
    ("exec_stdout", 0),  // 20
    ("exec_stdin", 2),   // 21
    ("exec_win", 1),     // 22
    ("pipe", 1),         // 23
    ("dup2", 2),         // 24
    ("dup", 1),          // 25
    ("fcntl", 3),        // 26
    ("spawn", 4),        // 27
    ("waitpid", 3),      // 28
    ("wait", 1),         // 29
    ("signal", 2),       // 30
    ("kill", 2),         // 31
    ("sigcheck", 1),     // 32
    ("clock", 1),        // 33
    ("getenv_r", 4),     // 34
    ("unsetenv", 2),     // 35
    ("environ", 3),      // 36
    ("mkdir", 3),        // 37
    ("rename", 4),       // 38
    ("rmdir", 2),        // 39
    ("sigprocmask", 3),  // 40
    ("sigaction", 3),    // 41
    ("sigaltstack", 2),  // 42
    ("spawn2", 1),       // 43
    ("getpid", 0),       // 44
    ("setpgid", 2),      // 45
    ("getpgid", 1),      // 46
    ("tcgetpgrp", 1),    // 47
    ("tcsetpgrp", 2),    // 48
    ("isatty", 1),       // 49
    ("getppid", 0),      // 50
    ("fork", 1),         // 51
    ("pipe_adopt", 3),   // 52
    ("exec_resolve", 2), // 53
    ("tcgetattr", 2),    // 54
    ("tcsetattr", 2),    // 55
    ("tcgetwinsize", 2), // 56
    ("ttyname", 3),      // 57
    ("fstat", 2),        // 58
    ("statp", 3),        // 59
    ("execve", 3),       // 60
    ("wait4", 4),        // 61
];

/// The vocabulary as `(import names, signatures)`, op-ordered: `__px_<name>` taking its `i64`
/// arguments and returning one `i64` — except `exit`, `(i64) -> ()`.
pub fn vtable() -> (Vec<String>, Vec<FuncType>) {
    OPS.iter()
        .map(|&(name, nargs)| {
            let sig = if name == "exit" {
                FuncType {
                    params: vec![I64],
                    results: Vec::new(),
                }
            } else {
                FuncType {
                    params: vec![I64; nargs],
                    results: vec![I64],
                }
            };
            (format!("__px_{name}"), sig)
        })
        .unzip()
}
