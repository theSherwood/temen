// ---- §14 spawn: the op-17 v1 record (#1863, #1864) ----
// Appended to every Rust-on-Temen driver guest's source by its test (`support/guest_vm_spawn.rs`), so
// there is one guest-side spawn to keep right. Not a module of the test crate: it is guest code, built
// by `rustc --emit=llvm-ir` with the guest. A pointer in a guest is a window offset, so a record,
// name or payload address is just the Rust pointer.

extern "C" {
    fn __vm_instantiate_rec(inst: i32, rec: i64) -> i64;
}

/// The args area a child reads its payload from (`module_args_end - module_args_base`).
const VM_ARGS_ROOM: usize = 16256;

#[repr(C, align(8))]
struct VmSpawnScratch {
    rec: [u32; 22],
    grants: [u32; 64],
    args: [u8; VM_ARGS_ROOM],
}

static mut VM_SPAWN: VmSpawnScratch = VmSpawnScratch {
    rec: [0; 22],
    grants: [0; 64],
    args: [0; VM_ARGS_ROOM],
};

/// Spawn `module` (a granted `Module` handle, or `-1` for this program) as a detached child: a window
/// of its own — its declared memory (`size_log2` 0) — paid from `budget` and returned to it when the
/// child ends; `grants` re-granted by name (`(name, handle)`, at most 16); `argv` the spawn's args
/// payload, `{argc, envc = 0}` then the NUL-terminated strings the child's `_start` reads. Returns the
/// child handle (join it with `__vm_join`), or -errno.
unsafe fn vm_spawn(inst: i32, budget: i32, module: i32, grants: &[(&[u8], i32)], argv: &[&[u8]]) -> i64 {
    let s = core::ptr::addr_of_mut!(VM_SPAWN);
    let a = core::ptr::addr_of_mut!((*s).args) as *mut u8;
    (a as *mut u32).write(argv.len() as u32);
    (a as *mut u32).add(1).write(0);
    let mut p = 8usize;
    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i];
        if p + arg.len() + 1 > VM_ARGS_ROOM {
            return -7; // -E2BIG
        }
        let mut j = 0;
        while j < arg.len() {
            a.add(p).write(arg[j]);
            p += 1;
            j += 1;
        }
        a.add(p).write(0);
        p += 1;
        i += 1;
    }
    if grants.len() > 16 {
        return -22; // -EINVAL
    }
    let g = core::ptr::addr_of_mut!((*s).grants) as *mut u32;
    let mut i = 0;
    while i < grants.len() {
        let (name, handle) = grants[i];
        g.add(4 * i).write(name.as_ptr() as u32);
        g.add(4 * i + 1).write(name.len() as u32);
        g.add(4 * i + 2).write(handle as u32);
        g.add(4 * i + 3).write(0);
        i += 1;
    }
    // `temen_ir::SpawnRec` v1, in u32 words: version 1 | entry 0, offset 0 (reserved), size_log2 0 (the
    // declared window) | no pager, module | budget, quota 0, grants, args, region -1 (none) | reserved
    // 0, child_off 0.
    let r = core::ptr::addr_of_mut!((*s).rec) as *mut u32;
    let words: [u32; 22] = [
        1, 0, 0, 0, 0, u32::MAX, module as u32, budget as u32, 0, 0,
        g as u32, 0, grants.len() as u32, 0, a as u32, 0, p as u32, 0,
        u32::MAX, 0, 0, 0,
    ];
    let mut k = 0;
    while k < 22 {
        r.add(k).write(words[k]);
        k += 1;
    }
    __vm_instantiate_rec(inst, r as i64)
}
