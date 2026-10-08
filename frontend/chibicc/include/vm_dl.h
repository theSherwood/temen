#ifndef VM_DL_H
#define VM_DL_H
// In-guest dynamic-linking loader — `vm_dlopen`/`vm_dlsym`/`vm_dlclose` over the `Jit` capability
// (DESIGN.md §22). A "shared object" is serialized Temen IR; a symbol is an installed `call.dyn`
// slot (an **unforgeable funcref**, §3c-checked at the call). This header is the ergonomic layer over
// the raw `__vm_jit_compile_linked` / `__vm_jit_install` primitives: it keeps a registry of loaded
// symbols and marshals it into the symbol-table buffer the host resolves against, so a loaded unit can
// reference any already-loaded symbol **by name** — and a mis-link is caught by the host's
// re-verification, never trusted.
//
// A **link unit** (chibicc `--emit-object`) may carry globals of its own (#2167). The loader asks the
// host how much room the unit's data needs (`__vm_jit_unit_info`) and reserves it with `malloc`; the
// host relocates the unit to that room and writes its initial data there. Each load gets its own room,
// so a hot-reloaded version keeps its own state. The unit's exported globals join the registry as data
// symbols, which a later unit can reference by name (`extern`), as can a global the program publishes
// with `vm_dl_export`. The room is `malloc`'s 16-byte alignment, so a global aligned beyond 16 bytes
// keeps its offset in the unit but not that alignment. Thread-locals and function pointers in a unit's
// data are refused (-22).
//
// Why this is better than POSIX `dlopen`: the object is re-verified on load (a malicious one can't
// escape — worst case it corrupts its own window), loading is capability-gated (you need the `Jit`
// handle, and the IR arrives through the powerbox — no ambient "load any file"), and `dlsym` yields a
// checked funcref slot, not a raw pointer.
#include <stdlib.h>
#include <temen.h>

#ifndef VM_DL_MAX
#define VM_DL_MAX 64 // max simultaneously-loaded symbols (units and data symbols)
#endif
#ifndef VM_DL_NAMEMAX
#define VM_DL_NAMEMAX 32 // max symbol-name length (incl. NUL)
#endif

typedef struct {
  char name[VM_DL_NAMEMAX];
  int used;
  int data;   // 0: a loaded unit (`slot`, `code`, `addr`); 1: a data symbol (`addr`, `owner`)
  int owner;  // a data symbol: the registry index of the unit that exports it; -1 for the program's
  long slot;  // the installed call.dyn slot (the funcref other units link to)
  long code;  // the Jit code handle (for invoking the symbol directly, via __vm_jit_invoke2)
  char *addr; // a unit: its data room (NULL if it has no data); a data symbol: its address
} VmDlSym;

static VmDlSym vm_dl_reg[VM_DL_MAX];
static int vm_dl_count;

static int vm_dl_streq(const char *a, const char *b) {
  while (*a && *a == *b) {
    a++;
    b++;
  }
  return *a == *b;
}

static int vm_dl_strlen(const char *s) {
  int n = 0;
  while (s[n])
    n++;
  return n;
}

// The registry index of the loaded unit (`data` 0) or data symbol (`data` 1) named `name`, or -1.
static int vm_dl_find(const char *name, int data) {
  for (int i = 0; i < VM_DL_MAX; i++)
    if (vm_dl_reg[i].used && vm_dl_reg[i].data == data && vm_dl_streq(vm_dl_reg[i].name, name))
      return i;
  return -1;
}

// The row `name` is registered in: its existing one — a reload, which the caller overwrites — or a
// free one. -1 if the registry is full.
static int vm_dl_put(const char *name, int data) {
  int i = vm_dl_find(name, data);
  if (i >= 0)
    return i;
  for (i = 0; i < VM_DL_MAX; i++) {
    if (vm_dl_reg[i].used)
      continue;
    int k = 0;
    while (name[k] && k < VM_DL_NAMEMAX - 1) {
      vm_dl_reg[i].name[k] = name[k];
      k++;
    }
    vm_dl_reg[i].name[k] = 0;
    vm_dl_reg[i].data = data;
    vm_dl_reg[i].used = 1;
    vm_dl_count++;
    return i;
  }
  return -1;
}

// Append an unsigned LEB128 to `buf` at `*pos`.
static void vm_dl_uleb(char *buf, long *pos, unsigned long v) {
  for (;;) {
    int b7 = v & 0x7f;
    v >>= 7;
    buf[(*pos)++] = (char)(v ? (b7 | 0x80) : b7);
    if (!v)
      return;
  }
}

// Read an unsigned LEB128 from `buf` at `*pos` (the host's well-formed `unit_info` reply).
static unsigned long vm_dl_uleb_get(const char *buf, long *pos) {
  unsigned long v = 0;
  for (int shift = 0;; shift += 7) {
    int b = (unsigned char)buf[(*pos)++];
    v |= (unsigned long)(b & 0x7f) << shift;
    if (!(b & 0x80))
      return v;
  }
}

// Marshal the whole registry into a `compile_linked` symbol table (DESIGN.md §22): `count`, then per
// row a `name` (uleb len + bytes), a `kind` byte and its payload (uleb) — `0` a unit's table slot, `2`
// a data symbol's address, and `3` (unnamed) `room`, where the unit's own data goes, when it has data.
// Passing the *whole* registry is fine — the host only binds the symbols a unit actually references.
static long vm_dl_build_symtab(char *buf, char *room) {
  long pos = 0;
  vm_dl_uleb(buf, &pos, (unsigned long)vm_dl_count + (room != 0));
  for (int i = 0; i < VM_DL_MAX; i++) {
    if (!vm_dl_reg[i].used)
      continue;
    int nlen = vm_dl_strlen(vm_dl_reg[i].name);
    vm_dl_uleb(buf, &pos, (unsigned long)nlen);
    for (int k = 0; k < nlen; k++)
      buf[pos++] = vm_dl_reg[i].name[k];
    buf[pos++] = vm_dl_reg[i].data ? 2 : 0;
    vm_dl_uleb(buf, &pos,
               vm_dl_reg[i].data ? (unsigned long)vm_dl_reg[i].addr : (unsigned long)vm_dl_reg[i].slot);
  }
  if (room) {
    vm_dl_uleb(buf, &pos, 0); // unnamed
    buf[pos++] = 3;
    vm_dl_uleb(buf, &pos, (unsigned long)room);
  }
  return pos;
}

// `vm_dlsym(name)` → the symbol's installed slot (a funcref another unit can `call.dyn`), or
// `-1` if it is not loaded.
static long vm_dlsym(const char *name) {
  int i = vm_dl_find(name, 0);
  return i < 0 ? -1 : vm_dl_reg[i].slot;
}

// `vm_dlsym_data(name)` → the address of a data symbol — a loaded unit's exported global, or one the
// program published with `vm_dl_export` — or NULL if none is loaded under that name.
static void *vm_dlsym_data(const char *name) {
  int i = vm_dl_find(name, 1);
  return i < 0 ? 0 : vm_dl_reg[i].addr;
}

// `vm_dl_export(name, addr)`: publish a global of the program's own under `name`, so a unit loaded
// later can reference it by name (an `extern` global). Returns 0, or -12 if the registry is full.
static int vm_dl_export(const char *name, void *addr) {
  int i = vm_dl_put(name, 1);
  if (i < 0)
    return -12;
  vm_dl_reg[i].addr = addr;
  vm_dl_reg[i].owner = -1;
  return 0;
}

// `vm_dlopen(name, ir, ir_len)`: load a unit (serialized Temen IR) that may reference already-loaded
// symbols **by name**. Reserve room for its data, resolve its references against the registry, compile
// (the host re-verifies), install it into the shared table, and register it under `name` and its
// exported globals under theirs. Returns the slot (>= 0), or a negative errno (-22 link/verify failed,
// -14 its room is not writable, -28 table full, -12 out of memory or registry full). Idempotent names
// are the caller's concern: a repeat `name` **hot-reloads** it (below).
static long vm_dlopen(const char *name, const void *ir, long ir_len) {
  static char symtab[(VM_DL_MAX + 1) * (VM_DL_NAMEMAX + 12) + 10];
  // What the unit's data needs: its room, then its exported globals and their offsets in that room.
  long n = __vm_jit_unit_info((void *)ir, ir_len, 0, 0); // the reply's length; nothing is written
  if (n < 0)
    return n;
  char *info = malloc(n);
  if (!info)
    return -12;
  __vm_jit_unit_info((void *)ir, ir_len, info, n);
  long pos = 0;
  unsigned long span = vm_dl_uleb_get(info, &pos);
  char *room = span ? malloc(span) : 0;
  long slot = span && !room ? -12 : 0;
  long code = 0;
  if (!slot) {
    long st_len = vm_dl_build_symtab(symtab, room);
    code = __vm_jit_compile_linked((void *)ir, ir_len, symtab, st_len);
    slot = code < 0 ? code : __vm_jit_install(code);
    if (code >= 0 && slot < 0)
      __vm_jit_release(code);
  }
  if (slot < 0) {
    free(room);
    free(info);
    return slot;
  }
  // Register — or **hot-reload**: if the name is already loaded, overwrite its slot+code in place.
  // The *previous* slot stays installed, so any unit already linked to it keeps working (it baked
  // that slot at link time); only a *later* `vm_dlopen`'s symbol table sees the new slot. That is
  // the live-patch shape: old callers pinned to the old version, new callers bound to the new. The
  // previous version's data room stays too — its code still uses it — so each version keeps its own
  // state; its exported globals, like its slot, are superseded by the new version's.
  int u = vm_dl_put(name, 0);
  if (u < 0) {
    free(info);
    return -12; // registry full
  }
  vm_dl_reg[u].slot = slot;
  vm_dl_reg[u].code = code;
  vm_dl_reg[u].addr = room;
  for (unsigned long k = vm_dl_uleb_get(info, &pos); k; k--) {
    long len = (long)vm_dl_uleb_get(info, &pos);
    char sym[VM_DL_NAMEMAX];
    for (long c = 0; c < len && c < VM_DL_NAMEMAX; c++)
      sym[c] = info[pos + c];
    pos += len;
    unsigned long off = vm_dl_uleb_get(info, &pos);
    if (len >= VM_DL_NAMEMAX)
      continue; // a name the registry can't hold: a unit that references it fails to link (-22)
    sym[len] = 0;
    int d = vm_dl_put(sym, 1);
    if (d < 0) {
      free(info);
      return -12; // registry full
    }
    vm_dl_reg[d].addr = room + off;
    vm_dl_reg[d].owner = u;
  }
  free(info);
  return slot;
}

// `vm_dlcall2(name, a, b)`: look the name up and invoke it (the REPL "eval"). The symbol's unit must
// have the raw `(i64, i64) -> (i64)` entry shape `__vm_jit_invoke2` requires. Returns the result, or
// `-1` if the name is not loaded.
static long vm_dlcall2(const char *name, long a, long b) {
  int i = vm_dl_find(name, 0);
  return i < 0 ? -1 : __vm_jit_invoke2(vm_dl_reg[i].code, a, b);
}

// `vm_dlclose(name)`: uninstall the unit's slot (a stale `call.dyn` of it then traps), free its data
// room, and drop it and its exported globals from the registry — a unit linked against those globals
// must not use them after. Returns 0, or `-1` if the name is not loaded. (The code memory itself is
// not reclaimed — the JIT arena has no per-function free; this frees the *slot*, the data and the
// names. A hot-reloaded name's earlier versions stay installed, with their data, as before.)
static int vm_dlclose(const char *name) {
  int u = vm_dl_find(name, 0);
  if (u < 0)
    return -1;
  __vm_jit_uninstall(vm_dl_reg[u].slot);
  __vm_jit_release(vm_dl_reg[u].code);
  free(vm_dl_reg[u].addr);
  for (int i = 0; i < VM_DL_MAX; i++) {
    if (vm_dl_reg[i].used && (i == u || (vm_dl_reg[i].data && vm_dl_reg[i].owner == u))) {
      vm_dl_reg[i].used = 0;
      vm_dl_count--;
    }
  }
  return 0;
}

#endif // VM_DL_H
