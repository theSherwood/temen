# WIRE.md — the unified TEMEN wire header

Every temen on-disk format — a compiled module, a link unit (object), a §12 durable-snapshot
artifact, a filesystem image, and any format that comes after — starts with the **same 16-byte
header**. One sniff identifies any temen blob; the `kind` field says what it is. The header is the
*envelope*; each format's payload keeps its own encoding and its own version sequence.

The single source of truth for the layout is `temen_encode::wire` (`crates/temen-encode/src/lib.rs`).
This document is the registry the code and its consumers are held to.

## Layout

All fields little-endian.

```text
offset  size  field    meaning
[0..8)   8    magic    b"TEMEN\0\0\0"  — "TEMEN" NUL-padded to 8 bytes (8-byte aligned, greppable)
[8..10)  2    kind     u16 — what the payload IS (registry below)
[10..12) 2    version  u16 — the payload's format version, per kind
[12..16) 4    flags    u32 — per-kind modifiers; reserved bits must be 0 and fail closed
```

`temen_encode::wire::read_header` checks **only the magic** and hands back the header plus the
payload. Every consumer then enforces its own `kind`, `version`, and `flags` at the header — before
reading a single payload byte. `write_header` is the one writer.

## The one rule: unknown is rejected, never guessed

This is a sandbox whose value is a small, trustworthy core, and the decoders are the
untrusted-input TCB. So:

- An **execution path** accepts exactly its expected `kind` and nothing else. `decode_module`
  accepts `module`; `decode_unit` accepts `module` or `object`; snapshot restore accepts
  `snapshot`; `decode_image` accepts `fs-image`. A foreign or unknown kind fails closed at the
  header (`BadKind` / "not a … container").
- `version` is exact-match per kind; there are no compatibility windows unless a wire rev opens one
  deliberately (and closes it once every committed asset has regenerated — the #900 precedent).
- Reserved `flags` bits fail closed.
- The **only** place an unknown kind is tolerated is as *inert payload inside a container* (a
  future `bundle` may carry a sub-blob it never dispatches). Extensibility lives at the carrier
  layer; the execution decoders stay strict.

## Kind registry

### Core, stable — `0x0000..=0x00FF`

Assigned here, by the project. Implemented kinds are marked; the rest are reserved names so their
numbers can't be reused for something else.

| kind     | name       | status      | payload / owner |
|----------|------------|-------------|-----------------|
| `0x0000` | `module`   | implemented | a runnable module — `temen-encode` `encode_module`/`decode_module`; today `version` = 10 |
| `0x0001` | `object`   | implemented | a link unit, the pre-link dialect — `encode_unit`/`decode_unit`; same `version` sequence as `module` |
| `0x0002` | `snapshot` | implemented | a §12 durable-snapshot artifact — `temen-snapshot` (`DURABILITY.md` §12); today `version` = 18 |
| `0x0003` | `fs-image` | implemented | a filesystem image a guest mounts — `temen-fs::encode_image`; today `version` = 1 |
| `0x0004` | `bundle`   | reserved    | a container of self-delimited sub-blobs (e.g. fs-image + snapshot = a resumable machine). Layout to be defined when a consumer appears. |

Before `0x0004` is implemented, or when a new core kind is needed, add it to this table first.
Plausible future core kinds (not yet reserved by number): `link-archive` (a collection of
objects), `debug-info` (the DWARF/source-map sidecar), `compile-cache` (cached native/wasm for a
module, keyed by digest), `region-image` (a §13 shared-region snapshot), `profile`, `corpus`,
`delta` (an incremental snapshot/module patch).

The `object` dialect used to be a *flag bit* on the module header; it is now its own kind. The
module's `flags` word has no defined bits today.

### Core, reserved — `0x0100..=0x7FFF`

Held for future first-party kinds. Decoders reject them (fail closed), so the space can be assigned
later without ambiguity.

### Community / experimental — `0x8000..=0xFEFF`

Usable without central coordination, for local experiments and quick prototypes. Two authors *can*
collide here — that is the author's risk, like a reserved machine range in ELF. The core runtime
never dispatches these; a container may carry them inert.

### Reserved — `0xFF00..=0xFFFE`

Held.

### Namespaced extension — `0xFFFF`

The collision-free path for a real third-party format. Not yet implemented; the intended shape:
the payload begins with a self-chosen **16-byte format id** (a UUID, or a truncated hash of the
format's spec), so independent authors never collide and no central authority hands out numbers.
Define the exact layout in this document before implementing.

## Migration (history)

The header replaced three unrelated pre-`temen` magics — `b"SVM\0"` (module), `b"SVMD"` (snapshot),
`b"SVMFSIM1"` (fs-image) — in a clean break (#1178): old-magic blobs fail at the magic check, which
is the desired behavior. Every committed `.temen` asset was migrated by rewriting its header in
place (the old 6-byte module header → the 16-byte TEMEN header; payload bytes untouched, exactly
what a rebuild emits for a header-only change) and re-validated through the new decoder. `version`
numbers carried over unchanged — the magic already distinguishes old from new, so no bump was
needed.

## Open questions

- **Should `snapshot` and `fs-image` consolidate?** They carry different things today — a
  snapshot is a paused domain's memory + protections + continuation + capability *authority*
  (D-scope: not the host-side resources handles name); an fs-image is a tree of input files a guest
  mounts. A guest with a live host-side memfs is not captured by a snapshot (the memfs is a host
  resource behind a capability). The candidate answers are (a) a `bundle` composing the two, or
  (b) a durable-filesystem capability whose state rides the snapshot. Deferred; decide when a
  consumer needs fs + CPU state together.
- The `bundle` layout and the `0xFFFF` extension layout are undefined until needed (do less).
