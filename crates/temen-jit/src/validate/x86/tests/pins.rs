//! #2799 layer 1 — the pins. For each accept-side hole #2603's review found, and for the two
//! published Cranelift sandbox escapes the validator exists to stop, a fixed machine-code sequence
//! the validator must reject, with the finding that rejects it. They live in this one table, so a
//! refactor that reopens one fails here, by name. A pin with a control (`ok`) also carries the same
//! code without the hole, which the validator accepts: the rejection is the hole's, not an accident
//! of the snippet.
//!
//! Hole 15 (code a `CodeDir` replays is never checked) is not a property of a function's bytes.
//! Its pin is #2536's single gate, `define_checked`, the only path that defines code.

use super::*;

/// One pin: what it would reopen, the validator's findings on the code, the finding that must
/// reject it, and the findings on the same code without the hole, which must be none.
struct Pin {
    hole: &'static str,
    run: fn() -> Vec<Finding>,
    why: &'static str,
    ok: Option<fn() -> Vec<Finding>>,
}

/// The findings on `bytes`, a function of [`G`]'s module with `relocs`.
fn guest(bytes: &[u8], relocs: &[Reloc]) -> Vec<Finding> {
    check(bytes, relocs, Entry::Guest(&G), Frame::UNKNOWN)
}

/// The D63 confinement, then the access it guards: the code holes 2–4 break.
fn d63() -> Vec<Finding> {
    findings(&after_guard(&[
        0x4c, 0x3b, 0xc0, // cmp r8, rax
        0x4c, 0x0f, 0x43, 0xc0, // cmovae r8, rax
        0x4a, 0x8b, 0x04, 0x07, // mov rax, [rdi + r8*1]
        0xc3,
    ]))
}

/// A function that spills `mem_base` to a frame slot at `store`'s width, reloads it at `load`'s,
/// and accesses the window through it.
fn spill_reload(store: &[u8], load: &[u8]) -> Vec<Finding> {
    let mut b = vec![0x48, 0x83, 0xec, 0x10]; // sub rsp, 16
    b.extend(store);
    b.extend(load);
    b.extend([0x48, 0x8b, 0x07, 0x48, 0x83, 0xc4, 0x10, 0xc3]); // mov rax, [rdi]; add rsp, 16; ret
    findings(&b)
}

/// CVE-2021-32629's shape: a 32-bit index (`mov ecx, r8d` zero-extends) spilled by `store`,
/// reloaded by `load`, and used as the index of a window access that relies on its upper half
/// being zero.
fn index_spill(store: &[u8], load: &[u8]) -> Vec<Finding> {
    let mut b = vec![
        0x44, 0x89, 0xc1, // mov ecx, r8d
        0x48, 0x83, 0xec, 0x10, // sub rsp, 16
    ];
    b.extend(store);
    b.extend(load);
    b.extend([
        0x48, 0x8b, 0x04, 0x0f, // mov rax, [rdi + rcx*1]
        0x48, 0x83, 0xc4, 0x10, // add rsp, 16
        0xc3,
    ]);
    findings(&b)
}

/// The module of CVE-2023-26489: wasm's 32-bit window, a 4 GiB reservation with its guard at 2^32,
/// where a 33-bit index escapes.
const G4: Guest<'static> = Guest {
    window: Some(WindowLayout {
        mask: 0xFFFF_FFFF,
        guard_offset: 1 << 32,
        pages: 0,
    }),
    table: 0,
    types: &[],
    own: &[],
};

/// A callee that takes the context in its second argument register (rsi): a host function.
const CAP: Contract = Contract {
    clobbers: SYSV_CLOBBERS,
    pops: Some(0),
    args: &[ArgTag::Any, ArgTag::Vmctx],
    safepoint: false,
    ret: Ret::Unknown,
};

/// A call to [`CAP`] through `movabs rax, <sym>; call rax`, after `setup`, under the stack check.
fn cap_call(setup: &[u8]) -> Vec<Finding> {
    let mut body = setup.to_vec();
    let at = (CHECK_LEN + body.len() + 2) as u32;
    body.extend([0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xd0, 0xc3]);
    let reloc = Reloc {
        at,
        kind: R::Abs8,
        addend: 0,
        callee: Some(CAP),
    };
    guest(&checked(&body), &[reloc])
}

fn pins() -> Vec<Pin> {
    vec![
        Pin {
            hole: "#2603 hole 1: a call clobbers the caller-saved registers",
            run: || {
                let body = after_guard(&[
                    0x4c, 0x3b, 0xc0, // cmp r8, rax
                    0x4c, 0x0f, 0x43, 0xc0, // cmovae r8, rax
                    0xe8, 0, 0, 0, 0, // call
                    0x4a, 0x8b, 0x04, 0x07, // mov rax, [rdi + r8*1]   (both clobbered)
                    0xc3,
                ]);
                guest(&checked(&body), &[call_reloc((CHECK_LEN + 18) as u32)])
            },
            why: "unrecognized provenance",
            ok: Some(|| {
                let body = after_guard(&[
                    0x4c, 0x3b, 0xc0, // cmp r8, rax
                    0x4c, 0x0f, 0x43, 0xc0, // cmovae r8, rax
                    0x4a, 0x8b, 0x04, 0x07, // mov rax, [rdi + r8*1]
                    0xe8, 0, 0, 0, 0, // call
                    0xc3,
                ]);
                guest(&checked(&body), &[call_reloc((CHECK_LEN + 22) as u32)])
            }),
        },
        Pin {
            hole: "#2603 hole 2: the cmov's condition",
            run: || {
                findings(&after_guard(&[
                    0x4c, 0x3b, 0xc0, // cmp r8, rax
                    0x4c, 0x0f, 0x42,
                    0xc0, // cmovb r8, rax   (keeps the index when out of range)
                    0x4a, 0x8b, 0x04, 0x07, // mov rax, [rdi + r8*1]
                    0xc3,
                ]))
            },
            why: "unconfined index",
            ok: Some(d63),
        },
        Pin {
            hole: "#2603 hole 3: a flag write between the cmp and the cmov",
            run: || {
                findings(&after_guard(&[
                    0x4c, 0x3b, 0xc0, // cmp r8, rax
                    0x48, 0x83, 0xc1, 0x01, // add rcx, 1   (writes CF)
                    0x4c, 0x0f, 0x43, 0xc0, // cmovae r8, rax
                    0x4a, 0x8b, 0x04, 0x07, // mov rax, [rdi + r8*1]
                    0xc3,
                ]))
            },
            why: "unconfined index",
            ok: Some(d63),
        },
        Pin {
            hole: "#2603 hole 4: an unknown compare bound",
            run: || {
                findings(&after_guard(&[
                    0x4d, 0x3b, 0xc1, // cmp r8, r9   (r9: a guest argument)
                    0x4c, 0x0f, 0x43, 0xc0, // cmovae r8, rax
                    0x4a, 0x8b, 0x04, 0x07, // mov rax, [rdi + r8*1]
                    0xc3,
                ]))
            },
            why: "unconfined index",
            ok: Some(d63),
        },
        Pin {
            hole: "#2603 hole 4: a compare bound past the reservation",
            run: || {
                findings(&after_guard(&[
                    0x49, 0xb9, 0, 0, 0, 0, 0, 0, 0x10, 0, // mov r9, 2^52
                    0x4d, 0x3b, 0xc1, // cmp r8, r9
                    0x4c, 0x0f, 0x43, 0xc0, // cmovae r8, rax
                    0x4a, 0x8b, 0x04, 0x07, // mov rax, [rdi + r8*1]
                    0xc3,
                ]))
            },
            why: "past the reservation",
            ok: Some(d63),
        },
        Pin {
            hole: "#2603 hole 5: a 4-byte spill restores no 8-byte fact",
            run: || spill_reload(&[0x89, 0x3c, 0x24], &[0x48, 0x8b, 0x3c, 0x24]), // edi; rdi
            why: "unrecognized provenance",
            ok: Some(|| spill_reload(&[0x48, 0x89, 0x3c, 0x24], &[0x48, 0x8b, 0x3c, 0x24])),
        },
        Pin {
            hole: "#2603 hole 5: a 4-byte reload truncates mem_base",
            run: || spill_reload(&[0x48, 0x89, 0x3c, 0x24], &[0x8b, 0x3c, 0x24]), // rdi; edi
            why: "unrecognized provenance",
            ok: Some(|| spill_reload(&[0x48, 0x89, 0x3c, 0x24], &[0x48, 0x8b, 0x3c, 0x24])),
        },
        Pin {
            hole: "CVE-2021-32629: a 32-bit index spilled, then reloaded sign-extended",
            run: || index_spill(&[0x89, 0x0c, 0x24], &[0x48, 0x63, 0x0c, 0x24]), // ecx; movsxd
            why: "unconfined index",
            ok: Some(|| index_spill(&[0x48, 0x89, 0x0c, 0x24], &[0x48, 0x8b, 0x0c, 0x24])),
        },
        Pin {
            hole: "CVE-2021-32629: a 32-bit index spilled, then reloaded at 64 bits",
            run: || index_spill(&[0x89, 0x0c, 0x24], &[0x48, 0x8b, 0x0c, 0x24]), // ecx; rcx
            why: "unconfined index",
            ok: Some(|| index_spill(&[0x48, 0x89, 0x0c, 0x24], &[0x48, 0x8b, 0x0c, 0x24])),
        },
        Pin {
            hole: "CVE-2023-26489: a 32-bit shift folded into the address's scale",
            run: || {
                let bytes = [
                    0x44, 0x89, 0xc1, // mov ecx, r8d
                    0x48, 0x8b, 0x04, 0xcf, // mov rax, [rdi + rcx*8]   (a 35-bit offset)
                    0xc3,
                ];
                check(&bytes, &[], Entry::Guest(&G4), Frame::UNKNOWN)
            },
            why: "past the reservation",
            ok: Some(|| {
                let bytes = [
                    0x44, 0x89, 0xc1, // mov ecx, r8d
                    0xc1, 0xe1, 0x03, // shl ecx, 3   (wraps at 32 bits)
                    0x48, 0x8b, 0x04, 0x0f, // mov rax, [rdi + rcx*1]
                    0xc3,
                ];
                check(&bytes, &[], Entry::Guest(&G4), Frame::UNKNOWN)
            }),
        },
        Pin {
            hole: "#2603 hole 6: frame slots follow rsp",
            run: || {
                findings(&[
                    0x48, 0x83, 0xec, 0x10, // sub rsp, 16
                    0x48, 0x89, 0x3c, 0x24, // mov [rsp], rdi   (slot −16 := mem_base)
                    0x50, // push rax          (rsp → −24)
                    0x48, 0x8b, 0x0c, 0x24, // mov rcx, [rsp]   (slot −24: rax)
                    0x48, 0x8b, 0x01, // mov rax, [rcx]
                    0x48, 0x83, 0xc4, 0x18, // add rsp, 24
                    0xc3,
                ])
            },
            why: "unrecognized provenance",
            ok: Some(|| {
                findings(&[
                    0x48, 0x83, 0xec, 0x10, // sub rsp, 16
                    0x48, 0x89, 0x3c, 0x24, // mov [rsp], rdi
                    0x50, // push rax
                    0x48, 0x8b, 0x4c, 0x24, 0x08, // mov rcx, [rsp + 8]   (slot −16)
                    0x48, 0x8b, 0x01, // mov rax, [rcx]
                    0x48, 0x83, 0xc4, 0x18, // add rsp, 24
                    0xc3,
                ])
            }),
        },
        Pin {
            hole: "#2603 hole 7: a 16-bit immediate keeps the register's upper bits",
            run: || {
                findings(&[
                    0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff, // mov rax, -1
                    0x66, 0xb8, 0x05, 0x00, // mov ax, 5   (rax = 0xffff_ffff_ffff_0005)
                    0x48, 0x8b, 0x04, 0x07, // mov rax, [rdi + rax*1]
                    0xc3,
                ])
            },
            why: "unconfined index",
            ok: Some(|| {
                findings(&[
                    0xb8, 0x05, 0, 0, 0, // mov eax, 5
                    0x48, 0x8b, 0x04, 0x07, // mov rax, [rdi + rax*1]
                    0xc3,
                ])
            }),
        },
        Pin {
            hole: "#2603 hole 8: a segment override",
            run: || findings(&[0x65, 0x48, 0x8b, 0x04, 0x07, 0xc3]), // mov rax, gs:[rdi + rax]
            why: "fs/gs",
            ok: None,
        },
        Pin {
            hole: "#2603 hole 8: an address-size override",
            run: || findings(&[0x67, 0x48, 0x8b, 0x04, 0x07, 0xc3]), // mov rax, [edi + eax]
            why: "address-size",
            ok: None,
        },
        Pin {
            hole: "#2603 hole 9: an implicit memory operand",
            run: || findings(&[0xd7, 0xc3]), // xlat: [rbx + al]
            why: "xlat",
            ok: None,
        },
        Pin {
            hole: "#2603 hole 9: a bit offset into memory",
            run: || findings(&[0x48, 0x0f, 0xab, 0x07, 0xc3]), // bts [rdi], rax
            why: "bit offset",
            ok: None,
        },
        Pin {
            hole: "#2603 hole 9: a far transfer",
            run: || findings(&[0xff, 0x1f, 0xc3]), // call far [rdi]
            why: "far call",
            ok: None,
        },
        Pin {
            hole: "#2603 hole 10: a stack access's index",
            run: || findings(&[0x48, 0x8b, 0x04, 0xcc, 0xc3]), // mov rax, [rsp + rcx*8]
            why: "unknown frame offset",
            ok: None,
        },
        Pin {
            hole: "#2603 hole 10: a rip-relative store",
            run: || findings(&[0x48, 0x89, 0x05, 0, 0, 0, 0, 0xc3]), // mov [rip], rax
            why: "store to the function's rodata",
            ok: None,
        },
        Pin {
            hole: "#2603 hole 10: a vmctx offset past its fields",
            run: || findings(&[0x48, 0x8b, 0x82, 0, 0, 1, 0, 0xc3]), // mov rax, [rdx + 0x10000]
            why: "no field compiled code may access",
            ok: None,
        },
        Pin {
            hole: "#2603 hole 11: an indirect jump through no jump table",
            run: || findings(&[0xff, 0xe0]), // jmp rax
            why: "holds no jump-table target",
            ok: None,
        },
        Pin {
            hole: "#2603 hole 12: verify trusts no proposal",
            run: || {
                let bytes = [
                    0x48, 0x31, 0xff, // xor rdi, rdi
                    0xeb, 0x00, // jmp +0
                    0x48, 0x8b, 0x07, // mov rax, [rdi]
                    0xc3,
                ];
                let ctx = Ctx {
                    bytes: &bytes,
                    relocs: &[],
                    guest: G,
                    frame: Frame::UNKNOWN,
                    reentry: None,
                };
                let cfg = Cfg::build(ctx, &mut Vec::new());
                let mut lie = propose(&cfg, ctx);
                lie[1] = Some(State::entry(Frame::UNKNOWN)); // "rdi is still mem_base"
                let mut f = Vec::new();
                verify(&cfg, &lie, &[], ctx, &mut f);
                f
            },
            why: "does not hold on the edge",
            ok: None,
        },
        Pin {
            hole: "#2603 hole 13: control falls off the end",
            run: || findings(&[0x48, 0x8b, 0x07]), // mov rax, [rdi]
            why: "off the end",
            ok: None,
        },
        Pin {
            hole: "#2603 hole 14: a direct call names its callee",
            run: || findings(&checked(&[0xe8, 0, 0, 0, 0, 0xc3])), // call <no relocation>
            why: "direct call without a relocation",
            ok: None,
        },
        Pin {
            hole: "#2603 hole 14 (#2534): a host call's context argument",
            run: || cap_call(&[]), // rsi is still fn_table_base
            why: "call argument 1",
            ok: Some(|| cap_call(&[0x48, 0x89, 0xd6])), // mov rsi, rdx
        },
        Pin {
            hole: "#2603 hole 14: an exit restores the callee-saved registers",
            run: || {
                findings(&[
                    0x53, // push rbx
                    0x48, 0x89, 0xfb, // mov rbx, rdi
                    0x48, 0x83, 0xc4, 0x08, // add rsp, 8   (rbx is not reloaded)
                    0xc3,
                ])
            },
            why: "callee-saved r3",
            ok: Some(|| findings(&[0x53, 0x48, 0x89, 0xfb, 0x5b, 0xc3])), // …; pop rbx; ret
        },
        Pin {
            hole: "#2603 hole 16 (I5): a call the stack check doesn't dominate",
            run: || guest(&[0xe8, 0, 0, 0, 0, 0xc3], &[call_reloc(1)]),
            why: "not dominated by the stack-overflow check",
            ok: Some(|| {
                guest(
                    &checked(&[0xe8, 0, 0, 0, 0, 0xc3]),
                    &[call_reloc(1 + CHECK_LEN as u32)],
                )
            }),
        },
        Pin {
            hole: "#2603 hole 16 (I5): a frame past the red zone",
            run: || {
                findings(&[
                    0x48, 0x81, 0xec, 0x00, 0x50, 0x00, 0x00, // sub rsp, 0x5000
                    0x48, 0x81, 0xc4, 0x00, 0x50, 0x00, 0x00, // add rsp, 0x5000
                    0xc3,
                ])
            },
            why: "past the red zone",
            ok: None,
        },
        Pin {
            hole: "#2603 hole 17: the decoder knows popcnt writes its destination",
            run: || {
                findings(&[
                    0x48, 0x89, 0xf8, // mov rax, rdi
                    0xf3, 0x48, 0x0f, 0xb8, 0xc1, // popcnt rax, rcx
                    0x48, 0x8b, 0x10, // mov rdx, [rax]
                    0xc3,
                ])
            },
            why: "unrecognized provenance",
            ok: Some(|| findings(&[0x48, 0x89, 0xf8, 0x48, 0x8b, 0x10, 0xc3])),
        },
        Pin {
            hole: "#2603 hole 17: pushf/popf desync rsp",
            run: || findings(&[0x9c, 0xc3]), // pushf; ret
            why: "pushf",
            ok: None,
        },
        Pin {
            hole: "#2603 hole 17: a memory form whose width the decoder cannot size",
            run: || {
                findings(&[
                    0x48, 0x83, 0xec, 0x40, // sub rsp, 0x40
                    0x0f, 0xae, 0x04,
                    0x24, // fxsave [rsp]   (512 bytes, not the 64 the frame has)
                    0x48, 0x83, 0xc4, 0x40, // add rsp, 0x40
                    0xc3,
                ])
            },
            why: "fxsave/xsave family",
            ok: Some(|| findings(&[0x0f, 0xae, 0xf0, 0xc3])), // mfence; ret
        },
    ]
}

/// Every pin is rejected for its hole, and every control passes.
#[test]
fn every_pin_is_rejected() {
    for p in pins() {
        let f = (p.run)();
        assert!(
            has(&f, p.why),
            "{}: not rejected for \"{}\": {f:?}",
            p.hole,
            p.why
        );
        if let Some(ok) = p.ok {
            let f = ok();
            assert!(
                f.is_empty(),
                "{}: the code without the hole is rejected: {f:?}",
                p.hole
            );
        }
    }
}
