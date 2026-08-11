# Issue tracking

How we track work, bugs, and investigations. **TL;DR: live tracking lives in
GitHub issues on a Project board.** The in-tree design docs (`DESIGN.md`,
`DURABILITY.md`, …) hold the design and get *linked* from issues — they are not
the tracker. (The old `ISSUES.md` registry is **retired**; see [The split](#the-split).)

## Why GitHub issues (and not more markdown)

- **CI cost.** Editing a markdown tracker triggers the CI matrix, which is slow.
  Issue create/edit/comment costs nothing — so investigation and triage churn
  belongs in issues, not in doc edits.
- Threading, assignees, notifications, cross-links, and a board view you don't
  get in a flat file.
- When a root-cause writeup must live beside the code (reviewed in the fixing
  PR), it goes in the relevant **design doc**; otherwise the issue body carries
  it. See [The split](#the-split).

## The board

One GitHub Project, two axes:

- **Columns = the native `Status` field** — Backlog / Active / Blocked /
  In-Review / Done. Managed by dragging cards.
- **Swimlanes = group by Parent issue.** Every issue is a sub-issue of exactly
  one **epic**, and the epic *is* the workstream. So the parent link is the
  workstream classification — there is no separate "Workstream" field to keep in
  sync.

## Workstreams (epics)

Eleven. File every issue as a sub-issue of exactly one:

| Epic | Workstream | Owning docs |
|---|---|---|
| #705 | **Verify** — verifier & confinement masking (the security hinge) | DESIGN §3–5; INVARIANTS 2; `svm-verify`, `svm-mask` |
| #706 | **Backends** — JIT/bytecode/interp tiers, codegen, differential parity | LLVM.md, OPT.md, OPS_PARITY.md; DESIGN §17/§22 |
| #707 | **IR/substrate** — binary format, text round-trip, op spec | DESIGN §3a/3b; SPEC.md; `svm-ir`, `svm-text` |
| #708 | **Concurrency** — fibers, scheduling, migration, fork mechanism | THREADS.md, FORK.md; DESIGN §23; `svm-fiber` |
| #709 | **Process** — domains, offers, serving, capabilities, nesting | PROCESS.md, IMPORTS.md, POWERBOX.md; DESIGN §12a |
| #710 | **Durability** — freeze/thaw, snapshots, C-ABI embedding | DURABILITY.md, INTERACTIVE_EMBEDDING.md; DESIGN §21 |
| #711 | **Debugging** — stepping, DAP, observability | DEBUGGING.md; DESIGN §19; `svm-dap` |
| #712 | **Consumers** — fork/exec/POSIX shell + language on-ramps | FORK/EXEC/POSIX/STAGE1.md, SELFHOST_C.md, NIM/GO/TYPESCRIPT.md; DESIGN §20 |
| #713 | **Web/Playground** — browser deploy, Pages, boot speed | FRONTEND.md, BROWSER.md, BOOTSPEED.md; `browser/` |
| #714 | **Perf** — benchmark harness, regressions vs wasmtime | INTERP_PERF.md, OPT.md; DESIGN §1a; `bench/` |
| #702 | **CI** — CI matrix, fuzzing, flakiness, dev tooling | AGENTS.md, `.github/` |

`Perf` and `CI` are cross-cutting — an issue homed elsewhere carries a
`touches:perf` / `touches:ci` label instead of moving there.

## Labels

- **`area:<workstream>`** — **exactly one**, matches the parent epic. Its home.
  (The board groups by the parent link; `area:` is how you slice by workstream
  from the Issues tab, off the board.)
- **`touches:<workstream>`** — **zero or more.** Cross-cutting overlap. An issue
  has one home but may touch other workstreams; the touched epic's filter still
  surfaces it.
- **`sev:S1|S2|S3|S4`** — severity: S1 corruption/escape · S2 host crash or wrong
  result · S3 robustness/quality · S4 cosmetic/flake.
- **`kind:epic|bug|task|flaky-ci`**.
- **`topic:*`** — **zero or more**, optional, additive, created on demand.
  Fine-grained subject tags *orthogonal* to the workstream: languages (`c`, `nim`,
  `go`, `rust`, `typescript`, `lua`, `tcl`, `quickjs`), engines/codegen (`jit`,
  `bytecode`, `tree-walker`, `guest-jit`, `cranelift`, `llvm`, `wasm`, `simd`,
  `gpu`, `peval`), runtime themes (`nesting`, `fork`, `serving`, `snapshot`,
  `futex`), consumer/demo surfaces (`bash`, `shell`, `chibicc`, `doom`, `sqlite`,
  `postgres`, `playground`), and quality/meta (`ergonomics`, `benchmark`, `test`).
  A topic never *homes* an issue (that's `area:`/parent); a topic that accumulates
  sustained work can graduate to a workstream.
- **`invariant`** — touches a rule in `INVARIANTS.md` (verifier/masking/grant-graph).
  Flags the sensitive changes.

The taxonomy is reproducible and idempotent: **`scripts/setup-labels.sh`**.

## Filing an issue

1. **Pick the parent epic** = the workstream it belongs to.
2. **Labels:** `area:` matching the parent; `sev:`; `kind:`; a `touches:` for
   each other workstream it affects; `invariant` if it touches `INVARIANTS.md`.
3. **Body:** What / Why it matters (or why it's a limitation, not a bug) / Root
   cause (or why deferred) / Fix sketch / Links.
4. **Set the parent** — GitHub's "add sub-issue" on the epic, or the sub-issues
   API. This is what puts it in the right swimlane.
5. Put the root-cause detail in the issue body. If it must live beside the code
   (reviewed in the fixing PR), add it to the relevant **design doc** and link it
   — don't split status across two places.

## The split

What lives where, so nothing is double-maintained:

- **GitHub issues** — status, triage, discussion, the live work list. The source
  of truth for *what's open and where it stands*.
- **Design docs** (`DESIGN.md`, `DURABILITY.md`, …) — the design itself, and the
  home for a root-cause writeup that must live beside the code (reviewed in the
  fixing PR), linked from the issue. The docs don't track status.
- **`ISSUES.md`** — **retired.** The old numbered `I##` registry was migrated to
  issues (live items) and otherwise left to git history. Do not recreate it; file
  an issue instead. Old `ISSUES.md I##` references in code comments are frozen
  provenance — resolve them via `git log`.
- **`TODO.md`** — **retired.** Its live rows were migrated to issues; parked /
  reserved / future rows remain in git history. The board is the index of open work.

## Closing

Close with a `state_reason` (completed / not planned / duplicate) and move the
card to Done. If the fix changes a design doc, note it there in the same PR.

## Attribution

Every agent-authored issue/PR comment ends with the Claude Code attribution
footer (a `---` rule then `_Generated by [Claude Code](https://claude.ai/code)_`).
