# Issue tracking

How we track work, bugs, and investigations. **TL;DR: live tracking lives in
GitHub issues on a Project board.** The in-tree design docs (`DESIGN.md`,
`DURABILITY.md`, …) and the deep root-cause registry (`ISSUES.md`) stay where
they are and get *linked* from issues — they are not the tracker.

## Why GitHub issues (and not more markdown)

- **CI cost.** Editing a markdown tracker (`ISSUES.md`, `TODO.md`) triggers the
  CI matrix, which is slow. Issue create/edit/comment costs nothing — so
  investigation and triage churn belongs in issues, not in doc edits.
- Threading, assignees, notifications, cross-links, and a board view you don't
  get in a flat file.
- The deep, code-adjacent record (a root-cause essay a reviewer reads beside the
  fix) still belongs in-tree; the issue links to it. See [The split](#the-split).

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
- **`sev:S1|S2|S3|S4`** — severity, mirroring `ISSUES.md`: S1 corruption/escape ·
  S2 host crash or wrong result · S3 robustness/quality · S4 cosmetic/flake.
- **`kind:epic|bug|task|flaky-ci`**.
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
5. If the bug needs a long root-cause essay a reviewer should read beside the
   fix, put that essay in `ISSUES.md` as an `I##` entry and **link it** from the
   issue — don't duplicate the prose.

## The split

What lives where, so nothing is double-maintained:

- **GitHub issues** — status, triage, discussion, the live work list. The source
  of truth for *what's open and where it stands*.
- **`ISSUES.md`** — deep root-cause registry. Keep an `I##` entry when the
  writeup benefits from living beside the code (reviewed in the fixing PR); link
  it from the issue. **Not** a status tracker — the issue's state is.
- **Design docs** (`DESIGN.md`, `DURABILITY.md`, …) — the design itself. Issues
  point at them; the docs don't track status.
- **`TODO.md`** — the legacy index of deferred work; the board supersedes it for
  indexing open work. Migrate a row to an issue when it goes active.

## Closing

Close with a `state_reason` (completed / not planned / duplicate) and move the
card to Done. If the issue had an `ISSUES.md` `I##` entry, move that entry to the
Resolved section (or delete it and note the fix in the relevant design doc), same
as before.

## Attribution

Every agent-authored issue/PR comment ends with the Claude Code attribution
footer (a `---` rule then `_Generated by [Claude Code](https://claude.ai/code)_`).
