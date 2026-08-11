#!/usr/bin/env bash
# Reproducible label taxonomy for the issue tracker (see ISSUE_TRACKING.md).
# Sets name + color + description together; idempotent (`--force` updates if it
# exists). Run once with owner creds:  bash scripts/setup-labels.sh
set -euo pipefail
REPO="theSherwood/vm"

label() { gh label create "$1" --color "$2" --description "$3" --repo "$REPO" --force; }

# --- area:*  (workstream home — exactly one per issue; = the parent epic) ---
label "area:verify"         "1D76DB" "Verifier & confinement masking (the security hinge)"
label "area:backends"       "1D76DB" "JIT/bytecode/interp tiers, codegen, differential parity"
label "area:ir"             "1D76DB" "IR/substrate: binary format, text round-trip, op spec"
label "area:concurrency"    "1D76DB" "Fibers, scheduling, migration, fork mechanism"
label "area:process"        "1D76DB" "Domains, offers, serving, capabilities, powerbox, nesting"
label "area:durability"     "1D76DB" "Freeze/thaw, snapshots, C-ABI embedding"
label "area:debugging"      "1D76DB" "Stepping, DAP, observability"
label "area:consumers"      "1D76DB" "fork/exec/POSIX shell + language on-ramps (C/Nim/Go/TS/wasm)"
label "area:web-playground" "1D76DB" "Browser deploy, Pages, playground, boot speed"
label "area:perf"           "1D76DB" "Benchmark harness, regressions vs wasmtime (cross-cutting)"
label "area:ci"             "1D76DB" "CI matrix, fuzzing, flakiness, dev tooling (cross-cutting)"

# --- touches:*  (secondary/cross-cutting — zero or more) ---
label "touches:verify"         "C5DEF5" "Secondary: also touches verify/masking"
label "touches:backends"       "C5DEF5" "Secondary: also touches backends/codegen"
label "touches:ir"             "C5DEF5" "Secondary: also touches IR/substrate"
label "touches:concurrency"    "C5DEF5" "Secondary: also touches concurrency/scheduling"
label "touches:process"        "C5DEF5" "Secondary: also touches process/capabilities"
label "touches:durability"     "C5DEF5" "Secondary: also touches durability/snapshots"
label "touches:debugging"      "C5DEF5" "Secondary: also touches debugging/observability"
label "touches:consumers"      "C5DEF5" "Secondary: also touches consumers/on-ramps"
label "touches:web-playground" "C5DEF5" "Secondary: also touches browser/playground"
label "touches:perf"           "C5DEF5" "Secondary: also touches performance"
label "touches:ci"             "C5DEF5" "Secondary: also touches CI/tooling"

# --- sev:*  (mirrors ISSUES.md severity) ---
label "sev:S1" "B60205" "Corruption / escape"
label "sev:S2" "D93F0B" "Host crash or wrong result"
label "sev:S3" "FBCA04" "Robustness / quality"
label "sev:S4" "C2E0C6" "Cosmetic / flake"

# --- status:*  DROPPED — board columns use the native Status field, not labels.
# Delete any that were auto-created earlier (ignore errors if absent).
for s in active blocked in-review parked done; do
  gh label delete "status:$s" --repo "$REPO" --yes 2>/dev/null || true
done

# --- kind:*  (issue type) ---
label "kind:epic"     "0052CC" "Workstream tracker"
label "kind:bug"      "D73A4A" "An ISSUES.md-style defect"
label "kind:task"     "A2EEEF" "A TODO.md-style unit of work"
label "kind:flaky-ci" "E4E669" "CI infra flake (the recurring S4 class)"

# --- invariant flag ---
label "invariant" "5319E7" "Touches a rule in INVARIANTS.md (verifier/masking/grant-graph)"

echo "Labels set for $REPO."
