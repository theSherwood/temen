# workflows_src — the editable mirror of `.github/workflows/`

The CI token used by agent sessions cannot push files under `.github/workflows/`
(GitHub requires the `workflow` OAuth scope). This directory is the workaround:
agents edit workflow files **here**, and the repo owner applies them by copying
the directory over:

```sh
cp .github/workflows_src/*.yml .github/workflows/
```

(then commit and push with owner credentials). Keep the two in sync in that
direction only — `workflows_src` is the source of truth for *pending* changes;
`.github/workflows/` is what actually runs. After a copy-over the two are
identical until the next agent edit.

**How pending changes are tracked — not here.** The `workflows_src == workflows` CI check *is* the
to-do list: it goes red on a PR the moment the mirror differs from the live workflows, names the file,
and drains on copy-over. Describe a workflow edit in the **PR description** (what changed and why),
exactly like any other change in the PR. Do **not** add an entry to this file: per-change entries made
every workflow-touching branch edit the same lines of this README, which produced merge conflicts
between concurrent PRs over a file that carries nothing the diff and the PR don't already carry. (The
`fuzz targets wired` check plays the same role for fuzz-target matrix rows.)

## Notes

> **Reminder for whoever drains this next:** `miri` no longer runs on PRs. If it is still listed as
> a *required* status check in branch protection, remove it there — a skipped required check blocks
> merges.
