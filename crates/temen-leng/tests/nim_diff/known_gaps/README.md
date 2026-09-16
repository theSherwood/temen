# `known_gaps/`

Corpus programs that are **expected to diverge** from native nimony. A `.nim` file here is run by
`nim_differential_corpus` exactly like one in the parent directory, but the expectation is inverted:
diverging is fine, *matching* is a failure telling you to move the file up into `tests/nim_diff/` and
close the issue its header names.

Every file starts with a `# KNOWN GAP — …` comment naming the issue. The directory is the whole
expectation mechanism — there is no per-case list to keep in sync.

Currently empty — #1488 (sub-word arithmetic) was the last open gap and is fixed.
