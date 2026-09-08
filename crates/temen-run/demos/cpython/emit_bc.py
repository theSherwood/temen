#!/usr/bin/env python3
"""Emit per-TU LLVM bitcode for CPython, reusing the makefile's exact compile flags — the
CPython twin of demos/postgres/emit_bc.py. For each native `.o`, ask `make -n` for its clang
command (after bumping the source mtime so the rule prints), rewrite it to `-emit-llvm -o X.bc`,
append the on-ramp vectorizer knobs, and run it.

    TEMEN_CPY_SRC=/path/to/cpython-3.13.1 python3 emit_bc.py
"""
import os, subprocess, shlex, concurrent.futures

SRC = os.environ.get("TEMEN_CPY_SRC")
assert SRC and os.path.isdir(SRC), "set TEMEN_CPY_SRC to the built cpython tree"
DIRS = ["Modules", "Objects", "Python", "Parser", "Programs"]
EXTRA = ["-emit-llvm", "-fno-vectorize", "-fno-slp-vectorize"]


def find_objs():
    objs = []
    for d in DIRS:
        for root, _, files in os.walk(os.path.join(SRC, d)):
            objs += [os.path.join(root, f) for f in files if f.endswith(".o")]
    return objs


def compile_cmd(obj):
    d, base = os.path.dirname(obj), os.path.basename(obj)
    src = obj[:-2] + ".c"
    if os.path.exists(src):
        os.utime(src, None)
    # CPython builds from the top-level Makefile (objects carry dir-relative names); run `make -n`
    # at $SRC and match the rule that produces this object path relative to $SRC.
    rel = os.path.relpath(obj, SRC)
    out = subprocess.run(["make", "-n", rel], cwd=SRC, capture_output=True, text=True, timeout=180).stdout
    line = next((l.strip() for l in out.splitlines()
                 if l.strip().startswith("clang ") and " -c " in l and rel.rsplit("/", 1)[-1][:-2] + ".c" in l), None)
    if not line:
        return (obj, None)
    toks, out_toks, bc, i = shlex.split(line), [], obj[:-2] + ".bc", 0
    while i < len(toks):
        t = toks[i]
        if t == "-o":
            out_toks += ["-o", bc]; i += 2; continue
        if t in ("-ftree-vectorize",):
            i += 1; continue
        out_toks.append(t); i += 1
    out_toks.extend(EXTRA)
    return (obj, (d, out_toks, bc))


def main():
    objs = find_objs()
    print(f"object set: {len(objs)} objects", flush=True)
    cmds = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        for _obj, cmd in ex.map(compile_cmd, objs):
            if cmd:
                cmds.append(cmd)
    print(f"compile commands recovered: {len(cmds)}", flush=True)

    def run(c):
        d, toks, bc = c
        return subprocess.run(toks, cwd=SRC, capture_output=True, text=True).returncode

    fails = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        for rc in ex.map(run, cmds):
            fails += rc != 0
    print(f"bitcode: {len(cmds) - fails}/{len(cmds)} ok, {fails} failed", flush=True)
    with open(os.path.join(SRC, "..", "bc_manifest.txt"), "w") as f:
        for _d, _t, bc in cmds:
            if os.path.exists(bc):
                f.write(os.path.abspath(bc) + "\n")


main()
