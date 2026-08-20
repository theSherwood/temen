#!/usr/bin/env bash
# GNU bash on the LLVM on-ramp — the whole-program bitcode pipeline (#802 slice 2), mirroring
# the Tcl/Postgres/QuickJS capstones. Fetch → configure → native oracle → per-TU bitcode
# (reusing each Makefile's exact flags) → llvm-link the full link-line object set + the shim +
# the reused waist → translate through the on-ramp. Fetched-not-vendored (GPLv3 — bash is never
# vendored into this tree; the pipeline builds from the upstream tarball).
#
# Unlike Tcl/QuickJS there is NO driver: bash IS a program — its own `main` (shell.c) is the
# entry the on-ramp's synthesized `_start` calls, argv parsed from the powerbox args buffer.
#
#   needs: clang, llvm-link, cc, make, curl, tar
#   env:   SVM_BASH_CACHE (default /tmp/svm_bash_cache), SVM_BASH_VER (default 5.2.21)
#
#   ./build_bitcode.sh            # → $CACHE/bash_linked.ll  (+ a native oracle at $SRC/bash)
set -uo pipefail

VER="${SVM_BASH_VER:-5.2.21}"
CACHE="${SVM_BASH_CACHE:-/tmp/svm_bash_cache}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$CACHE/bash-$VER"
OUT="$CACHE/bc"
mkdir -p "$CACHE" "$OUT"
cd "$CACHE"

echo "=== [1/6] fetch bash-$VER ==="
if [ ! -f "$SRC/shell.c" ]; then
  curl -sfL --max-time 180 -o "bash-$VER.tar.gz" \
    "https://ftp.gnu.org/gnu/bash/bash-$VER.tar.gz" || { echo "FETCH FAILED"; exit 11; }
  tar xf "bash-$VER.tar.gz"
fi

echo "=== [2/6] configure (the bring-up config) ==="
cd "$SRC"
# --without-bash-malloc: bash's sbrk-based allocator would need a brk shim; the waist malloc is
#   the proven path (every prior capstone). --disable-readline: non-interactive first (slice 4
#   brings line editing to the #797 terminal). --disable-nls: no locale catalogs in-guest.
# --disable-net-redirections: /dev/tcp opens sockets — no socket surface yet.
# Job control stays ON — it is the point of the personality's #798 machinery.
# ac_cv_type_long_double=no: the printf builtin's `%Lf` path uses x86_fp80, which the on-ramp
# does not lower (Milestone 1+); autoconf-denying the type keeps floatmax_t = double in BOTH the
# guest bitcode and the native oracle, so the differential stays honest.
[ -f config.h ] || ac_cv_type_long_double=no ./configure --without-bash-malloc \
  --disable-readline --disable-nls --disable-net-redirections >"$CACHE/configure.log" 2>&1 \
  || { echo "CONFIGURE FAILED"; tail -5 "$CACHE/configure.log"; exit 12; }

echo "=== [3/6] native oracle build (bash + the generated sources) ==="
# The build also GENERATES sources the bitcode step needs: y.tab.c (yacc), builtins/*.c
# (mkbuiltins from .def), version.c, signames.h — so the oracle build is not optional.
[ -x bash ] || make -j"$(nproc)" >"$CACHE/make.log" 2>&1 || { echo "BUILD FAILED"; tail -6 "$CACHE/make.log"; exit 13; }
echo "oracle: $(./bash -c 'echo $BASH_VERSION')"

echo "=== [4/6] per-TU bitcode (reusing each Makefile's exact flags) ==="
# The authoritative object set is bash's own link line: the top-level OBJECTS plus the members
# of libbuiltins/libglob/libsh/libhistory/libtilde. Each directory compiles its members with one
# common flag set — capture it per directory from a representative TU (the Tcl-recipe trick),
# swap gcc→clang -emit-llvm, compile every member uniformly FROM THAT DIRECTORY (several sources
# use `../`-relative includes).
rm -f "$OUT"/*.ll

# capture_flags <dir> <representative.o> <source.c> — sets FLAGS[] from the make -n line.
capture_flags() {
  local dir="$1" obj="$2" srcc="$3" line
  line=$(cd "$dir" && make -n -B "$obj" 2>/dev/null | grep -m1 -- "$srcc")
  line=${line/gcc/}
  line=${line/cc /}
  eval "set -- $line"
  FLAGS=()
  local a
  for a in "$@"; do
    case "$a" in
      -c | -o | -g | "$obj" | *"$srcc") ;; # strip the object/source and debug (bitcode is -S)
      *) FLAGS+=("$a") ;;
    esac
  done
}

# emit <dir> <base> — compile $dir/$base.c → $OUT/<tag>_<base>.ll with the captured FLAGS.
emit() {
  local dir="$1" base="$2" tag="$3"
  clang -emit-llvm -S -O2 -fno-vectorize -fno-slp-vectorize "${FLAGS[@]}" \
    "$dir/$base.c" -o "$OUT/${tag}_$base.ll" 2>"$OUT/${tag}_$base.err" \
    || { echo "  CLANG FAIL: $tag/$base"; head -2 "$OUT/${tag}_$base.err"; fail=1; }
}

fail=0
# Top-level objects, verbatim from the link line (xmalloc included; mksignames/buildversion are
# build-time generators, never linked).
TOP="shell eval y.tab general make_cmd print_cmd dispose_cmd execute_cmd variables copy_cmd
     error expr flags jobs subst hashcmd hashlib mailcheck trap input unwind_prot pathexp sig
     test version alias array arrayfunc assoc braces bracecomp bashhist bashline list stringlib
     locale findcmd redir pcomplete pcomplib syntax xmalloc"
capture_flags "$SRC" shell.o shell.c
( cd "$SRC" && true )
for b in $TOP; do (cd "$SRC" && emit . "$b" top); done

# The `.def`-generated builtins sources are transient (the Makefile's `.def.o` rule can leave no
# `.c` behind); regenerate any missing one with the built `mkbuiltins` before compiling.
(cd "$SRC/builtins" && for d in *.def; do
  [ -f "${d%.def}.c" ] || ./mkbuiltins -D . "$d" >/dev/null 2>&1
done)

# Archive members: source dir == archive dir for all five libraries. libhistory carries
# readline's STANDALONE support shims (shell/xmalloc/xfree/savestring/mbutil) that duplicate
# bash's own definitions — the native static link never pulls them (archive member semantics),
# so the whole-program link must skip them: take only the four hist* members.
for spec in "builtins:libbuiltins.a" "lib/glob:libglob.a" "lib/sh:libsh.a" \
            "lib/readline:libhistory.a" "lib/tilde:libtilde.a"; do
  dir="$SRC/${spec%%:*}" lib="${spec##*:}"
  first=$(ar t "$dir/$lib" | head -1)
  capture_flags "$dir" "$first" "${first%.o}.c"
  for obj in $(ar t "$dir/$lib"); do
    base=${obj%.o}
    case "$lib:$base" in libhistory.a:hist* | libhistory.a:history) ;; libhistory.a:*) continue ;; esac
    [ -f "$dir/$base.c" ] || { echo "  NO SRC: ${spec%%:*}/$base"; fail=1; continue; }
    (cd "$dir" && emit . "$base" "$(basename "${spec%%:*}")")
  done
done
echo "compiled $(ls "$OUT"/*.ll 2>/dev/null | wc -l) TUs (fail=$fail)"

echo "=== [5/6] shim + reused waist → bitcode ==="
CF=(-O2 -emit-llvm -S -fno-vectorize -fno-slp-vectorize -DNDEBUG -D_GNU_SOURCE)
DEMOS="$HERE/.."
clang "${CF[@]}" "$HERE/bash_shim.c" -o "$OUT/_bashshim.ll" || { echo "bash_shim FAIL"; exit 14; }
# The personality's own guest libc: bash's fnmatch/regcomp/regexec are the #800 implementations,
# linked as ordinary guest code (their `__px_malloc`/`__px_free` externs bridge in bash_shim.c).
for f in fnmatch regex; do
  clang "${CF[@]}" "$DEMOS/posix_libc/$f.c" -o "$OUT/_px_$f.ll" || { echo "posix_libc/$f FAIL"; exit 14; }
done
# The reused waist (the Tcl set): the Postgres printf/scanf engines + the guest strtod.
for s in "postgres/printf_shim:_printf" "strtod/strtod:_strtod"; do
  src="$DEMOS/${s%%:*}.c"; tag="${s##*:}"
  clang "${CF[@]}" "$src" -o "$OUT/$tag.ll" 2>/dev/null || echo "  note: optional shim ${s%%:*} skipped"
done

echo "=== [6/6] llvm-link → translate through the on-ramp ==="
LINKED="$CACHE/bash_linked.ll"
llvm-link -S "$OUT"/*.ll -o "$LINKED" 2>"$CACHE/llvm-link.err" \
  || { echo "LINK FAILED:"; tail -5 "$CACHE/llvm-link.err"; exit 15; }
echo "linked: $(stat -c%s "$LINKED") B → $LINKED"
TR="$HERE/../../../svm-llvm/target/release/examples/try_translate"
if [ -x "$TR" ]; then
  # SVM_STUB_EXTERNS=1: trap-stub the OS surface the guest never reaches. The surface bash DOES
  # reach binds to the svm-posix ops by bare libc name (Lane C) — slice 3's run wires the rest.
  echo "translate + verify:"
  SVM_STUB_EXTERNS=1 "$TR" "$LINKED" 2>&1 | grep -v '\[stub\]' | head -5
else
  echo "note: build the translator first:  (cd crates/svm-llvm && cargo build --release --example try_translate)"
fi
