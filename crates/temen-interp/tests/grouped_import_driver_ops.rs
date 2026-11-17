//! #2958 slice 5 review — a grouped import reaches a `Jit` driver op on the bytecode engine as its
//! flat form does. The bytecode op resolves a `call.import` through [`Host::import_target`], the
//! dispatch's own translation, so a grouped binding's packed op (`slot | consumer op << 16`)
//! reaches its remapped op. Through `import_binding(packed op)` it missed, and a grouped `install`
//! fell through to the generic dispatch, which answered `-EINVAL` where the tree-walker and the
//! flat form answer `CapFault`.

use temen_interp::{BoundImport, Host};

/// A program that calls `install` with code handle 7 through import 0, `grouped` (one method of a
/// two-method interface) or flat.
fn installing(grouped: bool) -> temen_ir::Module {
    let src = match grouped {
        true => {
            "memory 16
type 0 func (i64) -> (i64)
type 1 interface { first: 0, install: 0 }
import 0 interface \"jit\" 1
func 0 () -> (i64) {
block 0 () {
  v0 = i64.const 7
  v1 = call.import 0.install (v0)
  return v1
  }
}
"
        }
        false => {
            "memory 16
import 0 \"install\" (i64) -> (i64)
func 0 () -> (i64) {
block 0 () {
  v0 = i64.const 7
  v1 = call.import 0 (v0)
  return v1
  }
}
"
        }
    };
    temen_text::parse_module(src).expect("parse")
}

/// A host whose import 0 is the `Jit` cap's `install` (op 3), through a remap for `grouped`.
fn host(grouped: bool) -> Host {
    let mut h = Host::new();
    let jit = h.grant_jit(Some(16));
    h.set_import_bindings(vec![BoundImport::required(temen_ir::cap_id::JIT, 3, jit)]);
    if grouped {
        h.set_import_remap(0, std::sync::Arc::from(vec![3u32, 3]));
    }
    h
}

#[test]
fn a_grouped_jit_install_answers_as_its_flat_form_on_the_bytecode_engine() {
    for grouped in [false, true] {
        let m = installing(grouped);
        let tree = temen_interp::run_with_host(&m, 0, &[], &mut 1_000_000, &mut host(grouped));
        let bytecode = temen_interp::bytecode::compile_and_run_with_host(
            &m,
            0,
            &[],
            &mut 1_000_000,
            &mut host(grouped),
        )
        .expect("the bytecode engine runs it");
        assert_eq!(
            tree,
            Err(temen_interp::Trap::CapFault),
            "grouped {grouped}: the tree-walker"
        );
        assert_eq!(bytecode, tree, "grouped {grouped}: the bytecode engine");
    }
}
