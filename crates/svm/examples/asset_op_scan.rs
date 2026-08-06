//! One-shot §3d survey tool: decode each committed `.svmb` asset and report which
//! `Instantiator` (iface 6) ops its code actually contains — the fact base for which legacy
//! spawn ops the deletion may remove now vs must defer to the I64 asset-regeneration event.
fn main() {
    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    for dir in ["browser/web/assets", "browser/tests/fixtures"] {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().is_some_and(|x| x == "svmb") {
                    paths.push(p);
                }
            }
        }
    }
    paths.sort();
    for p in paths {
        let bytes = std::fs::read(&p).expect("read");
        let m = match svm_encode::decode_module(&bytes).or_else(|_| svm_encode::decode_unit(&bytes))
        {
            Ok(m) => m,
            Err(e) => {
                println!("{}: DECODE ERR {e:?}", p.display());
                continue;
            }
        };
        let mut ops = std::collections::BTreeSet::new();
        let mut ifaces = std::collections::BTreeMap::new();
        for f in &m.funcs {
            for b in &f.blocks {
                for i in &b.insts {
                    if let svm_ir::Inst::CapCall { type_id, op, .. } = i {
                        if *type_id == 6 {
                            ops.insert(*op);
                        }
                        ifaces
                            .entry(*type_id)
                            .or_insert_with(std::collections::BTreeSet::new)
                            .insert(*op);
                    }
                }
            }
        }
        println!(
            "{}: instantiator ops {:?} | all cap ifaces {:?}",
            p.display(),
            ops,
            ifaces
        );
    }
}
