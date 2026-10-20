//! temen#2452 — compiled code kept on disk ([`CodeDir`]), so a later process's compile of the same
//! image defines the code an earlier one emitted instead of compiling it.
//!
//! The code a compile emits names no address in its process (D70): each host function it calls is
//! a symbol, bound when the code is linked. So each function's bytes and relocations, the targets
//! by name ([`EmittedCode`]), link into any later process through the same relocation step
//! `JITModule` runs on code it just compiled. A [`CodeDir`] keeps them in a directory only its owner
//! can write, one file per compile, named by what the code depends on ([`root_key`],
//! [`child_key`]). A file that fails its checksum, or was kept by another build of temen or for
//! another CPU, is a miss, and the compile that follows overwrites it.

use crate::{host_import, Imports, JitError};
use cranelift_codegen::binemit::Reloc;
use cranelift_codegen::ir::LibCall;
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Linkage, Module, ModuleReloc, ModuleRelocTarget};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The code a compile emitted for the module's functions, in definition order: what a
/// [`CodeDir`] keeps.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct EmittedCode {
    funcs: Vec<EmittedFn>,
}

#[derive(Debug, PartialEq)]
struct EmittedFn {
    align: u64,
    bytes: Vec<u8>,
    relocs: Vec<EmittedReloc>,
}

#[derive(Debug, PartialEq)]
struct EmittedReloc {
    offset: u32,
    kind: Reloc,
    addend: i64,
    target: Target,
}

/// What a relocation names, by what a later process can bind it by.
#[derive(Debug, PartialEq)]
enum Target {
    /// Function `i` of the module, which every compile declares `i`th ([`crate::declare_all_funcs`]).
    Func(u32),
    /// A host function, by its symbol ([`crate::HostFn`], or a fast-path target).
    Import(String),
    /// A Cranelift libcall, by name.
    LibCall(String),
}

/// Where a compile's code comes from.
pub(crate) enum Codegen<'a> {
    /// The compile, which keeps each function's code while the record holds code
    /// ([`EmittedCode::push`]).
    Compile(Option<EmittedCode>),
    /// Code a [`CodeDir`] kept, linked against the host functions the compile's `imports` binds.
    Replay(&'a EmittedCode, &'a Imports),
}

/// The relocations kept code may carry: those the JIT's code uses on the targets it runs on. Code
/// with any other is not kept.
const KINDS: [Reloc; 7] = [
    Reloc::Abs4,
    Reloc::Abs8,
    Reloc::X86PCRel4,
    Reloc::X86CallPCRel4,
    Reloc::Arm64Call,
    Reloc::Aarch64AdrPrelPgHi21,
    Reloc::Aarch64AddAbsLo12Nc,
];

impl EmittedCode {
    /// Keep the function `module` defined from `bytes` (aligned to `align`) and `relocs`, whose
    /// module declared its functions as `ids`. `false` when the code names something a later
    /// process could not bind by name: then the compile's code is not kept.
    pub(crate) fn push(
        &mut self,
        module: &JITModule,
        ids: &[FuncId],
        align: u64,
        bytes: &[u8],
        relocs: &[ModuleReloc],
    ) -> bool {
        let relocs = relocs.iter().map(|r| {
            let target = match &r.name {
                ModuleRelocTarget::User {
                    namespace: 0,
                    index,
                } => {
                    let decl = module
                        .declarations()
                        .get_function_decl(FuncId::from_u32(*index));
                    if decl.linkage == Linkage::Import {
                        Target::Import(decl.name.clone()?)
                    } else if ids.get(*index as usize)?.as_u32() == *index {
                        Target::Func(*index)
                    } else {
                        return None;
                    }
                }
                ModuleRelocTarget::LibCall(lc) => Target::LibCall(lc.to_string()),
                _ => return None,
            };
            KINDS.contains(&r.kind).then_some(EmittedReloc {
                offset: r.offset,
                kind: r.kind,
                addend: r.addend,
                target,
            })
        });
        match relocs.collect() {
            Some(relocs) => {
                self.funcs.push(EmittedFn {
                    align,
                    bytes: bytes.to_vec(),
                    relocs,
                });
                true
            }
            None => false,
        }
    }

    /// Whether this code can be defined in `module`, which declared the module's functions as
    /// `ids` and binds the host functions `imports` names: every function's code is here, and
    /// every relocation names something its link binds. Checked before any function is defined,
    /// so a replay cannot fail halfway.
    pub(crate) fn fits(&self, module: &JITModule, ids: &[FuncId], imports: &Imports) -> bool {
        let cc = module.isa().default_call_conv();
        self.funcs.len() == ids.len()
            && self
                .funcs
                .iter()
                .flat_map(|f| &f.relocs)
                .all(|r| match &r.target {
                    Target::Func(i) => (*i as usize) < ids.len(),
                    Target::Import(name) => host_import(name, cc, imports).is_some(),
                    Target::LibCall(name) => name.parse::<LibCall>().is_ok(),
                })
    }

    /// Define the module's functions, which `module` declared as `ids`, from this code, as a compile
    /// defines the code it emits; what [`Self::fits`] checked is assumed. Returns the bytes of code
    /// defined.
    pub(crate) fn replay(
        &self,
        module: &mut JITModule,
        ids: &[FuncId],
        imports: &Imports,
    ) -> Result<usize, JitError> {
        let backend = |e: cranelift_module::ModuleError| JitError::Backend(e.to_string());
        let cc = module.isa().default_call_conv();
        let mut bytes = 0;
        for (f, &id) in self.funcs.iter().zip(ids) {
            let mut relocs = Vec::with_capacity(f.relocs.len());
            for r in &f.relocs {
                let name = match &r.target {
                    Target::Func(i) => ModuleRelocTarget::user(0, ids[*i as usize].as_u32()),
                    Target::Import(name) => {
                        let sig = host_import(name, cc, imports).ok_or_else(|| {
                            JitError::Backend(format!("kept code names {name}, which is not bound"))
                        })?;
                        let id = module
                            .declare_function(name, Linkage::Import, &sig)
                            .map_err(backend)?;
                        ModuleRelocTarget::user(0, id.as_u32())
                    }
                    Target::LibCall(name) => {
                        ModuleRelocTarget::LibCall(name.parse().map_err(|()| {
                            JitError::Backend(format!("kept code names libcall {name}"))
                        })?)
                    }
                };
                relocs.push(ModuleReloc {
                    offset: r.offset,
                    kind: r.kind,
                    name,
                    addend: r.addend,
                });
            }
            module
                .define_function_bytes(id, f.align, &f.bytes, &relocs)
                .map_err(backend)?;
            bytes += f.bytes.len();
        }
        Ok(bytes)
    }

    fn encode(&self, out: &mut Vec<u8>) {
        fn str(out: &mut Vec<u8>, s: &str) {
            out.extend((s.len() as u32).to_le_bytes());
            out.extend(s.as_bytes());
        }
        out.extend((self.funcs.len() as u32).to_le_bytes());
        for f in &self.funcs {
            out.extend(f.align.to_le_bytes());
            out.extend((f.bytes.len() as u32).to_le_bytes());
            out.extend(&f.bytes);
            out.extend((f.relocs.len() as u32).to_le_bytes());
            for r in &f.relocs {
                out.extend(r.offset.to_le_bytes());
                out.push(KINDS.iter().position(|k| *k == r.kind).unwrap_or(0) as u8);
                out.extend(r.addend.to_le_bytes());
                match &r.target {
                    Target::Func(i) => {
                        out.push(0);
                        out.extend(i.to_le_bytes());
                    }
                    Target::Import(name) => {
                        out.push(1);
                        str(out, name);
                    }
                    Target::LibCall(name) => {
                        out.push(2);
                        str(out, name);
                    }
                }
            }
        }
    }

    /// The code `bytes` encodes, all of it; `None` for anything else.
    fn decode(bytes: &[u8]) -> Option<EmittedCode> {
        let mut r = Reader(bytes);
        let mut funcs = Vec::new();
        for _ in 0..r.u32()? {
            let align = r.u64()?;
            let len = r.u32()? as usize;
            let code = r.take(len)?.to_vec();
            let mut relocs = Vec::new();
            for _ in 0..r.u32()? {
                let offset = r.u32()?;
                let kind = *KINDS.get(r.u8()? as usize)?;
                let addend = r.u64()? as i64;
                let target = match r.u8()? {
                    0 => Target::Func(r.u32()?),
                    1 => Target::Import(r.str()?),
                    2 => Target::LibCall(r.str()?),
                    _ => return None,
                };
                relocs.push(EmittedReloc {
                    offset,
                    kind,
                    addend,
                    target,
                });
            }
            funcs.push(EmittedFn {
                align,
                bytes: code,
                relocs,
            });
        }
        r.0.is_empty().then_some(EmittedCode { funcs })
    }
}

/// Reads an encoding front to back; each read is `None` past its end.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.0.len() < n {
            return None;
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Some(head)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn str(&mut self) -> Option<String> {
        let len = self.u32()? as usize;
        String::from_utf8(self.take(len)?.to_vec()).ok()
    }
}

/// The key a root compile's code is kept under: the module, by its content `digest`, and what else
/// the compile bakes into its code ([`crate::CompiledModule::compile_kept`]).
pub(crate) fn root_key(
    digest: &[u8; 32],
    entry: u32,
    polls: bool,
    fuel: bool,
    fast: bool,
    reserved_log2: u8,
    table_reserve_log2: u8,
) -> [u8; 32] {
    let mut k = b"temen-jit root\0".to_vec();
    k.extend(digest);
    k.extend(entry.to_le_bytes());
    k.extend([
        polls as u8,
        fuel as u8,
        fast as u8,
        reserved_log2,
        table_reserve_log2,
    ]);
    temen_encode::digest256(&k)
}

/// The key a §14 child's code is kept under: what its in-memory key names but the `call.cap`
/// thunk, which only the code's link binds ([`crate::ChildCodeCache`]).
pub(crate) fn child_key(
    digest: &[u8; 32],
    entry: u32,
    mapped_log2: u8,
    reserved_log2: u8,
    cells: [bool; 4],
    table_reserve_log2: u8,
) -> [u8; 32] {
    let mut k = b"temen-jit child\0".to_vec();
    k.extend(digest);
    k.extend(entry.to_le_bytes());
    k.extend([mapped_log2, reserved_log2, table_reserve_log2]);
    k.extend(cells.map(u8::from));
    temen_encode::digest256(&k)
}

/// A directory of compiled code (temen#2452). A compile handed one keeps its code there, and a later
/// compile of the same image, in this process or another, defines that code instead of compiling
/// it.
///
/// The code is input to the escape-TCB, so it is kept only where its owner alone can write: the
/// directory must be this user's and writable by no one else, and so must each file read from it.
/// A file whose checksum fails, or that was kept by another build of temen or for another CPU, is a
/// miss. temen's build is the executable it runs in, by device, inode, size and modification time:
/// a temen linked into a shared library must not keep code. The directory holds one file per image
/// and compile shape, which a later build overwrites; what else it holds is its owner's to clear.
#[derive(Clone)]
pub struct CodeDir(Arc<DirInner>);

struct DirInner {
    path: PathBuf,
    /// What kept code must have been kept for: this build of temen on this CPU ([`stamp`]).
    stamp: [u8; 32],
}

/// The bytes a kept file starts with, its format's version after them.
const MAGIC: &[u8; 8] = b"temenjit";
const VERSION: u32 = 1;

impl CodeDir {
    /// The directory at `path`, made (only its owner can enter it) if it does not exist. Fails
    /// unless it is a directory owned by this user and writable by no one else, and on a target
    /// with no such permissions (only unix keeps code).
    pub fn open(path: impl Into<PathBuf>) -> std::io::Result<CodeDir> {
        let path = path.into();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            std::fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(&path)?;
            let meta = std::fs::metadata(&path)?;
            if !meta.is_dir() || !owner_only(&meta) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    format!(
                        "{}: compiled code is kept only in a directory its owner alone can write",
                        path.display()
                    ),
                ));
            }
            Ok(CodeDir(Arc::new(DirInner {
                path,
                stamp: stamp()?,
            })))
        }
        #[cfg(not(unix))]
        {
            let _ = path;
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "compiled code is kept on unix only",
            ))
        }
    }

    /// Where it is.
    pub fn path(&self) -> &Path {
        &self.0.path
    }

    fn file(&self, key: &[u8; 32]) -> PathBuf {
        let name: String = key.iter().map(|b| format!("{b:02x}")).collect();
        self.0.path.join(name)
    }

    /// The code kept under `key`, if a file holds it intact, its owner's alone, for this build on
    /// this CPU.
    pub(crate) fn load(&self, key: &[u8; 32]) -> Option<EmittedCode> {
        use std::io::Read;
        let mut bytes = Vec::new();
        open_owned(&self.file(key))?.read_to_end(&mut bytes).ok()?;
        let body = bytes.len().checked_sub(32)?;
        let (body, sum) = bytes.split_at(body);
        if temen_encode::digest256(body) != *sum {
            return None;
        }
        let mut r = Reader(body);
        (r.take(8)? == MAGIC
            && r.u32()? == VERSION
            && r.take(32)? == key
            && r.take(32)? == self.0.stamp.as_slice())
        .then(|| EmittedCode::decode(r.0))?
    }

    /// Keep `code` under `key`, replacing what the directory held under it: written aside, then
    /// renamed into place, so a reader sees one file or the other whole. A failure keeps nothing:
    /// a later compile compiles.
    pub(crate) fn keep(&self, key: &[u8; 32], code: &EmittedCode) {
        use std::io::Write;
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let mut bytes = MAGIC.to_vec();
        bytes.extend(VERSION.to_le_bytes());
        bytes.extend(key);
        bytes.extend(self.0.stamp);
        code.encode(&mut bytes);
        let sum = temen_encode::digest256(&bytes);
        bytes.extend(sum);
        let file = self.file(key);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let aside = file.with_extension(format!("{}.{n}.tmp", std::process::id()));
        let kept = create_owned(&aside)
            .and_then(|mut f| f.write_all(&bytes))
            .and_then(|()| std::fs::rename(&aside, &file));
        if kept.is_err() {
            let _ = std::fs::remove_file(&aside);
        }
    }
}

/// The regular file at `path`, not through a symlink, if it is this user's and writable by no one
/// else.
#[cfg(unix)]
fn open_owned(path: &Path) -> Option<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .ok()?;
    let meta = file.metadata().ok()?;
    (meta.is_file() && owner_only(&meta)).then_some(file)
}

/// A new file at `path`, not through a symlink, that only this user can read or write.
#[cfg(unix)]
fn create_owned(path: &Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
}

/// Whether `meta` is this user's and writable by no one else.
#[cfg(unix)]
fn owner_only(meta: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    // SAFETY: `geteuid` reads the process's effective user id; it cannot fail.
    meta.uid() == unsafe { libc::geteuid() } && meta.mode() & 0o022 == 0
}

/// Off unix no file is owned in that sense, so none is read and none kept ([`CodeDir::open`]
/// refuses there).
#[cfg(not(unix))]
fn open_owned(_: &Path) -> Option<std::fs::File> {
    None
}

#[cfg(not(unix))]
fn create_owned(_: &Path) -> std::io::Result<std::fs::File> {
    Err(std::io::ErrorKind::Unsupported.into())
}

/// What code this process keeps is valid for: this build of temen (the executable it runs in) on
/// this CPU (the ISA the JIT compiles for, with its flags).
#[cfg(unix)]
fn stamp() -> std::io::Result<[u8; 32]> {
    use std::os::unix::fs::MetadataExt;
    let exe = std::fs::metadata(std::env::current_exe()?)?;
    let isa = crate::jit_isa().map_err(|e| std::io::Error::other(format!("{e:?}")))?;
    let mut s = Vec::new();
    for n in [exe.dev(), exe.ino(), exe.size()] {
        s.extend(n.to_le_bytes());
    }
    s.extend(exe.mtime().to_le_bytes());
    s.extend(exe.mtime_nsec().to_le_bytes());
    s.extend(format!("{}\n{}\n", isa.triple(), isa.flags()).as_bytes());
    for flag in isa.isa_flags() {
        s.extend(format!("{flag}\n").as_bytes());
    }
    Ok(temen_encode::digest256(&s))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Code of two functions, the first naming each kind of target.
    fn sample() -> EmittedCode {
        let reloc = |offset, kind, addend, target| EmittedReloc {
            offset,
            kind,
            addend,
            target,
        };
        EmittedCode {
            funcs: vec![
                EmittedFn {
                    align: 16,
                    bytes: (0..40).collect(),
                    relocs: vec![
                        reloc(2, Reloc::Abs8, 0, Target::Import("temen.cap_thunk".into())),
                        reloc(13, Reloc::X86CallPCRel4, -4, Target::Func(1)),
                        reloc(30, Reloc::Abs8, 0, Target::LibCall("CeilF64".into())),
                    ],
                },
                EmittedFn {
                    align: 1,
                    bytes: vec![0xc3],
                    relocs: vec![],
                },
            ],
        }
    }

    #[test]
    fn kept_code_decodes_to_what_was_encoded_and_nothing_else() {
        let mut bytes = Vec::new();
        sample().encode(&mut bytes);
        assert_eq!(EmittedCode::decode(&bytes), Some(sample()));
        for n in 0..bytes.len() {
            assert_eq!(
                EmittedCode::decode(&bytes[..n]),
                None,
                "its first {n} bytes"
            );
        }
        bytes.push(0);
        assert_eq!(EmittedCode::decode(&bytes), None, "a byte past its end");
    }

    #[cfg(unix)]
    #[test]
    fn code_is_read_back_only_under_its_key_for_its_build() {
        let path = std::env::temp_dir().join(format!("temen-jit-keys-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        let dir = CodeDir::open(&path).expect("open");
        let (key, other) = ([7; 32], [8; 32]);
        dir.keep(&key, &sample());
        assert_eq!(dir.load(&key), Some(sample()));
        assert_eq!(dir.load(&other), None, "nothing is kept under another key");
        std::fs::rename(dir.file(&key), dir.file(&other)).expect("rename");
        assert_eq!(
            dir.load(&other),
            None,
            "a file names the key its code was kept under"
        );
        std::fs::rename(dir.file(&other), dir.file(&key)).expect("rename");
        let elsewhere = CodeDir(Arc::new(DirInner {
            path: path.clone(),
            stamp: [0; 32],
        }));
        assert_eq!(
            elsewhere.load(&key),
            None,
            "code kept by another build of temen, or for another CPU"
        );
        let _ = std::fs::remove_dir_all(&path);
    }
}
