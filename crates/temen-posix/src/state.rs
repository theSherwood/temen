//! #1894 — the personality's **guest-visible state** as bytes, so a POSIX guest freezes
//! (DURABILITY §4: everything inside the cut is data). The personality is a [`CapState::Captured`]
//! capability: a freeze calls [`capture`] and the artifact carries the bytes; a thaw builds a fresh
//! personality and hands them to [`restore`].
//!
//! What rides is what the guest can observe and nothing the embedder configures:
//!
//! - **The world:** the memfs (files, their write clock, empty directories), the unread stdin, the
//!   process table's zombies, the pid counter, the foreground process group.
//! - **The root process:** its pids, its heap allocator (its blocks live in the window, which rides),
//!   its fd table, its directory streams, argv, cwd, environment, and signal and stop state.
//!
//! The embedder's configuration (the command registry, the stdout sink, the spawn and net
//! delegates, the terminal, a pinned clock) does not ride: the thawing embedder grants it again, as
//! it did for the frozen run. Output already written does not ride either: the embedder has read it.
//!
//! Descriptions an fd table shares — an open file's offset, a pipe's buffer, an adopted core pipe
//! end's dup group — are written once and named by index, so a restored `dup` still shares with its
//! original. A socket is an edge the thaw does not re-bind: its fd comes back closed, the ordinary
//! failure a guest already handles (DURABILITY §4, "Across the boundary"). A live fork twin's process
//! is not carried here; the freeze declines while one is live (#1688).
//!
//! Every map is written in key order, so the same state always writes the same bytes (§12.6).

use super::*;

/// The format's own version: bumped on any change, and a mismatch fails the restore.
const VERSION: u8 = 1;

/// Why a restore refused the bytes: they are not this format, or not well formed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StateError;

/// The personality's state as bytes: the world's and the root process's (see the module doc).
pub(crate) fn capture(w: &World, p: &Proc) -> Vec<u8> {
    let mut b = Writer(vec![VERSION]);

    // The world.
    b.bytes(&w.stdin[w.stdin_pos.min(w.stdin.len())..]);
    let mut files: Vec<(&String, &MemFile)> = w.files.iter().collect();
    files.sort_by(|a, b| a.0.cmp(b.0));
    b.uleb(files.len() as u64);
    for (path, f) in files {
        b.str(path);
        b.bytes(&f.bytes);
        b.ileb(f.mtime);
    }
    b.ileb(w.fs_clock);
    let mut dirs: Vec<&String> = w.explicit_dirs.iter().collect();
    dirs.sort();
    b.strs(dirs.into_iter());
    b.uleb(w.net_next_port as u64);
    let mut zombies: Vec<(i32, i32, i32, i32)> = w
        .procs
        .iter()
        .filter_map(|(&pid, e)| match e {
            ProcEntry::Zombie { status, pgid, ppid } => Some((pid, *status, *pgid, *ppid)),
            ProcEntry::Live(_) => None,
        })
        .collect();
    zombies.sort();
    b.uleb(zombies.len() as u64);
    for (pid, status, pgid, ppid) in zombies {
        for v in [pid, status, pgid, ppid] {
            b.ileb(v as i64);
        }
    }
    b.ileb(w.next_pid as i64);
    b.ileb(w.fg_pgid as i64);

    // The descriptions the fd table shares, in first-appearance order.
    let mut files_d: Vec<&Arc<Mutex<OpenFile>>> = Vec::new();
    let mut pipes_d: Vec<&PipeBuf> = Vec::new();
    let mut tokens_d: Vec<&Arc<CorePipeToken>> = Vec::new();
    fn index<'a, T: ?Sized>(table: &mut Vec<&'a Arc<T>>, a: &'a Arc<T>) -> u64 {
        match table.iter().position(|x| Arc::ptr_eq(x, a)) {
            Some(i) => i as u64,
            None => {
                table.push(a);
                (table.len() - 1) as u64
            }
        }
    }
    let mut fds: Vec<(u8, u64)> = Vec::with_capacity(p.fds.len());
    for e in &p.fds {
        fds.push(match e {
            None => (0, 0),
            Some(FdEntry::Stdin) => (1, 0),
            Some(FdEntry::Stdout) => (2, 0),
            Some(FdEntry::Stderr) => (3, 0),
            Some(FdEntry::File(f)) => (4, index(&mut files_d, f)),
            Some(FdEntry::PipeRead(q)) => (5, index(&mut pipes_d, q)),
            Some(FdEntry::PipeWrite(q)) => (6, index(&mut pipes_d, q)),
            Some(FdEntry::CorePipe(t)) => (7, index(&mut tokens_d, t)),
            // A socket's other end is outside the cut: it comes back closed.
            Some(FdEntry::NetSock(_) | FdEntry::NetStream(_) | FdEntry::NetListener(_)) => (0, 0),
        });
    }
    b.uleb(files_d.len() as u64);
    for f in files_d {
        let f = f.lock().unwrap_or_else(|e| e.into_inner());
        b.str(&f.path);
        b.uleb(f.pos as u64);
        b.bool(f.writable);
    }
    b.uleb(pipes_d.len() as u64);
    for q in pipes_d {
        let q = q.lock().unwrap_or_else(|e| e.into_inner());
        let (x, y) = q.as_slices();
        b.uleb((x.len() + y.len()) as u64);
        b.0.extend_from_slice(x);
        b.0.extend_from_slice(y);
    }
    b.uleb(tokens_d.len() as u64);
    for t in tokens_d {
        b.ileb(t.get() as i64);
    }

    // The root process.
    for v in [p.pid, p.ppid, p.pgid] {
        b.ileb(v as i64);
    }
    b.opt(p.pending_exec_heap, |b, (base, end)| {
        b.uleb(base);
        b.uleb(end);
    });
    b.uleb(p.heap_next);
    b.uleb(p.heap_end);
    let mut allocated: Vec<(u64, u64)> = p.allocated.iter().map(|(&a, &n)| (a, n)).collect();
    allocated.sort();
    b.pairs_u(&allocated);
    b.pairs_u(&p.free_list);
    b.uleb(fds.len() as u64);
    for (tag, idx) in fds {
        b.0.push(tag);
        if tag >= 4 {
            b.uleb(idx);
        }
    }
    b.uleb(p.dirs.len() as u64);
    for d in &p.dirs {
        b.opt(d.as_ref(), |b, d| {
            b.strs(d.entries.iter());
            b.uleb(d.pos as u64);
        });
    }
    b.strs(p.args.iter());
    b.str(&p.cwd);
    let mut env: Vec<(&String, &String)> = p.env.iter().collect();
    env.sort();
    b.uleb(env.len() as u64);
    for (k, v) in env {
        b.str(k);
        b.str(v);
    }
    let mut env_ptrs: Vec<(&String, &u64)> = p.env_ptrs.iter().collect();
    env_ptrs.sort();
    b.uleb(env_ptrs.len() as u64);
    for (k, v) in env_ptrs {
        b.str(k);
        b.uleb(*v);
    }
    b.uleb(p.sig_pending);
    b.map_i(&p.sig_handler);
    b.uleb(p.sig_mask);
    let masks: HashMap<i32, i64> = p
        .sig_action_mask
        .iter()
        .map(|(&s, &m)| (s, m as i64))
        .collect();
    b.map_i(&masks);
    b.map_i(&p.sig_action_flags);
    b.uleb(p.sig_stack_base);
    b.opt(p.stopped_sig, |b, s| b.ileb(s as i64));
    b.bool(p.stop_fresh);
    b.bool(p.cont_fresh);
    b.bool(p.reap_wake);
    b.opt(p.term_sig, |b, s| b.ileb(s as i64));
    b.uleb(p.handler_mask_stack.len() as u64);
    for &m in &p.handler_mask_stack {
        b.uleb(m);
    }
    b.bool(p.restart_ok);
    b.opt(p.pending_exec.as_ref(), |b, (blob, argv)| {
        b.bytes(blob);
        b.strs(argv.iter());
    });
    b.opt(p.pending_exec_image.as_ref(), |b, img| b.bytes(img));
    b.bool(p.core_task);
    b.opt(p.term_in.as_ref(), |b, t| b.ileb(t.get() as i64));
    b.0
}

/// Put captured state back: into a fresh personality on a thaw, or into the live one on an
/// in-session rewind. Only the state [`capture`] writes is replaced; the embedder's configuration and
/// the run's doors (wake, stop, kill, park request) stay as they are.
pub(crate) fn restore(w: &mut World, p: &mut Proc, bytes: &[u8]) -> Result<(), StateError> {
    let mut r = Reader(bytes);
    if r.u8()? != VERSION {
        return Err(StateError);
    }

    // The world, into locals first: nothing changes unless the whole of it decodes.
    let stdin = r.bytes()?;
    let mut files = HashMap::new();
    for _ in 0..r.len()? {
        let path = r.str()?;
        let bytes = r.bytes()?;
        let mtime = r.ileb()?;
        files.insert(path, MemFile { bytes, mtime });
    }
    let fs_clock = r.ileb()?;
    let explicit_dirs: HashSet<String> = r.strs()?.into_iter().collect();
    let net_next_port = u16::try_from(r.uleb()?).map_err(|_| StateError)?;
    let mut zombies = Vec::new();
    for _ in 0..r.len()? {
        let (pid, status, pgid, ppid) = (r.i32()?, r.i32()?, r.i32()?, r.i32()?);
        zombies.push((pid, ProcEntry::Zombie { status, pgid, ppid }));
    }
    let next_pid = r.i32()?;
    let fg_pgid = r.i32()?;

    let mut files_d = Vec::new();
    for _ in 0..r.len()? {
        let path = r.str()?;
        let pos = r.uleb()? as usize;
        let writable = r.bool()?;
        files_d.push(Arc::new(Mutex::new(OpenFile {
            path,
            pos,
            writable,
        })));
    }
    let mut pipes_d: Vec<PipeBuf> = Vec::new();
    for _ in 0..r.len()? {
        pipes_d.push(Arc::new(Mutex::new(r.bytes()?.into())));
    }
    let mut tokens_d = Vec::new();
    for _ in 0..r.len()? {
        tokens_d.push(Arc::new(CorePipeToken::new(r.i32()?)));
    }

    let (pid, ppid, pgid) = (r.i32()?, r.i32()?, r.i32()?);
    let pending_exec_heap = r.opt(|r| Ok((r.uleb()?, r.uleb()?)))?;
    let heap_next = r.uleb()?;
    let heap_end = r.uleb()?;
    let allocated: HashMap<u64, u64> = r.pairs_u()?.into_iter().collect();
    let free_list = r.pairs_u()?;
    let mut fds = Vec::new();
    for _ in 0..r.len()? {
        let tag = r.u8()?;
        let idx = if tag >= 4 { r.uleb()? as usize } else { 0 };
        let get = |i: usize, n: usize| (i < n).then_some(i).ok_or(StateError);
        fds.push(match tag {
            0 => None,
            1 => Some(FdEntry::Stdin),
            2 => Some(FdEntry::Stdout),
            3 => Some(FdEntry::Stderr),
            4 => Some(FdEntry::File(Arc::clone(
                &files_d[get(idx, files_d.len())?],
            ))),
            5 => Some(FdEntry::PipeRead(Arc::clone(
                &pipes_d[get(idx, pipes_d.len())?],
            ))),
            6 => Some(FdEntry::PipeWrite(Arc::clone(
                &pipes_d[get(idx, pipes_d.len())?],
            ))),
            7 => Some(FdEntry::CorePipe(Arc::clone(
                &tokens_d[get(idx, tokens_d.len())?],
            ))),
            _ => return Err(StateError),
        });
    }
    let mut dirs = Vec::new();
    for _ in 0..r.len()? {
        dirs.push(r.opt(|r| {
            Ok(DirStream {
                entries: r.strs()?,
                pos: r.uleb()? as usize,
            })
        })?);
    }
    let args = r.strs()?;
    let cwd = r.str()?;
    let mut env = HashMap::new();
    for _ in 0..r.len()? {
        let k = r.str()?;
        env.insert(k, r.str()?);
    }
    let mut env_ptrs = HashMap::new();
    for _ in 0..r.len()? {
        let k = r.str()?;
        env_ptrs.insert(k, r.uleb()?);
    }
    let sig_pending = r.uleb()?;
    let sig_handler = r.map_i()?;
    let sig_mask = r.uleb()?;
    let sig_action_mask = r.map_i()?.into_iter().map(|(s, m)| (s, m as u64)).collect();
    let sig_action_flags = r.map_i()?;
    let sig_stack_base = r.uleb()?;
    let stopped_sig = r.opt(|r| r.i32())?;
    let stop_fresh = r.bool()?;
    let cont_fresh = r.bool()?;
    let reap_wake = r.bool()?;
    let term_sig = r.opt(|r| r.i32())?;
    let mut handler_mask_stack = Vec::new();
    for _ in 0..r.len()? {
        handler_mask_stack.push(r.uleb()?);
    }
    let restart_ok = r.bool()?;
    let pending_exec = r.opt(|r| Ok((r.bytes()?, r.strs()?)))?;
    let pending_exec_image = r.opt(|r| Ok(Arc::<[u8]>::from(r.bytes()?)))?;
    let core_task = r.bool()?;
    let term_in = r.opt(|r| Ok(CorePipeToken::new(r.i32()?)))?;
    if !r.0.is_empty() {
        return Err(StateError);
    }

    // It decoded whole: apply it.
    w.stdin = stdin;
    w.stdin_pos = 0;
    w.files = files;
    w.fs_clock = fs_clock;
    w.explicit_dirs = explicit_dirs;
    w.net_listeners.clear();
    w.net_next_port = net_next_port;
    w.procs.retain(|_, e| matches!(e, ProcEntry::Live(_)));
    w.procs.extend(zombies);
    w.next_pid = next_pid;
    w.fg_pgid = fg_pgid;
    w.last_fork_mint = None;

    // A stop the restore adds or ends reaches the engine's mirror when one is installed (an
    // in-session rewind); a thaw's run publishes a carried stop when it installs one.
    let was_stopped = p.stopped_sig.is_some();
    p.pid = pid;
    p.ppid = ppid;
    p.pgid = pgid;
    p.pending_exec_heap = pending_exec_heap;
    p.heap_next = heap_next;
    p.heap_end = heap_end;
    p.allocated = allocated;
    p.free_list = free_list;
    p.fds = fds;
    p.dirs = dirs;
    p.args = args;
    p.cwd = cwd;
    p.env = env;
    p.env_ptrs = env_ptrs;
    p.sig_pending = sig_pending;
    p.sig_handler = sig_handler;
    p.sig_mask = sig_mask;
    p.sig_action_mask = sig_action_mask;
    p.sig_action_flags = sig_action_flags;
    p.sig_stack_base = sig_stack_base;
    p.stopped_sig = stopped_sig;
    p.stop_fresh = stop_fresh;
    p.cont_fresh = cont_fresh;
    p.reap_wake = reap_wake;
    p.term_sig = term_sig;
    p.handler_mask_stack = handler_mask_stack;
    p.restart_ok = restart_ok;
    p.pending_exec = pending_exec;
    p.pending_exec_image = pending_exec_image;
    p.core_task = core_task;
    p.term_in = term_in;
    // A pending caught signal is deliverable again at the next poll.
    p.arm_signals();
    if p.stopped_sig.is_some() != was_stopped {
        p.fire_stop_apply(!was_stopped);
    }
    Ok(())
}

struct Writer(Vec<u8>);

impl Writer {
    fn uleb(&mut self, mut v: u64) {
        loop {
            let byte = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                self.0.push(byte);
                return;
            }
            self.0.push(byte | 0x80);
        }
    }
    fn ileb(&mut self, v: i64) {
        self.uleb(((v << 1) ^ (v >> 63)) as u64);
    }
    fn bool(&mut self, v: bool) {
        self.0.push(v as u8);
    }
    fn bytes(&mut self, v: &[u8]) {
        self.uleb(v.len() as u64);
        self.0.extend_from_slice(v);
    }
    fn str(&mut self, v: &str) {
        self.bytes(v.as_bytes());
    }
    fn strs<'s>(&mut self, v: impl ExactSizeIterator<Item = &'s String>) {
        self.uleb(v.len() as u64);
        for s in v {
            self.str(s);
        }
    }
    fn opt<T>(&mut self, v: Option<T>, f: impl FnOnce(&mut Self, T)) {
        match v {
            None => self.bool(false),
            Some(x) => {
                self.bool(true);
                f(self, x);
            }
        }
    }
    fn pairs_u(&mut self, v: &[(u64, u64)]) {
        self.uleb(v.len() as u64);
        for &(a, b) in v {
            self.uleb(a);
            self.uleb(b);
        }
    }
    /// A signal-keyed map, in signal order.
    fn map_i(&mut self, m: &HashMap<i32, i64>) {
        let mut v: Vec<(i32, i64)> = m.iter().map(|(&k, &v)| (k, v)).collect();
        v.sort();
        self.uleb(v.len() as u64);
        for (k, v) in v {
            self.ileb(k as i64);
            self.ileb(v);
        }
    }
}

struct Reader<'b>(&'b [u8]);

impl Reader<'_> {
    fn u8(&mut self) -> Result<u8, StateError> {
        let (&b, rest) = self.0.split_first().ok_or(StateError)?;
        self.0 = rest;
        Ok(b)
    }
    fn uleb(&mut self) -> Result<u64, StateError> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = self.u8()?;
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err(StateError)
    }
    fn ileb(&mut self) -> Result<i64, StateError> {
        let u = self.uleb()?;
        Ok(((u >> 1) as i64) ^ -((u & 1) as i64))
    }
    fn i32(&mut self) -> Result<i32, StateError> {
        i32::try_from(self.ileb()?).map_err(|_| StateError)
    }
    /// A count, bounded by the bytes left (every element takes at least one), so a forged count
    /// cannot make a restore allocate without limit.
    fn len(&mut self) -> Result<usize, StateError> {
        let n = self.uleb()? as usize;
        if n > self.0.len() {
            return Err(StateError);
        }
        Ok(n)
    }
    fn bool(&mut self) -> Result<bool, StateError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(StateError),
        }
    }
    fn bytes(&mut self) -> Result<Vec<u8>, StateError> {
        let n = self.len()?;
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head.to_vec())
    }
    fn str(&mut self) -> Result<String, StateError> {
        String::from_utf8(self.bytes()?).map_err(|_| StateError)
    }
    fn strs(&mut self) -> Result<Vec<String>, StateError> {
        (0..self.len()?).map(|_| self.str()).collect()
    }
    fn opt<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, StateError>,
    ) -> Result<Option<T>, StateError> {
        if self.bool()? {
            f(self).map(Some)
        } else {
            Ok(None)
        }
    }
    fn pairs_u(&mut self) -> Result<Vec<(u64, u64)>, StateError> {
        (0..self.len()?)
            .map(|_| Ok((self.uleb()?, self.uleb()?)))
            .collect()
    }
    fn map_i(&mut self) -> Result<HashMap<i32, i64>, StateError> {
        (0..self.len()?)
            .map(|_| Ok((self.i32()?, self.ileb()?)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A personality with state in every shape the codec carries: a file two fds share, a pipe whose
    /// ends are two fds, an environment, a heap block, and a socket.
    fn seeded() -> Posix {
        let posix = new_posix(4096, 8192, b"input".to_vec());
        {
            let mut w = posix.world.lock().unwrap();
            w.file_put("/a".into(), b"hello".to_vec());
            w.explicit_dirs.insert("/d".into());
            w.stdin_pos = 2;
        }
        let mut p = posix.root.lock().unwrap();
        let file = Arc::new(Mutex::new(OpenFile {
            path: "/a".into(),
            pos: 3,
            writable: true,
        }));
        let pipe: PipeBuf = Arc::new(Mutex::new(b"xy".iter().copied().collect()));
        p.fds.push(Some(FdEntry::File(Arc::clone(&file))));
        p.fds.push(Some(FdEntry::File(file)));
        p.fds.push(Some(FdEntry::PipeRead(Arc::clone(&pipe))));
        p.fds.push(Some(FdEntry::PipeWrite(pipe)));
        let (sock, _) = mem_pair(NetAddr::loopback(1), NetAddr::loopback(2));
        p.fds.push(Some(FdEntry::NetSock(sock)));
        p.env.insert("K".into(), "v".into());
        p.cwd = "/d".into();
        p.allocated.insert(4096, 16);
        p.heap_next = 4112;
        drop(p);
        posix
    }

    #[test]
    fn state_round_trips_and_shared_descriptions_stay_shared() {
        let posix = seeded();
        let bytes = posix.capture_state();
        let back = Posix::from_state(&bytes).expect("restore");
        assert_eq!(
            back.capture_state(),
            bytes,
            "canonical: the same bytes again"
        );
        assert_eq!(back.read_file("/a"), Some(b"hello".to_vec()));
        assert_eq!(back.cwd(), "/d");
        let p = back.root.lock().unwrap();
        match (&p.fds[3], &p.fds[4], &p.fds[5], &p.fds[6]) {
            (
                Some(FdEntry::File(a)),
                Some(FdEntry::File(b)),
                Some(FdEntry::PipeRead(r)),
                Some(FdEntry::PipeWrite(w)),
            ) => {
                assert!(Arc::ptr_eq(a, b), "a dup'd file still shares its offset");
                assert_eq!(a.lock().unwrap().pos, 3);
                assert!(Arc::ptr_eq(r, w), "a pipe's two ends share one buffer");
                assert_eq!(r.lock().unwrap().len(), 2);
            }
            other => panic!("unexpected fds {:?}", other.0.is_some()),
        }
        assert!(p.fds[7].is_none(), "a socket comes back closed");
        assert_eq!(p.heap_next, 4112);
        assert_eq!(p.allocated.get(&4096), Some(&16));
        drop(p);
        assert_eq!(
            back.world.lock().unwrap().stdin,
            b"put".to_vec(),
            "only the unread stdin rides"
        );
    }

    /// A stopped job's stop rides, and the thawed run's engine hears of it the moment it installs its
    /// stop mirror, so the domain parks at its first op instead of running on.
    #[test]
    fn a_restored_stop_reaches_the_engine_when_the_run_installs_its_mirror() {
        let posix = seeded();
        posix.root.lock().unwrap().stopped_sig = Some(19);
        let back = Posix::from_state(&posix.capture_state()).expect("restore");
        let applied: Arc<Mutex<Vec<bool>>> = Arc::default();
        let log = Arc::clone(&applied);
        let (door, _) = cap_signal_source(&back);
        door.set_stop_apply(Arc::new(move |s| log.lock().unwrap().push(s)));
        assert_eq!(
            *applied.lock().unwrap(),
            vec![true],
            "the stop was published"
        );

        // An in-session rewind to a running state publishes the continue.
        back.restore_state(&seeded().capture_state())
            .expect("restore");
        assert_eq!(*applied.lock().unwrap(), vec![true, false]);
    }

    #[test]
    fn a_truncated_or_foreign_state_is_refused_and_changes_nothing() {
        let bytes = seeded().capture_state();
        let posix = new_posix(0, 0, Vec::new());
        let before = posix.capture_state();
        assert_eq!(
            posix.restore_state(&bytes[..bytes.len() - 1]),
            Err(StateError)
        );
        let mut foreign = bytes.clone();
        foreign[0] = VERSION + 1;
        assert_eq!(posix.restore_state(&foreign), Err(StateError));
        assert_eq!(posix.capture_state(), before, "nothing changed");
    }
}
