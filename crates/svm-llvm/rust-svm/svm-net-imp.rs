//! svm networking: `TcpStream`/`TcpListener`/`lookup_host` over the **`net` capability** (POSIX.md
//! §5a) — a named handle separate from `posix`, holding only the authority surface (`connect`/`bind`/
//! `accept`/`shutdown`/`resolve`). A connected socket is an ordinary personality **fd**: read/write
//! ride the posix `read`/`write` ops (`host::read_fd`/`write_fd`), close is the posix `close`, so the
//! data plane needs no socket surface at all. Loopback is the in-personality memnet (deterministic;
//! empty reads are `WouldBlock`, never a deadlocking block); beyond loopback the embedder's
//! `NetDelegate` grants or refuses — no `net` cap ⇒ everything here is `Unsupported`, fail closed.
//!
//! Socket knobs with no memnet meaning (nodelay/ttl/linger/keepalive, timeouts, nonblocking) are
//! accept-and-ignore setters with fixed-answer getters; `UdpSocket` awaits the memnet's datagram
//! slice and fails closed. A delegate-backed stream's `recv` may block host-side (the embedder owns
//! the real I/O), so read timeouts are advisory-ignored — document, don't pretend.
#![deny(unsafe_op_in_unsafe_fn)]
use crate::fmt;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut};
use crate::net::{Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, SocketAddrV4, SocketAddrV6, ToSocketAddrs};
use crate::sys::pal::host;
use crate::sys::{unsupported, unsupported_err};
use crate::time::Duration;

// The errnos the net/read/write ops return, mapped to the `ErrorKind`s programs match on.
const EAGAIN: i64 = -11;
const EACCES: i64 = -13;
const EPIPE: i64 = -32;
const EADDRINUSE: i64 = -98;
const ECONNREFUSED: i64 = -111;

fn err(code: i64) -> io::Error {
    match code {
        EAGAIN => io::ErrorKind::WouldBlock.into(),
        EACCES => io::ErrorKind::PermissionDenied.into(),
        EPIPE => io::ErrorKind::BrokenPipe.into(),
        EADDRINUSE => io::ErrorKind::AddrInUse.into(),
        ECONNREFUSED => io::ErrorKind::ConnectionRefused.into(),
        _ => io::Error::from_raw_os_error((-code) as i32),
    }
}

/// Encode a `SocketAddr` as the wire blob `[family u8 (4|6), port u16 LE, addr 4|16 bytes]`.
/// Returns the buffer and the used length (7 or 19).
fn encode_addr(a: &SocketAddr) -> ([u8; 19], usize) {
    let mut b = [0u8; 19];
    b[1..3].copy_from_slice(&a.port().to_le_bytes());
    match a {
        SocketAddr::V4(v4) => {
            b[0] = 4;
            b[3..7].copy_from_slice(&v4.ip().octets());
            (b, 7)
        }
        SocketAddr::V6(v6) => {
            b[0] = 6;
            b[3..19].copy_from_slice(&v6.ip().octets());
            (b, 19)
        }
    }
}

/// Decode one wire blob at the front of `b` → the address and the bytes consumed.
fn decode_addr(b: &[u8]) -> Option<(SocketAddr, usize)> {
    let port = u16::from_le_bytes([*b.get(1)?, *b.get(2)?]);
    match b.first()? {
        4 => {
            let o: [u8; 4] = b.get(3..7)?.try_into().ok()?;
            Some((SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::from(o), port)), 7))
        }
        6 => {
            let o: [u8; 16] = b.get(3..19)?.try_into().ok()?;
            Some((SocketAddr::V6(SocketAddrV6::new(Ipv6Addr::from(o), port, 0, 0)), 19))
        }
        _ => None,
    }
}

pub struct TcpStream {
    fd: i32,
    local: SocketAddr,
    peer: SocketAddr,
}

impl TcpStream {
    pub fn connect<A: ToSocketAddrs>(addr: A) -> io::Result<TcpStream> {
        super::each_addr(addr, TcpStream::connect_one)
    }

    fn connect_one(addr: &SocketAddr) -> io::Result<TcpStream> {
        if !host::have_net() {
            return Err(unsupported_err());
        }
        let (blob, len) = encode_addr(addr);
        let mut out = [0u8; 19];
        let fd = host::net_connect(blob.as_ptr(), len as i64, out.as_mut_ptr(), out.len() as i64);
        if fd < 0 {
            return Err(err(fd));
        }
        let local = decode_addr(&out).map(|(a, _)| a).unwrap_or(*addr);
        Ok(TcpStream { fd: fd as i32, local, peer: *addr })
    }

    pub fn connect_timeout(addr: &SocketAddr, _timeout: Duration) -> io::Result<TcpStream> {
        // The memnet connects synchronously and a delegate connect completes inside the host call;
        // there is no partial-connect state for a timeout to bound.
        TcpStream::connect_one(addr)
    }

    pub fn set_read_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        Ok(()) // advisory: the memnet never blocks; a delegate recv blocks host-side (see header)
    }

    pub fn set_write_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        Ok(())
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        Ok(None)
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        Ok(None)
    }

    pub fn peek(&self, _: &mut [u8]) -> io::Result<usize> {
        unsupported() // no non-consuming read op on the memnet
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let n = host::read_fd(self.fd as i64, buf.as_mut_ptr(), buf.len() as i64);
        if n < 0 { Err(err(n)) } else { Ok(n as usize) }
    }

    pub fn read_buf(&self, cursor: BorrowedCursor<'_, u8>) -> io::Result<()> {
        crate::io::default_read_buf(|b| self.read(b), cursor)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        for b in bufs {
            if !b.is_empty() {
                return self.read(b);
            }
        }
        Ok(0)
    }

    pub fn is_read_vectored(&self) -> bool {
        false
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let n = host::write_fd(self.fd as i64, buf.as_ptr(), buf.len() as i64);
        if n < 0 { Err(err(n)) } else { Ok(n as usize) }
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        for b in bufs {
            if !b.is_empty() {
                return self.write(b);
            }
        }
        Ok(0)
    }

    pub fn is_write_vectored(&self) -> bool {
        false
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        Ok(self.peer)
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        Ok(self.local)
    }

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        let how = match how {
            Shutdown::Read => 0,
            Shutdown::Write => 1,
            Shutdown::Both => 2,
        };
        let r = host::net_shutdown(self.fd as i64, how);
        if r < 0 { Err(err(r)) } else { Ok(()) }
    }

    pub fn duplicate(&self) -> io::Result<TcpStream> {
        let fd = host::dup(self.fd as i64);
        if fd < 0 {
            return Err(err(fd));
        }
        Ok(TcpStream { fd: fd as i32, local: self.local, peer: self.peer })
    }

    pub fn set_linger(&self, _: Option<Duration>) -> io::Result<()> {
        Ok(())
    }

    pub fn linger(&self) -> io::Result<Option<Duration>> {
        Ok(None)
    }

    pub fn set_keepalive(&self, _: bool) -> io::Result<()> {
        Ok(())
    }

    pub fn keepalive(&self) -> io::Result<bool> {
        Ok(false)
    }

    pub fn set_nodelay(&self, _: bool) -> io::Result<()> {
        Ok(()) // the memnet has no Nagle to disable
    }

    pub fn nodelay(&self) -> io::Result<bool> {
        Ok(true)
    }

    pub fn set_ttl(&self, _: u32) -> io::Result<()> {
        Ok(())
    }

    pub fn ttl(&self) -> io::Result<u32> {
        Ok(64)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        Ok(None)
    }

    pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
        Ok(()) // the memnet is effectively non-blocking already (empty reads are WouldBlock)
    }
}

impl Drop for TcpStream {
    fn drop(&mut self) {
        let _ = host::close(self.fd as i64);
    }
}

impl fmt::Debug for TcpStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TcpStream")
            .field("fd", &self.fd)
            .field("local", &self.local)
            .field("peer", &self.peer)
            .finish()
    }
}

pub struct TcpListener {
    fd: i32,
    local: SocketAddr,
}

impl TcpListener {
    pub fn bind<A: ToSocketAddrs>(addr: A) -> io::Result<TcpListener> {
        super::each_addr(addr, TcpListener::bind_one)
    }

    fn bind_one(addr: &SocketAddr) -> io::Result<TcpListener> {
        if !host::have_net() {
            return Err(unsupported_err());
        }
        let (blob, len) = encode_addr(addr);
        let mut out = [0u8; 19];
        let fd = host::net_bind(blob.as_ptr(), len as i64, out.as_mut_ptr(), out.len() as i64);
        if fd < 0 {
            return Err(err(fd));
        }
        // The written blob is the *actual* bound address — a `:0` request learns its ephemeral port.
        let local = decode_addr(&out).map(|(a, _)| a).unwrap_or(*addr);
        Ok(TcpListener { fd: fd as i32, local })
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        Ok(self.local)
    }

    pub fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        let mut out = [0u8; 19];
        let fd = host::net_accept(self.fd as i64, out.as_mut_ptr(), out.len() as i64);
        if fd < 0 {
            return Err(err(fd)); // -EAGAIN → WouldBlock: a cooperative guest cannot block on itself
        }
        let peer = decode_addr(&out)
            .map(|(a, _)| a)
            .ok_or_else(|| io::const_error!(io::ErrorKind::InvalidData, "bad peer address blob"))?;
        Ok((TcpStream { fd: fd as i32, local: self.local, peer }, peer))
    }

    pub fn duplicate(&self) -> io::Result<TcpListener> {
        let fd = host::dup(self.fd as i64);
        if fd < 0 {
            return Err(err(fd));
        }
        Ok(TcpListener { fd: fd as i32, local: self.local })
    }

    pub fn set_ttl(&self, _: u32) -> io::Result<()> {
        Ok(())
    }

    pub fn ttl(&self) -> io::Result<u32> {
        Ok(64)
    }

    pub fn set_only_v6(&self, _: bool) -> io::Result<()> {
        Ok(())
    }

    pub fn only_v6(&self) -> io::Result<bool> {
        Ok(false)
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        Ok(None)
    }

    pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
        Ok(())
    }
}

impl Drop for TcpListener {
    fn drop(&mut self) {
        let _ = host::close(self.fd as i64);
    }
}

impl fmt::Debug for TcpListener {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TcpListener")
            .field("fd", &self.fd)
            .field("local", &self.local)
            .finish()
    }
}

// UDP awaits the memnet's datagram slice (POSIX.md §5a follow-up) — uninhabited, every entry point
// fails closed, exactly the `unsupported` PAL's shape.
pub struct UdpSocket(!);

impl UdpSocket {
    pub fn bind<A: ToSocketAddrs>(_: A) -> io::Result<UdpSocket> {
        unsupported()
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        self.0
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        self.0
    }

    pub fn recv_from(&self, _: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.0
    }

    pub fn peek_from(&self, _: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.0
    }

    pub fn send_to(&self, _: &[u8], _: &SocketAddr) -> io::Result<usize> {
        self.0
    }

    pub fn duplicate(&self) -> io::Result<UdpSocket> {
        self.0
    }

    pub fn set_read_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        self.0
    }

    pub fn set_write_timeout(&self, _: Option<Duration>) -> io::Result<()> {
        self.0
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        self.0
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        self.0
    }

    pub fn set_broadcast(&self, _: bool) -> io::Result<()> {
        self.0
    }

    pub fn broadcast(&self) -> io::Result<bool> {
        self.0
    }

    pub fn set_multicast_loop_v4(&self, _: bool) -> io::Result<()> {
        self.0
    }

    pub fn multicast_loop_v4(&self) -> io::Result<bool> {
        self.0
    }

    pub fn set_multicast_ttl_v4(&self, _: u32) -> io::Result<()> {
        self.0
    }

    pub fn multicast_ttl_v4(&self) -> io::Result<u32> {
        self.0
    }

    pub fn set_multicast_loop_v6(&self, _: bool) -> io::Result<()> {
        self.0
    }

    pub fn multicast_loop_v6(&self) -> io::Result<bool> {
        self.0
    }

    pub fn join_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr) -> io::Result<()> {
        self.0
    }

    pub fn join_multicast_v6(&self, _: &Ipv6Addr, _: u32) -> io::Result<()> {
        self.0
    }

    pub fn leave_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr) -> io::Result<()> {
        self.0
    }

    pub fn leave_multicast_v6(&self, _: &Ipv6Addr, _: u32) -> io::Result<()> {
        self.0
    }

    pub fn set_ttl(&self, _: u32) -> io::Result<()> {
        self.0
    }

    pub fn ttl(&self) -> io::Result<u32> {
        self.0
    }

    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        self.0
    }

    pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
        self.0
    }

    pub fn recv(&self, _: &mut [u8]) -> io::Result<usize> {
        self.0
    }

    pub fn peek(&self, _: &mut [u8]) -> io::Result<usize> {
        self.0
    }

    pub fn send(&self, _: &[u8]) -> io::Result<usize> {
        self.0
    }

    pub fn connect<A: ToSocketAddrs>(&self, _: A) -> io::Result<()> {
        self.0
    }
}

impl fmt::Debug for UdpSocket {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0
    }
}

pub struct LookupHost {
    addrs: crate::vec::IntoIter<SocketAddr>,
}

impl Iterator for LookupHost {
    type Item = SocketAddr;
    fn next(&mut self) -> Option<SocketAddr> {
        self.addrs.next()
    }
}

/// Resolve `host` via the `net` cap's `resolve` op (`localhost` answers in-personality; anything
/// else needs the embedder's delegate). An IP literal short-circuits without a host call.
pub fn lookup_host(host: &str, port: u16) -> io::Result<LookupHost> {
    if let Ok(ip) = host.parse::<Ipv4Addr>() {
        return Ok(LookupHost { addrs: vec![SocketAddr::V4(SocketAddrV4::new(ip, port))].into_iter() });
    }
    if let Ok(ip) = host.parse::<Ipv6Addr>() {
        return Ok(LookupHost {
            addrs: vec![SocketAddr::V6(SocketAddrV6::new(ip, port, 0, 0))].into_iter(),
        });
    }
    if !host::have_net() {
        return Err(unsupported_err());
    }
    let hb = host.as_bytes();
    // Size-then-fetch, like `getenv_r`: probe for the total blob length, then read it.
    let need = host::net_resolve(hb.as_ptr(), hb.len() as i64, crate::ptr::null_mut(), 0);
    if need < 0 {
        return Err(io::const_error!(io::ErrorKind::NotFound, "failed to look up address"));
    }
    let mut buf = vec![0u8; need as usize];
    let got = host::net_resolve(hb.as_ptr(), hb.len() as i64, buf.as_mut_ptr(), buf.len() as i64);
    if got < 0 {
        return Err(io::const_error!(io::ErrorKind::NotFound, "failed to look up address"));
    }
    buf.truncate(got as usize);
    let mut addrs = Vec::new();
    let mut rest = &buf[..];
    while let Some((mut a, used)) = decode_addr(rest) {
        a.set_port(port);
        addrs.push(a);
        rest = &rest[used..];
    }
    Ok(LookupHost { addrs: addrs.into_iter() })
}
