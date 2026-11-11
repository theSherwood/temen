//! A surface follows its embedder's size (#2837). `surface.c` draws a frame at each size the
//! embedder gives, asking for the next through a `resize` cap of this test's: 64x48, 96x80, no area,
//! then 40x30. It checks every pixel it reads back, and that a surface with no area gives nothing to
//! draw. It runs on an offscreen surface, whose frames this test reads, and on a window's, under a
//! display (an Xvfb of the test's own when `DISPLAY` is unset). On all three engines.

mod common;

use std::sync::Arc;

use temen_webgpu::{Frames, Size, Surface};

/// A display for a window test: the one `DISPLAY` names, or else an Xvfb of the test's own, which
/// stops when this drops. With neither, the test fails, saying what to install, as with no adapter.
struct Display(Option<std::process::Child>);

impl Drop for Display {
    fn drop(&mut self) {
        if let Some(x) = &mut self.0 {
            let _ = x.kill();
            let _ = x.wait();
        }
    }
}

fn display() -> Display {
    use std::path::Path;
    use std::time::{Duration, Instant};
    if std::env::var_os("DISPLAY").is_some() {
        return Display(None);
    }
    // The first display number no other server holds. One another process takes first exits, and
    // the next is tried.
    for n in 100..200 {
        if Path::new(&format!("/tmp/.X{n}-lock")).exists() {
            continue;
        }
        let mut x = std::process::Command::new("Xvfb")
            .arg(format!(":{n}"))
            .args(["-screen", "0", "1024x768x24", "-nolisten", "tcp"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("no display: set DISPLAY, or install Xvfb (`xvfb`)");
        let socket = format!("/tmp/.X11-unix/X{n}");
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if x.try_wait().ok().flatten().is_some() {
                break;
            }
            if Path::new(&socket).exists() {
                std::env::set_var("DISPLAY", format!(":{n}"));
                return Display(Some(x));
            }
            assert!(
                Instant::now() < deadline,
                "Xvfb :{n} made no socket in 20 s"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    panic!("no free display for Xvfb among :100 to :199");
}

/// The test's `resize` cap: op 0 sets the surface's size to its two arguments.
fn resize_cap(size: Size) -> temen_run::HostCap {
    temen_run::HostCap::host_proc(0, move |_, _| {
        let size = size.clone();
        let h: temen_interp::HostProc = Box::new(move |_op, args, out, _mem, _minter| {
            size.set(args[0] as u32, args[1] as u32);
            temen_interp::put_results(out, &[0])
        });
        Some((h, temen_interp::CapState::Uncaptured))
    })
}

fn run(surface: Surface, size: Size, backend: temen_run::Backend) -> String {
    let inst = common::guest("surface");
    let caps = [
        ("webgpu", temen_webgpu::webgpu_cap(surface)),
        ("resize", resize_cap(size)),
    ];
    let stdout = common::run(&inst, backend, &caps);
    assert!(
        stdout.contains("frame 64x48: OK")
            && stdout.contains("frame 96x80: OK")
            && stdout.contains("no area: OK")
            && stdout.contains("frame 40x30: OK")
            && stdout.contains("surface: OK"),
        "surface ({backend:?}) self-check failed; stdout:\n{stdout}"
    );
    stdout
}

#[test]
fn offscreen_surface_follows_its_size() {
    common::require_adapter();
    for backend in common::ENGINES {
        let size = Size::new(64, 48);
        let frames = Frames::default();
        let surface = Surface::Offscreen {
            size: size.clone(),
            format: temen_gpu::TextureFormat::Rgba8Unorm,
            frames: frames.clone(),
        };
        run(surface, size, backend);
        // Three frames shown, not the one with no area; the last at the last size, all blue.
        let (shown, last) = frames.last();
        let last = last.expect("a presented frame");
        assert_eq!((shown, last.width, last.height), (3, 40, 30), "{backend:?}");
        assert!(
            last.texels.chunks(4).all(|t| t == [0, 0, 255, 255]),
            "the last frame is the guest's blue ({backend:?})"
        );
    }
}

/// An X11 window of the test's own: its connection, which wgpu presents through, and its id.
struct XWindow {
    conn: x11rb::xcb_ffi::XCBConnection,
    screen: i32,
    window: u32,
}

impl XWindow {
    fn new(width: u16, height: u16) -> XWindow {
        use x11rb::connection::Connection;
        use x11rb::protocol::xproto::{ConnectionExt, CreateWindowAux, WindowClass};
        let (conn, screen) = x11rb::xcb_ffi::XCBConnection::connect(None).expect("an X connection");
        let root = &conn.setup().roots[screen];
        let (parent, black) = (root.root, root.black_pixel);
        let window = conn.generate_id().expect("a window id");
        conn.create_window(
            x11rb::COPY_DEPTH_FROM_PARENT,
            window,
            parent,
            0,
            0,
            width,
            height,
            0,
            WindowClass::INPUT_OUTPUT,
            0,
            &CreateWindowAux::new().background_pixel(black),
        )
        .expect("a window");
        conn.map_window(window).expect("map the window");
        conn.flush().expect("flush");
        XWindow {
            conn,
            screen: screen as i32,
            window,
        }
    }
}

impl raw_window_handle::HasWindowHandle for XWindow {
    fn window_handle(
        &self,
    ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        let w = raw_window_handle::XcbWindowHandle::new(self.window.try_into().unwrap());
        // SAFETY: the window lives as long as `self`, and so does the handle borrowed from it.
        Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(w.into()) })
    }
}

impl raw_window_handle::HasDisplayHandle for XWindow {
    fn display_handle(
        &self,
    ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        let c = std::ptr::NonNull::new(self.conn.get_raw_xcb_connection());
        let d = raw_window_handle::XcbDisplayHandle::new(c, self.screen);
        // SAFETY: the connection lives as long as `self`, and so does the handle borrowed from it.
        Ok(unsafe { raw_window_handle::DisplayHandle::borrow_raw(d.into()) })
    }
}

#[test]
fn window_surface_follows_its_size() {
    common::require_adapter();
    let _display = display();
    let window: Arc<dyn temen_webgpu::WindowHandle> = Arc::new(XWindow::new(64, 48));
    for backend in common::ENGINES {
        let size = Size::new(64, 48);
        let surface = Surface::Window {
            window: Arc::clone(&window),
            size: size.clone(),
        };
        run(surface, size, backend);
    }
}

/// A host that runs the protocol in-process, with no guest, as a native compositor's tests do,
/// drives `temen_gpu::Gpu` over [`temen_webgpu::Wgpu`] itself: the same checks and surfaces.
#[test]
fn an_in_process_host_presents() {
    use temen_gpu::command::{ColorAttachment, Command, Encoder};
    use temen_gpu::enums::{LoadOp, StoreOp};
    common::require_adapter();
    let frames = Frames::default();
    let surface = Surface::Offscreen {
        size: Size::new(8, 4),
        format: temen_gpu::TextureFormat::Rgba8Unorm,
        frames: frames.clone(),
    };
    let mut gpu = temen_gpu::Gpu::new(temen_webgpu::Wgpu::new(surface));
    let mut e = Encoder::new();
    e.push(&Command::Acquire {
        texture: 1,
        view: 2,
    });
    e.push(&Command::BeginRenderPass {
        colors: vec![ColorAttachment {
            view: 2,
            resolve: None,
            load: LoadOp::Clear,
            store: StoreOp::Store,
            clear: [0.0, 1.0, 0.0, 1.0],
        }],
    });
    e.push(&Command::EndPass);
    e.push(&Command::Present {
        texture: 1,
        view: 2,
    });
    assert_eq!(gpu.submit(&e.bytes), 0, "the stream runs");
    let (shown, frame) = frames.last();
    let frame = frame.expect("a presented frame");
    assert_eq!((shown, frame.width, frame.height), (1, 8, 4));
    assert!(frame.texels.chunks(4).all(|t| t == [0, 255, 0, 255]));
}
