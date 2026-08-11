//! svm allocator `imp`: forward to the C `malloc` family. The svm-llvm on-ramp
//! synthesizes `malloc`/`free`/`realloc`/`calloc` as an in-window guest bump
//! allocator (`synth_malloc`, LLVM.md slice S), so `std`'s global allocator
//! reaches the heap with no host crossing per call. Over-aligned requests
//! (`align > MIN_ALIGN`) are served by over-allocation with the base pointer
//! stashed in the word before the returned block.
use super::MIN_ALIGN;
use crate::alloc::Layout;
use crate::mem;
use crate::ptr;

unsafe extern "C" {
    #[link_name = "malloc"]
    fn c_malloc(size: usize) -> *mut u8;
    #[link_name = "calloc"]
    fn c_calloc(count: usize, size: usize) -> *mut u8;
    #[link_name = "realloc"]
    fn c_realloc(ptr: *mut u8, new_size: usize) -> *mut u8;
    #[link_name = "free"]
    fn c_free(ptr: *mut u8);
}

#[inline]
unsafe fn aligned_malloc(layout: Layout) -> *mut u8 {
    let align = layout.align();
    let hdr = mem::size_of::<usize>();
    // Enough room to align up and to stash the base pointer just below the result.
    let total = layout.size().wrapping_add(align).wrapping_add(hdr);
    let base = unsafe { c_malloc(total) };
    if base.is_null() {
        return ptr::null_mut();
    }
    let raw = base as usize;
    let aligned = (raw + hdr + align - 1) & !(align - 1);
    unsafe { ((aligned - hdr) as *mut usize).write(raw) };
    aligned as *mut u8
}

#[inline]
pub unsafe fn alloc(layout: Layout) -> *mut u8 {
    if layout.align() <= MIN_ALIGN && layout.align() <= layout.size() {
        unsafe { c_malloc(layout.size()) }
    } else {
        unsafe { aligned_malloc(layout) }
    }
}

#[inline]
pub unsafe fn alloc_zeroed(layout: Layout) -> *mut u8 {
    if layout.align() <= MIN_ALIGN && layout.align() <= layout.size() {
        unsafe { c_calloc(layout.size(), 1) }
    } else {
        let ptr = unsafe { alloc(layout) };
        if !ptr.is_null() {
            unsafe { ptr::write_bytes(ptr, 0, layout.size()) };
        }
        ptr
    }
}

#[inline]
pub unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
    if layout.align() <= MIN_ALIGN && layout.align() <= layout.size() {
        unsafe { c_free(ptr) }
    } else {
        let hdr = mem::size_of::<usize>();
        let base = unsafe { ((ptr as usize - hdr) as *mut usize).read() };
        unsafe { c_free(base as *mut u8) }
    }
}

#[inline]
pub unsafe fn realloc(ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    if layout.align() <= MIN_ALIGN && layout.align() <= new_size {
        unsafe { c_realloc(ptr, new_size) }
    } else {
        // Can't safely grow an over-aligned block in place; alloc + copy + free.
        let new_layout = match Layout::from_size_align(new_size, layout.align()) {
            Ok(l) => l,
            Err(_) => return ptr::null_mut(),
        };
        let dst = unsafe { alloc(new_layout) };
        if !dst.is_null() {
            let n = core::cmp::min(layout.size(), new_size);
            unsafe { ptr::copy_nonoverlapping(ptr, dst, n) };
            unsafe { dealloc(ptr, layout) };
        }
        dst
    }
}
