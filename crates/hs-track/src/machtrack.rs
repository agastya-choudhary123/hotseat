//! Dirty-page tracking on macOS, via write protection and a Mach exception
//! handler.
//!
//! macOS has no soft-dirty bit and no `userfaultfd`. What it does have is the
//! Mach exception mechanism: a task can nominate a port that receives
//! `EXC_BAD_ACCESS` as a *message*, handled by an ordinary thread, while the
//! faulting thread stays parked inside the kernel waiting for a reply. So the
//! tracker takes write permission away from the KV cache, and every store the
//! model makes into a clean page turns into a message naming the address. The
//! handler records the page, hands the permission back, and replies
//! `KERN_SUCCESS`, at which point the kernel re-executes the store.
//!
//! The model never learns any of this happened. That is the property that
//! matters: the engine has no write log, no barriers, no cooperation with the
//! tracker at all. The dirty set comes from the hardware.
//!
//! # Handing a round over without losing a write
//!
//! Re-arming is the subtle part. A round boundary has to (a) make every page
//! fault again and (b) take the accumulated set, and a write can land between
//! those two steps. The scheme here is a two-buffer epoch:
//!
//! ```text
//! handler:   e0 = epoch                  harvest:  epoch += 1
//!            mark(buf[e0 & 1], page)               protect(whole region)
//!            unprotect(page)                       drain buf[old & 1]
//!            e1 = epoch
//!            if e1 != e0 { mark(buf[e1 & 1], page) }
//! ```
//!
//! If the handler's `unprotect` lands after the bulk `protect`, then the epoch
//! bump — which happens before the protect — is already visible, so `e1 != e0`
//! and the page is re-marked into the round that is now current. If it lands
//! before, the bulk protect re-arms it anyway. Either way a page that is
//! writable is a page that is marked, which is the only invariant that has to
//! hold. Reporting a page twice is harmless; missing one corrupts the transfer.

use crate::pageset::PageSet;
use crate::region::Region;
use crate::{Stats, Tracker};
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Mach bindings. Declared here rather than pulled from a crate so that the
// message layouts this file depends on are visible in this file.
// ---------------------------------------------------------------------------

type KernReturn = i32;
type MachPort = u32;
type VmProt = i32;

const KERN_SUCCESS: KernReturn = 0;
const KERN_FAILURE: KernReturn = 5;

const VM_PROT_READ: VmProt = 1;
const VM_PROT_WRITE: VmProt = 2;

const MACH_PORT_NULL: MachPort = 0;
const MACH_PORT_RIGHT_RECEIVE: u32 = 1;
const MACH_MSG_TYPE_MAKE_SEND: u32 = 20;

const EXC_MASK_BAD_ACCESS: u32 = 1 << 1;
const EXCEPTION_DEFAULT: i32 = 1;
const MACH_EXCEPTION_CODES: i32 = 0x8000_0000u32 as i32;

const MACH_SEND_MSG: i32 = 1;
const MACH_RCV_MSG: i32 = 2;
const MACH_RCV_TIMEOUT: i32 = 0x100;
const MACH_RCV_TIMED_OUT: i32 = 0x1000_4003u32 as i32;

const MACH_MSGH_BITS_REMOTE_MASK: u32 = 0x1f;

/// `mach_exception_raise`, the message id MIG assigns to
/// `EXCEPTION_DEFAULT | MACH_EXCEPTION_CODES`. Replies are id + 100.
const MSG_ID_EXCEPTION_RAISE: i32 = 2405;

#[cfg(target_arch = "aarch64")]
const THREAD_STATE_NONE: i32 = 5;
#[cfg(not(target_arch = "aarch64"))]
const THREAD_STATE_NONE: i32 = 13;

const EXC_SLOTS: usize = 32;

extern "C" {
    /// `mach_task_self()` is a macro over this global in C.
    static mach_task_self_: MachPort;

    fn mach_vm_protect(
        target: MachPort,
        address: u64,
        size: u64,
        set_maximum: i32,
        new_protection: VmProt,
    ) -> KernReturn;
    fn mach_port_allocate(task: MachPort, right: u32, name: *mut MachPort) -> KernReturn;
    fn mach_port_insert_right(
        task: MachPort,
        name: MachPort,
        poly: MachPort,
        poly_poly: u32,
    ) -> KernReturn;
    fn mach_port_deallocate(task: MachPort, name: MachPort) -> KernReturn;
    fn mach_port_mod_refs(task: MachPort, name: MachPort, right: u32, delta: i32) -> KernReturn;
    fn task_set_exception_ports(
        task: MachPort,
        exception_mask: u32,
        new_port: MachPort,
        behavior: i32,
        new_flavor: i32,
    ) -> KernReturn;
    fn task_get_exception_ports(
        task: MachPort,
        exception_mask: u32,
        masks: *mut u32,
        masks_cnt: *mut u32,
        old_handlers: *mut MachPort,
        old_behaviors: *mut i32,
        old_flavors: *mut i32,
    ) -> KernReturn;
    fn mach_msg(
        msg: *mut MachMsgHeader,
        option: i32,
        send_size: u32,
        rcv_size: u32,
        rcv_name: MachPort,
        timeout: u32,
        notify: MachPort,
    ) -> KernReturn;
}

#[inline(always)]
fn task_self() -> MachPort {
    unsafe { mach_task_self_ }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct MachMsgHeader {
    bits: u32,
    size: u32,
    remote_port: MachPort,
    local_port: MachPort,
    voucher_port: MachPort,
    id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PortDescriptor {
    name: MachPort,
    pad1: u32,
    pad2: u16,
    disposition: u8,
    kind: u8,
}

/// The `mach_exception_raise` request body.
///
/// ```text
///   0  mach_msg_header_t            24
///  24  msgh_descriptor_count         4
///  28  thread port descriptor       12
///  40  task port descriptor         12
///  52  NDR_record_t                  8
///  60  exception_type_t              4
///  64  codeCnt                       4
///  68  code[0]  kern_return sub-code 8   <-- not 8-byte aligned
///  76  code[1]  faulting address     8
///      total                        84
/// ```
///
/// `packed(4)` is load-bearing: MIG lays these out at four-byte alignment, so
/// the two 64-bit codes straddle an eight-byte boundary. Letting Rust align
/// them naturally shifts everything after `codeCnt` by four bytes, and the
/// symptom is a faulting address that looks like the real one rotated — which
/// is exactly how this was found.
#[repr(C, packed(4))]
#[derive(Clone, Copy)]
struct ExcRequest {
    head: MachMsgHeader,
    descriptor_count: u32,
    thread: PortDescriptor,
    task: PortDescriptor,
    ndr: [u8; 8],
    exception: i32,
    code_cnt: u32,
    /// The `kern_return_t` reason: `KERN_PROTECTION_FAILURE` for our writes.
    code_reason: i64,
    /// The address that faulted.
    code_addr: i64,
}

#[repr(C)]
struct ExcReply {
    head: MachMsgHeader,
    ndr: [u8; 8],
    ret_code: KernReturn,
}

/// Receive buffer: the request plus room for whatever trailer the kernel
/// appends.
#[repr(C, align(8))]
struct RecvBuf([u8; 512]);

// ---------------------------------------------------------------------------

struct Inner {
    region: Arc<Region>,
    /// Two dirty sets; `epoch & 1` selects the one currently accumulating.
    bufs: [PageSet; 2],
    epoch: AtomicU64,
    /// Pages made writable per fault. 1 is exact. Larger values trade a coarser
    /// dirty set for fewer faults and fewer VM map entries — see `docs/`.
    cluster: usize,
    armed: AtomicBool,
    stop: AtomicBool,
    port: MachPort,
    faults: AtomicU64,
    fault_ns: AtomicU64,
    pages_marked: AtomicU64,
    harvests: AtomicU64,
    rearm_ns: AtomicU64,
    /// Faults seen at addresses outside the region — these are real crashes and
    /// get handed back to whoever was handling them before.
    foreign: AtomicU64,
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

pub struct MachTracker {
    inner: Arc<Inner>,
    handler: Option<std::thread::JoinHandle<()>>,
    saved: SavedPorts,
}

/// The task's previous `EXC_BAD_ACCESS` handlers, restored on drop so that the
/// tracker does not permanently take over the process's crash handling.
struct SavedPorts {
    count: u32,
    masks: [u32; EXC_SLOTS],
    handlers: [MachPort; EXC_SLOTS],
    behaviors: [i32; EXC_SLOTS],
    flavors: [i32; EXC_SLOTS],
}

fn kr(what: &'static str, r: KernReturn) -> io::Result<()> {
    if r == KERN_SUCCESS {
        Ok(())
    } else {
        Err(io::Error::other(format!("{what} failed: kern_return {r}")))
    }
}

impl MachTracker {
    pub fn new(region: Arc<Region>, cluster: usize) -> io::Result<MachTracker> {
        assert!(cluster >= 1, "fault cluster must be at least one page");
        let task = task_self();

        let mut port: MachPort = 0;
        unsafe {
            kr("mach_port_allocate", mach_port_allocate(task, MACH_PORT_RIGHT_RECEIVE, &mut port))?;
            kr(
                "mach_port_insert_right",
                mach_port_insert_right(task, port, port, MACH_MSG_TYPE_MAKE_SEND),
            )?;
        }

        let mut saved = SavedPorts {
            count: EXC_SLOTS as u32,
            masks: [0; EXC_SLOTS],
            handlers: [0; EXC_SLOTS],
            behaviors: [0; EXC_SLOTS],
            flavors: [0; EXC_SLOTS],
        };
        unsafe {
            kr(
                "task_get_exception_ports",
                task_get_exception_ports(
                    task,
                    EXC_MASK_BAD_ACCESS,
                    saved.masks.as_mut_ptr(),
                    &mut saved.count,
                    saved.handlers.as_mut_ptr(),
                    saved.behaviors.as_mut_ptr(),
                    saved.flavors.as_mut_ptr(),
                ),
            )?;
        }

        let n_pages = region.n_pages();
        let inner = Arc::new(Inner {
            region,
            bufs: [PageSet::new(n_pages), PageSet::new(n_pages)],
            epoch: AtomicU64::new(0),
            cluster,
            armed: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            port,
            faults: AtomicU64::new(0),
            fault_ns: AtomicU64::new(0),
            pages_marked: AtomicU64::new(0),
            harvests: AtomicU64::new(0),
            rearm_ns: AtomicU64::new(0),
            foreign: AtomicU64::new(0),
        });

        unsafe {
            kr(
                "task_set_exception_ports",
                task_set_exception_ports(
                    task,
                    EXC_MASK_BAD_ACCESS,
                    port,
                    EXCEPTION_DEFAULT | MACH_EXCEPTION_CODES,
                    THREAD_STATE_NONE,
                ),
            )?;
        }

        let h = inner.clone();
        let handler = std::thread::Builder::new()
            .name("hs-mach-exc".into())
            .spawn(move || handler_loop(h))?;

        Ok(MachTracker { inner, handler: Some(handler), saved })
    }

    fn protect_all(&self, prot: VmProt) -> io::Result<()> {
        let r = &self.inner.region;
        unsafe {
            kr(
                "mach_vm_protect",
                mach_vm_protect(
                    task_self(),
                    r.addr() as u64,
                    r.len() as u64,
                    0,
                    prot,
                ),
            )
        }
    }
}

impl Tracker for MachTracker {
    fn name(&self) -> &'static str {
        "mach-vm-protect"
    }

    fn region(&self) -> &Arc<Region> {
        &self.inner.region
    }

    fn arm(&self) -> io::Result<()> {
        self.inner.bufs[0].clear();
        self.inner.bufs[1].clear();
        self.protect_all(VM_PROT_READ)?;
        self.inner.armed.store(true, Ordering::Release);
        Ok(())
    }

    fn harvest(&self, out: &PageSet) -> io::Result<()> {
        let t0 = std::time::Instant::now();
        // Order matters; see the module comment.
        let old = self.inner.epoch.fetch_add(1, Ordering::AcqRel);
        self.protect_all(VM_PROT_READ)?;
        self.inner.bufs[(old & 1) as usize].drain_into(out);
        self.inner.harvests.fetch_add(1, Ordering::Relaxed);
        self.inner.rearm_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn disarm(&self) -> io::Result<()> {
        self.inner.armed.store(false, Ordering::Release);
        self.protect_all(VM_PROT_READ | VM_PROT_WRITE)
    }

    fn stats(&self) -> Stats {
        let i = &self.inner;
        Stats {
            faults: i.faults.load(Ordering::Relaxed),
            fault_ns: i.fault_ns.load(Ordering::Relaxed),
            pages_marked: i.pages_marked.load(Ordering::Relaxed),
            harvests: i.harvests.load(Ordering::Relaxed),
            rearm_ns: i.rearm_ns.load(Ordering::Relaxed),
            foreign_faults: i.foreign.load(Ordering::Relaxed),
        }
    }
}

impl Drop for MachTracker {
    fn drop(&mut self) {
        let _ = self.disarm();
        self.inner.stop.store(true, Ordering::Release);
        if let Some(h) = self.handler.take() {
            let _ = h.join();
        }
        unsafe {
            let task = task_self();
            // Put the previous EXC_BAD_ACCESS handlers back before dropping our
            // port, so a later genuine crash is reported the way it would have
            // been if this tracker had never existed.
            for i in 0..self.saved.count as usize {
                task_set_exception_ports(
                    task,
                    self.saved.masks[i],
                    self.saved.handlers[i],
                    self.saved.behaviors[i],
                    self.saved.flavors[i],
                );
                if self.saved.handlers[i] != MACH_PORT_NULL {
                    mach_port_deallocate(task, self.saved.handlers[i]);
                }
            }
            if self.saved.count == 0 {
                task_set_exception_ports(
                    task,
                    EXC_MASK_BAD_ACCESS,
                    MACH_PORT_NULL,
                    EXCEPTION_DEFAULT,
                    THREAD_STATE_NONE,
                );
            }
            mach_port_deallocate(task, self.inner.port);
            mach_port_mod_refs(task, self.inner.port, MACH_PORT_RIGHT_RECEIVE, -1);
        }
    }
}

/// Receives exception messages until the tracker is dropped.
///
/// The receive has a timeout only so that this thread notices `stop`; a
/// timed-out receive is not an error.
fn handler_loop(inner: Arc<Inner>) {
    let task = task_self();
    let trace = std::env::var_os("HS_TRACK_TRACE").is_some();
    if trace {
        eprintln!("hs-track: handler thread up on port {}", inner.port);
    }
    let mut buf = RecvBuf([0u8; 512]);
    while !inner.stop.load(Ordering::Acquire) {
        let rc = unsafe {
            mach_msg(
                buf.0.as_mut_ptr() as *mut MachMsgHeader,
                MACH_RCV_MSG | MACH_RCV_TIMEOUT,
                0,
                buf.0.len() as u32,
                inner.port,
                50, // ms
                MACH_PORT_NULL,
            )
        };
        if rc == MACH_RCV_TIMED_OUT {
            continue;
        }
        if trace {
            let h = unsafe { &*(buf.0.as_ptr() as *const MachMsgHeader) };
            eprintln!("hs-track: msg rc={rc:#x} id={} size={} bits={:#x}", h.id, h.size, h.bits);
        }
        if rc != KERN_SUCCESS {
            // Nothing sensible to do here, and swallowing it would hide a real
            // problem behind a mysteriously stalled migration.
            eprintln!("hs-track: mach_msg receive failed: {rc}");
            continue;
        }

        let t0 = std::time::Instant::now();
        // Copied out rather than referenced: the buffer holds a packed struct.
        let req: ExcRequest = unsafe { std::ptr::read_unaligned(buf.0.as_ptr() as *const ExcRequest) };
        let ret = if { req.head }.id == MSG_ID_EXCEPTION_RAISE {
            if trace {
                let (e, r, a) = (req.exception, req.code_reason, req.code_addr);
                eprintln!("hs-track: exception={e} reason={r} at {a:#x}");
            }
            handle_fault(&inner, req.code_addr as usize)
        } else {
            KERN_FAILURE
        };
        if trace {
            eprintln!("hs-track: replying {ret}");
        }

        // The request carried send rights for the faulting thread and its task.
        // They are ours now, and there is one pair per fault, so failing to
        // drop them would exhaust the port name space in a few million faults.
        unsafe {
            if req.descriptor_count >= 2 {
                mach_port_deallocate(task, req.thread.name);
                mach_port_deallocate(task, req.task.name);
            }
        }

        let head = req.head;
        let mut reply = ExcReply {
            head: MachMsgHeader {
                bits: head.bits & MACH_MSGH_BITS_REMOTE_MASK,
                size: std::mem::size_of::<ExcReply>() as u32,
                remote_port: head.remote_port,
                local_port: MACH_PORT_NULL,
                voucher_port: MACH_PORT_NULL,
                id: head.id + 100,
            },
            ndr: req.ndr,
            ret_code: ret,
        };
        let src = unsafe {
            mach_msg(
                &mut reply.head,
                MACH_SEND_MSG,
                std::mem::size_of::<ExcReply>() as u32,
                0,
                MACH_PORT_NULL,
                0,
                MACH_PORT_NULL,
            )
        };
        if src != KERN_SUCCESS {
            eprintln!("hs-track: mach_msg reply failed: {src}");
        }
        inner.fault_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// Record the write and hand the permission back.
///
/// Returning `KERN_FAILURE` for an address outside the tracked region makes the
/// kernel pass the exception along to the next handler, which is what turns a
/// genuine null-pointer dereference elsewhere in the process back into an
/// ordinary crash instead of a silent hang.
fn handle_fault(inner: &Inner, addr: usize) -> KernReturn {
    let Some(page) = inner.region.page_of(addr) else {
        inner.foreign.fetch_add(1, Ordering::Relaxed);
        return KERN_FAILURE;
    };
    if !inner.armed.load(Ordering::Acquire) {
        inner.foreign.fetch_add(1, Ordering::Relaxed);
        return KERN_FAILURE;
    }

    let ps = inner.region.page_size();
    let first = page / inner.cluster * inner.cluster;
    let count = inner.cluster.min(inner.region.n_pages() - first);

    let e0 = inner.epoch.load(Ordering::Acquire);
    inner.bufs[(e0 & 1) as usize].mark_range(first, count);

    let r = unsafe {
        mach_vm_protect(
            task_self(),
            (inner.region.addr() + first * ps) as u64,
            (count * ps) as u64,
            0,
            VM_PROT_READ | VM_PROT_WRITE,
        )
    };
    if r != KERN_SUCCESS {
        eprintln!("hs-track: mach_vm_protect(rw) failed at page {page}: {r}");
        return KERN_FAILURE;
    }

    let e1 = inner.epoch.load(Ordering::Acquire);
    if e1 != e0 {
        // A harvest ran while we were unprotecting; make sure the round that is
        // now current also knows this page is writable.
        inner.bufs[(e1 & 1) as usize].mark_range(first, count);
    }

    inner.faults.fetch_add(1, Ordering::Relaxed);
    inner.pages_marked.fetch_add(count as u64, Ordering::Relaxed);
    KERN_SUCCESS
}
