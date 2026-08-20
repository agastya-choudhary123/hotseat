//! Process-wide state: the model, the thread pool, and the single sequence slot.
//!
//! One sequence at a time, on purpose. The macOS tracker installs a *task*
//! exception port, so a process has one dirty-page tracker, and giving each
//! sequence its own would mean one tracker owning several regions and
//! dispatching faults by address. That is a straightforward extension and it is
//! not what this project is about, so the slot holds one.

use hs_engine::kv::KvLayout;
use hs_engine::model::Model;
use hs_engine::pool::Pool;
use hs_track::Kind;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::seq::Seq;

pub struct Worker {
    pub name: String,
    pub model: Arc<Model>,
    pub model_name: String,
    pub pool: Arc<Pool>,
    pub max_ctx: usize,
    pub layout: KvLayout,
    pub tracker_kind: Kind,
    pub cluster: usize,
    pub token: String,
    pub slot: Mutex<Option<Arc<Seq>>>,
    next_id: AtomicU64,
}

impl Worker {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        model: Arc<Model>,
        model_name: String,
        pool: Arc<Pool>,
        max_ctx: usize,
        layout: KvLayout,
        tracker_kind: Kind,
        cluster: usize,
        token: String,
    ) -> Worker {
        Worker {
            name,
            model,
            model_name,
            pool,
            max_ctx,
            layout,
            tracker_kind,
            cluster,
            token,
            slot: Mutex::new(None),
            next_id: AtomicU64::new(1),
        }
    }

    pub fn new_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    pub fn current(&self) -> Option<Arc<Seq>> {
        self.slot.lock().unwrap().clone()
    }

    /// Claim the slot, refusing if a sequence is still running there.
    pub fn take_slot(&self, seq: Arc<Seq>) -> Result<(), String> {
        let mut s = self.slot.lock().unwrap();
        if let Some(old) = s.as_ref() {
            let st = old.state.lock().unwrap();
            if !st.done && !st.migrated_away {
                return Err(format!("sequence {} is still running here", old.id));
            }
        }
        *s = Some(seq);
        Ok(())
    }

    /// Constant-time-ish comparison of the shared secret.
    ///
    /// The migration port accepts memory into another process's address space,
    /// so it is not open to anyone who can reach it. This is a shared token,
    /// not authentication worth the name — the deployment story is a private
    /// network, exactly as for the container runtime this grew out of.
    pub fn check_token(&self, given: &str) -> bool {
        let a = self.token.as_bytes();
        let b = given.as_bytes();
        let mut diff = (a.len() ^ b.len()) as u8;
        for i in 0..a.len().max(b.len()) {
            diff |= a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0);
        }
        diff == 0
    }
}
