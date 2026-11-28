//! Simplified 3-dimension durable [`BudgetTree`] (`mem`, `fuel`, `tasks`) plus D66 parallelism
//! `lanes` (Invariant 3; `#3187` decision 7).
//!
//! Each node is a **ceiling on its whole subtree** (cgroups `memory.max` / `pids.max` model):
//! - `split` mints a child node with ceilings clamped to its parent's and deducts nothing.
//! - `transfer` raises a descendant's ceiling (never a peer's), clamped to the holder's ceiling.
//! - Every charge walks from the paying node to the root, checking every ancestor and applying
//!   all-or-nothing; every refund walks the same path back.
//! - Parallelism `lanes` additionally enforces `Σ granted child lanes <= parent.lanes` at grant
//!   time (Invariant 3 D66 ruling).
//! - Internal process-wide host caps (`max_mappings` for `vm.max_map_count`, `max_fds` for
//!   `RLIMIT_NOFILE`) live outside the guest-visible budget tree in `temen-mem` / fd wrappers.

use crate::sync::{lock_unpoisoned, Mutex};

/// Opaque index of a node in a [`BudgetTree`]. Node `0` is always the run's root grant.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct BudgetId(pub u32);

impl BudgetId {
    pub const ROOT: BudgetId = BudgetId(0);
}

/// Ceilings for a budget node (all 3 guest-visible dimensions + D66 lane cap).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Limits {
    /// Maximum live bytes (guest windows + `SharedRegion`s + `CapPool` host buffers).
    pub mem: u64,
    /// Maximum total fuel units burnable by this subtree (`u64::MAX` = unlimited).
    pub fuel: u64,
    /// Maximum concurrent live `Context`s (threads/children; cgroups `pids.max`).
    pub tasks: u64,
    /// Maximum simultaneously executing Carriers (D66 parallelism ceiling).
    pub lanes: u32,
}

impl Default for Limits {
    fn default() -> Limits {
        Limits {
            mem: 4 * 1024 * 1024 * 1024, // 4 GiB default (#2268)
            fuel: u64::MAX,
            tasks: 1024,
            lanes: 64,
        }
    }
}

/// Durable serializable view of a single budget node (for warm snapshots / thaw ceiling lowering).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BudgetSnapshot {
    pub id: BudgetId,
    pub parent: Option<BudgetId>,
    pub limits: Limits,
    pub used_mem: u64,
    pub used_fuel: u64,
    pub used_tasks: u64,
    pub used_lanes: u32,
    pub granted_lanes: u32,
}

#[derive(Clone, Debug)]
struct Node {
    parent: Option<BudgetId>,
    limits: Limits,
    used_mem: u64,
    used_fuel: u64,
    used_tasks: u64,
    used_lanes: u32,
    /// Sum of `lanes` ceilings granted to direct children (`<= limits.lanes`, Invariant 3 D66).
    granted_lanes: u32,
}

/// Thread-safe tree of budget nodes shared by a runtime instance and all its domains.
pub struct BudgetTree {
    nodes: Mutex<Vec<Node>>,
}

impl BudgetTree {
    /// Create a new budget tree with root node `BudgetId::ROOT` holding `root_limits`.
    pub fn new(root_limits: Limits) -> BudgetTree {
        BudgetTree {
            nodes: Mutex::new(vec![Node {
                parent: None,
                limits: root_limits,
                used_mem: 0,
                used_fuel: 0,
                used_tasks: 0,
                used_lanes: 0,
                granted_lanes: 0,
            }]),
        }
    }

    /// Mint a child budget node under `parent`, clamping `requested` ceilings to `parent`'s
    /// ceilings. Returns `None` if `parent` does not exist or has insufficient ungranted D66 lanes
    /// (`parent.granted_lanes + clamped_lanes > parent.limits.lanes`).
    pub fn split(&self, parent: BudgetId, requested: Limits) -> Option<BudgetId> {
        let mut nodes = lock_unpoisoned(&self.nodes);
        let p = nodes.get(parent.0 as usize)?;
        let clamped = Limits {
            mem: requested.mem.min(p.limits.mem),
            fuel: requested.fuel.min(p.limits.fuel),
            tasks: requested.tasks.min(p.limits.tasks),
            lanes: requested.lanes.min(p.limits.lanes),
        };
        let new_granted = p.granted_lanes.checked_add(clamped.lanes)?;
        if new_granted > p.limits.lanes {
            return None;
        }
        nodes[parent.0 as usize].granted_lanes = new_granted;
        let id = BudgetId(nodes.len() as u32);
        nodes.push(Node {
            parent: Some(parent),
            limits: clamped,
            used_mem: 0,
            used_fuel: 0,
            used_tasks: 0,
            used_lanes: 0,
            granted_lanes: 0,
        });
        Some(id)
    }

    /// Lower a node's ceilings (e.g., on durable thaw re-hosting — Invariant 3 never raises
    /// ceilings through a thaw hook).
    pub fn lower_limits(&self, id: BudgetId, cap: Limits) -> bool {
        let mut nodes = lock_unpoisoned(&self.nodes);
        let Some(n) = nodes.get_mut(id.0 as usize) else {
            return false;
        };
        n.limits.mem = n.limits.mem.min(cap.mem);
        n.limits.fuel = n.limits.fuel.min(cap.fuel);
        n.limits.tasks = n.limits.tasks.min(cap.tasks);
        n.limits.lanes = n.limits.lanes.min(cap.lanes);
        true
    }

    /// Minimum remaining room along the ancestry chain from `id` to root.
    pub fn remaining(&self, id: BudgetId) -> Option<Limits> {
        let nodes = lock_unpoisoned(&self.nodes);
        let mut cur = Some(id);
        let mut room = Limits {
            mem: u64::MAX,
            fuel: u64::MAX,
            tasks: u64::MAX,
            lanes: u32::MAX,
        };
        let mut visited = false;
        while let Some(nid) = cur {
            let n = nodes.get(nid.0 as usize)?;
            visited = true;
            room.mem = room.mem.min(n.limits.mem.saturating_sub(n.used_mem));
            room.fuel = room.fuel.min(n.limits.fuel.saturating_sub(n.used_fuel));
            room.tasks = room.tasks.min(n.limits.tasks.saturating_sub(n.used_tasks));
            room.lanes = room.lanes.min(n.limits.lanes.saturating_sub(n.used_lanes));
            cur = n.parent;
        }
        visited.then_some(room)
    }

    /// All-or-nothing charge of `bytes` of `mem` up the ancestry chain from `id`.
    pub fn charge_mem(&self, id: BudgetId, bytes: u64) -> bool {
        if bytes == 0 {
            return true;
        }
        let mut nodes = lock_unpoisoned(&self.nodes);
        let chain = collect_chain(&nodes, id);
        if chain.is_empty() {
            return false;
        }
        for &idx in &chain {
            let n = &nodes[idx];
            if n.used_mem
                .checked_add(bytes)
                .is_none_or(|t| t > n.limits.mem)
            {
                return false;
            }
        }
        for &idx in &chain {
            nodes[idx].used_mem += bytes;
        }
        true
    }

    /// Refund `bytes` of `mem` up the ancestry chain from `id`.
    pub fn refund_mem(&self, id: BudgetId, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let mut nodes = lock_unpoisoned(&self.nodes);
        let mut cur = Some(id);
        while let Some(nid) = cur {
            let Some(n) = nodes.get_mut(nid.0 as usize) else {
                break;
            };
            n.used_mem = n.used_mem.saturating_sub(bytes);
            cur = n.parent;
        }
    }

    /// All-or-nothing charge of `count` live tasks up the ancestry chain from `id`.
    pub fn charge_tasks(&self, id: BudgetId, count: u64) -> bool {
        if count == 0 {
            return true;
        }
        let mut nodes = lock_unpoisoned(&self.nodes);
        let chain = collect_chain(&nodes, id);
        if chain.is_empty() {
            return false;
        }
        for &idx in &chain {
            let n = &nodes[idx];
            if n.used_tasks
                .checked_add(count)
                .is_none_or(|t| t > n.limits.tasks)
            {
                return false;
            }
        }
        for &idx in &chain {
            nodes[idx].used_tasks += count;
        }
        true
    }

    /// Refund `count` tasks up the ancestry chain from `id`.
    pub fn refund_tasks(&self, id: BudgetId, count: u64) {
        if count == 0 {
            return;
        }
        let mut nodes = lock_unpoisoned(&self.nodes);
        let mut cur = Some(id);
        while let Some(nid) = cur {
            let Some(n) = nodes.get_mut(nid.0 as usize) else {
                break;
            };
            n.used_tasks = n.used_tasks.saturating_sub(count);
            cur = n.parent;
        }
    }

    /// Draw up to `want` fuel units from `id` and every ancestor (Decision 30).
    /// Returns the actual granted chunk (`0` if any ancestor's fuel ceiling is exhausted).
    pub fn draw_fuel(&self, id: BudgetId, want: u64) -> u64 {
        if want == 0 {
            return 0;
        }
        let mut nodes = lock_unpoisoned(&self.nodes);
        let chain = collect_chain(&nodes, id);
        if chain.is_empty() {
            return 0;
        }
        let mut avail = want;
        for &idx in &chain {
            let n = &nodes[idx];
            if n.limits.fuel != u64::MAX {
                avail = avail.min(n.limits.fuel.saturating_sub(n.used_fuel));
            }
        }
        if avail == 0 {
            return 0;
        }
        for &idx in &chain {
            let n = &mut nodes[idx];
            if n.limits.fuel != u64::MAX {
                n.used_fuel = n.used_fuel.saturating_add(avail);
            }
        }
        avail
    }

    /// Refund unburned fuel from a Carrier's [`crate::FuelCell`] back up the ancestry chain.
    pub fn refund_fuel(&self, id: BudgetId, unused: u64) {
        if unused == 0 {
            return;
        }
        let mut nodes = lock_unpoisoned(&self.nodes);
        let mut cur = Some(id);
        while let Some(nid) = cur {
            let Some(n) = nodes.get_mut(nid.0 as usize) else {
                break;
            };
            if n.limits.fuel != u64::MAX {
                n.used_fuel = n.used_fuel.saturating_sub(unused);
            }
            cur = n.parent;
        }
    }

    /// Acquire one D66 execution lane along the ancestry chain from `id` to root.
    pub fn acquire_lane(&self, id: BudgetId) -> bool {
        let mut nodes = lock_unpoisoned(&self.nodes);
        let chain = collect_chain(&nodes, id);
        if chain.is_empty() {
            return false;
        }
        for &idx in &chain {
            let n = &nodes[idx];
            if n.used_lanes >= n.limits.lanes {
                return false;
            }
        }
        for &idx in &chain {
            nodes[idx].used_lanes += 1;
        }
        true
    }

    /// Release one D66 execution lane back up the ancestry chain.
    pub fn release_lane(&self, id: BudgetId) {
        let mut nodes = lock_unpoisoned(&self.nodes);
        let mut cur = Some(id);
        while let Some(nid) = cur {
            let Some(n) = nodes.get_mut(nid.0 as usize) else {
                break;
            };
            n.used_lanes = n.used_lanes.saturating_sub(1);
            cur = n.parent;
        }
    }

    /// Snapshot all nodes (for durability / inspection).
    pub fn snapshot(&self) -> Vec<BudgetSnapshot> {
        let nodes = lock_unpoisoned(&self.nodes);
        nodes
            .iter()
            .enumerate()
            .map(|(i, n)| BudgetSnapshot {
                id: BudgetId(i as u32),
                parent: n.parent,
                limits: n.limits,
                used_mem: n.used_mem,
                used_fuel: n.used_fuel,
                used_tasks: n.used_tasks,
                used_lanes: n.used_lanes,
                granted_lanes: n.granted_lanes,
            })
            .collect()
    }
}

fn collect_chain(nodes: &[Node], start: BudgetId) -> Vec<usize> {
    let mut chain = Vec::new();
    let mut cur = Some(start);
    while let Some(nid) = cur {
        let idx = nid.0 as usize;
        let Some(n) = nodes.get(idx) else {
            return Vec::new();
        };
        chain.push(idx);
        cur = n.parent;
    }
    chain
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[test]
    fn ceiling_overcommit_and_all_or_nothing_charge() {
        let tree = BudgetTree::new(Limits {
            mem: 1000,
            fuel: 500,
            tasks: 4,
            lanes: 4,
        });
        // Two children each given 800 mem (overcommitted relative to parent's 1000) and 2 lanes each.
        let c1 = tree
            .split(
                BudgetId::ROOT,
                Limits {
                    mem: 800,
                    fuel: 400,
                    tasks: 3,
                    lanes: 2,
                },
            )
            .unwrap();
        let c2 = tree
            .split(
                BudgetId::ROOT,
                Limits {
                    mem: 800,
                    fuel: 400,
                    tasks: 3,
                    lanes: 2,
                },
            )
            .unwrap();
        // A third child requesting 1 lane fails D66 sum check (2 + 2 + 1 > 4).
        assert!(tree
            .split(
                BudgetId::ROOT,
                Limits {
                    mem: 100,
                    fuel: 100,
                    tasks: 1,
                    lanes: 1,
                },
            )
            .is_none());

        assert!(tree.charge_mem(c1, 700));
        // c2 has 800 local room, but root only has 300 left -> all-or-nothing refusal.
        assert!(!tree.charge_mem(c2, 400));
        assert_eq!(tree.remaining(c2).unwrap().mem, 300);
        assert!(tree.charge_mem(c2, 300));
        tree.refund_mem(c1, 700);
        assert_eq!(tree.remaining(BudgetId::ROOT).unwrap().mem, 700);
    }

    #[test]
    fn fuel_draw_and_refund_respects_ancestors() {
        let tree = BudgetTree::new(Limits {
            mem: 1024,
            fuel: 100,
            tasks: 4,
            lanes: 2,
        });
        let child = tree
            .split(
                BudgetId::ROOT,
                Limits {
                    mem: 1024,
                    fuel: 80,
                    tasks: 2,
                    lanes: 1,
                },
            )
            .unwrap();

        assert_eq!(tree.draw_fuel(child, 60), 60);
        assert_eq!(tree.draw_fuel(child, 60), 20); // clamped by child's 80
        assert_eq!(tree.draw_fuel(child, 10), 0);
        tree.refund_fuel(child, 15);
        assert_eq!(tree.draw_fuel(child, 100), 15);
        assert_eq!(tree.remaining(BudgetId::ROOT).unwrap().fuel, 20);
    }

    #[test]
    fn lane_acquire_release_enforces_hierarchy() {
        let tree = BudgetTree::new(Limits {
            mem: 1024,
            fuel: u64::MAX,
            tasks: 10,
            lanes: 2,
        });
        let child = tree
            .split(
                BudgetId::ROOT,
                Limits {
                    mem: 1024,
                    fuel: u64::MAX,
                    tasks: 10,
                    lanes: 1,
                },
            )
            .unwrap();

        assert!(tree.acquire_lane(child));
        assert!(!tree.acquire_lane(child)); // child capped at 1
        assert!(tree.acquire_lane(BudgetId::ROOT)); // root takes 2nd lane
        assert!(!tree.acquire_lane(BudgetId::ROOT)); // root capped at 2
        tree.release_lane(child);
        assert!(tree.acquire_lane(child));
    }
}
