//! Fixed-pool page allocator and per-sequence block table.
//!
//! This module is plain Rust with no Python dependency, so its invariants are
//! tested with `cargo test` alone. `lib.rs` wraps it for Python.
//!
//! Invariants (checked by `PageForge::check_invariants` and the tests):
//! 1. every page id in `0..total` is either on the free list or owned by
//!    exactly one sequence, never both and never neither;
//! 2. `in_use[id]` is true exactly when `id` is owned by some sequence;
//! 3. `free_pages() + used_pages() == total_pages()`.

use std::collections::{HashMap, VecDeque};
use std::fmt;

pub type PageId = u32;
pub type SeqId = u64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocError {
    /// Asked for more pages than are free. Nothing was allocated.
    OutOfPages { requested: usize, free: usize },
    /// A page id outside `0..total` was handed back.
    InvalidPage(PageId),
    /// A page that is already free was handed back.
    DoubleFree(PageId),
    /// The sequence owns no pages.
    UnknownSequence(SeqId),
    /// Pool or page size of zero.
    InvalidConfig(&'static str),
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfPages { requested, free } => {
                write!(f, "out of pages: requested {requested}, {free} free")
            }
            Self::InvalidPage(id) => write!(f, "page id {id} is outside the pool"),
            Self::DoubleFree(id) => write!(f, "page {id} is already free (double free)"),
            Self::UnknownSequence(id) => write!(f, "sequence {id} owns no pages"),
            Self::InvalidConfig(msg) => write!(f, "invalid configuration: {msg}"),
        }
    }
}

impl std::error::Error for AllocError {}

/// FIFO free list over a fixed set of page ids.
///
/// FIFO (rather than LIFO) reuse spreads writes across the pool, and makes
/// reuse order deterministic, which the tests rely on.
#[derive(Debug, Clone)]
pub struct PageAllocator {
    free_list: VecDeque<PageId>,
    in_use: Vec<bool>,
}

impl PageAllocator {
    pub fn new(total: usize) -> Result<Self, AllocError> {
        if total == 0 {
            return Err(AllocError::InvalidConfig("n_pages must be > 0"));
        }
        if total > PageId::MAX as usize {
            return Err(AllocError::InvalidConfig("n_pages exceeds u32 range"));
        }
        Ok(Self {
            free_list: (0..total as PageId).collect(),
            in_use: vec![false; total],
        })
    }

    pub fn total(&self) -> usize {
        self.in_use.len()
    }

    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Take `n` pages. All-or-nothing: on error the free list is unchanged.
    /// O(1) per page.
    pub fn alloc(&mut self, n: usize) -> Result<Vec<PageId>, AllocError> {
        if n > self.free_list.len() {
            return Err(AllocError::OutOfPages {
                requested: n,
                free: self.free_list.len(),
            });
        }
        let pages: Vec<PageId> = self.free_list.drain(..n).collect();
        for &p in &pages {
            self.in_use[p as usize] = true;
        }
        Ok(pages)
    }

    /// Return pages to the free list. Validates every id before changing
    /// anything, so a bad id (or a duplicate within `ids`) leaves the
    /// allocator untouched. O(1) per page.
    pub fn free(&mut self, ids: &[PageId]) -> Result<(), AllocError> {
        let mut seen = Vec::with_capacity(ids.len());
        for &p in ids {
            let i = p as usize;
            if i >= self.in_use.len() {
                return Err(AllocError::InvalidPage(p));
            }
            if !self.in_use[i] || seen.contains(&p) {
                return Err(AllocError::DoubleFree(p));
            }
            seen.push(p);
        }
        for &p in ids {
            self.in_use[p as usize] = false;
            self.free_list.push_back(p);
        }
        Ok(())
    }

    pub fn is_in_use(&self, id: PageId) -> bool {
        self.in_use.get(id as usize).copied().unwrap_or(false)
    }

    pub fn free_ids(&self) -> impl Iterator<Item = &PageId> {
        self.free_list.iter()
    }
}

/// Logical-to-physical page mapping per sequence. Page `i` of a sequence's
/// table holds tokens `i*page_size .. (i+1)*page_size`.
#[derive(Debug, Clone, Default)]
pub struct BlockTable {
    table: HashMap<SeqId, Vec<PageId>>,
}

impl BlockTable {
    pub fn append(&mut self, seq: SeqId, pages: &[PageId]) {
        self.table.entry(seq).or_default().extend_from_slice(pages);
    }

    pub fn evict(&mut self, seq: SeqId) -> Option<Vec<PageId>> {
        self.table.remove(&seq)
    }

    pub fn get(&self, seq: SeqId) -> Option<&Vec<PageId>> {
        self.table.get(&seq)
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&SeqId, &Vec<PageId>)> {
        self.table.iter()
    }
}

/// Allocator plus block table: the object Python talks to.
#[derive(Debug, Clone)]
pub struct PageForgeCore {
    allocator: PageAllocator,
    blocks: BlockTable,
    page_size: usize,
}

impl PageForgeCore {
    pub fn new(n_pages: usize, page_size: usize) -> Result<Self, AllocError> {
        if page_size == 0 {
            return Err(AllocError::InvalidConfig("page_size must be > 0"));
        }
        Ok(Self {
            allocator: PageAllocator::new(n_pages)?,
            blocks: BlockTable::default(),
            page_size,
        })
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn total_pages(&self) -> usize {
        self.allocator.total()
    }

    pub fn free_pages(&self) -> usize {
        self.allocator.free_count()
    }

    pub fn used_pages(&self) -> usize {
        self.total_pages() - self.free_pages()
    }

    pub fn num_sequences(&self) -> usize {
        self.blocks.len()
    }

    /// Pages needed to hold `n_tokens` tokens.
    pub fn pages_needed(&self, n_tokens: usize) -> usize {
        n_tokens.div_ceil(self.page_size)
    }

    /// Append `n` fresh pages to `seq`'s table and return them.
    pub fn alloc_for_seq(&mut self, seq: SeqId, n: usize) -> Result<Vec<PageId>, AllocError> {
        let pages = self.allocator.alloc(n)?;
        self.blocks.append(seq, &pages);
        Ok(pages)
    }

    /// Grow `seq` so it can hold `n_tokens`; returns only the new pages.
    pub fn ensure_capacity(
        &mut self,
        seq: SeqId,
        n_tokens: usize,
    ) -> Result<Vec<PageId>, AllocError> {
        let have = self.blocks.get(seq).map_or(0, Vec::len);
        let need = self.pages_needed(n_tokens);
        if need <= have {
            return Ok(Vec::new());
        }
        self.alloc_for_seq(seq, need - have)
    }

    /// Release every page `seq` owns and return them.
    pub fn free_seq(&mut self, seq: SeqId) -> Result<Vec<PageId>, AllocError> {
        let pages = self
            .blocks
            .get(seq)
            .ok_or(AllocError::UnknownSequence(seq))?
            .clone();
        self.allocator.free(&pages)?;
        self.blocks.evict(seq);
        Ok(pages)
    }

    pub fn pages_for(&self, seq: SeqId) -> Option<&Vec<PageId>> {
        self.blocks.get(seq)
    }

    /// Verify the module-level invariants. O(total pages).
    pub fn check_invariants(&self) -> Result<(), String> {
        let total = self.total_pages();
        let mut owner = vec![0u8; total];
        for id in self.allocator.free_ids() {
            let i = *id as usize;
            if i >= total {
                return Err(format!("free list holds out-of-range page {id}"));
            }
            if self.allocator.is_in_use(*id) {
                return Err(format!("page {id} is on the free list but marked in use"));
            }
            owner[i] += 1;
        }
        for (seq, pages) in self.blocks.iter() {
            for id in pages {
                let i = *id as usize;
                if !self.allocator.is_in_use(*id) {
                    return Err(format!(
                        "sequence {seq} owns page {id}, which is not marked in use"
                    ));
                }
                owner[i] += 1;
            }
        }
        if let Some((id, n)) = owner.iter().enumerate().find(|(_, &n)| n != 1) {
            return Err(format!(
                "page {id} has {n} owners (free list + sequences); expected exactly 1"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn core(n: usize) -> PageForgeCore {
        PageForgeCore::new(n, 16).unwrap()
    }

    #[test]
    fn alloc_and_free_round_trip() {
        let mut pf = core(8);
        let a = pf.alloc_for_seq(0, 3).unwrap();
        assert_eq!(a, vec![0, 1, 2]);
        assert_eq!(pf.free_pages(), 5);
        assert_eq!(pf.free_seq(0).unwrap(), a);
        assert_eq!(pf.free_pages(), 8);
        pf.check_invariants().unwrap();
    }

    #[test]
    fn freed_pages_are_reused_in_fifo_order() {
        let mut pf = core(4);
        pf.alloc_for_seq(0, 2).unwrap(); // 0,1
        pf.alloc_for_seq(1, 2).unwrap(); // 2,3
        pf.free_seq(0).unwrap();
        assert_eq!(pf.alloc_for_seq(2, 2).unwrap(), vec![0, 1]);
        pf.check_invariants().unwrap();
    }

    #[test]
    fn out_of_pages_is_all_or_nothing() {
        let mut pf = core(4);
        pf.alloc_for_seq(0, 3).unwrap();
        let err = pf.alloc_for_seq(1, 2).unwrap_err();
        assert_eq!(
            err,
            AllocError::OutOfPages {
                requested: 2,
                free: 1
            }
        );
        assert_eq!(pf.free_pages(), 1);
        assert!(pf.pages_for(1).is_none());
        pf.check_invariants().unwrap();
    }

    #[test]
    fn double_free_and_invalid_page_are_rejected_without_side_effects() {
        let mut a = PageAllocator::new(4).unwrap();
        let p = a.alloc(2).unwrap();
        a.free(&p).unwrap();
        assert_eq!(a.free(&p[..1]), Err(AllocError::DoubleFree(p[0])));
        let q = a.alloc(1).unwrap();
        assert_eq!(a.free(&[q[0], q[0]]), Err(AllocError::DoubleFree(q[0])));
        assert_eq!(a.free(&[99]), Err(AllocError::InvalidPage(99)));
        assert!(a.is_in_use(q[0]));
        assert_eq!(a.free_count(), 3);
    }

    #[test]
    fn unknown_sequence_and_bad_config() {
        let mut pf = core(4);
        assert_eq!(pf.free_seq(7), Err(AllocError::UnknownSequence(7)));
        assert!(PageForgeCore::new(0, 16).is_err());
        assert!(PageForgeCore::new(4, 0).is_err());
    }

    #[test]
    fn ensure_capacity_allocates_only_missing_pages() {
        let mut pf = core(16);
        assert_eq!(pf.ensure_capacity(0, 1).unwrap().len(), 1);
        assert!(pf.ensure_capacity(0, 16).unwrap().is_empty());
        assert_eq!(pf.ensure_capacity(0, 17).unwrap().len(), 1);
        assert_eq!(pf.ensure_capacity(0, 59).unwrap().len(), 2);
        assert_eq!(pf.pages_for(0).unwrap().len(), 4);
        assert_eq!(pf.pages_needed(0), 0);
        pf.check_invariants().unwrap();
    }

    /// Randomised alloc/free against a simple model, checking invariants at
    /// every step. Deterministic xorshift so failures reproduce.
    #[test]
    fn randomised_operations_preserve_invariants() {
        let mut pf = core(64);
        let mut model: HashMap<SeqId, usize> = HashMap::new();
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..20_000 {
            let seq = next() % 12;
            if next() % 3 == 0 {
                let res = pf.free_seq(seq);
                match model.remove(&seq) {
                    Some(n) => assert_eq!(res.unwrap().len(), n),
                    None => assert_eq!(res, Err(AllocError::UnknownSequence(seq))),
                }
            } else {
                let n = (next() % 6) as usize;
                let free_before = pf.free_pages();
                match pf.alloc_for_seq(seq, n) {
                    Ok(p) => {
                        assert_eq!(p.len(), n);
                        *model.entry(seq).or_default() += n;
                    }
                    Err(AllocError::OutOfPages { .. }) => {
                        assert!(n > free_before);
                        assert_eq!(pf.free_pages(), free_before);
                    }
                    Err(e) => panic!("unexpected error {e}"),
                }
            }
            pf.check_invariants().unwrap();
            assert_eq!(pf.used_pages(), model.values().sum::<usize>());
        }
    }
}
