//! PageForge Rust core: a fixed-pool page allocator and block table.
//!
//! Built as the Python extension module `_pageforge` when the `python` feature
//! is on (maturin does this). Without it, the crate is plain Rust and
//! `cargo test` exercises the allocator with no interpreter involved.

pub mod allocator;

pub use allocator::{AllocError, BlockTable, PageAllocator, PageForgeCore, PageId, SeqId};

#[cfg(feature = "python")]
mod python {
    use super::allocator::{AllocError, PageForgeCore, PageId, SeqId};
    use pyo3::exceptions::{PyKeyError, PyMemoryError, PyRuntimeError, PyValueError};
    use pyo3::prelude::*;

    fn to_py(err: AllocError) -> PyErr {
        let msg = err.to_string();
        match err {
            AllocError::OutOfPages { .. } => PyMemoryError::new_err(msg),
            AllocError::UnknownSequence(_) => PyKeyError::new_err(msg),
            AllocError::InvalidPage(_)
            | AllocError::DoubleFree(_)
            | AllocError::InvalidConfig(_) => PyValueError::new_err(msg),
        }
    }

    /// Page allocator + block table for one KV-cache pool.
    ///
    /// Out of pages raises MemoryError (and allocates nothing); freeing an
    /// unknown sequence raises KeyError.
    #[pyclass(name = "PageForge", module = "_pageforge")]
    pub struct PyPageForge {
        inner: PageForgeCore,
    }

    #[pymethods]
    impl PyPageForge {
        #[new]
        fn new(n_pages: usize, page_size: usize) -> PyResult<Self> {
            Ok(Self {
                inner: PageForgeCore::new(n_pages, page_size).map_err(to_py)?,
            })
        }

        /// Append `n` pages to `seq_id`'s block table; returns the new page ids.
        fn alloc_for_seq(&mut self, seq_id: SeqId, n: usize) -> PyResult<Vec<PageId>> {
            self.inner.alloc_for_seq(seq_id, n).map_err(to_py)
        }

        /// Grow `seq_id` to hold `n_tokens`; returns only newly allocated pages.
        fn ensure_capacity(&mut self, seq_id: SeqId, n_tokens: usize) -> PyResult<Vec<PageId>> {
            self.inner.ensure_capacity(seq_id, n_tokens).map_err(to_py)
        }

        /// Release all of `seq_id`'s pages; returns them.
        fn free_seq(&mut self, seq_id: SeqId) -> PyResult<Vec<PageId>> {
            self.inner.free_seq(seq_id).map_err(to_py)
        }

        /// `seq_id`'s pages in logical order (empty if it owns none).
        fn pages_for(&self, seq_id: SeqId) -> Vec<PageId> {
            self.inner.pages_for(seq_id).cloned().unwrap_or_default()
        }

        fn pages_needed(&self, n_tokens: usize) -> usize {
            self.inner.pages_needed(n_tokens)
        }

        fn free_pages(&self) -> usize {
            self.inner.free_pages()
        }

        fn used_pages(&self) -> usize {
            self.inner.used_pages()
        }

        fn total_pages(&self) -> usize {
            self.inner.total_pages()
        }

        fn num_sequences(&self) -> usize {
            self.inner.num_sequences()
        }

        #[getter]
        fn page_size(&self) -> usize {
            self.inner.page_size()
        }

        /// Raise RuntimeError if any allocator invariant is broken.
        fn check_invariants(&self) -> PyResult<()> {
            self.inner
                .check_invariants()
                .map_err(PyRuntimeError::new_err)
        }

        fn __repr__(&self) -> String {
            format!(
                "PageForge(total_pages={}, page_size={}, free={}, sequences={})",
                self.inner.total_pages(),
                self.inner.page_size(),
                self.inner.free_pages(),
                self.inner.num_sequences()
            )
        }
    }

    #[pymodule]
    fn _pageforge(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyPageForge>()?;
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        Ok(())
    }
}
