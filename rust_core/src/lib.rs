//! lib.rs — SymboLR Rust Core: root crate module + PyO3 FFI boundary.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  Python                                                     │
//! │    gp/rust_bridge.py  →  symbolr_rust.EvolutionStream       │
//! │                           ↕  JSON telemetry (serde_json)    │
//! ├─────────────────────────────────────────────────────────────┤
//! │  PyO3 FFI Boundary                                          │
//! │    #[pyclass] EvolutionStream  ←  probe_data: Vec<f64>      │
//! │    __new__:  numpy → Vec<f64>  (single copy, held in Rust)  │
//! │    __next__: one parallel generation, zero FFI per-call     │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Rust Core (no Python/GC/FFI on hot path)                   │
//! │    ast.rs       — Expr enum, Box<Expr> heap ASTs            │
//! │    archive.rs   — HashMap-backed MAP-Elites archive          │
//! │    operators.rs — O(1)-pruned genetic operators              │
//! │    evolution.rs — rayon par_iter() fitness evaluation        │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## FFI Contract
//!
//! Probe data enters Rust **once** in `EvolutionStream::new()` via
//! `PyReadonlyArray1::as_slice()?.to_vec()`.  Every subsequent `__next__`
//! call operates on Rust-owned memory — no GIL acquisition, no numpy
//! reference counting, no serialization on the critical evolutionary path.
//!
//! Per-generation telemetry exits as a compact **JSON string** so Python
//! consumes exactly one string allocation per generation, not one per
//! formula.

// ─────────────────────────────────────────────────────────────────────────────
// Module declarations
// ─────────────────────────────────────────────────────────────────────────────

pub mod ast;
pub mod archive;
pub mod operators;
pub mod evolution;

// ─────────────────────────────────────────────────────────────────────────────
// Imports
// ─────────────────────────────────────────────────────────────────────────────

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use rayon::prelude::*;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use ast::{Expr, parse_prefix};
use archive::{ArchiveConfig, MapElitesArchive};
use evolution::EvolutionConfig;

// ─────────────────────────────────────────────────────────────────────────────
// Legacy memoization cache  (evaluate_fast / evaluate_batch hot path)
// ─────────────────────────────────────────────────────────────────────────────

static CACHE: Lazy<DashMap<String, Vec<f64>>> = Lazy::new(DashMap::new);

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Serialize an f64 that may be non-finite (Inf/NaN) to a JSON Value.
/// Non-finite values map to `null` — the only valid JSON representation.
#[inline]
fn finite_f64(v: f64) -> serde_json::Value {
    serde_json::Number::from_f64(v)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

// ─────────────────────────────────────────────────────────────────────────────
// EvolutionEngine — the Phase 3 PyO3 Ask-and-Tell iterator
// ─────────────────────────────────────────────────────────────────────────────

use serde::{Serialize, Deserialize};

fn default_rng() -> SmallRng {
    SmallRng::seed_from_u64(0)
}

#[derive(Serialize, Deserialize)]
pub struct EvolutionState {
    archive:    MapElitesArchive,
    evo_config: EvolutionConfig,
    #[serde(skip, default = "default_rng")]
    rng:        SmallRng,
}

#[pyclass]
pub struct EvolutionEngine {
    state: Option<EvolutionState>,
    current_generation: usize,
    max_generations:    usize,
    pending_offspring:  Vec<Expr>,
}

#[pymethods]
impl EvolutionEngine {
    #[new]
    #[pyo3(signature = (
        max_generations = 50,
        pop_size        = 50,
        seed            = 42,
        crossover_rate  = 0.20,
        mutation_rate   = 0.70,
    ))]
    fn new(
        max_generations: usize,
        pop_size:        usize,
        seed:            u64,
        crossover_rate:  f64,
        mutation_rate:   f64,
    ) -> PyResult<Self> {
        let archive_config = ArchiveConfig::default();
        let evo_config = EvolutionConfig {
            pop_size,
            crossover_rate,
            mutation_rate,
            max_tree_depth: 4,
            archive_config: archive_config.clone(),
        };
        let archive = MapElitesArchive::new(archive_config);
        let rng = SmallRng::seed_from_u64(seed);

        let state = EvolutionState {
            archive,
            evo_config,
            rng,
        };

        Ok(Self {
            state: Some(state),
            current_generation: 0,
            max_generations,
            pending_offspring: Vec::new(),
        })
    }

    fn ask(&mut self) -> PyResult<Vec<String>> {
        if self.current_generation >= self.max_generations {
            return Ok(Vec::new());
        }

        let mut state = self.state.take().expect("Stream state poisoned or accessed concurrently");

        // 1. Generate offspring on CPU
        let offspring = crate::evolution::generate_offspring(
            &state.archive,
            &state.evo_config,
            &mut state.rng,
        );

        // 2. Extract prefix formulas
        let formulas: Vec<String> = offspring.iter().map(|e| e.to_prefix()).collect();
        
        self.pending_offspring = offspring;
        self.state = Some(state);

        Ok(formulas)
    }

    fn tell(&mut self, fitnesses: Vec<f64>) -> PyResult<String> {
        if self.pending_offspring.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("No pending offspring. Call ask() first."));
        }

        let gen = self.current_generation + 1;
        let mut state = self.state.take().expect("Stream state poisoned or accessed concurrently");
        let offspring = std::mem::take(&mut self.pending_offspring);

        // 4. Update the MAP-Elites Archive
        let stats = crate::evolution::update_archive(
            &mut state.archive,
            offspring,
            fitnesses,
            gen,
            0, // gen_time_ms will be handled by python side now
        );

        // Compute gradient-sensitivity telemetry before moving state back
        let grad_sensitivity = state.archive.mean_gradient_sensitivity();

        // Restore state
        self.state = Some(state);
        self.current_generation = gen;

        let json = serde_json::json!({
            "generation_number":        stats.generation,
            "best_mse":                 finite_f64(stats.best_loss),
            "average_mse":              finite_f64(stats.mean_loss),
            "top_formula_latex":        stats.best_formula_latex,
            "top_formula_prefix":       stats.best_formula_prefix,
            "archive_size":             stats.archive_size,
            "new_entries":              stats.new_entries,
            "gen_time_ms":              stats.gen_time_ms,
            // Phase 2: gradient awareness telemetry
            "gradient_sensitivity_mean": finite_f64(grad_sensitivity),
        });

        Ok(json.to_string())
    }

    fn archive_stats(&self) -> String {
        let state = self.state.as_ref().unwrap();
        let s = state.archive.stats();
        serde_json::json!({
            "occupied_niches":            s.occupied_niches,
            "max_niches":                 s.max_niches,
            "occupancy_pct":              finite_f64(s.occupancy_pct),
            "total_attempts":             s.total_attempts,
            "total_additions":            s.total_additions,
            "best_loss":                  finite_f64(s.best_loss),
            "median_loss":                finite_f64(s.median_loss),
            "mean_elite_age":             finite_f64(s.mean_elite_age),
            "gradient_sensitivity_mean":  finite_f64(s.gradient_sensitivity_mean),
        })
        .to_string()
    }

    fn hall_of_fame(&self, top_k: usize) -> String {
        let state = self.state.as_ref().unwrap();
        let hof: Vec<serde_json::Value> = state.archive
            .hall_of_fame(top_k)
            .iter()
            .map(|n| serde_json::json!({
                "latex":      format!("{}", n.expr),
                "prefix":     n.expr.to_prefix(),
                "loss":       finite_f64(n.raw_loss),
                "complexity": n.complexity,
                "age":        n.age,
            }))
            .collect();
        serde_json::json!(hof).to_string()
    }
    
    fn save_checkpoint(&self, path: &str) -> PyResult<()> {
        let state = self.state.as_ref().expect("Engine state poisoned");
        let checkpoint = serde_json::json!({
            "state": state,
            "current_generation": self.current_generation,
            "max_generations": self.max_generations,
        });
        let file = std::fs::File::create(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        serde_json::to_writer(file, &checkpoint).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    #[staticmethod]
    fn load_checkpoint(path: &str) -> PyResult<Self> {
        let file = std::fs::File::open(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let checkpoint: serde_json::Value = serde_json::from_reader(file).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        
        let state: EvolutionState = serde_json::from_value(checkpoint["state"].clone()).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let current_generation = checkpoint["current_generation"].as_u64().unwrap_or(0) as usize;
        let max_generations = checkpoint["max_generations"].as_u64().unwrap_or(50) as usize;

        Ok(Self {
            state: Some(state),
            current_generation,
            max_generations,
            pending_offspring: Vec::new(),
        })
    }

    #[getter]
    fn generation(&self) -> usize {
        self.current_generation
    }

    #[getter]
    fn max_generations(&self) -> usize {
        self.max_generations
    }

    #[getter]
    fn archive_size(&self) -> usize {
        self.state.as_ref().unwrap().archive.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Legacy PyO3 Bridge  (evaluate_fast / evaluate_batch)
// Preserved for zero-regression compatibility with existing Python test suite.
// ─────────────────────────────────────────────────────────────────────────────

fn compute_single(prefix_expr: &str, t_view: &[f64]) -> Result<Vec<f64>, String> {
    // Cache key includes t_array length to prevent cross-call contamination
    // when the same formula is evaluated with different time-step counts.
    let cache_key = format!("{}|{}", prefix_expr, t_view.len());
    if let Some(cached) = CACHE.get(&cache_key) {
        return Ok(cached.clone());
    }
    let ast: Expr = parse_prefix(prefix_expr)
        .map_err(|e| format!("parse_prefix error: {e}"))?;
    let mut result = Vec::with_capacity(t_view.len());
    for &t_val in t_view.iter() {
        // Legacy path: g=0.0, dl=0.0 — backward compatible with SyntheticEvaluator
        let mut val = ast.eval(t_val, 0.0, 0.0);
        if !val.is_finite() { val = 1.0; }
        result.push(val);
    }
    CACHE.insert(cache_key, result.clone());
    Ok(result)
}

#[pyfunction]
fn evaluate_fast<'py>(
    py: Python<'py>,
    prefix_expr: &str,
    t_array: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t_view = t_array.as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result_vec = compute_single(prefix_expr, t_view)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(PyArray1::from_slice_bound(py, &result_vec))
}

#[pyfunction]
fn evaluate_batch<'py>(
    py: Python<'py>,
    prefix_exprs: Vec<String>,
    t_array: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let t_view = t_array.as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let results: Result<Vec<Vec<f64>>, String> = prefix_exprs
        .par_iter()
        .map(|expr| compute_single(expr, t_view))
        .collect();

    let results = results.map_err(|e| PyValueError::new_err(e))?;

    Ok(PyArray2::from_vec2_bound(py, &results)
        .map_err(|e| PyValueError::new_err(e.to_string()))?)
}

// ─────────────────────────────────────────────────────────────────────────────
// PyO3 module registration
// ─────────────────────────────────────────────────────────────────────────────

#[pymodule]
fn symbolr_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Legacy schedule evaluators (Phase 1 compatibility).
    m.add_function(wrap_pyfunction!(evaluate_fast, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_batch, m)?)?;
    // Phase 3: ask-and-tell evolution iterator.
    m.add_class::<EvolutionEngine>()?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (Phase 3 Validation)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod ffi_tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn test_evolution_engine_ffi() {
        // Initialize the Python interpreter for testing outside of a module context
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|_py| {
            // 1. Test: Engine Init
            let max_generations = 3;
            let mut engine = EvolutionEngine::new(
                max_generations,
                10, // pop_size
                42, // seed
                0.2, // crossover_rate
                0.7, // mutation_rate
            ).expect("Failed to initialize EvolutionEngine");

            assert_eq!(engine.generation(), 0, "Initial generation should be 0");
            assert_eq!(engine.max_generations(), 3, "max_generations should be 3");

            // 2. Test: Ask-and-Tell Protocol
            for expected_gen in 1..=3 {
                let formulas = engine.ask().expect("ask() should return formulas");
                assert_eq!(formulas.len(), 10, "Should generate exactly pop_size offspring");

                let fitnesses = vec![1.0; 10]; // mock Python callback evaluation
                let json_str = engine.tell(fitnesses).expect("tell() should return telemetry JSON");

                let parsed: Value = serde_json::from_str(&json_str).expect("Result should be valid JSON");
                
                // Assert generation number increments correctly
                let gen_num = parsed["generation_number"].as_u64().expect("Missing generation_number");
                assert_eq!(gen_num, expected_gen as u64, "Generation number mismatch");
                
                // Ensure other telemetry fields are present
                assert!(parsed.get("best_mse").is_some(), "Missing best_mse");
                assert!(parsed.get("average_mse").is_some(), "Missing average_mse");
                assert!(parsed.get("top_formula_latex").is_some(), "Missing top_formula_latex");
            }

            // 3. Test: End of evolution
            let formulas = engine.ask().expect("ask() should return empty at end");
            assert!(formulas.is_empty(), "Expected empty vector after max_generations");
        });
    }
}