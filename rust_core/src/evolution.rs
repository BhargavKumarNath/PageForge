//! evolution.rs — Parallel evolutionary loop for the SymboLR Thick Rust Core.
//!
//! ## Architecture
//!
//! ```text
//! run_evolution_step()
//!   ├── sample_parents()        ← fitness-proportionate + uniform selection
//!   ├── generate offspring[]    ← sequential (needs mutable RNG)
//!   ├── evaluate_population_parallel()   ← RAYON par_iter() over all CPU cores
//!   │     └── evaluate_synthetic()       ← quadratic landscape fitness, seeded RNG
//!   └── archive.try_add() × pop_size    ← O(1) niche insertion
//! ```
//!
//! ## Zero-Copy Dataset Design
//!
//! `evaluate_population_parallel` accepts the time-step array as `&[f64]` —
//! a contiguous memory slice that maps directly to Python's numpy array via
//! `as_slice()` with zero copy.  No `Vec` allocation, no FFI serialization.
//! Phase 3 will extend this to `ndarray::ArrayView2<f64>` for the full probe
//! dataset, maintaining the same zero-copy contract.
//!
//! ## Thread Safety
//!
//! Rayon's `par_iter()` distributes `Expr` evaluations across all CPU threads.
//! Each thread independently:
//!   • Calls `Expr::eval_schedule()` (pure, no shared state)
//!   • Calls `evaluate_synthetic()` with a per-formula seeded `SmallRng`
//! No mutexes, no atomic counters, no false sharing.

use crate::archive::{ArchiveConfig, MapElitesArchive};
use crate::ast::Expr;
use crate::operators::apply_random_operator;
use serde::{Serialize, Deserialize};

use rand::Rng;

// ─────────────────────────────────────────────────────────────────────────────
// 1.  Configuration Structs
// ─────────────────────────────────────────────────────────────────────────────

/// Hyperparameters for the synthetic quadratic-landscape fitness function.
///
/// Mirrors Python's `evaluate_synthetic` from `gp/fitness.py` exactly,
/// enabling direct numerical comparison during transition testing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitnessConfig {
    /// Number of quadratic loss landscape dimensions (default: 5).
    pub n_dims: usize,
    /// Ensemble evaluations per formula for noise robustness (default: 3).
    pub n_evaluations: usize,
    /// Gradient-noise scale (default: 0.02).
    pub noise_scale: f64,
    /// Parsimony pressure coefficient — added as `coeff * tree.size()` (default: 0.01).
    pub parsimony_coeff: f64,
}

impl Default for FitnessConfig {
    fn default() -> Self {
        Self {
            n_dims: 5,
            n_evaluations: 3,
            noise_scale: 0.02,
            parsimony_coeff: 0.01,
        }
    }
}

/// Hyperparameters for the evolutionary loop.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvolutionConfig {
    /// Number of offspring generated per generation.
    pub pop_size: usize,
    /// Probability of subtree crossover (vs mutation).
    pub crossover_rate: f64,
    /// Probability of any mutation variant (conditional on not crossing over).
    pub mutation_rate: f64,
    /// Max depth for randomly generated initial trees.
    pub max_tree_depth: usize,
    /// Archive configuration (bins, age penalty, etc.).
    pub archive_config: ArchiveConfig,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            pop_size: 50,
            crossover_rate: 0.20,
            mutation_rate: 0.70,
            max_tree_depth: 4,
            archive_config: ArchiveConfig::default(),
        }
    }
}

/// Per-generation telemetry snapshot — streamed to Python in Phase 3.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationStats {
    pub generation: usize,
    pub best_loss: f64,
    pub mean_loss: f64,
    pub archive_size: usize,
    pub new_entries: usize,
    pub gen_time_ms: u128,
    /// LaTeX string of the best formula in this generation.
    pub best_formula_latex: String,
    /// Prefix string of the best formula (for FFI handoff).
    pub best_formula_prefix: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Offspring Generation
// ─────────────────────────────────────────────────────────────────────────────

pub fn generate_offspring<R: Rng>(
    archive: &MapElitesArchive,
    config: &EvolutionConfig,
    rng: &mut R,
) -> Vec<Expr> {
    let parents = if archive.is_empty() {
        crate::operators::ramped_half_and_half(config.pop_size, 2, config.max_tree_depth, rng)
    } else {
        let mut pool = crate::evolution::sample_parents(archive, config.pop_size, rng);
        while pool.len() < config.pop_size {
            pool.push(crate::operators::generate_tree(config.max_tree_depth, rng));
        }
        pool
    };

    (0..config.pop_size)
        .map(|_| apply_random_operator(&parents, config.crossover_rate, config.mutation_rate, rng))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Parent Sampling
// ─────────────────────────────────────────────────────────────────────────────

fn sample_parents<R: Rng>(archive: &MapElitesArchive, n: usize, rng: &mut R) -> Vec<Expr> {
    if archive.is_empty() {
        return Vec::new();
    }

    let niches: Vec<(&Expr, f64)> = archive
        .niches
        .values()
        .map(|n| (&n.expr, n.raw_loss))
        .collect();

    let max_loss = niches.iter().map(|(_, l)| *l).fold(f64::NEG_INFINITY, f64::max) + 1e-8;
    let weights: Vec<f64> = niches.iter().map(|(_, l)| max_loss - l + 1e-8).collect();
    let total_w: f64 = weights.iter().sum();

    let n_fitness = (n as f64 * 0.6) as usize;
    let n_uniform = n - n_fitness;

    let mut parents = Vec::with_capacity(n);

    for _ in 0..n_fitness {
        let mut r = rng.gen::<f64>() * total_w;
        let mut chosen = 0;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            chosen = i;
            if r <= 0.0 { break; }
        }
        parents.push(niches[chosen].0.clone());
    }

    for _ in 0..n_uniform {
        let i = rng.gen_range(0..niches.len());
        parents.push(niches[i].0.clone());
    }

    parents
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Archive Update
// ─────────────────────────────────────────────────────────────────────────────

pub fn update_archive(
    archive: &mut MapElitesArchive,
    offspring: Vec<Expr>,
    fitnesses: Vec<f64>,
    generation: usize,
    gen_time_ms: u128,
) -> GenerationStats {
    let mut new_entries = 0usize;
    for (expr, loss) in offspring.into_iter().zip(fitnesses.iter().copied()) {
        if loss.is_finite() && archive.try_add(expr, loss, loss) {
            new_entries += 1;
        }
    }

    archive.increment_ages();

    let stats = archive.stats();
    let (latex, prefix) = archive
        .best()
        .map(|n| (format!("{}", n.expr), n.expr.to_prefix()))
        .unwrap_or_default();

    let finite_fitnesses: Vec<f64> = fitnesses.iter().copied().filter(|f| f.is_finite()).collect();
    let mean_loss = if finite_fitnesses.is_empty() {
        f64::INFINITY
    } else {
        finite_fitnesses.iter().sum::<f64>() / finite_fitnesses.len() as f64
    };

    GenerationStats {
        generation,
        best_loss: stats.best_loss,
        mean_loss,
        archive_size: archive.len(),
        new_entries,
        gen_time_ms,
        best_formula_latex: latex,
        best_formula_prefix: prefix,
    }
}

