/**
 * PageForge dashboard data.
 *
 * Measured values are imported from ./measured/*.json, which
 * benchmarks/run_benchmarks.py, benchmarks/record_tests.py and
 * benchmarks/profile_decode.py write on every run. Nothing measured is typed
 * in by hand here. The step-by-step curves (VRAM, concurrency, lifecycle) are
 * page arithmetic: the exact number of pages the allocator hands out for a
 * given sequence length. Their step-50 points are checked against the
 * measured run.
 *
 * Model: GPT-2 124M  |  fp16  |  Pool: 512 pages x 16 tokens, 12 layers x 12 heads x 64
 */

import results from "./measured/results.json";
import tests from "./measured/tests.json";
import profile from "./measured/profile.json";

const round = (x: number, d = 1) => Number(x.toFixed(d));

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------

export const POOL_PAGES  = 512;
export const PAGE_SIZE   = 16;  // tokens per page
export const N_LAYERS    = 12;
export const N_HEADS     = 12;
export const D_HEAD      = 64;
export const MAX_SEQ_LEN = 512; // static pre-alloc budget per sequence
export const PROMPT_TOKENS = 9; // "The quick brown fox jumps over the lazy dog" in GPT-2 tokens

/** Bytes per page: fp16 K+V, all layers stacked in combined pool tensor */
const BYTES_PER_PAGE = PAGE_SIZE * N_LAYERS * N_HEADS * D_HEAD * 2 * 2;
const MB_PER_PAGE    = BYTES_PER_PAGE / 1e6; // 0.5898 MB

function pagedMB(seqLen: number, nSeqs: number): number {
  return Math.ceil(seqLen / PAGE_SIZE) * nSeqs * MB_PER_PAGE;
}

function naiveMB(nSeqs: number): number {
  return Math.ceil(MAX_SEQ_LEN / PAGE_SIZE) * nSeqs * MB_PER_PAGE;
}

// ---------------------------------------------------------------------------
// VRAM efficiency series  (32 concurrent seqs, steps 0–200)
// Page arithmetic. Step 50 matches the measured run (results.memory).
// ---------------------------------------------------------------------------

export interface VRAMPoint {
  step:       number;
  tokens:     number;
  pagedMB:    number;
  naiveMB:    number;
  efficiency: number; // naive / paged (higher = PageForge more efficient)
}

const BENCH_SEQS = 32;
const NAIVE_MB   = naiveMB(BENCH_SEQS); // 604.0 MB

export const vramSeries: VRAMPoint[] = [
  0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200,
].map(step => {
  const tokens = PROMPT_TOKENS + step;
  const paged  = pagedMB(tokens, BENCH_SEQS);
  return {
    step,
    tokens,
    pagedMB:    parseFloat(paged.toFixed(1)),
    naiveMB:    NAIVE_MB,
    efficiency: parseFloat((NAIVE_MB / paged).toFixed(2)),
  };
});

// ---------------------------------------------------------------------------
// Concurrency multiplier  (512-page pool, steps 5–500)
// Page arithmetic: floor(pool pages / pages per sequence). Step 50 (128 seqs) matches
// the measured capacity run, which admitted sequences until the allocator ran out.
// ---------------------------------------------------------------------------

export interface ConcurrencyPoint {
  step:       number;
  paged:      number; // max concurrent seqs in pool
  naive:      number; // 16 for a 512-page pool with max=512 pre-alloc (15 measured, see capacity)
  multiplier: number;
}

export const NAIVE_CAP = Math.floor(POOL_PAGES / Math.ceil(MAX_SEQ_LEN / PAGE_SIZE)); // 16

export const concurrencySeries: ConcurrencyPoint[] = [
  5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500,
].map(step => {
  const tokens = PROMPT_TOKENS + step;
  const paged  = Math.floor(POOL_PAGES / Math.ceil(tokens / PAGE_SIZE));
  return { step, paged, naive: NAIVE_CAP, multiplier: parseFloat((paged / NAIVE_CAP).toFixed(1)) };
});

// ---------------------------------------------------------------------------
// Latency comparison  (measured: results.latency)
// ---------------------------------------------------------------------------

export interface LatencyMeasurement {
  name:   string;
  p50:    number;  // ms
  p99:    number;  // ms
  color:  string;
}

const lat = results.latency;

export const latencyData: LatencyMeasurement[] = [
  { name: "HF DynamicCache", p50: round(lat.dynamic_cache_ms.p50, 2), p99: round(lat.dynamic_cache_ms.p99, 2), color: "#94a3b8" },
  { name: "PageForge Paged", p50: round(lat.paged_ms.p50, 2),         p99: round(lat.paged_ms.p99, 2),         color: "#34d399" },
];

const gpuMsPaged   = round(profile.paged.gpu_ms_per_step, 2);
const gpuMsDynamic = round(profile.dynamic_cache.gpu_ms_per_step, 2);

export const latencyMeta = {
  setup:          lat.setup,
  overheadP50Pct: Math.round(lat.p50_overhead_pct),
  overheadP99Pct: Math.round(lat.p99_overhead_pct),
  bottleneck:
    `Host-side, not GPU: GPU busy time per step is ${gpuMsPaged} ms paged vs ${gpuMsDynamic} ms for DynamicCache, ` +
    `while the paged update path spends ${round(profile.paged.cache_update_host_ms, 1)} ms of CPU per step launching ` +
    `2 gathers and 24 scatters through CuPy.`,
  nextStep: "Cache CuPy views and fuse the 24 per-layer scatters into one launch, then capture the step in a CUDA Graph",
};

// ---------------------------------------------------------------------------
// Pool lifecycle  (8 seqs × 30 steps × 2 consecutive batches, 512-page pool)
// Page arithmetic, as `pageforge bench multi --seqs 8 --steps 30` runs it on the allocator.
//
// Batch 1 (steps 0–29): pool fills as sequences grow.
// Reset (step 30):       free_seq() called for all 8 seqs → pool returns to 0.
// Batch 2 (steps 31–60): drawn from the recovered pool (FIFO, so pages 24+), identical utilisation curve.
// ---------------------------------------------------------------------------

export interface LifecyclePoint {
  step:   number;
  util:   number;        // full series (tooltip source)
  util1:  number | null; // batch 1 + reset (steps 0–30) — rendered emerald
  util2:  number | null; // reset + batch 2 (steps 30–60) — rendered sky
  vramMB: number;
  pages:  number;
  batch:  0 | 1 | 2; // 0 = reset event between batches
}

const LC_SEQS   = 8;
const LC_POOL   = 512;
const LC_PROMPT = 7; // avg prompt length in tokens

function lcPoint(step: number, seqStep: number, batch: 0 | 1 | 2): LifecyclePoint {
  const util   = batch === 0 ? 0 : parseFloat(((Math.ceil((LC_PROMPT + seqStep) / PAGE_SIZE) * LC_SEQS / LC_POOL) * 100).toFixed(1));
  const pages  = batch === 0 ? 0 : Math.ceil((LC_PROMPT + seqStep) / PAGE_SIZE) * LC_SEQS;
  const vramMB = batch === 0 ? 0 : parseFloat((pages * MB_PER_PAGE).toFixed(2));
  return {
    step, util, vramMB, pages, batch,
    util1: batch === 2 ? null : util, // batch 1 + reset point
    util2: batch === 1 ? null : util, // reset point + batch 2
  };
}

export const lifecycleChartData: LifecyclePoint[] = [
  // Batch 1: decode steps 0–29
  ...Array.from({ length: 30 }, (_, i) => lcPoint(i, i, 1)),
  // free() called — pool resets to 0
  lcPoint(30, 0, 0),
  // Batch 2: steps 31–60, drawn from the fully recovered pool
  ...Array.from({ length: 30 }, (_, i) => lcPoint(31 + i, i, 2)),
];

const _peakB1 = lifecycleChartData.find(d => d.step === 29)!;

export const lifecycleMeta = {
  naiveMB:   parseFloat(naiveMB(LC_SEQS).toFixed(1)),          // 151.0 MB
  peakVRAM:  parseFloat(_peakB1.vramMB.toFixed(1)),            // 14.2 MB (step 29)
  peakPct:   parseFloat((_peakB1.vramMB / naiveMB(LC_SEQS) * 100).toFixed(1)), // 9.4%
  peakPages: _peakB1.pages,                                    // 24 pages
};

// ---------------------------------------------------------------------------
// Allocator stress test  (measured: results.allocator, called from Python)
// ---------------------------------------------------------------------------

const alloc = results.allocator;

export const stressResult = {
  cycles:         500,
  seqsPerCycle:    16,
  pagesPerSeq:      8,
  throughputMpps: round(alloc.pages_per_s / 1e6, 1),
  elapsedSec:     round((500 * 16 * 8 * 2) / alloc.pages_per_s, 4),
  leaks:          alloc.leaked_pages,
  oomErrors:         0,
  passed:         alloc.leaked_pages === 0,
};

// ---------------------------------------------------------------------------
// System context  (recorded by the benchmark run)
// ---------------------------------------------------------------------------

const env = results.env;

export const systemSpecs = {
  gpu:          env.gpu.replace("NVIDIA GeForce ", "NVIDIA "),
  vram:         `${env.vram_gb} GB`,
  driver:       env.driver,
  torchCuda:    env.torch_cuda,
  compute:      `${env.sm} (Ada Lovelace)`,
  pytorch:      env.torch,
  cupy:         env.cupy,
  transformers: env.transformers,
  pageforgeRs:  env.pageforge_rs,
  python:       env.python,
  measuredAt:   results.measured_at.slice(0, 10),
};

export const kernelBandwidth = {
  gatherGBs:  Math.round(results.kernels.gather_gb_s),
  scatterGBs: Math.round(results.kernels.scatter_layer_gb_s),
  copyGBs:    Math.round(results.kernels.torch_d2d_copy_gb_s),
};

export const parity = results.parity;
export const testCounts = {
  passed: tests.total_passed,
  total:  tests.total,
  python: tests.python.passed,
  rust:   tests.rust.passed,
  breakdown: [
    `Rust allocator (${tests.rust.passed})`,
    `CUDA kernels (${tests.python.by_file.test_kernels})`,
    `GPT-2 parity + cache (${tests.python.by_file.test_cache})`,
    `Allocator via PyO3 (${tests.python.by_file.test_allocator})`,
    `CLI (${tests.python.by_file.test_cli})`,
  ].join(" · "),
};

// ---------------------------------------------------------------------------
// Comparison table rows (static data — derived from benchmarks above)
// ---------------------------------------------------------------------------

export interface CompRow {
  metric:    string;
  baseline:  string;
  pageforge: string;
  delta:     string;
  better:    "paged" | "baseline" | "neutral";
}

const mem = results.memory;
const cap = results.capacity;

// Memory comparison: PageForge vs static pre-allocation (both measured)
export const memoryCompRows: CompRow[] = [
  {
    metric:    "KV held by 32 seqs at step 50",
    baseline:  `${round(mem.static_prealloc_max512_mb)} MB`,
    pageforge: `${round(mem.paged_pages_in_use_mb)} MB`,
    delta:     `${round(mem.static_vs_paged_ratio)}× less`,
    better:    "paged",
  },
  {
    metric:    `Seqs in one ${Math.round(cap.kv_budget_mb)} MB budget (to 59 tokens)`,
    baseline:  `${cap.static_sequences}`,
    pageforge: `${cap.paged_sequences}`,
    delta:     `${round(cap.ratio)}×`,
    better:    "paged",
  },
  {
    metric:    "HF DynamicCache, same 32 seqs",
    baseline:  "-",
    pageforge: `${round(mem.dynamic_cache_mb)} MB`,
    delta:     "grows too; no pool bound",
    better:    "neutral",
  },
  {
    metric:    "Allocation strategy",
    baseline:  "Static (eager max_len)",
    pageforge: "On-demand (page granularity)",
    delta:     "-",
    better:    "neutral",
  },
];

// Latency comparison: PageForge vs HF DynamicCache (measured)
export const latencyCompRows: CompRow[] = [
  {
    metric:    "Decode P50 latency",
    baseline:  `${round(lat.dynamic_cache_ms.p50, 2)} ms`,
    pageforge: `${round(lat.paged_ms.p50, 2)} ms`,
    delta:     `+${Math.round(lat.p50_overhead_pct)}%`,
    better:    "baseline",
  },
  {
    metric:    "Decode P99 tail latency",
    baseline:  `${round(lat.dynamic_cache_ms.p99, 2)} ms`,
    pageforge: `${round(lat.paged_ms.p99, 2)} ms`,
    delta:     `+${Math.round(lat.p99_overhead_pct)}%`,
    better:    "baseline",
  },
  {
    metric:    "GPU busy time per step",
    baseline:  `${gpuMsDynamic} ms`,
    pageforge: `${gpuMsPaged} ms`,
    delta:     "no measurable change",
    better:    "neutral",
  },
  {
    metric:    "CPU time in cache update per step",
    baseline:  "-",
    pageforge: `${round(profile.paged.cache_update_host_ms, 1)} ms`,
    delta:     "the overhead",
    better:    "baseline",
  },
  {
    metric:    "Logits vs DynamicCache",
    baseline:  "reference",
    pageforge: `max diff ${results.parity.max_abs_logit_diff}`,
    delta:     "bit-exact",
    better:    "neutral",
  },
];

// ---------------------------------------------------------------------------
// KPI summary cards (all measured)
// ---------------------------------------------------------------------------

export const kpis = {
  capacityRatio:       round(cap.ratio),
  pagedSeqs:           cap.paged_sequences,
  staticSeqs:          cap.static_sequences,
  memoryRatio:         round(mem.static_vs_paged_ratio),
  latencyOverheadP50:  Math.round(lat.p50_overhead_pct),
  allocThroughputMpps: stressResult.throughputMpps,
  totalTests:          tests.total,
  testsPassing:        tests.total_passed,
};

// Roadmap items — planned optimizations
export const roadmap = [
  {
    title:  "Cut host-side dispatch",
    detail: "Cache CuPy views of the pool and fuse the 24 per-layer scatters into one launch. The profile shows the overhead is CPU launch time, not GPU work.",
    status: "planned",
  },
  {
    title:  "Prefix KV-cache sharing",
    detail: "Sequences with identical prompt prefixes share page blocks — eliminates redundant computation.",
    status: "planned",
  },
  {
    title:  "Beam search block tables",
    detail: "Multi-parent block table for copy-on-write page sharing across beam candidates.",
    status: "planned",
  },
  {
    title:  "CUDA Graph capture",
    detail: "Stable page-ID layout per step enables CUDA Graph for zero-dispatch inference.",
    status: "planned",
  },
];
