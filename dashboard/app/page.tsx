import { CheckCircle } from "lucide-react";
import NavBar           from "@/components/NavBar";
import MetricCard       from "@/components/MetricCard";
import VRAMChart        from "@/components/VRAMChart";
import ConcurrencyChart from "@/components/ConcurrencyChart";
import LatencyChart     from "@/components/LatencyChart";
import LifecycleChart   from "@/components/LifecycleChart";
import ArchitectureCard from "@/components/ArchitectureCard";
import SystemSpecs      from "@/components/SystemSpecs";
import ComparisonTable  from "@/components/ComparisonTable";
import {
  kpis, vramSeries, concurrencySeries, latencyData, latencyMeta,
  lifecycleChartData, lifecycleMeta, stressResult, systemSpecs,
  memoryCompRows, latencyCompRows, roadmap, NAIVE_CAP,
} from "@/lib/data";

// ── Sparkline data (0-100 normalised) ─────────────────────────────────────
const CONCURRENCY_SPARK = [16, 25, 30, 38, 44, 50, 62, 75, 100, 100].reverse();
const VRAM_SPARK        = [100, 80, 65, 55, 48, 42, 36, 30, 22, 16];
const OVERHEAD_SPARK    = [0, 8, 15, 20, 25, 28, 31, 33, 33, 33];
const ALLOC_SPARK       = [40, 55, 68, 80, 88, 93, 97, 99, 100, 100];

// ── Tech badge colour map (static — Tailwind purges dynamic class strings) ─
const BADGE: Record<string, string> = {
  violet:  "text-violet-400  border-violet-400/25  bg-violet-400/8",
  emerald: "text-emerald-400 border-emerald-400/25 bg-emerald-400/8",
  sky:     "text-sky-400     border-sky-400/25     bg-sky-400/8",
  amber:   "text-amber-400   border-amber-400/25   bg-amber-400/8",
  rose:    "text-rose-400    border-rose-400/25    bg-rose-400/8",
};

const TECH_BADGES = [
  { label: "Rust · PyO3",       color: "violet"  },
  { label: "CUDA C · nvrtc",    color: "emerald" },
  { label: "CuPy RawKernel",    color: "sky"     },
  { label: "PyTorch fp16",      color: "amber"   },
  { label: "DLPack zero-copy",  color: "rose"    },
];

// ── Problem / Solution / Result explainer cards ────────────────────────────
const EXPLAINER = [
  {
    step:      "01",
    label:     "The Problem",
    border:    "border-rose-400/15",
    bg:        "bg-rose-400/5",
    statColor: "text-rose-400",
    heading:   "Naive pre-allocation wastes 90%+ of VRAM",
    body:
      "Traditional KV caches reserve max_seq_len (512) tokens of memory at sequence start — regardless of how many tokens are actually generated. " +
      "For GPT-2, that's 18.9 MB per sequence, all committed upfront. A 512-page pool can serve just 16 sequences simultaneously.",
    stat:    "16 seqs max",
    statSub: "512-page pool · naive pre-alloc",
  },
  {
    step:      "02",
    label:     "The Solution",
    border:    "border-sky-400/15",
    bg:        "bg-sky-400/5",
    statColor: "text-sky-400",
    heading:   "On-demand paged allocation",
    body:
      "PageForge allocates memory in fixed-size pages (16 tokens = 0.59 MB) on each decode step. " +
      "A Rust PageAllocator manages an O(1) VecDeque free-list. Sequences claim pages as they grow and return them immediately when complete — " +
      "enabling the same pool to serve many more sequences concurrently.",
    stat:    "0.59 MB / page",
    statSub: "16 tokens · allocated on demand",
  },
  {
    step:      "03",
    label:     "The Result",
    border:    "border-emerald-400/15",
    bg:        "bg-emerald-400/5",
    statColor: "text-emerald-400",
    heading:   "8–32× more concurrent sequences",
    body:
      "At decode step 10 (20 tokens), each sequence uses just 2 pages vs 32 pre-allocated — freeing 94% of the pool for other sequences. " +
      "The same 512-page budget that handles 16 naive sequences can now serve 128–512 concurrent sequences, " +
      "depending on how far along each sequence is.",
    stat:    "128–512 seqs",
    statSub: "same 512-page pool · 8–32×",
  },
];

export default function Page() {
  return (
    <div className="min-h-screen">
      <NavBar />

      <main className="max-w-7xl mx-auto px-6 py-14 space-y-12">

        {/* ── Hero ──────────────────────────────────────────────────── */}
        <section>
          <p className="label-tag mb-4">Paged KV-Cache Memory Manager</p>
          <div className="flex flex-col md:flex-row md:items-start justify-between gap-8">
            <div className="flex-1">
              <h1 className="text-6xl font-light tracking-tight text-zinc-100 mb-4 leading-none">
                PageForge
              </h1>
              <p className="text-lg text-zinc-400 font-light leading-relaxed max-w-xl mb-6">
                A from-scratch implementation of vLLM-style paged KV-cache attention in
                Rust + CUDA. On-demand page allocation eliminates static VRAM waste,
                enabling 8–32× more concurrent LLM inference sequences per GPU.
              </p>
              <div className="flex flex-wrap gap-2">
                {TECH_BADGES.map(({ label, color }) => (
                  <span
                    key={label}
                    className={`px-2.5 py-1 rounded-full text-[11px] font-medium border ${BADGE[color]}`}
                  >
                    {label}
                  </span>
                ))}
              </div>
            </div>

            {/* Hero stats */}
            <div className="shrink-0 grid grid-cols-2 gap-3 md:w-[340px]">
              {[
                { val: "8–32×", sub: "more concurrent seqs / GB", color: "text-emerald-400" },
                { val: "94%",   sub: "less active VRAM per seq",  color: "text-sky-400"     },
                { val: "+33%",  sub: "P50 latency overhead",      color: "text-amber-400"   },
                { val: "75/75", sub: "tests passing",             color: "text-violet-400"  },
              ].map(({ val, sub, color }) => (
                <div key={sub} className="glass-card px-4 py-4">
                  <p className={`text-3xl font-light tabular-nums ${color}`}>{val}</p>
                  <p className="text-[11px] text-zinc-500 mt-1 leading-snug">{sub}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── Problem / Solution / Result ───────────────────────────── */}
        <section>
          <p className="label-tag mb-5">System Design</p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {EXPLAINER.map(card => (
              <div
                key={card.step}
                className={`glass-card p-5 border ${card.border} ${card.bg} flex flex-col gap-3`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[10px] uppercase tracking-widest text-zinc-600">{card.label}</span>
                  <span className="text-xs font-mono text-zinc-700">{card.step}</span>
                </div>
                <h3 className="text-sm font-semibold text-zinc-200 leading-snug">
                  {card.heading}
                </h3>
                <p className="text-[12px] text-zinc-500 leading-relaxed flex-1">
                  {card.body}
                </p>
                <div className={`mt-1 px-3 py-2 rounded-lg bg-zinc-900/60 border ${card.border}`}>
                  <p className={`text-base font-light tabular-nums ${card.statColor}`}>{card.stat}</p>
                  <p className="text-[10px] text-zinc-600 mt-0.5">{card.statSub}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ── KPI Cards ─────────────────────────────────────────────── */}
        <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            label="Concurrency Multiplier"
            value={`${kpis.concurrencyMultiplier}×`}
            sub="max seqs · 512-page pool · step 10"
            delta="vs 16 naive"
            deltaType="positive"
            spark={CONCURRENCY_SPARK}
            accentColor="#34d399"
          />
          <MetricCard
            label="Active VRAM vs Naive"
            value={`${kpis.activeVRAMPct}%`}
            sub="14.2 MB active vs 151 MB naive (8 seqs)"
            delta="−90.6% VRAM"
            deltaType="positive"
            spark={VRAM_SPARK}
            accentColor="#38bdf8"
          />
          <MetricCard
            label="P50 Latency Overhead"
            value={`+${kpis.latencyOverheadP50}%`}
            sub="7.5 ms → 10.0 ms vs HF DynamicCache"
            delta="scatter cost"
            deltaType="neutral"
            spark={OVERHEAD_SPARK}
            accentColor="#fbbf24"
          />
          <MetricCard
            label="Allocator Throughput"
            value={`${kpis.allocThroughputMpps}M/s`}
            sub="pages/sec · 500 cycles · 0 leaks"
            delta="PASS"
            deltaType="positive"
            spark={ALLOC_SPARK}
            accentColor="#a78bfa"
          />
        </section>

        {/* ── Comparison Tables ─────────────────────────────────────── */}
        <section className="glass-card p-6">
          <p className="label-tag mb-1">At a Glance</p>
          <h2 className="section-title mb-6">PageForge vs Alternatives</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <ComparisonTable
              title="Memory Efficiency"
              baseLabel="Naive Static"
              rows={memoryCompRows}
              note="Naive baseline pre-allocates max_seq_len=512 tokens per sequence at start, regardless of actual generation length."
            />
            <ComparisonTable
              title="Decode Latency"
              baseLabel="HF DynamicCache"
              rows={latencyCompRows}
              note="HF DynamicCache grows KV tensors by torch.cat each step (contiguous layout). PageForge scatter-copies into non-contiguous pages — additional overhead per step."
            />
          </div>
        </section>

        {/* ── VRAM Efficiency Chart ─────────────────────────────────── */}
        <section className="glass-card p-6">
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-6">
            <div>
              <p className="label-tag mb-1">Memory Efficiency</p>
              <h2 className="section-title">VRAM Consumption vs Decode Step</h2>
              <p className="section-sub mt-1">
                32 concurrent sequences · GPT-2 124M · fp16 · Naive budget = 512 tokens / sequence (fixed)
              </p>
            </div>
            <div className="text-right shrink-0">
              <p className="text-3xl font-light text-emerald-400">8.0×</p>
              <p className="text-xs text-zinc-500">savings at step 50</p>
            </div>
          </div>
          <VRAMChart data={vramSeries} />
        </section>

        {/* ── Concurrency + Latency ─────────────────────────────────── */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <section className="glass-card p-6">
            <p className="label-tag mb-1">Concurrency</p>
            <h2 className="section-title mb-0.5">Max Concurrent Sequences</h2>
            <p className="section-sub mb-5">
              512-page pool · page_size=16 · prompt ≈ 10 tokens · shows PageForge advantage degrades as seqs grow
            </p>
            <ConcurrencyChart data={concurrencySeries} naiveCap={NAIVE_CAP} />
          </section>

          <section className="glass-card p-6">
            <p className="label-tag mb-1">Latency Analysis</p>
            <h2 className="section-title mb-0.5">Decode Step Latency</h2>
            <p className="section-sub mb-5">
              50 iterations · 5 warmup · RTX 4070 Laptop · prompt: "The quick brown fox…"
            </p>
            <LatencyChart
              data={latencyData}
              overheadP50Pct={latencyMeta.overheadP50Pct}
              overheadP99Pct={latencyMeta.overheadP99Pct}
              bottleneck={latencyMeta.bottleneck}
            />
          </section>
        </div>

        {/* ── Pool Lifecycle ────────────────────────────────────────── */}
        <section className="glass-card p-6">
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-6">
            <div>
              <p className="label-tag mb-1">Pool Lifecycle</p>
              <h2 className="section-title">Sequential Batch Reuse</h2>
              <p className="section-sub mt-1">
                8 concurrent seqs · 30 decode steps · free() at step 30 · Batch 2 reuses exact same pages
              </p>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-emerald-400 shrink-0">
              <CheckCircle className="w-4 h-4" />
              <span className="font-medium">Zero fragmentation verified</span>
            </div>
          </div>
          <LifecycleChart
            data={lifecycleChartData}
            naiveMB={lifecycleMeta.naiveMB}
            peakVRAM={lifecycleMeta.peakVRAM}
            peakPct={lifecycleMeta.peakPct}
          />
        </section>

        {/* ── Architecture + System Specs ───────────────────────────── */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <ArchitectureCard />
          <SystemSpecs specs={systemSpecs} stress={stressResult} />
        </div>

        {/* ── Roadmap ───────────────────────────────────────────────── */}
        <section className="glass-card p-6">
          <p className="label-tag mb-1">Engineering Roadmap</p>
          <h2 className="section-title mb-6">Planned Optimisations</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {roadmap.map((item, i) => (
              <div
                key={item.title}
                className="flex gap-4 px-4 py-4 rounded-xl bg-zinc-900/50 border border-zinc-800/60"
              >
                <span className="text-xs font-mono text-zinc-700 shrink-0 mt-0.5">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <div>
                  <p className="text-sm font-semibold text-zinc-200 mb-1">{item.title}</p>
                  <p className="text-xs text-zinc-500 leading-relaxed">{item.detail}</p>
                </div>
                <span className="text-[10px] px-2 py-0.5 rounded-full border border-zinc-700 text-zinc-600 self-start shrink-0">
                  {item.status}
                </span>
              </div>
            ))}
          </div>
          <p className="text-[11px] text-zinc-600 mt-4 leading-relaxed">
            Inspired by vLLM PagedAttention (Kwon et al., 2023). This implementation demonstrates the
            core paging mechanism from first principles — Rust allocator, hand-written CUDA scatter/gather
            kernels, and a DLPack zero-copy bridge to PyTorch.
          </p>
        </section>

      </main>

      {/* ── Footer ──────────────────────────────────────────────────── */}
      <footer className="border-t border-zinc-800/50 mt-8 py-8">
        <div className="max-w-7xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="text-xs text-zinc-600">PageForge v0.1.0</span>
            <span className="text-zinc-800">·</span>
            <span className="text-xs text-zinc-600">RTX 4070 Laptop · sm_89 · CUDA 12.8</span>
            <span className="text-zinc-800">·</span>
            <span className="text-xs text-zinc-600">Hardware-verified benchmarks · GPT-2 124M fp16</span>
          </div>
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors shrink-0"
          >
            Source on GitHub →
          </a>
        </div>
      </footer>
    </div>
  );
}
