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
  kernelBandwidth, parity, testCounts,
} from "@/lib/data";

// ── Sparkline data (0-100 normalised) ─────────────────────────────────────
const CONCURRENCY_SPARK = [16, 25, 30, 38, 44, 50, 62, 75, 100, 100].reverse();
const VRAM_SPARK        = [100, 80, 65, 55, 48, 42, 36, 30, 22, 16];
const OVERHEAD_SPARK    = [0, 10, 20, 30, 40, 50, 60, 70, 80, 88];
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
    heading:   "Static pre-allocation reserves memory nobody uses",
    body:
      "A static KV cache reserves max_seq_len (512) tokens per sequence at the start, however few tokens it generates. " +
      "For GPT-2 that measured 18.9 MB per sequence, so a 302 MB budget fits only " + kpis.staticSeqs + " sequences.",
    stat:    `${kpis.staticSeqs} seqs`,
    statSub: "302 MB budget · static max_len=512 · measured",
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
      "A Rust allocator keeps a VecDeque free list (O(1) per page) and a block table per sequence. Sequences claim pages as they grow and return them when done, " +
      "so the same pool serves more sequences at once.",
    stat:    "0.59 MB / page",
    statSub: "16 tokens · allocated on demand",
  },
  {
    step:      "03",
    label:     "The Result",
    border:    "border-emerald-400/15",
    bg:        "bg-emerald-400/5",
    statColor: "text-emerald-400",
    heading:   `${kpis.capacityRatio}× more sequences in the same memory`,
    body:
      "Measured by admitting GPT-2 sequences into a 512-page pool until the allocator ran out: " + kpis.pagedSeqs +
      " sequences fit at 59 tokens each. The cost is decode latency: +" + kpis.latencyOverheadP50 +
      "% at P50, almost all of it Python and CuPy launch time on the CPU.",
    stat:    `${kpis.pagedSeqs} seqs`,
    statSub: `same 302 MB budget · ${kpis.capacityRatio}× · measured`,
  },
];

export default function Page() {
  return (
    <div className="min-h-screen overflow-x-hidden">
      <NavBar />

      <main className="max-w-7xl mx-auto px-3 xs:px-4 sm:px-6 py-6 xs:py-8 sm:py-14 space-y-6 xs:space-y-8 sm:space-y-12">

        {/* ── Hero ──────────────────────────────────────────────────── */}
        <section>
          <p className="label-tag mb-4">Paged KV-Cache Memory Manager</p>
          <div className="flex flex-col md:flex-row md:items-start justify-between gap-6 md:gap-8">
            <div className="flex-1 min-w-0">
              <h1 className="text-3xl xs:text-4xl sm:text-5xl md:text-6xl font-light tracking-tight text-zinc-100 mb-3 xs:mb-4 leading-tight">
                PageForge
              </h1>
              <p className="text-sm xs:text-base sm:text-lg text-zinc-400 font-light leading-relaxed max-w-xl mb-4 xs:mb-6">
                A from-scratch implementation of vLLM-style paged KV-cache attention in
                Rust and CUDA, plugged into Hugging Face transformers. Pages are allocated on
                demand, so the same KV memory holds {kpis.capacityRatio}× more GPT-2 sequences than static
                pre-allocation, with logits bit-identical to the stock cache.
              </p>
              <div className="flex flex-wrap gap-1.5 xs:gap-2">
                {TECH_BADGES.map(({ label, color }) => (
                  <span
                    key={label}
                    className={`px-2 xs:px-2.5 py-0.5 xs:py-1 rounded-full text-[10px] xs:text-[11px] font-medium border ${BADGE[color]}`}
                  >
                    {label}
                  </span>
                ))}
              </div>
            </div>

            {/* Hero stats */}
            <div className="grid grid-cols-2 gap-2 xs:gap-3 w-full md:w-[340px] md:shrink-0">
              {[
                { val: `${kpis.capacityRatio}×`, sub: "more seqs in one KV budget", color: "text-emerald-400" },
                { val: `${parity.max_abs_logit_diff}`, sub: "max logit diff vs DynamicCache", color: "text-sky-400" },
                { val: `+${kpis.latencyOverheadP50}%`, sub: "P50 decode latency overhead", color: "text-amber-400" },
                { val: `${testCounts.passed}/${testCounts.total}`, sub: "tests passing", color: "text-violet-400" },
              ].map(({ val, sub, color }) => (
                <div key={sub} className="glass-card px-2.5 py-2.5 xs:px-3 xs:py-3 sm:px-4 sm:py-4">
                  <p className={`text-xl xs:text-2xl sm:text-3xl font-light tabular-nums ${color}`}>{val}</p>
                  <p className="text-[9px] xs:text-[10px] sm:text-[11px] text-zinc-500 mt-1 leading-snug">{sub}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── Problem / Solution / Result ───────────────────────────── */}
        <section>
          <p className="label-tag mb-3 xs:mb-5">System Design</p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 xs:gap-4">
            {EXPLAINER.map(card => (
              <div
                key={card.step}
                className={`glass-card p-3 xs:p-5 border ${card.border} ${card.bg} flex flex-col gap-2 xs:gap-3`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[9px] xs:text-[10px] uppercase tracking-widest text-zinc-600">{card.label}</span>
                  <span className="text-xs font-mono text-zinc-700 shrink-0">{card.step}</span>
                </div>
                <h3 className="text-xs xs:text-sm font-semibold text-zinc-200 leading-snug">
                  {card.heading}
                </h3>
                <p className="text-[11px] xs:text-[12px] text-zinc-500 leading-relaxed flex-1">
                  {card.body}
                </p>
                <div className={`mt-1 px-2.5 xs:px-3 py-1.5 xs:py-2 rounded-lg bg-zinc-900/60 border ${card.border}`}>
                  <p className={`text-sm xs:text-base font-light tabular-nums ${card.statColor}`}>{card.stat}</p>
                  <p className="text-[9px] xs:text-[10px] text-zinc-600 mt-0.5">{card.statSub}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ── KPI Cards ─────────────────────────────────────────────── */}
        <section className="grid grid-cols-2 md:grid-cols-4 gap-2 xs:gap-3 sm:gap-4">
          <MetricCard
            label="Capacity Multiplier"
            value={`${kpis.capacityRatio}×`}
            sub={`${kpis.pagedSeqs} vs ${kpis.staticSeqs} seqs · 302 MB budget`}
            delta="measured"
            deltaType="positive"
            spark={CONCURRENCY_SPARK}
            accentColor="#34d399"
          />
          <MetricCard
            label="KV Memory vs Static"
            value={`${kpis.memoryRatio}× less`}
            sub="32 seqs at step 50 · measured"
            delta={`${memoryCompRows[0].pageforge} vs ${memoryCompRows[0].baseline}`}
            deltaType="positive"
            spark={VRAM_SPARK}
            accentColor="#38bdf8"
          />
          <MetricCard
            label="P50 Latency Overhead"
            value={`+${kpis.latencyOverheadP50}%`}
            sub={`${latencyData[0].p50} ms → ${latencyData[1].p50} ms vs HF DynamicCache`}
            delta="host dispatch"
            deltaType="neutral"
            spark={OVERHEAD_SPARK}
            accentColor="#fbbf24"
          />
          <MetricCard
            label="Allocator Throughput"
            value={`${kpis.allocThroughputMpps}M/s`}
            sub="pages/sec via Python · 500 cycles · 0 leaks"
            delta={stressResult.passed ? "PASS" : "FAIL"}
            deltaType="positive"
            spark={ALLOC_SPARK}
            accentColor="#a78bfa"
          />
        </section>

        {/* ── Comparison Tables ─────────────────────────────────────── */}
        <section className="glass-card p-3 xs:p-4 sm:p-6">
          <p className="label-tag mb-1">At a Glance</p>
          <h2 className="section-title mb-4 xs:mb-6">PageForge vs Alternatives</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 xs:gap-6 sm:gap-8">
            <ComparisonTable
              title="Memory Efficiency"
              baseLabel="Static max_len=512"
              rows={memoryCompRows}
              note="Baseline: transformers StaticCache(max_cache_len=512), bytes measured with torch.cuda.memory_allocated. HF's default DynamicCache also grows with the sequence; PageForge's gain over it is a bounded, reusable pool, not fewer bytes."
            />
            <ComparisonTable
              title="Decode Latency"
              baseLabel="HF DynamicCache"
              rows={latencyCompRows}
              note={latencyMeta.setup}
            />
          </div>
        </section>

        {/* ── VRAM Efficiency Chart ─────────────────────────────────── */}
        <section className="glass-card p-3 xs:p-4 sm:p-6">
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-3 xs:gap-4 mb-4 xs:mb-6">
            <div className="min-w-0">
              <p className="label-tag mb-1">Memory Efficiency</p>
              <h2 className="section-title">VRAM Consumption vs Decode Step</h2>
              <p className="section-sub text-[10px] xs:text-xs mt-1">
                32 concurrent sequences · GPT-2 124M · fp16 · page arithmetic; step 50 checked against the measured run
              </p>
            </div>
            <div className="text-right shrink-0">
              <p className="text-2xl xs:text-3xl font-light text-emerald-400">{kpis.memoryRatio}×</p>
              <p className="text-[10px] xs:text-xs text-zinc-500">savings at step 50</p>
            </div>
          </div>
          <VRAMChart data={vramSeries} />
        </section>

        {/* ── Concurrency + Latency ─────────────────────────────────── */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 xs:gap-4">
          <section className="glass-card p-3 xs:p-4 sm:p-6">
            <p className="label-tag mb-1">Concurrency</p>
            <h2 className="section-title mb-0.5">Max Concurrent Sequences</h2>
            <p className="section-sub text-[10px] xs:text-xs mb-3 xs:mb-5">
              512-page pool · page_size=16 · 9-token prompt · page arithmetic · the advantage shrinks as sequences grow
            </p>
            <ConcurrencyChart data={concurrencySeries} naiveCap={NAIVE_CAP} />
          </section>

          <section className="glass-card p-3 xs:p-4 sm:p-6">
            <p className="label-tag mb-1">Latency Analysis</p>
            <h2 className="section-title mb-0.5">Decode Step Latency</h2>
            <p className="section-sub text-[10px] xs:text-xs mb-3 xs:mb-5">
              {latencyMeta.setup}
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
        <section className="glass-card p-3 xs:p-4 sm:p-6">
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-3 xs:gap-4 mb-4 xs:mb-6">
            <div className="min-w-0">
              <p className="label-tag mb-1">Pool Lifecycle</p>
              <h2 className="section-title">Sequential Batch Reuse</h2>
              <p className="section-sub text-[10px] xs:text-xs mt-1">
                8 concurrent seqs · 30 decode steps · free() at step 30 · page arithmetic, as `pageforge bench multi` runs it
              </p>
            </div>
            <div className="flex items-center gap-1 xs:gap-1.5 text-[10px] xs:text-xs text-emerald-400 shrink-0">
              <CheckCircle className="w-3 xs:w-4 h-3 xs:h-4 shrink-0" />
              <span className="font-medium">Zero leaks (allocator invariants tested)</span>
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
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 xs:gap-4">
          <ArchitectureCard gatherGBs={kernelBandwidth.gatherGBs} scatterGBs={kernelBandwidth.scatterGBs} copyGBs={kernelBandwidth.copyGBs} />
          <SystemSpecs specs={systemSpecs} stress={stressResult} />
        </div>

        {/* ── Roadmap ───────────────────────────────────────────────── */}
        <section className="glass-card p-3 xs:p-4 sm:p-6">
          <p className="label-tag mb-1">Engineering Roadmap</p>
          <h2 className="section-title mb-4 xs:mb-6">Planned Optimisations</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 xs:gap-3 sm:gap-4">
            {roadmap.map((item, i) => (
              <div
                key={item.title}
                className="flex gap-2 xs:gap-3 sm:gap-4 px-2.5 xs:px-4 py-2.5 xs:py-4 rounded-xl bg-zinc-900/50 border border-zinc-800/60"
              >
                <span className="text-[9px] xs:text-xs font-mono text-zinc-700 shrink-0 mt-0.5">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <div className="min-w-0">
                  <p className="text-xs xs:text-sm font-semibold text-zinc-200 mb-0.5 xs:mb-1">{item.title}</p>
                  <p className="text-[10px] xs:text-xs text-zinc-500 leading-relaxed">{item.detail}</p>
                </div>
                <span className="text-[8px] xs:text-[10px] px-1.5 xs:px-2 py-0.5 rounded-full border border-zinc-700 text-zinc-600 self-start shrink-0 whitespace-nowrap">
                  {item.status}
                </span>
              </div>
            ))}
          </div>
          <p className="text-[10px] xs:text-[11px] text-zinc-600 mt-3 xs:mt-4 leading-relaxed">
            Inspired by vLLM PagedAttention (Kwon et al., 2023). This implementation demonstrates the
            core paging mechanism from first principles: Rust allocator, hand-written CUDA scatter/gather
            kernels, and a DLPack zero-copy bridge to PyTorch.
          </p>
        </section>

      </main>

      {/* ── Footer ──────────────────────────────────────────────────── */}
      <footer className="border-t border-zinc-800/50 mt-6 xs:mt-8 py-4 xs:py-6 sm:py-8">
        <div className="max-w-7xl mx-auto px-3 xs:px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-2 xs:gap-3 text-center sm:text-left">
          <div className="flex flex-wrap justify-center sm:justify-start items-center gap-x-2 xs:gap-x-3 gap-y-1 text-[10px] xs:text-xs">
            <span className="text-zinc-600">PageForge v0.1.0</span>
            <span className="text-zinc-800 hidden sm:inline">·</span>
            <span className="text-zinc-600 hidden xs:inline">{systemSpecs.gpu.replace("NVIDIA ", "")} · sm_89 · torch CUDA {systemSpecs.torchCuda}</span>
            <span className="text-zinc-800 hidden sm:inline">·</span>
            <span className="text-zinc-600 hidden sm:inline">Measured {systemSpecs.measuredAt} · GPT-2 124M fp16</span>
          </div>
          <a
            href="https://github.com/BhargavKumarNath/PageForge"
            target="_blank"
            rel="noreferrer"
            className="text-[10px] xs:text-xs text-zinc-500 hover:text-zinc-300 transition-colors shrink-0"
          >
            Source on GitHub →
          </a>
        </div>
      </footer>
    </div>
  );
}
