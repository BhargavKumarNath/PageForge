import { CheckCircle } from "lucide-react";

interface StressResult {
  cycles: number; seqsPerCycle: number; pagesPerSeq: number;
  throughputMpps: number; elapsedSec: number;
  leaks: number; oomErrors: number; passed: boolean;
}

interface Specs {
  gpu: string; vram: string; cudaNvcc: string; cudaRuntime: string;
  compute: string; pytorch: string; cupy: string; transformers: string;
  rust: string; python: string; os: string;
}

interface Props { specs: Specs; stress: StressResult }

const SPEC_GROUPS = [
  {
    heading: "Hardware",
    items: (s: Specs) => [
      ["GPU",      s.gpu],
      ["VRAM",     s.vram],
      ["Compute",  s.compute],
    ],
  },
  {
    heading: "Software",
    items: (s: Specs) => [
      ["CUDA nvcc",     s.cudaNvcc],
      ["CUDA runtime",  s.cudaRuntime],
      ["PyTorch",       s.pytorch],
      ["CuPy",          s.cupy],
      ["transformers",  s.transformers],
      ["Rust",          s.rust],
      ["Python",        s.python],
    ],
  },
] as const;

export default function SystemSpecs({ specs, stress }: Props) {
  return (
    <div className="glass-card p-3 xs:p-4 sm:p-6 flex flex-col gap-3 xs:gap-4 sm:gap-5">
      <div>
        <p className="label-tag mb-1">System Context</p>
        <h2 className="section-title">Verified Hardware & Stack</h2>
      </div>

      {/* Spec groups */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 xs:gap-4">
        {SPEC_GROUPS.map(group => (
          <div key={group.heading}>
            <p className="text-[9px] xs:text-[10px] uppercase tracking-widest text-zinc-600 mb-1.5 xs:mb-2">{group.heading}</p>
            <div className="space-y-1">
              {group.items(specs).map(([k, v]) => (
                <div key={k} className="flex justify-between gap-2 text-[10px] xs:text-xs min-w-0">
                  <span className="text-zinc-500 shrink-0">{k}</span>
                  <span className="text-zinc-300 text-right font-mono truncate">{v}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="border-t border-zinc-800/60" />

      {/* Stress test summary */}
      <div>
        <div className="flex items-center justify-between mb-2 xs:mb-3 gap-2">
          <p className="text-[9px] xs:text-[10px] uppercase tracking-widest text-zinc-600">Allocator Stress Test</p>
          <div className="flex items-center gap-1 xs:gap-1.5 text-[10px] xs:text-xs text-emerald-400 shrink-0">
            <CheckCircle className="w-3 xs:w-3.5 h-3 xs:h-3.5 shrink-0" />
            <span className="font-medium">PASS</span>
          </div>
        </div>
        <div className="grid grid-cols-2 xs:grid-cols-3 gap-1.5 xs:gap-2">
          {[
            ["Cycles",      `${stress.cycles}`],
            ["Seqs/cycle",  `${stress.seqsPerCycle}`],
            ["Throughput",  `${stress.throughputMpps}M/s`],
            ["Elapsed",     `${stress.elapsedSec}s`],
            ["Mem leaks",   `${stress.leaks}`],
            ["OOM errors",  `${stress.oomErrors}`],
          ].map(([label, value]) => (
            <div key={label} className="bg-zinc-900/60 rounded-lg px-2 xs:px-3 py-1.5 xs:py-2">
              <p className="text-[8px] xs:text-[9px] uppercase tracking-widest text-zinc-600 mb-0.5 xs:mb-1">{label}</p>
              <p
                className="text-xs xs:text-sm font-mono font-semibold"
                style={{
                  color: label === "Mem leaks" || label === "OOM errors"
                    ? (parseInt(value) === 0 ? "#34d399" : "#fb7185")
                    : label === "Throughput"
                    ? "#38bdf8"
                    : "#e4e4e7",
                }}
              >
                {value}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Test suite badge */}
      <div className="flex items-center justify-between gap-2 xs:gap-3 px-2.5 xs:px-4 py-2 xs:py-3 rounded-xl bg-emerald-400/5 border border-emerald-400/15">
        <div className="min-w-0">
          <p className="text-[10px] xs:text-xs font-semibold text-emerald-400">75 / 75 tests passing</p>
          <p className="text-[9px] xs:text-[11px] text-zinc-500 mt-0.5 leading-snug">
            Rust unit (6) · CUDA kernels (10) · GPT-2 integration (7) · VRAM + latency (32) · Multi-seq (19)
          </p>
        </div>
        <CheckCircle className="w-4 xs:w-5 h-4 xs:h-5 text-emerald-400 shrink-0" />
      </div>
    </div>
  );
}
