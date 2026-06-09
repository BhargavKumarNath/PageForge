"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LabelList } from "recharts";
import { LatencyMeasurement } from "@/lib/data";

interface Props {
  data:            LatencyMeasurement[];
  overheadP50Pct:  number;
  overheadP99Pct:  number;
  bottleneck:      string;
}

export default function LatencyChart({ data, overheadP50Pct, overheadP99Pct, bottleneck }: Props) {
  const p50 = data.map(d => ({ name: d.name, value: d.p50, color: d.color }));
  const p99 = data.map(d => ({ name: d.name, value: d.p99, color: d.color }));

  function HBar({ items, label }: { items: typeof p50; label: string }) {
    return (
      <div className="mb-4">
        <p className="text-[10px] uppercase tracking-widest text-zinc-600 mb-2">{label}</p>
        <div className="space-y-2">
          {items.map(item => {
            const maxVal = Math.max(...items.map(i => i.value));
            const pct = (item.value / maxVal) * 100;
            const isPageForge = item.name.includes("PageForge");
            return (
              <div key={item.name} className="group">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] text-zinc-400">{item.name}</span>
                  <span
                    className="text-[11px] font-mono font-semibold"
                    style={{ color: item.color }}
                  >
                    {item.value.toFixed(1)} ms
                  </span>
                </div>
                <div className="h-5 bg-zinc-900 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{
                      width: `${pct}%`,
                      background: isPageForge
                        ? "linear-gradient(90deg, #34d399cc, #34d39966)"
                        : "linear-gradient(90deg, #94a3b8cc, #94a3b866)",
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-1">
      <HBar items={p50} label="P50 (median latency)" />
      <HBar items={p99} label="P99 (tail latency)" />

      {/* Overhead summary */}
      <div className="mt-4 grid grid-cols-2 gap-3">
        <div className="bg-zinc-900/60 rounded-xl px-4 py-3 text-center">
          <p className="text-[10px] text-zinc-500 uppercase tracking-widest mb-1">P50 overhead</p>
          <p className="text-2xl font-light text-amber-400">+{overheadP50Pct}%</p>
        </div>
        <div className="bg-zinc-900/60 rounded-xl px-4 py-3 text-center">
          <p className="text-[10px] text-zinc-500 uppercase tracking-widest mb-1">P99 overhead</p>
          <p className="text-2xl font-light text-emerald-400">+{overheadP99Pct}%</p>
        </div>
      </div>

      <div className="mt-3 px-3 py-2.5 rounded-lg bg-zinc-900/40 border border-zinc-800/60">
        <p className="text-[10px] text-zinc-500 leading-relaxed">
          <span className="text-zinc-400 font-medium">Bottleneck: </span>
          {bottleneck}
        </p>
      </div>
    </div>
  );
}
