"use client";

import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, ReferenceArea,
} from "recharts";
import { LifecyclePoint } from "@/lib/data";

interface Props {
  data:     LifecyclePoint[];
  naiveMB:  number;
  peakVRAM: number;
  peakPct:  number;
}

interface TooltipPayload {
  active?:  boolean;
  payload?: Array<{ payload?: LifecyclePoint }>;
  label?:   number;
}

function CustomTooltip({ active, payload, label }: TooltipPayload) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;

  if (d.batch === 0) {
    return (
      <div className="glass-card-hi px-3 py-2.5 text-xs">
        <p className="text-amber-400 font-medium">free() called</p>
        <p className="text-zinc-500 mt-1">All pages returned to pool</p>
      </div>
    );
  }

  return (
    <div className="glass-card-hi px-3 py-2.5 text-xs space-y-1.5">
      <p className="text-zinc-400 mb-1">
        Step {label}
        <span className={`ml-2 text-[10px] px-1.5 py-0.5 rounded-full ${
          d.batch === 1
            ? "text-emerald-400 bg-emerald-400/10"
            : "text-sky-400 bg-sky-400/10"
        }`}>
          Batch {d.batch}
        </span>
      </p>
      <div className="flex justify-between gap-5">
        <span className="text-zinc-500">Pages allocated</span>
        <span className="text-zinc-300 font-mono">{d.pages} / 512</span>
      </div>
      <div className="flex justify-between gap-5">
        <span className="text-zinc-500">Pool utilisation</span>
        <span className="text-emerald-400 font-mono">{d.util.toFixed(1)}%</span>
      </div>
      <div className="flex justify-between gap-5">
        <span className="text-zinc-500">Active VRAM</span>
        <span className="text-sky-400 font-mono">{d.vramMB.toFixed(2)} MB</span>
      </div>
    </div>
  );
}

export default function LifecycleChart({ data, naiveMB, peakVRAM, peakPct }: Props) {
  const naivePct = (naiveMB / naiveMB) * 100; // 100% — used for reference line

  return (
    <div>
      {/* Stats row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 xs:gap-3 sm:gap-4 mb-4 xs:mb-6">
        {[
          { label: "Naive VRAM (8 seqs)",  value: `${naiveMB} MB`,   color: "text-rose-400"    },
          { label: "PageForge peak (B1)",  value: `${peakVRAM} MB`,  color: "text-emerald-400" },
          { label: "% of naive budget",    value: `${peakPct}%`,     color: "text-sky-400"     },
          { label: "Pages after free()",   value: "0",               color: "text-emerald-400" },
        ].map(({ label, value, color }) => (
          <div key={label} className="bg-zinc-900/50 rounded-xl px-2.5 xs:px-4 py-2 xs:py-3">
            <p className="label-tag text-[8px] xs:text-[10px] mb-1">{label}</p>
            <p className={`text-xl xs:text-2xl font-light tabular-nums ${color}`}>{value}</p>
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-x-3 xs:gap-x-5 gap-y-1.5 xs:gap-y-2 mb-3 xs:mb-4">
        <div className="flex items-center gap-1.5 xs:gap-2 min-w-0">
          <div className="w-8 h-0.5 bg-emerald-400 shrink-0" />
          <span className="text-[10px] xs:text-xs text-zinc-400">Batch 1: initial allocation</span>
        </div>
        <div className="flex items-center gap-1.5 xs:gap-2 min-w-0">
          <div className="w-8 h-0.5 bg-sky-400 shrink-0" />
          <span className="text-[10px] xs:text-xs text-zinc-400">Batch 2: same pages reused</span>
        </div>
        <div className="flex items-center gap-1.5 xs:gap-2 min-w-0">
          <div className="w-2 h-2 rounded-full bg-amber-400/60 shrink-0" />
          <span className="text-[10px] xs:text-xs text-zinc-400">free() called</span>
        </div>
        <div className="flex items-center gap-1.5 xs:gap-2 min-w-0">
          <div className="w-8 h-0.5 border-t border-dashed border-rose-400/50 shrink-0" />
          <span className="text-[10px] xs:text-xs text-zinc-400">Naive budget: {naiveMB} MB</span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: 40 }}>
          <defs>
            <linearGradient id="lcGrad1" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#34d399" stopOpacity={0.35} />
              <stop offset="100%" stopColor="#34d399" stopOpacity={0.02} />
            </linearGradient>
            <linearGradient id="lcGrad2" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#38bdf8" stopOpacity={0.30} />
              <stop offset="100%" stopColor="#38bdf8" stopOpacity={0.02} />
            </linearGradient>
          </defs>

          {/* Batch region shading */}
          <ReferenceArea x1={0}  x2={30} fill="rgba(52,211,153,0.03)"  fillOpacity={1} />
          <ReferenceArea x1={30} x2={60} fill="rgba(56,189,248,0.025)" fillOpacity={1} />

          <CartesianGrid stroke="rgba(255,255,255,0.04)" vertical={false} />

          <XAxis
            dataKey="step"
            tick={{ fill: "#52525b", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            tickMargin={6}
            label={{ value: "Decode step", position: "insideBottom", offset: -2, fill: "#52525b", fontSize: 10 }}
          />
          <YAxis
            tick={{ fill: "#52525b", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            width={40}
            tickFormatter={v => `${v.toFixed(0)}%`}
            domain={[0, 12]}
          />

          <Tooltip
            content={<CustomTooltip />}
            cursor={{ stroke: "rgba(255,255,255,0.07)", strokeWidth: 1 }}
          />

          {/* free() divider at step 30 */}
          <ReferenceLine
            x={30}
            stroke="rgba(251,191,36,0.4)"
            strokeWidth={1.5}
            strokeDasharray="3 3"
            label={{ value: "free()", position: "top", fill: "#fbbf24", fontSize: 9, dy: -2 }}
          />

          {/* Batch 1 — emerald */}
          <Area
            type="monotone"
            dataKey="util1"
            stroke="#34d399"
            strokeWidth={2}
            fill="url(#lcGrad1)"
            dot={false}
            activeDot={{ r: 3, fill: "#34d399", strokeWidth: 0 }}
            connectNulls={false}
          />
          {/* Batch 2 — sky */}
          <Area
            type="monotone"
            dataKey="util2"
            stroke="#38bdf8"
            strokeWidth={2}
            fill="url(#lcGrad2)"
            dot={false}
            activeDot={{ r: 3, fill: "#38bdf8", strokeWidth: 0 }}
            connectNulls={false}
          />
        </AreaChart>
      </ResponsiveContainer>

      <p className="text-center text-[10px] xs:text-[11px] text-zinc-600 mt-2 xs:mt-3">
        Batch 2 reuses the exact same physical pages freed by Batch 1. Pool never grows, zero fragmentation.
      </p>
    </div>
  );
}
