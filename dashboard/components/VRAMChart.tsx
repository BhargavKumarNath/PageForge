"use client";

import {
  ComposedChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { VRAMPoint } from "@/lib/data";

interface Props { data: VRAMPoint[] }

interface TooltipPayload {
  payload?: Array<{ payload?: { pagedMB?: number; naiveMB?: number } }>;
  active?: boolean;
  label?: number;
}

function CustomTooltip({ active, payload, label }: TooltipPayload) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload ?? {};
  const savings = d.naiveMB && d.pagedMB ? (d.naiveMB / d.pagedMB).toFixed(1) : "-";
  return (
    <div className="glass-card-hi px-4 py-3 text-xs space-y-1.5 min-w-[160px]">
      <p className="text-zinc-400 font-medium mb-2">Decode step {label}</p>
      <div className="flex justify-between gap-6">
        <span className="text-zinc-500">Naive (512-tok)</span>
        <span className="text-rose-400 font-mono">{d.naiveMB?.toFixed(0)} MB</span>
      </div>
      <div className="flex justify-between gap-6">
        <span className="text-zinc-500">PageForge</span>
        <span className="text-emerald-400 font-mono">{d.pagedMB?.toFixed(1)} MB</span>
      </div>
      <div className="border-t border-zinc-800 pt-1.5 flex justify-between">
        <span className="text-zinc-500">Savings</span>
        <span className="text-zinc-200 font-semibold">{savings}×</span>
      </div>
    </div>
  );
}

export default function VRAMChart({ data }: Props) {
  return (
    <div>
      {/* Legend */}
      <div className="flex items-center gap-6 mb-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-0.5 border-t-2 border-dashed border-rose-400/70" />
          <span className="text-xs text-zinc-400">Naive pre-alloc (max 512 tokens)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-8 h-0.5 bg-emerald-400" />
          <span className="text-xs text-zinc-400">PageForge paged</span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: 0 }}>
          <defs>
            <linearGradient id="pagedAreaGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#34d399" stopOpacity={0.35} />
              <stop offset="100%" stopColor="#34d399" stopOpacity={0.02} />
            </linearGradient>
            <linearGradient id="naiveAreaGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#fb7185" stopOpacity={0.12} />
              <stop offset="100%" stopColor="#fb7185" stopOpacity={0.01} />
            </linearGradient>
          </defs>

          <CartesianGrid
            strokeDasharray="0"
            stroke="rgba(255,255,255,0.04)"
            vertical={false}
          />

          <XAxis
            dataKey="step"
            tick={{ fill: "#52525b", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            tickMargin={8}
            label={{
              value: "Decode step",
              position: "insideBottom",
              offset: -2,
              fill: "#52525b",
              fontSize: 11,
            }}
          />

          <YAxis
            tick={{ fill: "#52525b", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={v => `${v} MB`}
            width={68}
          />

          <Tooltip content={<CustomTooltip />} cursor={{ stroke: "rgba(255,255,255,0.07)", strokeWidth: 1 }} />

          {/* Naive waste band (background) */}
          <Area
            type="monotone"
            dataKey="naiveMB"
            fill="url(#naiveAreaGrad)"
            stroke="#fb7185"
            strokeWidth={1.5}
            strokeDasharray="6 4"
            dot={false}
            activeDot={false}
          />

          {/* PageForge paged area (foreground, covers lower waste band) */}
          <Area
            type="monotone"
            dataKey="pagedMB"
            fill="url(#pagedAreaGrad)"
            stroke="#34d399"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: "#34d399", strokeWidth: 0 }}
          />

          {/* Efficiency annotation at step 50 */}
          <ReferenceLine
            x={50}
            stroke="rgba(255,255,255,0.06)"
            strokeDasharray="3 3"
            label={{ value: "8× at step 50", position: "top", fill: "#52525b", fontSize: 10 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
