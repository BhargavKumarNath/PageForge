"use client";

import {
  ComposedChart, Bar, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
  Cell,
} from "recharts";
import { ConcurrencyPoint } from "@/lib/data";

interface Props { data: ConcurrencyPoint[]; naiveCap: number }

interface TooltipPayload {
  active?: boolean;
  payload?: Array<{ value: number }>;
  label?: number;
}

function CustomTooltip({ active, payload, label }: TooltipPayload) {
  if (!active || !payload?.length) return null;
  const paged = payload[0]?.value ?? 0;
  const multiplier = (paged / 16).toFixed(1);
  return (
    <div className="glass-card-hi px-3 py-2.5 text-xs space-y-1">
      <p className="text-zinc-400 mb-1.5">Step {label}</p>
      <div className="flex justify-between gap-5">
        <span className="text-zinc-500">Naive (max=512)</span>
        <span className="text-zinc-400 font-mono">16</span>
      </div>
      <div className="flex justify-between gap-5">
        <span className="text-zinc-500">PageForge</span>
        <span className="text-emerald-400 font-mono">{paged}</span>
      </div>
      <div className="border-t border-zinc-800 pt-1.5 flex justify-between">
        <span className="text-zinc-500">Multiplier</span>
        <span className="text-zinc-100 font-semibold">{multiplier}×</span>
      </div>
    </div>
  );
}

export default function ConcurrencyChart({ data, naiveCap }: Props) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <ComposedChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: -10 }}>
        <defs>
          <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#34d399" stopOpacity={0.8} />
            <stop offset="100%" stopColor="#34d399" stopOpacity={0.3} />
          </linearGradient>
        </defs>

        <CartesianGrid stroke="rgba(255,255,255,0.04)" vertical={false} strokeDasharray="0" />

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
          width={36}
        />

        <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />

        {/* Naive reference line */}
        <ReferenceLine
          y={naiveCap}
          stroke="#fb7185"
          strokeDasharray="5 3"
          strokeWidth={1.5}
          label={{
            value: `Naive: ${naiveCap}`,
            position: "insideTopRight",
            fill: "#fb7185",
            fontSize: 10,
          }}
        />

        <Bar dataKey="paged" radius={[3, 3, 0, 0]} maxBarSize={24}>
          {data.map((_, i) => (
            <Cell
              key={i}
              fill={`url(#barGrad)`}
              opacity={data[i].multiplier > 4 ? 1 : 0.5 + data[i].multiplier / 8}
            />
          ))}
        </Bar>
      </ComposedChart>
    </ResponsiveContainer>
  );
}
