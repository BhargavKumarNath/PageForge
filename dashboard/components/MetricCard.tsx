import { TrendingDown, TrendingUp, Minus } from "lucide-react";

interface MetricCardProps {
  label:      string;
  value:      string;
  sub:        string;
  delta:      string;
  deltaType:  "positive" | "negative" | "neutral";
  /** Micro-sparkline data (0–100 normalised) */
  spark?:     number[];
  accentColor?: string;
}

function MiniSparkline({ data, color }: { data: number[]; color: string }) {
  const w = 64, h = 24;
  const max = Math.max(...data, 1);
  const pts = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = h - (v / max) * h;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg width={w} height={h} className="overflow-visible" aria-hidden>
      <defs>
        <linearGradient id={`sg-${color.replace("#", "")}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor={color} stopOpacity={0.3} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <polyline
        points={`${pts} ${w},${h} 0,${h}`}
        fill={`url(#sg-${color.replace("#", "")})`}
        stroke="none"
      />
      <polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function MetricCard({
  label,
  value,
  sub,
  delta,
  deltaType,
  spark,
  accentColor = "#34d399",
}: MetricCardProps) {
  const deltaStyle = {
    positive: "text-emerald-400 bg-emerald-400/8",
    negative: "text-rose-400   bg-rose-400/8",
    neutral:  "text-zinc-400   bg-zinc-400/8",
  }[deltaType];

  const DeltaIcon = deltaType === "positive"
    ? TrendingUp
    : deltaType === "negative"
    ? TrendingDown
    : Minus;

  return (
    <div className="glass-card p-3 xs:p-4 sm:p-5 flex flex-col gap-2 xs:gap-3 hover:border-zinc-700/60 transition-colors">

      {/* Label */}
      <p className="label-tag text-[8px] xs:text-[10px]">{label}</p>

      {/* Value row */}
      <div className="flex items-end justify-between gap-1.5 xs:gap-2">
        <span
          className="text-2xl xs:text-3xl sm:text-4xl font-light tracking-tight text-zinc-100 leading-none"
          style={{ fontVariantNumeric: "tabular-nums" }}
        >
          {value}
        </span>
        {spark && <span className="hidden xs:block scale-75 xs:scale-100 origin-bottom-right"><MiniSparkline data={spark} color={accentColor} /></span>}
      </div>

      {/* Footer */}
      <div className="flex flex-col xs:flex-row xs:items-center justify-between pt-1.5 xs:pt-2 border-t border-zinc-800/60 gap-1.5 xs:gap-2">
        <span className="text-[10px] xs:text-[11px] text-zinc-500 leading-snug flex-1 min-w-0">{sub}</span>
        <span className={`flex items-center gap-0.5 xs:gap-1 text-[9px] xs:text-[11px] font-medium px-1.5 xs:px-2 py-0.5 rounded-full shrink-0 ${deltaStyle}`}>
          <DeltaIcon className="w-2.5 xs:w-3 h-2.5 xs:h-3 shrink-0" />
          {delta}
        </span>
      </div>
    </div>
  );
}
