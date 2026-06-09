import { CompRow } from "@/lib/data";

interface Props {
  title:      string;
  baseLabel:  string;
  rows:       CompRow[];
  note?:      string;
}

export default function ComparisonTable({ title, baseLabel, rows, note }: Props) {
  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs font-semibold text-zinc-300">{title}</p>
        <div className="flex items-center gap-2 text-[10px] text-zinc-600">
          <span className="px-2 py-0.5 rounded bg-zinc-800/80">{baseLabel}</span>
          <span className="text-zinc-700">vs</span>
          <span className="px-2 py-0.5 rounded bg-emerald-400/10 text-emerald-400/70">PageForge</span>
        </div>
      </div>

      <div className="rounded-xl overflow-hidden border border-zinc-800/60">
        {/* Header */}
        <div className="grid grid-cols-[2fr_1fr_1fr_1fr] bg-zinc-900/60 px-4 py-2">
          <span className="text-[10px] uppercase tracking-widest text-zinc-600">Metric</span>
          <span className="text-[10px] uppercase tracking-widest text-zinc-600 text-right">{baseLabel}</span>
          <span className="text-[10px] uppercase tracking-widest text-emerald-400/60 text-right">PageForge</span>
          <span className="text-[10px] uppercase tracking-widest text-zinc-600 text-right">Δ</span>
        </div>

        {/* Rows */}
        {rows.map((row, i) => (
          <div
            key={row.metric}
            className={`grid grid-cols-[2fr_1fr_1fr_1fr] px-4 py-2.5 items-center border-t border-zinc-800/40 ${
              i % 2 === 0 ? "bg-zinc-950/30" : "bg-transparent"
            }`}
          >
            <span className="text-xs text-zinc-400">{row.metric}</span>
            <span className="text-xs font-mono text-zinc-500 text-right">{row.baseline}</span>
            <span className="text-xs font-mono text-right"
              style={{ color: row.better === "paged" ? "#34d399" : row.better === "baseline" ? "#94a3b8" : "#a1a1aa" }}
            >
              {row.pageforge}
            </span>
            <span
              className={`text-xs font-semibold text-right ${
                row.better === "paged"
                  ? "text-emerald-400"
                  : row.better === "baseline"
                  ? "text-amber-400"
                  : "text-zinc-600"
              }`}
            >
              {row.delta}
            </span>
          </div>
        ))}
      </div>

      {note && (
        <p className="text-[10px] text-zinc-600 mt-2 leading-relaxed">{note}</p>
      )}
    </div>
  );
}
