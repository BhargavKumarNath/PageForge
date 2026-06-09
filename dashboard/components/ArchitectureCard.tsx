interface Layer {
  name:     string;
  tech:     string;
  detail:   string;
  color:    string;
}

const layers: Layer[] = [
  {
    name:   "Python User Code",
    tech:   "pageforge.cache.PagedKVCache",
    detail: "DynamicCache subclass — drop-in for HuggingFace transformers",
    color:  "#a78bfa",
  },
  {
    name:   "Rust PageForge",
    tech:   "PyO3 · maturin",
    detail: "PageAllocator: O(1) VecDeque free-list · BlockTable: HashMap<seq_id, Vec<page_id>>",
    color:  "#fb7185",
  },
  {
    name:   "CUDA Kernels",
    tech:   "CuPy RawKernel (nvrtc)",
    detail: "gather_kv · scatter_kv_layer — 100–200 GB/s on sm_89",
    color:  "#34d399",
  },
  {
    name:   "GPU Pool",
    tech:   "fp16 · CuPy",
    detail: "(N_pages, page_size, n_layers × n_heads, d_head) — K and V combined",
    color:  "#38bdf8",
  },
];

const flow = [
  { from: "Python",  to: "Rust",  label: "alloc_for_seq / free_seq" },
  { from: "Rust",    to: "CUDA",  label: "page_ids → gather / scatter" },
  { from: "CUDA",    to: "Pool",  label: "non-contiguous ↔ contiguous" },
];

export default function ArchitectureCard() {
  return (
    <div className="glass-card p-6 h-full flex flex-col">
      <p className="label-tag mb-1">System Architecture</p>
      <h2 className="section-title mb-4">Component Stack</h2>

      {/* Layer stack */}
      <div className="flex-1 space-y-2">
        {layers.map((layer, i) => (
          <div key={layer.name} className="relative">
            <div
              className="rounded-xl px-4 py-3 border transition-colors"
              style={{
                background: `${layer.color}08`,
                borderColor: `${layer.color}20`,
              }}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-xs font-semibold text-zinc-200">{layer.name}</span>
                    <span
                      className="text-[10px] px-1.5 py-0.5 rounded-full font-medium"
                      style={{ color: layer.color, background: `${layer.color}15` }}
                    >
                      {layer.tech}
                    </span>
                  </div>
                  <p className="text-[11px] text-zinc-500 leading-snug">{layer.detail}</p>
                </div>
                <span
                  className="text-xs font-mono font-bold shrink-0 mt-0.5"
                  style={{ color: layer.color }}
                >
                  L{i}
                </span>
              </div>
            </div>

            {/* Connector arrow */}
            {i < layers.length - 1 && (
              <div className="flex items-center gap-2 px-4 py-1">
                <div className="w-px h-3 bg-zinc-800 ml-3" />
                <span className="text-[10px] text-zinc-600 ml-1">
                  {flow[i]?.label}
                </span>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* DLPack note */}
      <div className="mt-4 px-3 py-2 rounded-lg bg-zinc-900/60 border border-zinc-800/60">
        <p className="text-[10px] text-zinc-500">
          <span className="text-zinc-400 font-medium">DLPack bridge</span>
          {" "}— zero-copy PyTorch ↔ CuPy tensor interop on each decode step.
          No host round-trips. Verified bit-exact vs HF DynamicCache (max diff = 0.0).
        </p>
      </div>
    </div>
  );
}
