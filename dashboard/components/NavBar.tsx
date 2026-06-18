"use client";

import { Github, ExternalLink } from "lucide-react";

export default function NavBar() {
  return (
    <header className="sticky top-0 z-50 border-b border-zinc-800/60 bg-zinc-950/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-3 xs:px-4 sm:px-6 h-11 xs:h-12 flex items-center justify-between">

        {/* Left — brand */}
        <div className="flex items-center gap-1.5 xs:gap-2 sm:gap-3 min-w-0">
          <div className="flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse shrink-0" />
            <span className="text-xs xs:text-sm font-semibold text-zinc-100 tracking-tight truncate">
              PageForge
            </span>
          </div>
          <span className="text-zinc-700 hidden xs:inline text-xs">|</span>
          <span className="text-[10px] xs:text-xs text-zinc-500 hidden sm:inline">v0.1.0</span>
        </div>

        {/* Center — context chips */}
        <div className="hidden md:flex items-center gap-2">
          {[
            { label: "RTX 4070 Laptop", accent: false },
            { label: "GPT-2 124M",      accent: false },
            { label: "fp16",            accent: false },
            { label: "75 / 75 tests",   accent: true  },
          ].map(({ label, accent }) => (
            <span
              key={label}
              className={`px-2 py-0.5 rounded-full text-[10px] font-medium border ${
                accent
                  ? "text-emerald-400 border-emerald-400/20 bg-emerald-400/5"
                  : "text-zinc-500 border-zinc-800 bg-zinc-900/50"
              }`}
            >
              {label}
            </span>
          ))}
        </div>

        {/* Right — links */}
        <div className="flex items-center gap-2 xs:gap-4 shrink-0">
          <a
            href="https://github.com/BhargavKumarNath/PageForge"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1 xs:gap-1.5 text-[10px] xs:text-xs text-zinc-500 hover:text-zinc-200 transition-colors"
          >
            <Github className="w-3 h-3 xs:w-3.5 xs:h-3.5 shrink-0" />
            <span className="hidden xs:inline">GitHub</span>
            <ExternalLink className="w-2.5 h-2.5 xs:w-3 xs:h-3 opacity-50 hidden xs:inline shrink-0" />
          </a>
        </div>
      </div>
    </header>
  );
}
