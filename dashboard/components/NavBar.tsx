"use client";

import { Github, ExternalLink } from "lucide-react";

export default function NavBar() {
  return (
    <header className="sticky top-0 z-50 border-b border-zinc-800/60 bg-zinc-950/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-6 h-12 flex items-center justify-between">

        {/* Left — brand */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-sm font-semibold text-zinc-100 tracking-tight">
              PageForge
            </span>
          </div>
          <span className="text-zinc-700">|</span>
          <span className="text-xs text-zinc-500">v0.1.0</span>
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
        <div className="flex items-center gap-4">
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-200 transition-colors"
          >
            <Github className="w-3.5 h-3.5" />
            <span>GitHub</span>
            <ExternalLink className="w-3 h-3 opacity-50" />
          </a>
        </div>
      </div>
    </header>
  );
}
