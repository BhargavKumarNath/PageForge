import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
  weight: ["300", "400", "500", "600"],
});

export const metadata: Metadata = {
  title: "PageForge — Paged KV-Cache Memory Manager",
  description:
    "System performance dashboard for PageForge: a paged KV-cache memory manager for LLM inference built on Rust, CUDA, and PyTorch. 8× more concurrent sequences per GB vs naive pre-allocation.",
  keywords: ["LLM inference", "KV-cache", "PagedAttention", "CUDA", "Rust", "vLLM"],
  openGraph: {
    title: "PageForge — Paged KV-Cache Memory Manager",
    description: "8× more concurrent sequences per GB. 33% latency overhead. Built on Rust + CUDA.",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body>{children}</body>
    </html>
  );
}
