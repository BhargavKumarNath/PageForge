import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import "./globals.css";
import { kpis } from "@/lib/data";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
  weight: ["300", "400", "500", "600"],
});

export const metadata: Metadata = {
  title: "PageForge : Paged KV-Cache Memory Manager",
  description:
    `Measured dashboard for PageForge, a paged KV-cache for LLM inference built on Rust, CUDA and PyTorch: ${kpis.capacityRatio}× more GPT-2 sequences per KV budget than static pre-allocation, at +${kpis.latencyOverheadP50}% P50 decode latency.`,
  keywords: ["LLM inference", "KV-cache", "PagedAttention", "CUDA", "Rust", "vLLM"],
  openGraph: {
    title: "PageForge : Paged KV-Cache Memory Manager",
    description: `${kpis.capacityRatio}× more sequences per KV budget, +${kpis.latencyOverheadP50}% P50 latency. Rust + CUDA.`,
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const isProduction = process.env.NODE_ENV === "production";

  return (
    <html lang="en" className={inter.variable}>
      <body>
        {children}
        {isProduction && <Analytics />}
      </body>
    </html>
  );
}
