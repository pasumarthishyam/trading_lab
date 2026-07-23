import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Trading Lab — Research Dashboard",
  description:
    "Quantitative trading research platform for strategy development and validation",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen">
        <div className="flex min-h-screen">
          {/* Sidebar */}
          <aside className="w-60 border-r border-[var(--border)] bg-[var(--bg-secondary)] flex flex-col fixed h-screen">
            <div className="p-5 border-b border-[var(--border)]">
              <a href="/" className="block">
                <h1 className="text-lg font-bold tracking-tight text-white">
                  Trading Lab
                </h1>
                <p className="text-xs text-[var(--text-muted)] mt-0.5">
                  Research Dashboard
                </p>
              </a>
            </div>
            <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
              <a
                href="/"
                className="flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-[var(--text-secondary)] hover:text-white hover:bg-[var(--bg-card)] transition-colors"
              >
                <span className="text-base">🏠</span>
                Home
              </a>
              <a
                href="/strategy/vcf"
                className="flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-[var(--text-secondary)] hover:text-white hover:bg-[var(--bg-card)] transition-colors"
              >
                <span className="text-base">📊</span>
                VCF Strategy
              </a>
              <a
                href="/strategy/vcf/NIFTY"
                className="flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-[var(--text-secondary)] hover:text-white hover:bg-[var(--bg-card)] transition-colors ml-4"
              >
                <span className="text-base">📈</span>
                NIFTY Tests
              </a>
              <a
                href="/strategy/vcf/insights"
                className="flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-[var(--text-secondary)] hover:text-white hover:bg-[var(--bg-card)] transition-colors ml-4"
              >
                <span className="text-base">💡</span>
                Master Insights
              </a>
              <div className="my-2 h-px bg-[var(--border)] mx-3" />
              <a
                href="/data-status"
                className="flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-[var(--text-secondary)] hover:text-white hover:bg-[var(--bg-card)] transition-colors"
              >
                <span className="text-base">📁</span>
                Data Status
              </a>
            </nav>
            <div className="p-3 border-t border-[var(--border)] text-xs text-[var(--text-muted)]">
              v0.1.0
            </div>
          </aside>

          {/* Main content */}
          <main className="flex-1 ml-60">{children}</main>
        </div>
      </body>
    </html>
  );
}
