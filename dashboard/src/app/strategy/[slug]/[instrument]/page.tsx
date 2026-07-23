import {
    getStrategy,
    getCategories,
    getTests,
    getInstruments,
    getLatestRun,
    getInsights,
    getCharts,
    getDataTables,
} from "@/lib/db";
import { notFound } from "next/navigation";

export default async function InstrumentTestsPage({
    params,
}: {
    params: Promise<{ slug: string; instrument: string }>;
}) {
    const { slug, instrument: instrumentSymbol } = await params;
    const strategy = getStrategy(slug);
    if (!strategy) return notFound();

    const instruments = getInstruments();
    const instrument = instruments.find((i) => i.symbol === instrumentSymbol);
    if (!instrument) return notFound();

    const categories = getCategories(strategy.id);
    const tests = getTests(strategy.id);

    // Group tests by category
    const testsByCategory: Record<string, typeof tests> = {};
    for (const t of tests) {
        const cat = t.category_slug || "unknown";
        if (!testsByCategory[cat]) testsByCategory[cat] = [];
        testsByCategory[cat].push(t);
    }

    // Get run data for each test
    const testData = tests.map((t) => {
        const run = getLatestRun(t.id, instrument.id);
        const insights = run ? getInsights(run.id) : [];
        const charts = run ? getCharts(run.id) : [];
        const tables = run ? getDataTables(run.id) : [];
        return { test: t, run, insights, chartCount: charts.length, tableCount: tables.length };
    });

    const completed = testData.filter((td) => td.run).length;

    return (
        <div className="p-8 max-w-7xl">
            {/* Breadcrumb */}
            <div className="flex items-center gap-2 text-sm text-[var(--text-muted)] mb-6">
                <a href="/" className="hover:text-white transition-colors">Home</a>
                <span>→</span>
                <a href={`/strategy/${slug}`} className="hover:text-white transition-colors">
                    {strategy.name}
                </a>
                <span>→</span>
                <span className="text-[var(--text-secondary)]">{instrument.symbol}</span>
            </div>

            {/* Header */}
            <div className="flex items-center justify-between mb-8">
                <div>
                    <h1 className="text-2xl font-bold text-white mb-1">
                        {instrument.name} — Test Suite
                    </h1>
                    <p className="text-[var(--text-secondary)]">
                        {completed}/{tests.length} tests completed
                    </p>
                </div>
                <div className="flex items-center gap-2">
                    <span className="bg-[var(--accent-green)]/20 text-[var(--accent-green)] px-3 py-1 rounded-lg text-sm font-mono">
                        ✅ {completed}
                    </span>
                    <span className="bg-[var(--accent-amber)]/20 text-[var(--accent-amber)] px-3 py-1 rounded-lg text-sm font-mono">
                        ⏳ {tests.length - completed}
                    </span>
                </div>
            </div>

            {/* Test Grid by Category */}
            <div className="space-y-8">
                {categories.map((cat) => {
                    const catTests = testsByCategory[cat.slug] || [];
                    const catData = catTests.map((ct) =>
                        testData.find((td) => td.test.id === ct.id)!
                    );
                    const catCompleted = catData.filter((cd) => cd?.run).length;

                    return (
                        <div key={cat.id}>
                            <div className="flex items-center gap-3 mb-3">
                                <h2 className="text-sm font-medium text-[var(--text-muted)] uppercase tracking-wider">
                                    {cat.name}
                                </h2>
                                <span className="text-xs font-mono text-[var(--text-muted)]">
                                    {catCompleted}/{catTests.length}
                                </span>
                                <div className="flex-1 h-px bg-[var(--border)]" />
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                                {catData.map((td) => {
                                    if (!td) return null;
                                    const { test: t, run, insights, chartCount, tableCount } = td;
                                    const topInsight = insights.find(
                                        (i) => i.severity === "critical" || i.severity === "important"
                                    );

                                    return (
                                        <a
                                            key={t.id}
                                            href={`/strategy/${slug}/${instrumentSymbol}/${t.slug}`}
                                            className="card-hover bg-[var(--bg-card)] border border-[var(--border)] rounded-lg p-4 block"
                                        >
                                            <div className="flex items-start justify-between mb-2">
                                                <div className="flex items-center gap-2">
                                                    <span className="text-base">
                                                        {run ? "✅" : "⏳"}
                                                    </span>
                                                    <div>
                                                        <div className="flex items-center gap-1.5">
                                                            <span className="font-mono text-xs text-[var(--text-muted)]">
                                                                T{String(t.test_number).padStart(2, "0")}
                                                            </span>
                                                            <span className="text-sm font-medium text-white">
                                                                {t.name}
                                                            </span>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>

                                            <p className="text-xs text-[var(--text-muted)] line-clamp-2 mb-3">
                                                {t.description}
                                            </p>

                                            {run && (
                                                <div className="flex items-center gap-3 text-[10px] font-mono text-[var(--text-muted)] mb-2">
                                                    <span>📊 {chartCount} charts</span>
                                                    <span>📋 {tableCount} tables</span>
                                                    <span>💡 {insights.length} insights</span>
                                                </div>
                                            )}

                                            {topInsight && (
                                                <div
                                                    className={`rounded px-2 py-1.5 text-xs line-clamp-2 ${topInsight.severity === "critical"
                                                            ? "bg-[var(--accent-red)]/10 text-[var(--accent-red)]"
                                                            : "bg-[var(--accent-amber)]/10 text-[var(--accent-amber)]"
                                                        }`}
                                                >
                                                    {topInsight.insight_text}
                                                </div>
                                            )}

                                            {run && (
                                                <p className="text-[10px] text-[var(--text-muted)] mt-2 font-mono">
                                                    {run.run_at.split("T")[0]} · {run.duration_seconds?.toFixed(1)}s · {run.row_count?.toLocaleString()} rows
                                                </p>
                                            )}
                                        </a>
                                    );
                                })}
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
