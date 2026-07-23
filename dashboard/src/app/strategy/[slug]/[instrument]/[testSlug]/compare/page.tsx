import {
    getStrategy,
    getTest,
    getInstruments,
    getInsights,
    getCharts,
    getDataTables,
    type TestRun,
} from "@/lib/db";
import { notFound } from "next/navigation";

function getRun(runId: number): TestRun | undefined {
    const { getDb } = require("@/lib/db");
    return getDb()
        .prepare("SELECT * FROM test_runs WHERE id = ?")
        .get(runId) as TestRun | undefined;
}

export default async function ComparePage({
    params,
    searchParams,
}: {
    params: Promise<{ slug: string; instrument: string; testSlug: string }>;
    searchParams: Promise<{ run1?: string; run2?: string }>;
}) {
    const { slug, instrument: instrumentSymbol, testSlug } = await params;
    const { run1: run1Id, run2: run2Id } = await searchParams;

    const strategy = getStrategy(slug);
    const test = getTest(testSlug);
    const instruments = getInstruments();
    const instrument = instruments.find((i) => i.symbol === instrumentSymbol);

    if (!strategy || !test || !instrument || !run1Id || !run2Id) return notFound();

    const run1 = getRun(parseInt(run1Id));
    const run2 = getRun(parseInt(run2Id));
    if (!run1 || !run2) return notFound();

    const insights1 = getInsights(run1.id);
    const insights2 = getInsights(run2.id);
    const charts1 = getCharts(run1.id);
    const charts2 = getCharts(run2.id);

    // Find insight differences
    const insightMap1 = new Map(insights1.map((i) => [i.insight_key, i]));
    const insightMap2 = new Map(insights2.map((i) => [i.insight_key, i]));
    const allKeys = new Set([...insightMap1.keys(), ...insightMap2.keys()]);

    const insightDiffs = Array.from(allKeys).map((key) => {
        const i1 = insightMap1.get(key);
        const i2 = insightMap2.get(key);
        const changed = i1 && i2 && i1.insight_value !== i2.insight_value;
        const added = !i1 && i2;
        const removed = i1 && !i2;
        return { key, i1, i2, changed, added, removed };
    });

    return (
        <div className="p-8 max-w-7xl">
            {/* Breadcrumb */}
            <div className="flex items-center gap-2 text-sm text-[var(--text-muted)] mb-6">
                <a href="/" className="hover:text-white transition-colors">Home</a>
                <span>→</span>
                <a href={`/strategy/${slug}/${instrumentSymbol}/${testSlug}`} className="hover:text-white transition-colors">
                    T{String(test.test_number).padStart(2, "0")} {test.name}
                </a>
                <span>→</span>
                <span className="text-[var(--text-secondary)]">Compare Runs</span>
            </div>

            {/* Header */}
            <div className="mb-8">
                <h1 className="text-2xl font-bold text-white mb-2">
                    🔀 Run Comparison — T{String(test.test_number).padStart(2, "0")}
                </h1>
                <p className="text-[var(--text-secondary)]">
                    Comparing two runs of {test.name}
                </p>
            </div>

            {/* Run Metadata Comparison */}
            <div className="grid grid-cols-2 gap-4 mb-8">
                {[run1, run2].map((run, idx) => (
                    <div
                        key={run.id}
                        className={`bg-[var(--bg-card)] border rounded-xl p-5 ${idx === 0
                                ? "border-[var(--accent-blue)]/30"
                                : "border-[var(--accent-purple)]/30"
                            }`}
                    >
                        <div className="flex items-center gap-2 mb-3">
                            <span
                                className={`text-xs font-bold px-2 py-0.5 rounded ${idx === 0
                                        ? "bg-[var(--accent-blue)]/20 text-[var(--accent-blue)]"
                                        : "bg-[var(--accent-purple)]/20 text-[var(--accent-purple)]"
                                    }`}
                            >
                                Run {idx === 0 ? "A" : "B"}
                            </span>
                            <span className="text-xs text-[var(--text-muted)] font-mono">
                                #{run.id}
                            </span>
                        </div>
                        <div className="space-y-1.5 text-xs font-mono text-[var(--text-secondary)]">
                            <p>🕐 {run.run_at}</p>
                            <p>⏱ {run.duration_seconds?.toFixed(1)}s</p>
                            <p>📅 {run.data_window_start} → {run.data_window_end}</p>
                            <p>📑 {run.row_count?.toLocaleString()} rows</p>
                            <p>
                                <span
                                    className={`px-1 py-0.5 rounded text-[10px] ${run.status === "success"
                                            ? "bg-[var(--accent-green)]/20 text-[var(--accent-green)]"
                                            : "bg-[var(--accent-red)]/20 text-[var(--accent-red)]"
                                        }`}
                                >
                                    {run.status}
                                </span>
                            </p>
                        </div>
                    </div>
                ))}
            </div>

            {/* Insight Diff */}
            <section className="mb-8">
                <h2 className="text-lg font-semibold text-white mb-4">
                    💡 Insight Comparison
                </h2>
                <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl overflow-hidden">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b border-[var(--border)]">
                                <th className="px-4 py-2.5 text-left text-xs font-medium text-[var(--text-muted)] uppercase w-48">
                                    Insight
                                </th>
                                <th className="px-4 py-2.5 text-left text-xs font-medium text-[var(--accent-blue)] uppercase">
                                    Run A
                                </th>
                                <th className="px-4 py-2.5 text-left text-xs font-medium text-[var(--accent-purple)] uppercase">
                                    Run B
                                </th>
                                <th className="px-4 py-2.5 text-left text-xs font-medium text-[var(--text-muted)] uppercase w-24">
                                    Status
                                </th>
                            </tr>
                        </thead>
                        <tbody>
                            {insightDiffs.map(({ key, i1, i2, changed, added, removed }) => (
                                <tr
                                    key={key}
                                    className={`border-b border-[var(--border)]/50 ${changed
                                            ? "bg-[var(--accent-amber)]/5"
                                            : added
                                                ? "bg-[var(--accent-green)]/5"
                                                : removed
                                                    ? "bg-[var(--accent-red)]/5"
                                                    : ""
                                        }`}
                                >
                                    <td className="px-4 py-2 font-mono text-xs text-[var(--text-secondary)]">
                                        {key}
                                    </td>
                                    <td className="px-4 py-2 font-mono text-xs text-[var(--text-primary)]">
                                        {i1?.insight_value || "—"}
                                    </td>
                                    <td className="px-4 py-2 font-mono text-xs text-[var(--text-primary)]">
                                        {i2?.insight_value || "—"}
                                    </td>
                                    <td className="px-4 py-2">
                                        {changed && (
                                            <span className="text-xs text-[var(--accent-amber)]">
                                                ⚡ Changed
                                            </span>
                                        )}
                                        {added && (
                                            <span className="text-xs text-[var(--accent-green)]">
                                                ✨ New
                                            </span>
                                        )}
                                        {removed && (
                                            <span className="text-xs text-[var(--accent-red)]">
                                                🗑 Removed
                                            </span>
                                        )}
                                        {!changed && !added && !removed && (
                                            <span className="text-xs text-[var(--text-muted)]">
                                                = Same
                                            </span>
                                        )}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </section>

            {/* Chart Count Comparison */}
            <section className="mb-8">
                <h2 className="text-lg font-semibold text-white mb-4">
                    📊 Charts Comparison
                </h2>
                <div className="grid grid-cols-2 gap-4">
                    <div className="bg-[var(--bg-card)] border border-[var(--accent-blue)]/20 rounded-xl p-4 text-center">
                        <p className="text-2xl font-bold font-mono text-[var(--accent-blue)]">
                            {charts1.length}
                        </p>
                        <p className="text-xs text-[var(--text-muted)] mt-1">
                            Charts in Run A
                        </p>
                    </div>
                    <div className="bg-[var(--bg-card)] border border-[var(--accent-purple)]/20 rounded-xl p-4 text-center">
                        <p className="text-2xl font-bold font-mono text-[var(--accent-purple)]">
                            {charts2.length}
                        </p>
                        <p className="text-xs text-[var(--text-muted)] mt-1">
                            Charts in Run B
                        </p>
                    </div>
                </div>
            </section>

            {/* Performance Comparison */}
            <section>
                <h2 className="text-lg font-semibold text-white mb-4">
                    ⏱ Performance
                </h2>
                <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl overflow-hidden">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b border-[var(--border)]">
                                <th className="px-4 py-2.5 text-left text-xs font-medium text-[var(--text-muted)] uppercase">
                                    Metric
                                </th>
                                <th className="px-4 py-2.5 text-left text-xs font-medium text-[var(--accent-blue)] uppercase">
                                    Run A
                                </th>
                                <th className="px-4 py-2.5 text-left text-xs font-medium text-[var(--accent-purple)] uppercase">
                                    Run B
                                </th>
                                <th className="px-4 py-2.5 text-left text-xs font-medium text-[var(--text-muted)] uppercase">
                                    Delta
                                </th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr className="border-b border-[var(--border)]/50">
                                <td className="px-4 py-2 text-xs text-[var(--text-secondary)]">
                                    Duration
                                </td>
                                <td className="px-4 py-2 font-mono text-xs">
                                    {run1.duration_seconds?.toFixed(2)}s
                                </td>
                                <td className="px-4 py-2 font-mono text-xs">
                                    {run2.duration_seconds?.toFixed(2)}s
                                </td>
                                <td className="px-4 py-2 font-mono text-xs">
                                    {run1.duration_seconds && run2.duration_seconds && (
                                        <span
                                            className={
                                                run2.duration_seconds < run1.duration_seconds
                                                    ? "text-[var(--accent-green)]"
                                                    : "text-[var(--accent-red)]"
                                            }
                                        >
                                            {run2.duration_seconds < run1.duration_seconds
                                                ? "▼"
                                                : "▲"}{" "}
                                            {Math.abs(
                                                run2.duration_seconds - run1.duration_seconds
                                            ).toFixed(2)}
                                            s
                                        </span>
                                    )}
                                </td>
                            </tr>
                            <tr className="border-b border-[var(--border)]/50">
                                <td className="px-4 py-2 text-xs text-[var(--text-secondary)]">
                                    Data Rows
                                </td>
                                <td className="px-4 py-2 font-mono text-xs">
                                    {run1.row_count?.toLocaleString()}
                                </td>
                                <td className="px-4 py-2 font-mono text-xs">
                                    {run2.row_count?.toLocaleString()}
                                </td>
                                <td className="px-4 py-2 font-mono text-xs">
                                    {run1.row_count && run2.row_count && (
                                        <span
                                            className={
                                                run2.row_count > run1.row_count
                                                    ? "text-[var(--accent-green)]"
                                                    : run2.row_count < run1.row_count
                                                        ? "text-[var(--accent-red)]"
                                                        : "text-[var(--text-muted)]"
                                            }
                                        >
                                            {run2.row_count > run1.row_count ? "+" : ""}
                                            {(run2.row_count - run1.row_count).toLocaleString()}
                                        </span>
                                    )}
                                </td>
                            </tr>
                            <tr>
                                <td className="px-4 py-2 text-xs text-[var(--text-secondary)]">
                                    Insights
                                </td>
                                <td className="px-4 py-2 font-mono text-xs">
                                    {insights1.length}
                                </td>
                                <td className="px-4 py-2 font-mono text-xs">
                                    {insights2.length}
                                </td>
                                <td className="px-4 py-2 font-mono text-xs text-[var(--text-muted)]">
                                    {insights2.length - insights1.length >= 0 ? "+" : ""}
                                    {insights2.length - insights1.length}
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </section>
        </div>
    );
}
