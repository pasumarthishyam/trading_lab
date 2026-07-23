import {
    getStrategy,
    getTest,
    getInstruments,
    getLatestRun,
    getRunHistory,
    getInsights,
    getCharts,
    getDataTables,
} from "@/lib/db";
import { notFound } from "next/navigation";
import PlotlyChart from "@/components/PlotlyChart";
import SortableTable from "@/components/SortableTable";

export default async function TestDetailPage({
    params,
}: {
    params: Promise<{ slug: string; instrument: string; testSlug: string }>;
}) {
    const { slug, instrument: instrumentSymbol, testSlug } = await params;
    const strategy = getStrategy(slug);
    const test = getTest(testSlug);
    const instruments = getInstruments();
    const instrument = instruments.find((i) => i.symbol === instrumentSymbol);

    if (!strategy || !test || !instrument) return notFound();

    const latestRun = getLatestRun(test.id, instrument.id);
    const runHistory = getRunHistory(test.id, instrument.id);
    const insights = latestRun ? getInsights(latestRun.id) : [];
    const charts = latestRun ? getCharts(latestRun.id) : [];
    const dataTables = latestRun ? getDataTables(latestRun.id) : [];

    const severityColors: Record<string, string> = {
        info: "border-[var(--accent-blue)] bg-[var(--accent-blue)]/5 text-[var(--accent-blue)]",
        important:
            "border-[var(--accent-amber)] bg-[var(--accent-amber)]/5 text-[var(--accent-amber)]",
        critical:
            "border-[var(--accent-red)] bg-[var(--accent-red)]/5 text-[var(--accent-red)]",
    };

    const severityIcons: Record<string, string> = {
        info: "ℹ️",
        important: "⚠️",
        critical: "🔴",
    };

    // Check if we can enable run comparison
    const successfulRuns = runHistory.filter((r) => r.status === "success");
    const canCompare = successfulRuns.length >= 2;

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
                <a
                    href={`/strategy/${slug}/${instrumentSymbol}`}
                    className="hover:text-white transition-colors"
                >
                    {instrumentSymbol}
                </a>
                <span>→</span>
                <span className="text-[var(--text-secondary)]">
                    T{String(test.test_number).padStart(2, "0")} {test.name}
                </span>
            </div>

            {/* Header */}
            <div className="mb-8">
                <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3 mb-2">
                        <span className="text-2xl">{latestRun ? "✅" : "⏳"}</span>
                        <div>
                            <h1 className="text-2xl font-bold text-white">
                                T{String(test.test_number).padStart(2, "0")} — {test.name}
                            </h1>
                            <p className="text-[var(--text-secondary)] mt-1">
                                {test.description}
                            </p>
                            <p className="text-xs text-[var(--text-muted)] mt-1">
                                Category: {test.category_name}
                            </p>
                        </div>
                    </div>
                    {canCompare && (
                        <a
                            href={`/strategy/${slug}/${instrumentSymbol}/${testSlug}/compare?run1=${successfulRuns[0].id}&run2=${successfulRuns[1].id}`}
                            className="text-xs bg-[var(--accent-purple)]/10 border border-[var(--accent-purple)]/30 text-[var(--accent-purple)] px-3 py-1.5 rounded-lg hover:bg-[var(--accent-purple)]/20 transition-colors flex-shrink-0"
                        >
                            🔀 Compare Runs
                        </a>
                    )}
                </div>

                {latestRun && (
                    <div className="flex items-center gap-4 mt-4 text-xs font-mono text-[var(--text-muted)] bg-[var(--bg-card)] border border-[var(--border)] rounded-lg px-4 py-2.5 flex-wrap">
                        <span>🕐 {latestRun.run_at}</span>
                        <span>⏱ {latestRun.duration_seconds?.toFixed(1)}s</span>
                        <span>📊 {charts.length} charts</span>
                        <span>📋 {dataTables.length} tables</span>
                        <span>💡 {insights.length} insights</span>
                        {latestRun.data_window_start && (
                            <span>
                                📅 {latestRun.data_window_start} → {latestRun.data_window_end}
                            </span>
                        )}
                        {latestRun.row_count && (
                            <span>📑 {latestRun.row_count.toLocaleString()} rows</span>
                        )}
                    </div>
                )}
            </div>

            {!latestRun ? (
                <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-12 text-center">
                    <p className="text-lg text-[var(--text-muted)] mb-2">
                        ⏳ Test not yet run
                    </p>
                    <p className="text-sm text-[var(--text-muted)]">
                        Run this test from the terminal to see results here
                    </p>
                    <code className="block mt-4 text-sm font-mono text-[var(--accent-blue)] bg-[var(--bg-secondary)] px-4 py-2 rounded inline-block">
                        python -m strategies.VCF.tests.{test.category_slug}.{test.slug}
                    </code>
                </div>
            ) : (
                <div className="space-y-8">
                    {/* Section 1: Insights */}
                    {insights.length > 0 && (
                        <section>
                            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                💡 Key Insights
                                <span className="text-xs font-mono text-[var(--text-muted)]">
                                    ({insights.length})
                                </span>
                            </h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                {insights.map((insight) => (
                                    <div
                                        key={insight.id}
                                        className={`border-l-2 rounded-lg px-4 py-3 ${severityColors[insight.severity] || severityColors.info}`}
                                    >
                                        <div className="flex items-center gap-2 mb-1">
                                            <span>{severityIcons[insight.severity] || "ℹ️"}</span>
                                            <span className="font-mono text-xs opacity-80">
                                                {insight.insight_key}
                                            </span>
                                            {insight.insight_value && (
                                                <span className="font-mono text-sm font-bold">
                                                    {insight.insight_value}
                                                </span>
                                            )}
                                        </div>
                                        <p className="text-sm text-[var(--text-primary)] opacity-90">
                                            {insight.insight_text}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        </section>
                    )}

                    {/* Section 2: Charts (Static + Interactive Plotly) */}
                    {charts.length > 0 && (
                        <section>
                            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                📊 Charts
                                <span className="text-xs font-mono text-[var(--text-muted)]">
                                    ({charts.length})
                                </span>
                            </h2>
                            <div className="space-y-5">
                                {charts.map((chart) => (
                                    <div
                                        key={chart.id}
                                        className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl overflow-hidden"
                                    >
                                        <div className="px-5 py-3 border-b border-[var(--border)] flex items-center justify-between">
                                            <div>
                                                <h3 className="text-sm font-medium text-white">
                                                    {chart.title}
                                                </h3>
                                                {chart.description && (
                                                    <p className="text-xs text-[var(--text-muted)] mt-0.5">
                                                        {chart.description}
                                                    </p>
                                                )}
                                            </div>
                                            <div className="flex items-center gap-2">
                                                {chart.chart_type && (
                                                    <span className="text-xs font-mono text-[var(--text-muted)] bg-[var(--bg-secondary)] px-2 py-0.5 rounded">
                                                        {chart.chart_type}
                                                    </span>
                                                )}
                                                {chart.plotly_json && (
                                                    <span className="text-xs font-mono text-[var(--accent-purple)] bg-[var(--accent-purple)]/10 px-2 py-0.5 rounded">
                                                        interactive
                                                    </span>
                                                )}
                                            </div>
                                        </div>

                                        {/* Interactive Plotly chart if JSON is available */}
                                        {chart.plotly_json ? (
                                            <div className="p-4 bg-[#1a1a24]">
                                                <PlotlyChart
                                                    plotlyJson={chart.plotly_json}
                                                    title={chart.title || undefined}
                                                />
                                            </div>
                                        ) : (
                                            /* Static PNG fallback */
                                            <div className="p-4 flex justify-center bg-[#1a1a24]">
                                                {/* eslint-disable-next-line @next/next/no-img-element */}
                                                <img
                                                    src={`/api/chart-image?path=${encodeURIComponent(chart.filename)}`}
                                                    alt={chart.title || "Chart"}
                                                    className="max-w-full h-auto rounded"
                                                    style={{ maxHeight: "500px" }}
                                                />
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </section>
                    )}

                    {/* Section 3: Data Tables (Sortable + Exportable) */}
                    {dataTables.length > 0 && (
                        <section>
                            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                📋 Tables
                                <span className="text-xs font-mono text-[var(--text-muted)]">
                                    ({dataTables.length})
                                </span>
                            </h2>
                            <div className="space-y-5">
                                {dataTables.map((table) => (
                                    <div
                                        key={table.id}
                                        className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl overflow-hidden"
                                    >
                                        <div className="px-5 py-3 border-b border-[var(--border)] flex items-center justify-between">
                                            <div>
                                                <h3 className="text-sm font-medium text-white">
                                                    {table.title}
                                                </h3>
                                                {table.description && (
                                                    <p className="text-xs text-[var(--text-muted)] mt-0.5">
                                                        {table.description}
                                                    </p>
                                                )}
                                            </div>
                                            <span className="text-xs font-mono text-[var(--text-muted)]">
                                                {table.row_count} × {table.column_count}
                                            </span>
                                        </div>
                                        {table.data_json && (
                                            <SortableTable
                                                dataJson={table.data_json}
                                                title={table.title || undefined}
                                                filename={table.filename}
                                            />
                                        )}
                                    </div>
                                ))}
                            </div>
                        </section>
                    )}

                    {/* Section 4: Terminal Output (Result) */}
                    {latestRun.terminal_output && (
                        <section>
                            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                🖥️ Terminal Output
                            </h2>
                            <div className="bg-[#0d0d0d] border border-[var(--border)] rounded-xl overflow-hidden">
                                <div className="px-5 py-2.5 border-b border-[var(--border)] flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full bg-[#ff5f57]" />
                                    <div className="w-3 h-3 rounded-full bg-[#febc2e]" />
                                    <div className="w-3 h-3 rounded-full bg-[#28c840]" />
                                    <span className="ml-3 text-xs text-[var(--text-muted)] font-mono">
                                        python -m strategies.VCF.tests.{test.category_slug}.{test.slug}
                                    </span>
                                </div>
                                <pre className="p-5 text-xs font-mono text-green-400/90 whitespace-pre-wrap overflow-x-auto leading-relaxed max-h-[500px] overflow-y-auto">
                                    {latestRun.terminal_output}
                                </pre>
                            </div>
                        </section>
                    )}

                    {/* Section 5: Run History */}
                    {runHistory.length > 0 && (
                        <section>
                            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                📜 Run History
                                <span className="text-xs font-mono text-[var(--text-muted)]">
                                    ({runHistory.length} runs)
                                </span>
                            </h2>
                            <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl overflow-hidden">
                                <table className="w-full text-sm">
                                    <thead>
                                        <tr className="border-b border-[var(--border)]">
                                            <th className="px-4 py-2.5 text-left text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                                                Run At
                                            </th>
                                            <th className="px-4 py-2.5 text-left text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                                                Status
                                            </th>
                                            <th className="px-4 py-2.5 text-left text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                                                Duration
                                            </th>
                                            <th className="px-4 py-2.5 text-left text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                                                Data Window
                                            </th>
                                            <th className="px-4 py-2.5 text-left text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                                                Rows
                                            </th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {runHistory.map((run) => (
                                            <tr
                                                key={run.id}
                                                className={`border-b border-[var(--border)]/50 ${run.id === latestRun?.id
                                                        ? "bg-[var(--accent-blue)]/5"
                                                        : "hover:bg-[var(--bg-card-hover)]"
                                                    } transition-colors`}
                                            >
                                                <td className="px-4 py-2 font-mono text-xs text-[var(--text-secondary)]">
                                                    {run.run_at}
                                                    {run.id === latestRun?.id && (
                                                        <span className="ml-2 text-[var(--accent-blue)]">
                                                            (viewing)
                                                        </span>
                                                    )}
                                                </td>
                                                <td className="px-4 py-2">
                                                    <span
                                                        className={`text-xs font-medium px-2 py-0.5 rounded ${run.status === "success"
                                                                ? "bg-[var(--accent-green)]/20 text-[var(--accent-green)]"
                                                                : run.status === "failed"
                                                                    ? "bg-[var(--accent-red)]/20 text-[var(--accent-red)]"
                                                                    : "bg-[var(--accent-amber)]/20 text-[var(--accent-amber)]"
                                                            }`}
                                                    >
                                                        {run.status}
                                                    </span>
                                                </td>
                                                <td className="px-4 py-2 font-mono text-xs text-[var(--text-secondary)]">
                                                    {run.duration_seconds?.toFixed(1)}s
                                                </td>
                                                <td className="px-4 py-2 font-mono text-xs text-[var(--text-secondary)]">
                                                    {run.data_window_start} → {run.data_window_end}
                                                </td>
                                                <td className="px-4 py-2 font-mono text-xs text-[var(--text-secondary)]">
                                                    {run.row_count?.toLocaleString()}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </section>
                    )}
                </div>
            )}
        </div>
    );
}
