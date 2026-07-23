import { getStrategy, getAllInsights } from "@/lib/db";
import { notFound } from "next/navigation";

export default async function MasterInsightsPage({
    params,
}: {
    params: Promise<{ slug: string }>;
}) {
    const { slug } = await params;
    const strategy = getStrategy(slug);
    if (!strategy) return notFound();

    const insights = getAllInsights(slug);

    // Group insights by severity
    const critical = insights.filter((i) => i.severity === "critical");
    const important = insights.filter((i) => i.severity === "important");
    const info = insights.filter((i) => i.severity === "info");

    const severityColors: Record<string, string> = {
        info: "border-[var(--accent-blue)] bg-[var(--accent-blue)]/5",
        important: "border-[var(--accent-amber)] bg-[var(--accent-amber)]/5",
        critical: "border-[var(--accent-red)] bg-[var(--accent-red)]/5",
    };

    const severityIcons: Record<string, string> = {
        info: "ℹ️",
        important: "⚠️",
        critical: "🔴",
    };

    const renderInsightGroup = (
        title: string,
        items: typeof insights,
        severity: string
    ) => {
        if (items.length === 0) return null;
        return (
            <section className="mb-8">
                <div className="flex items-center gap-3 mb-4">
                    <h2 className="text-lg font-semibold text-white">{title}</h2>
                    <span className="text-xs font-mono text-[var(--text-muted)] bg-[var(--bg-secondary)] px-2 py-0.5 rounded">
                        {items.length}
                    </span>
                </div>
                <div className="space-y-3">
                    {items.map((insight) => (
                        <div
                            key={insight.id}
                            className={`border-l-2 rounded-lg px-5 py-4 ${severityColors[severity]}`}
                        >
                            <div className="flex items-start justify-between mb-1">
                                <div className="flex items-center gap-2">
                                    <span>{severityIcons[severity]}</span>
                                    <span className="font-mono text-xs text-[var(--text-muted)]">
                                        {insight.insight_key}
                                    </span>
                                    {insight.insight_value && (
                                        <span className="font-mono text-sm font-bold text-white">
                                            {insight.insight_value}
                                        </span>
                                    )}
                                </div>
                                <a
                                    href={`/strategy/${slug}/NIFTY/${insight.test_slug}`}
                                    className="text-xs text-[var(--accent-blue)] hover:underline flex-shrink-0"
                                >
                                    {insight.test_name} →
                                </a>
                            </div>
                            <p className="text-sm text-[var(--text-primary)] opacity-90">
                                {insight.insight_text}
                            </p>
                        </div>
                    ))}
                </div>
            </section>
        );
    };

    return (
        <div className="p-8 max-w-5xl">
            {/* Breadcrumb */}
            <div className="flex items-center gap-2 text-sm text-[var(--text-muted)] mb-6">
                <a href="/" className="hover:text-white transition-colors">Home</a>
                <span>→</span>
                <a href={`/strategy/${slug}`} className="hover:text-white transition-colors">
                    {strategy.name}
                </a>
                <span>→</span>
                <span className="text-[var(--text-secondary)]">Master Insights</span>
            </div>

            {/* Header */}
            <div className="mb-8">
                <h1 className="text-2xl font-bold text-white mb-2">
                    💡 Master Insights
                </h1>
                <p className="text-[var(--text-secondary)]">
                    All key findings from the {strategy.name} test suite, organized by
                    severity. {insights.length} insights from {new Set(insights.map((i) => i.test_name)).size} tests.
                </p>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-4 mb-8">
                <div className="bg-[var(--accent-red)]/5 border border-[var(--accent-red)]/20 rounded-xl p-4 text-center">
                    <p className="text-2xl font-bold font-mono text-[var(--accent-red)]">
                        {critical.length}
                    </p>
                    <p className="text-xs text-[var(--text-muted)] mt-1">Critical</p>
                </div>
                <div className="bg-[var(--accent-amber)]/5 border border-[var(--accent-amber)]/20 rounded-xl p-4 text-center">
                    <p className="text-2xl font-bold font-mono text-[var(--accent-amber)]">
                        {important.length}
                    </p>
                    <p className="text-xs text-[var(--text-muted)] mt-1">Important</p>
                </div>
                <div className="bg-[var(--accent-blue)]/5 border border-[var(--accent-blue)]/20 rounded-xl p-4 text-center">
                    <p className="text-2xl font-bold font-mono text-[var(--accent-blue)]">
                        {info.length}
                    </p>
                    <p className="text-xs text-[var(--text-muted)] mt-1">Info</p>
                </div>
            </div>

            {/* Insights grouped by severity */}
            {renderInsightGroup("🔴 Critical Findings", critical, "critical")}
            {renderInsightGroup("⚠️ Important Findings", important, "important")}
            {renderInsightGroup("ℹ️ Information", info, "info")}

            {insights.length === 0 && (
                <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-12 text-center">
                    <p className="text-lg text-[var(--text-muted)] mb-2">
                        No insights yet
                    </p>
                    <p className="text-sm text-[var(--text-muted)]">
                        Run tests to start discovering insights
                    </p>
                </div>
            )}
        </div>
    );
}
