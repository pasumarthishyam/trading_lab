import {
    getStrategy,
    getCategories,
    getTests,
    getInstruments,
    getTestCompletion,
    getLatestRun,
    getInsights,
} from "@/lib/db";
import { notFound } from "next/navigation";

export default async function StrategyPage({
    params,
}: {
    params: Promise<{ slug: string }>;
}) {
    const { slug } = await params;
    const strategy = getStrategy(slug);
    if (!strategy) return notFound();

    const categories = getCategories(strategy.id);
    const tests = getTests(strategy.id);
    const instruments = getInstruments();
    const instrument = instruments[0];

    const completion = instrument
        ? getTestCompletion(strategy.id, instrument.id)
        : { total: 0, completed: 0 };

    // Group tests by category
    const testsByCategory: Record<string, typeof tests> = {};
    for (const t of tests) {
        const cat = t.category_slug || "unknown";
        if (!testsByCategory[cat]) testsByCategory[cat] = [];
        testsByCategory[cat].push(t);
    }

    return (
        <div className="p-8 max-w-7xl">
            {/* Breadcrumb */}
            <div className="flex items-center gap-2 text-sm text-[var(--text-muted)] mb-6">
                <a href="/" className="hover:text-white transition-colors">
                    Home
                </a>
                <span>→</span>
                <span className="text-[var(--text-secondary)]">{strategy.name}</span>
            </div>

            {/* Header */}
            <div className="mb-8">
                <div className="flex items-start justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-white mb-2">
                            {strategy.name}
                        </h1>
                        <p className="text-[var(--text-secondary)] max-w-2xl">
                            {strategy.description}
                        </p>
                    </div>
                    <div className="flex gap-3">
                        {instrument && (
                            <a
                                href={`/strategy/${slug}/${instrument.symbol}`}
                                className="bg-[var(--accent-blue)] text-black px-4 py-2 rounded-lg text-sm font-medium hover:opacity-90 transition-opacity"
                            >
                                View {instrument.symbol} Tests →
                            </a>
                        )}
                        <a
                            href={`/strategy/${slug}/insights`}
                            className="bg-[var(--bg-card)] border border-[var(--border)] text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-[var(--bg-card-hover)] transition-colors"
                        >
                            💡 Master Insights
                        </a>
                    </div>
                </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-4 gap-4 mb-8">
                <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
                    <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">
                        Total Tests
                    </p>
                    <p className="text-2xl font-bold font-mono text-white">
                        {completion.total}
                    </p>
                </div>
                <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
                    <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">
                        Completed
                    </p>
                    <p className="text-2xl font-bold font-mono text-[var(--accent-green)]">
                        {completion.completed}
                    </p>
                </div>
                <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
                    <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">
                        Remaining
                    </p>
                    <p className="text-2xl font-bold font-mono text-[var(--accent-amber)]">
                        {completion.total - completion.completed}
                    </p>
                </div>
                <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
                    <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">
                        Categories
                    </p>
                    <p className="text-2xl font-bold font-mono text-white">
                        {categories.length}
                    </p>
                </div>
            </div>

            {/* Test Grid by Category */}
            <h2 className="text-lg font-semibold text-white mb-4">Test Suite</h2>
            <div className="space-y-6">
                {categories.map((cat) => {
                    const catTests = testsByCategory[cat.slug] || [];
                    return (
                        <div key={cat.id}>
                            <h3 className="text-sm font-medium text-[var(--text-muted)] uppercase tracking-wider mb-3">
                                {cat.name}
                            </h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                                {catTests.map((t) => {
                                    const run = instrument
                                        ? getLatestRun(t.id, instrument.id)
                                        : undefined;
                                    const insights = run ? getInsights(run.id) : [];
                                    const topInsight = insights.find(
                                        (i) => i.severity === "critical" || i.severity === "important"
                                    );
                                    const hasRun = !!run;

                                    return (
                                        <a
                                            key={t.id}
                                            href={
                                                instrument
                                                    ? `/strategy/${slug}/${instrument.symbol}/${t.slug}`
                                                    : "#"
                                            }
                                            className="card-hover bg-[var(--bg-card)] border border-[var(--border)] rounded-lg p-4"
                                        >
                                            <div className="flex items-center gap-2 mb-2">
                                                <span className="text-base">
                                                    {hasRun ? "✅" : "⏳"}
                                                </span>
                                                <span className="font-mono text-xs text-[var(--text-muted)]">
                                                    T{String(t.test_number).padStart(2, "0")}
                                                </span>
                                                <span className="text-sm font-medium text-white">
                                                    {t.name}
                                                </span>
                                            </div>
                                            <p className="text-xs text-[var(--text-muted)] line-clamp-1 mb-2">
                                                {t.description}
                                            </p>
                                            {topInsight && (
                                                <div className="bg-[var(--bg-secondary)] rounded px-2 py-1 text-xs text-[var(--accent-amber)] line-clamp-1">
                                                    💡 {topInsight.insight_text}
                                                </div>
                                            )}
                                            {run && (
                                                <p className="text-[10px] text-[var(--text-muted)] mt-2 font-mono">
                                                    Last run: {run.run_at.split("T")[0]}
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
