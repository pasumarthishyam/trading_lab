import { getDb } from "@/lib/db";
import fs from "fs";
import path from "path";

interface DataFileStatus {
    type: string;
    symbol: string;
    timeframe: string;
    filePath: string;
    exists: boolean;
    sizeBytes: number | null;
    modifiedAt: string | null;
    stale: boolean;
}

function scanDataFiles(): DataFileStatus[] {
    const dataDir = path.join(process.cwd(), "..", "data");
    const rawDir = path.join(dataDir, "raw");
    const processedDir = path.join(dataDir, "processed");

    const files: DataFileStatus[] = [];

    // Scan raw data — actual dirs: indices, volatility, options, stocks
    const assetTypes = ["indices", "volatility", "options", "stocks"];

    for (const assetType of assetTypes) {
        const assetDir = path.join(rawDir, assetType);
        if (!fs.existsSync(assetDir)) continue;

        try {
            const entries = fs.readdirSync(assetDir);
            for (const entry of entries) {
                if (entry.startsWith(".")) continue;
                const entryPath = path.join(assetDir, entry);
                const stat = fs.statSync(entryPath);

                if (stat.isDirectory()) {
                    // Symbol directory — scan for parquet files directly inside
                    try {
                        const parquetFiles = fs
                            .readdirSync(entryPath)
                            .filter((f: string) => f.endsWith(".parquet"));

                        for (const pf of parquetFiles) {
                            const pfPath = path.join(entryPath, pf);
                            const pfStat = fs.statSync(pfPath);
                            const timeframe = path.basename(pf, ".parquet");
                            const daysSince = (Date.now() - pfStat.mtime.getTime()) / (1000 * 60 * 60 * 24);

                            files.push({
                                type: assetType,
                                symbol: entry,
                                timeframe,
                                filePath: path.relative(dataDir, pfPath),
                                exists: true,
                                sizeBytes: pfStat.size,
                                modifiedAt: pfStat.mtime.toISOString().split("T")[0],
                                stale: daysSince > 7,
                            });
                        }
                    } catch { /* skip */ }
                } else if (entry.endsWith(".parquet")) {
                    // Direct parquet file at asset level
                    files.push({
                        type: assetType,
                        symbol: path.basename(entry, ".parquet"),
                        timeframe: "daily",
                        filePath: path.relative(dataDir, entryPath),
                        exists: true,
                        sizeBytes: stat.size,
                        modifiedAt: stat.mtime.toISOString().split("T")[0],
                        stale: (Date.now() - stat.mtime.getTime()) / (1000 * 60 * 60 * 24) > 7,
                    });
                }
            }
        } catch { /* directory doesn't exist or not readable */ }
    }

    // Scan for master files in processed/
    if (fs.existsSync(processedDir)) {
        try {
            const processed = fs.readdirSync(processedDir).filter((f: string) => f.endsWith(".parquet"));
            for (const f of processed) {
                const fPath = path.join(processedDir, f);
                const stat = fs.statSync(fPath);
                files.push({
                    type: "processed",
                    symbol: path.basename(f, ".parquet"),
                    timeframe: "master",
                    filePath: path.relative(dataDir, fPath),
                    exists: true,
                    sizeBytes: stat.size,
                    modifiedAt: stat.mtime.toISOString().split("T")[0],
                    stale: false,
                });
            }
        } catch { /* skip */ }
    }

    // Scan for VCF master
    const vcfMaster = path.join(
        process.cwd(),
        "..",
        "strategies",
        "VCF",
        "vcf_master.parquet"
    );
    if (fs.existsSync(vcfMaster)) {
        const stat = fs.statSync(vcfMaster);
        files.push({
            type: "feature",
            symbol: "VCF",
            timeframe: "master",
            filePath: "strategies/VCF/vcf_master.parquet",
            exists: true,
            sizeBytes: stat.size,
            modifiedAt: stat.mtime.toISOString().split("T")[0],
            stale: (Date.now() - stat.mtime.getTime()) / (1000 * 60 * 60 * 24) > 7,
        });
    }

    return files;
}

function formatBytes(bytes: number | null): string {
    if (bytes == null || bytes === 0) return "—";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DataStatusPage() {
    const files = scanDataFiles();
    const db = getDb();

    // DB stats
    const dbPath = path.join(process.cwd(), "..", "data", "trading_lab.db");
    let dbSize = 0;
    try {
        dbSize = fs.statSync(dbPath).size;
    } catch { /* skip */ }

    const runCount = (
        db.prepare("SELECT COUNT(*) as c FROM test_runs WHERE status='success'").get() as {
            c: number;
        }
    ).c;
    const insightCount = (
        db.prepare("SELECT COUNT(*) as c FROM insights").get() as { c: number }
    ).c;
    const chartCount = (
        db.prepare("SELECT COUNT(*) as c FROM charts").get() as { c: number }
    ).c;

    // Group files by type
    const grouped: Record<string, DataFileStatus[]> = {};
    for (const f of files) {
        if (!grouped[f.type]) grouped[f.type] = [];
        grouped[f.type].push(f);
    }

    const typeLabels: Record<string, string> = {
        indices: "📈 Index Data",
        volatility: "📊 Volatility Data",
        options: "📋 Options Data",
        stocks: "📉 Stock Data",
        processed: "⚙️ Processed Data",
        feature: "🧩 Feature Files",
    };

    return (
        <div className="p-8 max-w-6xl">
            {/* Breadcrumb */}
            <div className="flex items-center gap-2 text-sm text-[var(--text-muted)] mb-6">
                <a href="/" className="hover:text-white transition-colors">Home</a>
                <span>→</span>
                <span className="text-[var(--text-secondary)]">Data Status</span>
            </div>

            <div className="mb-8">
                <h1 className="text-2xl font-bold text-white mb-2">📁 Data Status</h1>
                <p className="text-[var(--text-secondary)]">
                    Freshness and health of all data sources, files, and the research
                    database
                </p>
            </div>

            {/* Database Stats */}
            <div className="grid grid-cols-4 gap-4 mb-8">
                <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
                    <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">
                        Database Size
                    </p>
                    <p className="text-xl font-bold font-mono text-white">
                        {formatBytes(dbSize)}
                    </p>
                </div>
                <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
                    <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">
                        Successful Runs
                    </p>
                    <p className="text-xl font-bold font-mono text-[var(--accent-green)]">
                        {runCount}
                    </p>
                </div>
                <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
                    <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">
                        Insights
                    </p>
                    <p className="text-xl font-bold font-mono text-[var(--accent-amber)]">
                        {insightCount}
                    </p>
                </div>
                <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
                    <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">
                        Charts Stored
                    </p>
                    <p className="text-xl font-bold font-mono text-[var(--accent-blue)]">
                        {chartCount}
                    </p>
                </div>
            </div>

            {/* Data Files by Category */}
            <div className="space-y-6">
                {Object.entries(grouped).map(([type, typeFiles]) => (
                    <section key={type}>
                        <h2 className="text-sm font-medium text-[var(--text-muted)] uppercase tracking-wider mb-3">
                            {typeLabels[type] || type}
                        </h2>
                        <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl overflow-hidden">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="border-b border-[var(--border)]">
                                        <th className="px-4 py-2.5 text-left text-xs font-medium text-[var(--text-muted)] uppercase">
                                            Symbol
                                        </th>
                                        <th className="px-4 py-2.5 text-left text-xs font-medium text-[var(--text-muted)] uppercase">
                                            Timeframe
                                        </th>
                                        <th className="px-4 py-2.5 text-left text-xs font-medium text-[var(--text-muted)] uppercase">
                                            Status
                                        </th>
                                        <th className="px-4 py-2.5 text-left text-xs font-medium text-[var(--text-muted)] uppercase">
                                            Size
                                        </th>
                                        <th className="px-4 py-2.5 text-left text-xs font-medium text-[var(--text-muted)] uppercase">
                                            Last Modified
                                        </th>
                                        <th className="px-4 py-2.5 text-left text-xs font-medium text-[var(--text-muted)] uppercase">
                                            Freshness
                                        </th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {typeFiles.map((f, idx) => (
                                        <tr
                                            key={idx}
                                            className="border-b border-[var(--border)]/50 hover:bg-[var(--bg-card-hover)] transition-colors"
                                        >
                                            <td className="px-4 py-2 font-mono text-xs text-white font-medium">
                                                {f.symbol}
                                            </td>
                                            <td className="px-4 py-2 font-mono text-xs text-[var(--text-secondary)]">
                                                {f.timeframe}
                                            </td>
                                            <td className="px-4 py-2">
                                                <span
                                                    className={`text-xs px-1.5 py-0.5 rounded ${f.exists
                                                        ? "bg-[var(--accent-green)]/20 text-[var(--accent-green)]"
                                                        : "bg-[var(--accent-red)]/20 text-[var(--accent-red)]"
                                                        }`}
                                                >
                                                    {f.exists ? "✓ present" : "✕ missing"}
                                                </span>
                                            </td>
                                            <td className="px-4 py-2 font-mono text-xs text-[var(--text-secondary)]">
                                                {formatBytes(f.sizeBytes)}
                                            </td>
                                            <td className="px-4 py-2 font-mono text-xs text-[var(--text-secondary)]">
                                                {f.modifiedAt || "—"}
                                            </td>
                                            <td className="px-4 py-2">
                                                {f.exists && (
                                                    <span
                                                        className={`text-xs px-1.5 py-0.5 rounded ${f.stale
                                                            ? "bg-[var(--accent-amber)]/20 text-[var(--accent-amber)]"
                                                            : "bg-[var(--accent-green)]/20 text-[var(--accent-green)]"
                                                            }`}
                                                    >
                                                        {f.stale ? "⚠ stale" : "✓ fresh"}
                                                    </span>
                                                )}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </section>
                ))}
            </div>
        </div>
    );
}
