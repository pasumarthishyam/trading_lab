"use client";

import { useState, useMemo } from "react";

interface SortableTableProps {
    dataJson: string;
    title?: string;
    filename?: string;
}

type SortDir = "asc" | "desc" | null;

export default function SortableTable({
    dataJson,
    title,
    filename,
}: SortableTableProps) {
    const [sortCol, setSortCol] = useState<string | null>(null);
    const [sortDir, setSortDir] = useState<SortDir>(null);
    const [filterText, setFilterText] = useState("");

    let rows: Record<string, unknown>[] = [];
    try {
        rows = JSON.parse(dataJson);
    } catch {
        return <p className="text-sm text-[var(--text-muted)] p-4">Invalid JSON</p>;
    }

    const columns = rows.length > 0 ? Object.keys(rows[0]) : [];

    const handleSort = (col: string) => {
        if (sortCol === col) {
            setSortDir(sortDir === "asc" ? "desc" : sortDir === "desc" ? null : "asc");
            if (sortDir === "desc") setSortCol(null);
        } else {
            setSortCol(col);
            setSortDir("asc");
        }
    };

    const filtered = useMemo(() => {
        if (!filterText) return rows;
        const lower = filterText.toLowerCase();
        return rows.filter((row) =>
            columns.some((col) =>
                String(row[col] ?? "")
                    .toLowerCase()
                    .includes(lower)
            )
        );
    }, [rows, columns, filterText]);

    const sorted = useMemo(() => {
        if (!sortCol || !sortDir) return filtered;
        return [...filtered].sort((a, b) => {
            const va = a[sortCol];
            const vb = b[sortCol];
            if (va == null && vb == null) return 0;
            if (va == null) return 1;
            if (vb == null) return -1;

            if (typeof va === "number" && typeof vb === "number") {
                return sortDir === "asc" ? va - vb : vb - va;
            }
            const sa = String(va).toLowerCase();
            const sb = String(vb).toLowerCase();
            return sortDir === "asc" ? sa.localeCompare(sb) : sb.localeCompare(sa);
        });
    }, [filtered, sortCol, sortDir]);

    const exportCsv = () => {
        const header = columns.join(",");
        const lines = sorted.map((row) =>
            columns.map((col) => {
                const val = String(row[col] ?? "");
                return val.includes(",") ? `"${val}"` : val;
            }).join(",")
        );
        const csv = [header, ...lines].join("\n");
        const blob = new Blob([csv], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename ? filename.replace(".csv", "") + "_export.csv" : "export.csv";
        a.click();
        URL.revokeObjectURL(url);
    };

    const sortIcon = (col: string) => {
        if (sortCol !== col) return "↕";
        return sortDir === "asc" ? "↑" : "↓";
    };

    return (
        <div>
            {/* Controls */}
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-[var(--border)]">
                <input
                    type="text"
                    placeholder="Filter rows..."
                    value={filterText}
                    onChange={(e) => setFilterText(e.target.value)}
                    className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded px-3 py-1.5 text-xs text-[var(--text-primary)] placeholder-[var(--text-muted)] w-64 focus:outline-none focus:border-[var(--accent-blue)] transition-colors"
                />
                <div className="flex items-center gap-3">
                    <span className="text-xs text-[var(--text-muted)] font-mono">
                        {sorted.length} rows
                    </span>
                    <button
                        onClick={exportCsv}
                        className="text-xs bg-[var(--bg-secondary)] border border-[var(--border)] text-[var(--text-secondary)] px-2.5 py-1.5 rounded hover:bg-[var(--bg-card-hover)] hover:text-white transition-colors flex items-center gap-1"
                    >
                        📥 Export CSV
                    </button>
                </div>
            </div>

            {/* Table */}
            <div className="overflow-x-auto">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b border-[var(--border)]">
                            {columns.map((col) => (
                                <th
                                    key={col}
                                    onClick={() => handleSort(col)}
                                    className="px-4 py-2.5 text-left text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider cursor-pointer hover:text-white transition-colors select-none"
                                >
                                    <span className="flex items-center gap-1">
                                        {col}
                                        <span className="text-[10px] opacity-50">
                                            {sortIcon(col)}
                                        </span>
                                    </span>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {sorted.map((row, idx) => (
                            <tr
                                key={idx}
                                className="border-b border-[var(--border)]/50 hover:bg-[var(--bg-card-hover)] transition-colors"
                            >
                                {columns.map((col) => (
                                    <td
                                        key={col}
                                        className="px-4 py-2 font-mono text-xs text-[var(--text-secondary)]"
                                    >
                                        {typeof row[col] === "number"
                                            ? (row[col] as number).toLocaleString(undefined, {
                                                maximumFractionDigits: 3,
                                            })
                                            : String(row[col] ?? "")}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
