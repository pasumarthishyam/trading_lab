"use client";

import dynamic from "next/dynamic";
import { useState } from "react";

// Dynamic import to avoid SSR issues with plotly
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface PlotlyChartProps {
    plotlyJson: string;
    title?: string;
}

export default function PlotlyChart({ plotlyJson, title }: PlotlyChartProps) {
    const [isExpanded, setIsExpanded] = useState(false);

    let plotData: { data: object[]; layout: Record<string, unknown> };
    try {
        plotData = JSON.parse(plotlyJson);
    } catch {
        return (
            <p className="text-sm text-[var(--text-muted)] p-4">
                Could not parse Plotly JSON
            </p>
        );
    }

    const darkLayout = {
        ...plotData.layout,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "#1a1a24",
        font: { color: "#e8e8f0", family: "Inter, sans-serif", size: 12 },
        xaxis: {
            ...(plotData.layout?.xaxis as object || {}),
            gridcolor: "#2a2a3a",
            zerolinecolor: "#2a2a3a",
        },
        yaxis: {
            ...(plotData.layout?.yaxis as object || {}),
            gridcolor: "#2a2a3a",
            zerolinecolor: "#2a2a3a",
        },
        margin: { l: 60, r: 30, t: 50, b: 50 },
        autosize: true,
    };

    return (
        <div className={isExpanded ? "fixed inset-0 z-50 bg-black/90 p-8 flex flex-col" : ""}>
            {isExpanded && (
                <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-medium text-white">{title}</h3>
                    <button
                        onClick={() => setIsExpanded(false)}
                        className="text-white text-sm bg-[var(--bg-card)] border border-[var(--border)] px-3 py-1.5 rounded-lg hover:bg-[var(--bg-card-hover)] transition-colors"
                    >
                        ✕ Close
                    </button>
                </div>
            )}
            <div className={isExpanded ? "flex-1" : "relative"}>
                {!isExpanded && (
                    <button
                        onClick={() => setIsExpanded(true)}
                        className="absolute top-2 right-2 z-10 text-xs bg-[var(--bg-card)] border border-[var(--border)] text-[var(--text-secondary)] px-2.5 py-1 rounded hover:bg-[var(--bg-card-hover)] hover:text-white transition-colors"
                    >
                        🔍 Expand
                    </button>
                )}
                <Plot
                    data={plotData.data as Plotly.Data[]}
                    layout={darkLayout as Partial<Plotly.Layout>}
                    config={{
                        responsive: true,
                        displayModeBar: true,
                        displaylogo: false,
                        modeBarButtonsToRemove: ["lasso2d", "select2d", "sendDataToCloud"],
                    }}
                    style={{ width: "100%", height: isExpanded ? "100%" : "450px" }}
                    useResizeHandler
                />
            </div>
        </div>
    );
}
