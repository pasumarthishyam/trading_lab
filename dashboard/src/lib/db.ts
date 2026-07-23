import Database from "better-sqlite3";
import path from "path";

// Path to the SQLite database (relative to the dashboard directory)
const DB_PATH = path.join(process.cwd(), "..", "data", "trading_lab.db");

let db: Database.Database | null = null;

export function getDb(): Database.Database {
    if (!db) {
        db = new Database(DB_PATH, { readonly: true });
        db.pragma("journal_mode = WAL");
    }
    return db;
}

// ── Type definitions ────────────────────────────────────────────────

export interface Strategy {
    id: number;
    name: string;
    slug: string;
    description: string;
    created_at: string;
}

export interface Instrument {
    id: number;
    name: string;
    symbol: string;
    asset_type: string;
}

export interface TestCategory {
    id: number;
    strategy_id: number;
    name: string;
    slug: string;
    display_order: number;
}

export interface TestDef {
    id: number;
    category_id: number;
    test_number: number;
    name: string;
    slug: string;
    description: string;
    category_name?: string;
    category_slug?: string;
}

export interface TestRun {
    id: number;
    test_id: number;
    instrument_id: number;
    run_at: string;
    duration_seconds: number | null;
    status: string;
    data_window_start: string | null;
    data_window_end: string | null;
    config_snapshot: string | null;
    row_count: number | null;
    terminal_output: string | null;
    notes: string | null;
}

export interface Insight {
    id: number;
    test_run_id: number;
    insight_key: string;
    insight_value: string | null;
    insight_text: string;
    severity: string;
    created_at: string;
}

export interface Chart {
    id: number;
    test_run_id: number;
    filename: string;
    title: string | null;
    chart_type: string | null;
    description: string | null;
    plotly_json: string | null;
}

export interface DataTable {
    id: number;
    test_run_id: number;
    filename: string;
    title: string | null;
    description: string | null;
    row_count: number | null;
    column_count: number | null;
    data_json: string | null;
}

// ── Query helpers ───────────────────────────────────────────────────

export function getStrategies(): Strategy[] {
    return getDb().prepare("SELECT * FROM strategies ORDER BY id").all() as Strategy[];
}

export function getStrategy(slug: string): Strategy | undefined {
    return getDb()
        .prepare("SELECT * FROM strategies WHERE slug = ?")
        .get(slug) as Strategy | undefined;
}

export function getInstruments(): Instrument[] {
    return getDb().prepare("SELECT * FROM instruments ORDER BY id").all() as Instrument[];
}

export function getCategories(strategyId: number): TestCategory[] {
    return getDb()
        .prepare(
            "SELECT * FROM test_categories WHERE strategy_id = ? ORDER BY display_order"
        )
        .all(strategyId) as TestCategory[];
}

export function getTests(strategyId: number): TestDef[] {
    return getDb()
        .prepare(
            `SELECT t.*, tc.name as category_name, tc.slug as category_slug
       FROM tests t
       JOIN test_categories tc ON t.category_id = tc.id
       WHERE tc.strategy_id = ?
       ORDER BY t.test_number`
        )
        .all(strategyId) as TestDef[];
}

export function getTest(slug: string): TestDef | undefined {
    return getDb()
        .prepare(
            `SELECT t.*, tc.name as category_name, tc.slug as category_slug
       FROM tests t
       JOIN test_categories tc ON t.category_id = tc.id
       WHERE t.slug = ?`
        )
        .get(slug) as TestDef | undefined;
}

export function getLatestRun(
    testId: number,
    instrumentId: number
): TestRun | undefined {
    return getDb()
        .prepare(
            `SELECT * FROM test_runs
       WHERE test_id = ? AND instrument_id = ? AND status = 'success'
       ORDER BY run_at DESC LIMIT 1`
        )
        .get(testId, instrumentId) as TestRun | undefined;
}

export function getRunHistory(
    testId: number,
    instrumentId: number
): TestRun[] {
    return getDb()
        .prepare(
            `SELECT * FROM test_runs
       WHERE test_id = ? AND instrument_id = ?
       ORDER BY run_at DESC`
        )
        .all(testId, instrumentId) as TestRun[];
}

export function getInsights(runId: number): Insight[] {
    return getDb()
        .prepare("SELECT * FROM insights WHERE test_run_id = ? ORDER BY id")
        .all(runId) as Insight[];
}

export function getCharts(runId: number): Chart[] {
    return getDb()
        .prepare("SELECT * FROM charts WHERE test_run_id = ? ORDER BY id")
        .all(runId) as Chart[];
}

export function getDataTables(runId: number): DataTable[] {
    return getDb()
        .prepare("SELECT * FROM data_tables WHERE test_run_id = ? ORDER BY id")
        .all(runId) as DataTable[];
}

export function getAllInsights(strategySlug: string): (Insight & { test_name: string; test_slug: string })[] {
    return getDb()
        .prepare(
            `SELECT i.*, t.name as test_name, t.slug as test_slug
       FROM insights i
       JOIN test_runs tr ON i.test_run_id = tr.id
       JOIN tests t ON tr.test_id = t.id
       JOIN test_categories tc ON t.category_id = tc.id
       JOIN strategies s ON tc.strategy_id = s.id
       WHERE s.slug = ? AND tr.status = 'success'
       ORDER BY
         CASE i.severity
           WHEN 'critical' THEN 1
           WHEN 'important' THEN 2
           ELSE 3
         END,
         i.id`
        )
        .all(strategySlug) as (Insight & { test_name: string; test_slug: string })[];
}

export function getTestCompletion(strategyId: number, instrumentId: number) {
    const total = getDb()
        .prepare(
            `SELECT COUNT(*) as count FROM tests t
       JOIN test_categories tc ON t.category_id = tc.id
       WHERE tc.strategy_id = ?`
        )
        .get(strategyId) as { count: number };

    const completed = getDb()
        .prepare(
            `SELECT COUNT(DISTINCT t.id) as count FROM tests t
       JOIN test_categories tc ON t.category_id = tc.id
       JOIN test_runs tr ON tr.test_id = t.id
       WHERE tc.strategy_id = ? AND tr.instrument_id = ? AND tr.status = 'success'`
        )
        .get(strategyId, instrumentId) as { count: number };

    return { total: total.count, completed: completed.count };
}
