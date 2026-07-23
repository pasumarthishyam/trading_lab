# Trading Lab

A research platform for systematic trading strategies on Indian equities — data ingestion, feature engineering, ML-based signal models, and a web dashboard for reviewing backtests, all in one repository.

## Data

- Six years of OHLCV history for 215 symbols across five timeframes (1-minute through daily), pulled through Zerodha's Kite Connect API.
- 25M+ rows stored as 1,160 partitioned Parquet files, queried with DuckDB — full-history scans across every symbol and timeframe without running a database server.
- A corporate-action layer adjusts for splits, bonuses, and exchange holidays, and repairs missing sessions, so price series stay continuous and correctly adjusted.

## Modeling

- XGBoost and LightGBM classifiers trained on 40+ engineered features to flag significant price moves.
- Purged walk-forward cross-validation with embargo gaps to prevent lookahead leakage; hyperparameters tuned with Optuna, feature contribution ranked with SHAP.
- Isolation Forest for anomaly detection on abnormal sessions; K-Means clustering groups ~1,250 trading days into market regimes, used as categorical features downstream.

## Dashboard

A Next.js + TypeScript app (`dashboard/`) for browsing backtest results — per-strategy, per-instrument breakdowns rendered with Plotly, reading off the same data layer as the research code.

## Structure

```
trading_lab/
├── infrastructure/    Shared data, feature, and ML pipeline code used across every strategy
│   ├── data/           Ingestion, corporate actions, validation
│   ├── features/       Feature engineering
│   ├── ml/              Classifiers, anomaly detection, clustering
│   └── rs/               Relative strength / sector computation
├── strategies/         Independent strategy modules, each self-contained
├── data/                Parquet-backed market data store (gitignored)
├── dashboard/           Next.js research UI for backtest results
└── scripts/             Ingestion and maintenance scripts
```

## Setup

```bash
git clone https://github.com/pasumarthishyam/trading_lab
cd trading_lab
python -m venv .venv
.venv\Scripts\activate      # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
pip install -e .
```

Data ingestion requires a `.env` with Zerodha Kite Connect credentials (not tracked in git).

For the dashboard:

```bash
cd dashboard
npm install
npm run dev
```
