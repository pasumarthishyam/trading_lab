import { getStrategies, getInstruments, getTestCompletion } from "@/lib/db";

export default function HomePage() {
  const strategies = getStrategies();
  const instruments = getInstruments();

  return (
    <div className="p-8 max-w-6xl">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white mb-2">Trading Lab</h1>
        <p className="text-[var(--text-secondary)]">
          Quantitative research platform for strategy development and validation
        </p>
      </div>

      {/* Strategy Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {strategies.map((s) => {
          const completion = instruments[0]
            ? getTestCompletion(s.id, instruments[0].id)
            : { total: 0, completed: 0 };
          const pct =
            completion.total > 0
              ? Math.round((completion.completed / completion.total) * 100)
              : 0;

          return (
            <a
              key={s.id}
              href={`/strategy/${s.slug}`}
              className="card-hover block bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-6"
            >
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h2 className="text-lg font-semibold text-white mb-1">
                    {s.name}
                  </h2>
                  <p className="text-sm text-[var(--text-secondary)] line-clamp-2">
                    {s.description}
                  </p>
                </div>

                {/* Completion ring */}
                <div className="relative w-14 h-14 flex-shrink-0 ml-4">
                  <svg
                    className="w-14 h-14 -rotate-90"
                    viewBox="0 0 56 56"
                  >
                    <circle
                      cx="28"
                      cy="28"
                      r="24"
                      fill="none"
                      stroke="var(--border)"
                      strokeWidth="3"
                    />
                    <circle
                      cx="28"
                      cy="28"
                      r="24"
                      fill="none"
                      stroke="var(--accent-green)"
                      strokeWidth="3"
                      strokeDasharray={`${(pct / 100) * 150.8} 150.8`}
                      strokeLinecap="round"
                    />
                  </svg>
                  <span className="absolute inset-0 flex items-center justify-center text-xs font-mono font-medium text-white">
                    {pct}%
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-4 text-xs text-[var(--text-muted)]">
                <span className="font-mono">
                  {completion.completed}/{completion.total} tests
                </span>
                {instruments.map((i) => (
                  <span
                    key={i.id}
                    className="bg-[var(--bg-secondary)] px-2 py-0.5 rounded text-[var(--text-secondary)]"
                  >
                    {i.symbol}
                  </span>
                ))}
              </div>
            </a>
          );
        })}
      </div>

      {/* Quick Stats */}
      <div className="mt-8 grid grid-cols-3 gap-4">
        <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
          <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">
            Strategies
          </p>
          <p className="text-2xl font-bold font-mono text-white">
            {strategies.length}
          </p>
        </div>
        <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
          <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">
            Instruments
          </p>
          <p className="text-2xl font-bold font-mono text-white">
            {instruments.length}
          </p>
        </div>
        <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
          <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">
            Platform
          </p>
          <p className="text-sm text-[var(--text-secondary)] mt-1">
            SQLite + Python
          </p>
        </div>
      </div>
    </div>
  );
}
