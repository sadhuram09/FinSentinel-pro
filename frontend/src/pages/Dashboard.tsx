/**
 * Dashboard — the authenticated home.
 *
 * Guards on mount via GET /auth/me (401 -> /login). Offers a single primary
 * action (run an analysis on a ticker) styled as writing on an index card,
 * with a loading state in the document's own voice, and a "Recent reads" list
 * of past analyses as compact evidence-card rows linking to /analysis/:id.
 */

import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import EvidenceCard from "../components/ui/EvidenceCard";
import ShaderButton from "../components/ui/ShaderButton";
import VerdictStamp, { type Verdict } from "../components/ui/VerdictStamp";

const API_BASE = "http://localhost:8000";

const LOADING_LINES = [
  "Reading the filings...",
  "Weighing the signals...",
  "Scoring the evidence...",
  "Consulting the judge...",
];

interface User {
  email: string;
  name: string | null;
}

interface HistoryItem {
  id: number;
  ticker: string;
  lookback_period: string;
  prediction: string;
  verdict: string;
  created_at: string;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? iso
    : d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

export default function Dashboard() {
  const navigate = useNavigate();

  const [user, setUser] = useState<User | null>(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [history, setHistory] = useState<HistoryItem[] | null>(null);

  const [ticker, setTicker] = useState("");
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [loadingLine, setLoadingLine] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // --- auth guard + initial history load -------------------------------
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/auth/me`, { credentials: "include" });
        if (!active) return;
        if (res.status === 401) {
          navigate("/login");
          return;
        }
        const me = (await res.json()) as User;
        setUser(me);
        setAuthChecked(true);

        const hres = await fetch(`${API_BASE}/history`, { credentials: "include" });
        if (active && hres.ok) setHistory((await hres.json()) as HistoryItem[]);
        else if (active) setHistory([]);
      } catch {
        if (active) navigate("/login");
      }
    })();
    return () => {
      active = false;
    };
  }, [navigate]);

  // Cycle the loading lines while a run is in flight.
  useEffect(() => {
    if (!running) return;
    setLoadingLine(0);
    const iv = window.setInterval(() => setLoadingLine((i) => (i + 1) % LOADING_LINES.length), 1500);
    return () => window.clearInterval(iv);
  }, [running]);

  async function handleLogout() {
    try {
      await fetch(`${API_BASE}/auth/logout`, { method: "POST", credentials: "include" });
    } finally {
      navigate("/login");
    }
  }

  async function handleRun(e: React.FormEvent) {
    e.preventDefault();
    const symbol = ticker.trim().toUpperCase();
    if (!symbol) return;

    setRunError(null);
    setRunning(true);
    try {
      const res = await fetch(`${API_BASE}/analysis/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ ticker: symbol, lookback_period: "1y" }),
      });
      if (res.status === 401) {
        navigate("/login");
        return;
      }
      if (!res.ok) {
        let detail = "Analysis failed. Please try again.";
        try {
          const data = await res.json();
          if (typeof data.detail === "string") detail = data.detail;
        } catch {
          /* non-JSON */
        }
        setRunError(detail);
        return;
      }
      const data = await res.json();
      // The run response carries the persisted history record id (same id used
      // by /history and /history/:id) — navigate straight to it.
      if (data.id != null) navigate(`/analysis/${data.id}`);
    } catch {
      setRunError("Network error — is the backend running?");
    } finally {
      setRunning(false);
    }
  }

  if (!authChecked) {
    return (
      <div className="flex min-h-screen items-center justify-center font-mono text-sm text-paper-muted">
        checking session...
      </div>
    );
  }

  const displayName = user?.name?.trim() || user?.email || "";

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-ink px-6 py-4 sm:px-10">
        <Link to="/" className="font-display text-xl text-paper">
          FinSentinel Pro
        </Link>
        <div className="flex items-center gap-5 font-sans text-sm text-paper-muted">
          <span>{displayName}</span>
          <button type="button" onClick={handleLogout} className="transition-colors hover:text-paper">
            Log out
          </button>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-6 sm:px-10">
        {/* Main action */}
        <section className="pt-28 text-center">
          <h1 className="font-display text-section text-paper">What do you want to look at?</h1>

          <form onSubmit={handleRun} className="mt-10 flex flex-col items-center gap-5">
            <div
              style={{ transform: "rotate(-0.6deg)" }}
              className="shadow-card flex items-center rounded-[3px] border border-black/15 bg-paper px-4 py-3"
            >
              <input
                ref={inputRef}
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
                placeholder="AAPL, MSFT, TSLA..."
                autoCapitalize="characters"
                spellCheck={false}
                disabled={running}
                className="w-64 bg-transparent text-center font-mono text-lg uppercase tracking-wide text-ink outline-none placeholder:normal-case placeholder:tracking-normal placeholder:text-ink/35"
              />
            </div>

            <ShaderButton type="submit" disabled={running}>
              Run analysis
            </ShaderButton>

            {running && (
              <p className="font-mono text-sm text-paper-muted" aria-live="polite">
                {LOADING_LINES[loadingLine]}
              </p>
            )}
            {runError && !running && (
              <p className="note-hand text-sm text-verdict-rejected">{runError}</p>
            )}
          </form>
        </section>

        {/* Recent reads */}
        <section className="mt-32 pb-24">
          <h2 className="font-display text-2xl text-paper">Recent reads</h2>

          <div className="mt-8 space-y-4">
            {history === null && (
              <p className="font-mono text-sm text-paper-muted">loading...</p>
            )}

            {history !== null && history.length === 0 && (
              <div className="rounded-[4px] border border-dashed border-paper-muted/40 px-6 py-10 text-center">
                <p className="note-hand text-base text-paper-muted">
                  No reads yet. Try a ticker above — the first one's always the most honest.
                </p>
              </div>
            )}

            {history?.map((item) => (
              <Link
                key={item.id}
                to={`/analysis/${item.id}`}
                className="block transition-transform hover:-translate-y-0.5"
              >
                <EvidenceCard id={`hist-${item.id}`} confidenceLevel="strong" compact>
                  <div className="flex items-center gap-4">
                    <span className="font-mono text-base font-bold text-ink">{item.ticker}</span>
                    <VerdictStamp verdict={item.verdict as Verdict} size="sm" />
                    <span className="ml-auto font-mono text-xs text-ink/45">
                      {formatDate(item.created_at)}
                    </span>
                  </div>
                </EvidenceCard>
              </Link>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
