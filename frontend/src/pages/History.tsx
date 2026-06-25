/**
 * History — the complete list of a user's analyses (/history).
 *
 * The backend's GET /history returns a FLAT array with only a `limit` query
 * param (no offset/page), so true server-side pagination isn't available. We
 * fetch a generous limit and paginate/filter on the client: a "Load more"
 * reveals more of the fetched list, and a ticker filter narrows it.
 *
 * Rows reuse the shared HistoryRow component (same as Dashboard's Recent reads).
 */

import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import AppHeader, { type AppUser } from "../components/ui/AppHeader";
import HistoryRow, { type HistoryEntry } from "../components/ui/HistoryRow";

const API_BASE = "http://localhost:8000";
const FETCH_LIMIT = 200; // well above expected volume; the endpoint caps here
const PAGE_SIZE = 20; // client-side reveal step

export default function History() {
  const navigate = useNavigate();
  const [user, setUser] = useState<AppUser | null>(null);
  const [history, setHistory] = useState<HistoryEntry[] | null>(null);
  const [filter, setFilter] = useState("");
  const [visible, setVisible] = useState(PAGE_SIZE);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const me = await fetch(`${API_BASE}/auth/me`, { credentials: "include" });
        if (!active) return;
        if (me.status === 401) return navigate("/login");
        setUser((await me.json()) as AppUser);

        const res = await fetch(`${API_BASE}/history?limit=${FETCH_LIMIT}`, { credentials: "include" });
        if (active && res.ok) setHistory((await res.json()) as HistoryEntry[]);
        else if (active) setHistory([]);
      } catch {
        if (active) navigate("/login");
      }
    })();
    return () => {
      active = false;
    };
  }, [navigate]);

  const all = history ?? [];
  const needle = filter.trim().toUpperCase();
  const filtered = needle ? all.filter((h) => h.ticker.toUpperCase().includes(needle)) : all;
  const shown = filtered.slice(0, visible);

  return (
    <div className="min-h-screen">
      <AppHeader user={user} showBack />

      <main className="mx-auto max-w-3xl px-6 pb-24 pt-16 sm:px-10">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <h1 className="font-display text-section text-paper">History</h1>
          <input
            value={filter}
            onChange={(e) => {
              setFilter(e.target.value);
              setVisible(PAGE_SIZE);
            }}
            placeholder="filter by ticker..."
            spellCheck={false}
            className="w-48 rounded-[3px] border border-ink bg-paper/[0.03] px-3 py-1.5 font-mono text-sm uppercase text-paper outline-none transition-colors placeholder:normal-case placeholder:text-paper-muted/50 focus:border-paper/30"
          />
        </div>

        <div className="mt-10 space-y-4">
          {history === null && <p className="font-mono text-sm text-paper-muted">loading...</p>}

          {history !== null && all.length === 0 && (
            <div className="rounded-[4px] border border-dashed border-paper-muted/40 px-6 py-10 text-center">
              <p className="note-hand text-base text-paper-muted">
                No reads yet. Try a ticker on the dashboard — the first one's always the most honest.
              </p>
            </div>
          )}

          {history !== null && all.length > 0 && filtered.length === 0 && (
            <p className="font-mono text-sm text-paper-muted">No analyses match &ldquo;{filter.trim()}&rdquo;.</p>
          )}

          {shown.map((item) => (
            <HistoryRow key={item.id} item={item} />
          ))}
        </div>

        {filtered.length > visible && (
          <button
            type="button"
            onClick={() => setVisible((v) => v + PAGE_SIZE)}
            className="mt-8 font-mono text-sm text-paper-muted underline-offset-4 hover:text-paper hover:underline"
          >
            Load more ({filtered.length - visible} remaining)
          </button>
        )}

        {history !== null && filtered.length > 0 && (
          <p className="mt-8 font-mono text-[11px] text-paper-muted/50">
            showing {shown.length} of {filtered.length}
            {needle ? ` (filtered from ${all.length})` : ""}
          </p>
        )}
      </main>
    </div>
  );
}
