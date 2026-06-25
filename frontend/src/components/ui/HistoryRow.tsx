/**
 * HistoryRow — one saved analysis as a compact, clickable evidence-card row.
 *
 * Shared by the Dashboard's "Recent reads" and the full History page so both
 * render identical rows (ticker, small verdict stamp, date), linking to the
 * analysis detail.
 */

import { Link } from "react-router-dom";

import EvidenceCard from "./EvidenceCard";
import VerdictStamp, { type Verdict } from "./VerdictStamp";

export interface HistoryEntry {
  id: number;
  ticker: string;
  lookback_period: string;
  prediction: string;
  verdict: string;
  created_at: string;
}

export function formatHistoryDate(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? iso
    : d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

export default function HistoryRow({ item }: { item: HistoryEntry }) {
  return (
    <Link to={`/analysis/${item.id}`} className="block transition-transform hover:-translate-y-0.5">
      <EvidenceCard id={`hist-${item.id}`} confidenceLevel="strong" compact>
        <div className="flex items-center gap-4">
          <span className="font-mono text-base font-bold text-ink">{item.ticker}</span>
          <VerdictStamp verdict={item.verdict as Verdict} size="sm" />
          <span className="ml-auto font-mono text-xs text-ink/45">{formatHistoryDate(item.created_at)}</span>
        </div>
      </EvidenceCard>
    </Link>
  );
}
