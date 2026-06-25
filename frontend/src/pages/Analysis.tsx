/**
 * Analysis detail — /analysis/:id.
 *
 * Fetches a saved run (GET /history/:id) and renders the full AnalysisResponse
 * stored in `result_json`: the judge verdict up top, five expandable evidence
 * cards (one per agent), a dependency-free SHAP bar chart, and an audit footer.
 *
 * Parsing prefers the nested objects (analyst / forecast / risk_report /
 * sentiment_report / evidence_sources / judge_verdict) and uses
 * forecast.horizons for the 5d/30d split rather than the duplicated top-level
 * forecast fields.
 */

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { Link, useNavigate, useParams } from "react-router-dom";

import AnalysisScrollReveal from "../components/ui/AnalysisScrollReveal";
import EvidenceCard, { type ConfidenceLevel } from "../components/ui/EvidenceCard";
import VerdictStamp, { type Verdict } from "../components/ui/VerdictStamp";

const API_BASE = "http://localhost:8000";

const APPROVED = "#3DDC84";
const REJECTED = "#E5484D";

// ---- response shape (subset we render) -------------------------------------
interface ShapItem { feature: string; value: number; contribution: number; }
interface Horizon { horizon_days: number; direction: string; probability_up: number; }
interface AnalysisResult {
  model_version: string;
  shap_explanation: ShapItem[];
  analyst: {
    last_close: number;
    commentary: string;
    technicals: { rsi: number; macd: number; bb_pct: number };
    fundamentals: { pe_ratio: number | null; roe: number | null; debt_to_equity: number | null };
  };
  forecast: { prediction: string; direction_probability: number; horizons: Horizon[] };
  risk_report: {
    var_95: number; var_99: number; cvar_95: number; sharpe_ratio: number;
    sortino_ratio: number; max_drawdown: number; beta: number; risk_tier: string;
  };
  sentiment_report: {
    sentiment_score: number; sentiment_label: string; headline_count: number;
    top_headlines: { headline: string; label: string; score: number }[];
    note: string | null;
  };
  evidence_sources: { section: string; snippet: string; similarity_score: number }[];
  retrieval_confidence: number;
  confidence_tier: "strong" | "moderate" | "insufficient";
  judge_verdict: {
    verdict: Verdict; consistency_score: number; flags: string[];
    override_reasoning: string; audit_id: string;
  };
}
interface HistoryDetail {
  id: number; ticker: string; created_at: string; result_json: AnalysisResult;
}

// ---- formatting helpers ----------------------------------------------------
const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`;
const num = (x: number, d = 2) => x.toFixed(d);
const naMono = (x: number | null, fmt: (v: number) => string) =>
  x === null || x === undefined ? "N/A" : fmt(x);
const fmtDate = (iso: string) => {
  const dt = new Date(iso);
  return Number.isNaN(dt.getTime())
    ? iso
    : dt.toLocaleString("en-US", { month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit" });
};

const TIER_TO_LEVEL: Record<AnalysisResult["confidence_tier"], ConfidenceLevel> = {
  strong: "strong",
  moderate: "moderate",
  insufficient: "low",
};

// ---- expandable agent card -------------------------------------------------
function AgentCard(props: {
  cardId: string;
  level: ConfidenceLevel;
  title: string;
  summary: React.ReactNode;
  detail: React.ReactNode;
  isOpen: boolean;
  onToggle: () => void;
  offset?: number;
}) {
  const { cardId, level, title, summary, detail, isOpen, onToggle, offset = 0 } = props;
  return (
    <div style={{ marginLeft: offset }}>
      <EvidenceCard id={cardId} confidenceLevel={level} label="" className="w-full max-w-2xl">
        <button onClick={onToggle} className="flex w-full items-center justify-between gap-4 text-left">
          <span>
            <span className="font-mono text-xs uppercase tracking-[0.18em] text-ink/45">{title}</span>
            <span className="mt-1 block text-[15px] text-ink/90">{summary}</span>
          </span>
          <ChevronDown
            size={18}
            className={`shrink-0 text-ink/40 transition-transform duration-200 ${isOpen ? "rotate-180" : ""}`}
          />
        </button>
        <AnimatePresence initial={false}>
          {isOpen && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.25, ease: "easeOut" }}
              style={{ overflow: "hidden" }}
            >
              <div className="mt-4 border-t border-black/10 pt-4 text-[14px] text-ink/85">{detail}</div>
            </motion.div>
          )}
        </AnimatePresence>
      </EvidenceCard>
    </div>
  );
}

function Stat({ label, value, mono = true }: { label: string; value: React.ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1">
      <span className="text-ink/55">{label}</span>
      <span className={mono ? "font-mono text-ink" : "text-ink"}>{value}</span>
    </div>
  );
}

export default function Analysis() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState<HistoryDetail | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "notfound">("loading");
  const [open, setOpen] = useState<Record<string, boolean>>({});

  const toggle = (k: string) => setOpen((o) => ({ ...o, [k]: !o[k] }));

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/history/${id}`, { credentials: "include" });
        if (!active) return;
        if (res.status === 401) return navigate("/login");
        if (res.status === 404) return setStatus("notfound");
        if (!res.ok) return setStatus("notfound");
        setData((await res.json()) as HistoryDetail);
        setStatus("ready");
      } catch {
        if (active) setStatus("notfound");
      }
    })();
    return () => {
      active = false;
    };
  }, [id, navigate]);

  if (status === "loading") {
    return (
      <div className="flex min-h-screen items-center justify-center font-mono text-sm text-paper-muted">
        retrieving the file...
      </div>
    );
  }
  if (status === "notfound" || !data) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-5 px-6 text-center">
        <p className="font-display text-section text-paper">That file isn&rsquo;t in the drawer.</p>
        <p className="note-hand text-paper-muted">No analysis with that id — it may never have been run.</p>
        <Link to="/dashboard" className="font-mono text-sm text-paper-muted underline hover:text-paper">
          ← back to dashboard
        </Link>
      </div>
    );
  }

  const a = data.result_json;
  const f = a.analyst.fundamentals;
  const maxAbs = Math.max(...a.shap_explanation.map((s) => Math.abs(s.contribution)), 1e-9);
  const shap = [...a.shap_explanation].sort((x, y) => Math.abs(y.contribution) - Math.abs(x.contribution));

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b border-ink px-6 py-5 sm:px-10">
        <Link to="/dashboard" className="font-mono text-xs text-paper-muted hover:text-paper">
          ← back to dashboard
        </Link>
        <div className="mt-3 flex flex-wrap items-baseline gap-x-5 gap-y-1">
          <h1 className="font-mono text-4xl font-bold tracking-tight text-paper">{data.ticker}</h1>
          <span className="font-mono text-sm text-paper-muted">{fmtDate(data.created_at)}</span>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-6 pb-24 sm:px-10">
        {/* THE VERDICT */}
        <section className="pt-16">
          <div className="flex flex-wrap items-center gap-5">
            <VerdictStamp verdict={a.judge_verdict.verdict} />
            <span className="font-mono text-sm text-paper-muted">
              consistency {num(a.judge_verdict.consistency_score)}
            </span>
            {a.judge_verdict.flags.map((flag) => (
              <span key={flag} className="rounded-[3px] border border-verdict-conflicted/50 px-2 py-0.5 font-mono text-[11px] text-verdict-conflicted">
                {flag}
              </span>
            ))}
          </div>
          <div className="mt-6 rounded-[4px] border border-ink bg-paper/[0.03] p-5">
            <p className="font-sans text-[15px] leading-relaxed text-paper-muted">
              {a.judge_verdict.override_reasoning}
            </p>
          </div>
        </section>

        {/* FIVE EVIDENCE CARDS (fanned vertical stack, expandable) */}
        <AnalysisScrollReveal className="mt-20">
          <h2 className="mb-8 font-display text-section text-paper">The five reads</h2>
          <div className="space-y-6">
            {/* Analyst */}
            <AgentCard
              cardId="card-analyst" level="strong" title="Analyst" offset={6}
              isOpen={!!open.analyst} onToggle={() => toggle("analyst")}
              summary={
                <>
                  last close <span className="font-mono">${num(a.analyst.last_close)}</span> —{" "}
                  {a.analyst.commentary.split(". ")[0]}.
                </>
              }
              detail={
                <div className="grid grid-cols-1 gap-x-10 gap-y-1 sm:grid-cols-2">
                  <div>
                    <p className="mb-1 font-mono text-[11px] uppercase tracking-wider text-ink/45">technicals</p>
                    <Stat label="RSI" value={num(a.analyst.technicals.rsi, 1)} />
                    <Stat label="MACD" value={num(a.analyst.technicals.macd, 3)} />
                    <Stat label="%B (bb_pct)" value={num(a.analyst.technicals.bb_pct, 3)} />
                  </div>
                  <div>
                    <p className="mb-1 font-mono text-[11px] uppercase tracking-wider text-ink/45">fundamentals</p>
                    <Stat label="P/E" value={naMono(f.pe_ratio, (v) => num(v, 1))} />
                    <Stat label="ROE" value={naMono(f.roe, (v) => pct(v))} />
                    <Stat label="D/E" value={naMono(f.debt_to_equity, (v) => num(v, 2))} />
                  </div>
                </div>
              }
            />

            {/* Forecast */}
            <AgentCard
              cardId="card-forecast" level="strong" title="Forecast" offset={-8}
              isOpen={!!open.forecast} onToggle={() => toggle("forecast")}
              summary={
                <>
                  <span className="font-mono uppercase">{a.forecast.prediction}</span> · P(up){" "}
                  <span className="font-mono">{pct(a.forecast.direction_probability)}</span>
                </>
              }
              detail={
                <div className="space-y-2">
                  {a.forecast.horizons.map((h) => (
                    <div key={h.horizon_days} className="flex items-baseline justify-between">
                      <span className="text-ink/55">{h.horizon_days}-day horizon</span>
                      <span className="font-mono text-ink">
                        {h.direction.toUpperCase()} · {pct(h.probability_up)}
                      </span>
                    </div>
                  ))}
                </div>
              }
            />

            {/* Risk */}
            <AgentCard
              cardId="card-risk" level="strong" title="Risk" offset={10}
              isOpen={!!open.risk} onToggle={() => toggle("risk")}
              summary={<>risk tier <span className="font-mono font-semibold">{a.risk_report.risk_tier}</span></>}
              detail={
                <div className="font-mono text-[13px]">
                  <Stat label="VaR 95%" value={pct(a.risk_report.var_95, 2)} />
                  <Stat label="VaR 99%" value={pct(a.risk_report.var_99, 2)} />
                  <Stat label="CVaR 95%" value={pct(a.risk_report.cvar_95, 2)} />
                  <Stat label="Sharpe" value={num(a.risk_report.sharpe_ratio)} />
                  <Stat label="Sortino" value={num(a.risk_report.sortino_ratio)} />
                  <Stat label="Max drawdown" value={pct(a.risk_report.max_drawdown, 1)} />
                  <Stat label="Beta (vs SPY)" value={num(a.risk_report.beta)} />
                </div>
              }
            />

            {/* Sentiment */}
            <AgentCard
              cardId="card-sentiment" level="strong" title="Sentiment" offset={-6}
              isOpen={!!open.sentiment} onToggle={() => toggle("sentiment")}
              summary={
                <>
                  <span className="font-mono">{a.sentiment_report.sentiment_label}</span> · score{" "}
                  <span className="font-mono">{num(a.sentiment_report.sentiment_score, 3)}</span>
                </>
              }
              detail={
                a.sentiment_report.note ? (
                  <div className="rounded-[4px] border border-dashed border-ink/25 p-3">
                    <p className="note-hand text-ink/50">{a.sentiment_report.note}</p>
                  </div>
                ) : (
                  <ul className="space-y-3">
                    {a.sentiment_report.top_headlines.map((h, i) => (
                      <li key={i} className="flex items-start gap-3">
                        <span
                          className="mt-0.5 shrink-0 font-mono text-[11px]"
                          style={{ color: h.score >= 0 ? APPROVED : REJECTED }}
                        >
                          {h.score >= 0 ? "+" : ""}{num(h.score, 2)}
                        </span>
                        <span className="text-ink/80">{h.headline}</span>
                      </li>
                    ))}
                  </ul>
                )
              }
            />

            {/* Evidence (RAG) */}
            <AgentCard
              cardId="card-evidence" level={TIER_TO_LEVEL[a.confidence_tier]} title={`Evidence (RAG) · ${a.confidence_tier}`} offset={8}
              isOpen={!!open.evidence} onToggle={() => toggle("evidence")}
              summary={<>retrieval confidence <span className="font-mono">{num(a.retrieval_confidence)}</span></>}
              detail={
                <div className="space-y-4">
                  {a.evidence_sources.map((e, i) => (
                    <blockquote key={i} className="border-l-2 border-ink/20 pl-3">
                      <div className="mb-1 flex items-center justify-between font-mono text-[11px] text-ink/45">
                        <span className="uppercase tracking-wider">{e.section}</span>
                        <span>sim {num(e.similarity_score)}</span>
                      </div>
                      <p className="text-[13px] leading-relaxed text-ink/75">
                        {e.snippet.length > 320 ? `${e.snippet.slice(0, 320)}…` : e.snippet}
                      </p>
                    </blockquote>
                  ))}
                </div>
              }
            />
          </div>
        </AnalysisScrollReveal>

        {/* SHAP chart */}
        <AnalysisScrollReveal className="mt-20">
          <h2 className="mb-2 font-display text-section text-paper">What moved the model</h2>
          <p className="mb-8 font-mono text-xs text-paper-muted">
            SHAP feature attributions · green pushes toward up, red toward down
          </p>
          <div className="space-y-3">
            {shap.map((s) => {
              const width = (Math.abs(s.contribution) / maxAbs) * 100;
              const positive = s.contribution >= 0;
              return (
                <div key={s.feature} className="flex items-center gap-3">
                  <span className="w-28 shrink-0 text-right font-mono text-xs text-paper-muted">{s.feature}</span>
                  <div className="h-5 flex-1 rounded-sm bg-paper/[0.04]">
                    <div
                      className="h-full rounded-sm"
                      style={{ width: `${width}%`, backgroundColor: positive ? `${APPROVED}59` : `${REJECTED}59` }}
                    />
                  </div>
                  <span
                    className="w-16 shrink-0 font-mono text-xs"
                    style={{ color: positive ? APPROVED : REJECTED }}
                  >
                    {positive ? "+" : ""}{num(s.contribution, 3)}
                  </span>
                </div>
              );
            })}
          </div>
        </AnalysisScrollReveal>

        {/* Audit footer */}
        <footer className="mt-24 border-t border-ink pt-6">
          <div className="flex flex-wrap gap-x-8 gap-y-1 font-mono text-[11px] text-paper-muted/70">
            <span>model_version: {a.model_version}</span>
            <span>audit_id: {a.judge_verdict.audit_id}</span>
            <span>run: {fmtDate(data.created_at)}</span>
          </div>
        </footer>
      </main>
    </div>
  );
}
