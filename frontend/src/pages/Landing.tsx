/**
 * Landing page — composed entirely from the existing design system.
 *
 * Eight sections (hero + seven revealed-on-scroll). Copy is fixed by spec; no
 * invented claims, stats, or testimonials. The hero reuses the design-system
 * demo's 3-EvidenceCard + Conflicted-stamp composition.
 */

import type { ReactNode } from "react";

import AnalysisScrollReveal from "../components/ui/AnalysisScrollReveal";
import EvidenceCard from "../components/ui/EvidenceCard";
import MarginNote from "../components/ui/MarginNote";
import ShaderButton from "../components/ui/ShaderButton";
import VerdictStamp from "../components/ui/VerdictStamp";

const GITHUB_URL = "https://github.com/sadhuram09/FinSentinel-pro";

const AGENTS = [
  { id: "agent-analyst", name: "Analyst", desc: "fundamentals: P/E, ROE, debt ratios" },
  { id: "agent-forecast", name: "Forecast", desc: "XGBoost + LightGBM ensemble, validated, not overfit" },
  { id: "agent-risk", name: "Risk", desc: "VaR, CVaR, Sharpe, drawdown, beta" },
  { id: "agent-sentiment", name: "Sentiment", desc: "FinBERT on real-time news" },
  { id: "agent-evidence", name: "Evidence (RAG)", desc: "retrieves actual SEC 10-K filings, scores its own retrieval confidence" },
];

const STATS = [
  { id: "stat-0-1", text: "AUC 0.61 · 30-day forecast" },
  { id: "stat-1-1", text: "5 agents · 1 verdict" },
  { id: "stat-2-1", text: "SEC 10-K · real filings, not headlines" },
  { id: "stat-3-2", text: "MLflow tracked · every training run" },
];

const TECH = [
  "FastAPI", "LangGraph", "Groq (Llama 3.3)", "XGBoost", "LightGBM", "SHAP",
  "FinBERT", "ChromaDB", "MLflow", "Evidently AI", "SQLAlchemy", "React", "Framer Motion",
];

function SectionHeading({ children }: { children: ReactNode }) {
  return <h2 className="font-display text-section text-paper">{children}</h2>;
}

export default function Landing() {
  return (
    <main className="relative mx-auto max-w-5xl px-6 pb-24 pt-20 sm:px-10">
      {/* 1 · HERO ----------------------------------------------------------- */}
      <section>
        <p className="font-mono text-[11px] uppercase tracking-[0.28em] text-paper-muted">
          FINSENTINEL PRO · EVIDENCE-WEIGHTED ANALYSIS
        </p>
        <h1 className="mt-5 max-w-3xl font-display text-hero text-paper">
          Five signals. One honest verdict.
        </h1>
        <p className="mt-6 max-w-xl font-sans text-lg leading-relaxed text-paper-muted">
          Most financial tools tell you what to do. This one tells you when it
          isn't sure.
        </p>

        {/* Hero visual: the design-system composition. */}
        <div className="relative mt-16 flex flex-wrap items-start justify-center gap-y-10 pb-10">
          <EvidenceCard id="hero-strong" confidenceLevel="strong" className="z-30 w-64">
            Services at a record high; management cites durable installed-base growth.
          </EvidenceCard>
          <EvidenceCard id="hero-moderate" confidenceLevel="moderate" className="z-20 -ml-5 mt-8 w-64">
            Component costs &ldquo;may&rdquo; pressure margins — direction unstated.
          </EvidenceCard>
          <EvidenceCard id="hero-low" confidenceLevel="low" className="z-10 -ml-5 mt-3 w-64">
            A passing mention of competitive pressure; too faint to anchor.
          </EvidenceCard>
          <div className="absolute bottom-0 right-6 z-40 sm:right-16">
            <VerdictStamp verdict="Conflicted" />
          </div>
        </div>

        <div className="mt-12">
          <ShaderButton to="/signup">Try a ticker</ShaderButton>
        </div>
      </section>

      {/* 2 · THE PROBLEM ---------------------------------------------------- */}
      <AnalysisScrollReveal className="mt-40">
        <section className="max-w-2xl">
          <SectionHeading>Confidence is cheap. We don't sell it.</SectionHeading>
          <p className="mt-6 font-sans text-lg leading-relaxed text-paper-muted">
            Most AI finance tools overstate certainty — a single signal, like a
            sentiment read or a chart pattern, presented as a verdict.
          </p>
          <div className="mt-8 rounded-[4px] border border-ink bg-paper/[0.03] p-5 font-mono text-[13px] leading-relaxed text-paper-muted">
            Our 5-day forecast: <span className="text-paper">AUC 0.52</span> —
            barely better than chance. We say so. The 30-day forecast
            (<span className="text-paper">AUC 0.61</span>) has real but modest
            signal — also disclosed, not inflated.
          </div>
        </section>
      </AnalysisScrollReveal>

      {/* 3 · HOW IT WORKS --------------------------------------------------- */}
      <AnalysisScrollReveal className="mt-40">
        <section>
          <SectionHeading>Five agents. No groupthink.</SectionHeading>
          <div className="mt-12 flex flex-wrap justify-center gap-x-2 gap-y-8">
            {AGENTS.map((a, i) => (
              <div key={a.id} style={{ marginTop: i % 2 === 0 ? 0 : 26 }}>
                <EvidenceCard id={a.id} confidenceLevel="strong" compact className="w-60">
                  <span className="font-semibold text-ink">{a.name}</span>
                  <span className="text-ink/70"> — {a.desc}</span>
                </EvidenceCard>
              </div>
            ))}
          </div>
        </section>
      </AnalysisScrollReveal>

      {/* 4 · THE JUDGE ------------------------------------------------------ */}
      <AnalysisScrollReveal className="mt-40">
        <section className="max-w-2xl">
          <SectionHeading>The agent that's allowed to say no.</SectionHeading>
          <div className="mt-6 space-y-5 font-sans text-lg leading-relaxed text-paper-muted">
            <p>
              The Judge doesn't forecast. It reads the other four agents and scores
              how much they actually agree — a single consistency number across
              analyst, forecast, risk, and sentiment.
            </p>
            <p>
              Agreement alone isn't enough. Verdict gating is strict: any raised
              flag — wide uncertainty, a near-coin-flip probability — blocks
              Approved, even when consistency is high. The default isn't optimism.
            </p>
            <p>
              And the Judge's own written reasoning is held to the same standard:
              its LLM output is regex-checked to strip any language claiming
              accuracy or confidence the system didn't earn.
            </p>
          </div>

          <div className="relative mt-12 flex flex-col gap-6 sm:flex-row sm:items-start">
            <div className="shrink-0">
              <VerdictStamp verdict="Conflicted" />
            </div>
            <div className="rounded-[4px] border border-ink bg-paper/[0.03] p-5">
              <p className="font-sans text-[15px] leading-relaxed text-paper-muted">
                Consistency 0.82 — analyst and forecast align and news sentiment is
                neutral, but a low-confidence flag on the headline probability blocks
                approval; document evidence was moderate, not conclusive.
              </p>
              <p className="mt-4 font-mono text-[11px] uppercase tracking-[0.16em] text-paper-muted/70">
                audit_id: 8f3a1c20-7e44-4b91-a0c2-1d41a9f3e2bd
              </p>
            </div>
          </div>
        </section>
      </AnalysisScrollReveal>

      {/* 5 · REAL NUMBERS -------------------------------------------------- */}
      <AnalysisScrollReveal className="mt-40">
        <section>
          <SectionHeading>Built numbers, not marketing numbers.</SectionHeading>
          {/* Each stat is a small paper card — same texture, dog-ear and seeded
              rotation as the evidence cards, just sized for a short stat. */}
          <div className="mt-14 flex flex-wrap justify-center gap-x-4 gap-y-10">
            {STATS.map((s, i) => (
              <div key={s.id} style={{ marginTop: i % 2 === 0 ? 0 : 22 }}>
                <EvidenceCard id={s.id} confidenceLevel="strong" compact className="w-52">
                  <span className="font-mono text-[13px] leading-snug text-ink">{s.text}</span>
                </EvidenceCard>
              </div>
            ))}
          </div>
        </section>
      </AnalysisScrollReveal>

      {/* 6 · TECH STACK ---------------------------------------------------- */}
      <AnalysisScrollReveal className="mt-40">
        <section>
          <SectionHeading>Under the hood</SectionHeading>
          {/* A quiet mono list with middot dividers — the same muted cream-on-dark
              voice as the kicker, reusing the page's "·" motif. */}
          <div className="mt-10 flex flex-wrap items-center gap-x-3 gap-y-2 font-mono text-sm text-paper-muted">
            {TECH.flatMap((t, i) =>
              i === 0
                ? [
                    <span key={t} className="transition-colors hover:text-paper">
                      {t}
                    </span>,
                  ]
                : [
                    <span key={`${t}-div`} aria-hidden className="select-none text-paper-muted/40">
                      ·
                    </span>,
                    <span key={t} className="transition-colors hover:text-paper">
                      {t}
                    </span>,
                  ]
            )}
          </div>
        </section>
      </AnalysisScrollReveal>

      {/* 7 · CTA ----------------------------------------------------------- */}
      <AnalysisScrollReveal className="mt-40">
        <section className="relative max-w-2xl">
          <SectionHeading>See what the evidence says.</SectionHeading>
          <div className="mt-8 flex flex-wrap gap-4">
            <ShaderButton to="/signup">Sign up</ShaderButton>
            <ShaderButton href={GITHUB_URL} variant="secondary">
              View on GitHub
            </ShaderButton>
          </div>
          <MarginNote className="left-[60%] top-0 hidden lg:block" side="left">
            no card required to look around
          </MarginNote>
        </section>
      </AnalysisScrollReveal>

      {/* 8 · FOOTER -------------------------------------------------------- */}
      <footer className="mt-40 border-t border-ink pt-10">
        <p className="font-display text-2xl text-paper">FinSentinel Pro</p>
        <p className="mt-3 font-sans text-sm text-paper-muted">
          Built by Sadhuram Agarwal.
        </p>
        <div className="mt-4 flex gap-6 font-mono text-[13px] text-paper-muted">
          <a href={GITHUB_URL} target="_blank" rel="noopener noreferrer" className="hover:text-paper">
            GitHub
          </a>
          <a href="#" className="hover:text-paper">
            Portfolio
          </a>
        </div>
      </footer>
    </main>
  );
}
