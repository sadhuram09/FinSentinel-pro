/**
 * Design-system proof composition.
 *
 * Renders the reading-room ground (DocumentTexture), a Fraunces hero, a desk of
 * overlapping EvidenceCards at the three confidence levels (so the visual
 * fade/feather is obvious), a MarginNote with its hand-drawn connector, a
 * Conflicted VerdictStamp (to show the stamp-down impact), and a second section
 * wrapped in AnalysisScrollReveal.
 */

import DocumentTexture from "./components/ui/DocumentTexture";
import EvidenceCard from "./components/ui/EvidenceCard";
import VerdictStamp from "./components/ui/VerdictStamp";
import MarginNote from "./components/ui/MarginNote";
import AnalysisScrollReveal from "./components/ui/AnalysisScrollReveal";

function App() {
  return (
    <>
      <DocumentTexture />

      <main className="mx-auto min-h-screen max-w-5xl px-6 py-20 sm:px-10">
        {/* Hero */}
        <header className="max-w-3xl">
          <p className="font-mono text-[11px] uppercase tracking-[0.28em] text-paper-muted">
            FinSentinel · Design System
          </p>
          <h1 className="mt-5 font-display text-hero text-paper">
            Evidence, weighed in a quiet room.
          </h1>
          <p className="mt-6 max-w-xl font-sans text-base leading-relaxed text-paper-muted">
            Every judgment arrives as paper on a desk — cards you can trust at a
            glance, because the ones we&rsquo;re unsure of look unsure.
          </p>
        </header>

        {/* The desk: cards placed absolutely at different x/y/rotation, so they
            read like physical objects set down at different times — overlapping,
            staggered, not a neat row. MarginNote lives in the clear space BELOW
            the moderate card, never over any card body. */}
        <section className="relative mx-auto mt-24 h-[600px] w-full max-w-[880px]">
          <EvidenceCard
            id="ev-strong-7"
            confidenceLevel="strong"
            className="w-72"
            style={{ position: "absolute", left: 8, top: 0, zIndex: 30 }}
          >
            Q4 net sales rose on iPhone strength, with Services at a record high —
            management cites durable installed-base growth.
          </EvidenceCard>

          <EvidenceCard
            id="ev-moderate-0"
            confidenceLevel="moderate"
            className="w-72"
            style={{ position: "absolute", left: 296, top: 104, zIndex: 20 }}
          >
            Supply-chain commentary is mixed; component costs &ldquo;may&rdquo;
            pressure margins in the near term, though direction is unstated.
          </EvidenceCard>

          <EvidenceCard
            id="ev-low-0"
            confidenceLevel="low"
            className="w-72"
            style={{ position: "absolute", left: 584, top: 28, zIndex: 10 }}
          >
            A passing mention of competitive pressure in adjacent markets; too
            faint to anchor a directional read.
          </EvidenceCard>

          {/* Annotates the moderate card from below, connector reaching up. */}
          <MarginNote className="left-[330px] top-[330px]" side="top">
            partially relevant — the filing hedges here
          </MarginNote>
        </section>

        {/* Verdict stamp. */}
        <section className="mt-20 flex items-center gap-6">
          <span className="font-mono text-[11px] uppercase tracking-[0.24em] text-paper-muted">
            Judge verdict
          </span>
          <VerdictStamp verdict="Conflicted" />
        </section>

        {/* Scroll-reveal proof, far enough down to trigger on scroll. */}
        <div className="h-[60vh]" aria-hidden />
        <AnalysisScrollReveal>
          <section className="max-w-2xl border-t border-ink pt-12">
            <h2 className="font-display text-section text-paper">
              Read calmly, down the page.
            </h2>
            <p className="mt-5 font-sans leading-relaxed text-paper-muted">
              This block faded up only as it entered view — a short, settled
              motion, not a bounce. The same wrapper composes around any section.
            </p>
            <p className="mt-6 font-mono text-sm text-paper-muted">
              direction_probability <span className="text-paper">0.612</span> ·
              confidence <span className="text-verdict-conflicted">moderate</span>
            </p>
          </section>
        </AnalysisScrollReveal>
      </main>
    </>
  );
}

export default App;
