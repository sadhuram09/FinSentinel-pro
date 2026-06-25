/**
 * VerdictStamp — the Judge's verdict as a rubber-stamp impression.
 *
 * Rotated -8deg, a bold border in the verdict colour, and a faintly irregular
 * border-radius so it reads as hand-stamped rather than a perfect rounded rect.
 * It animates like a physical stamp hitting paper: scale 1.4 -> 1 with opacity,
 * fast (150ms) ease-out — an impact, not a smooth fade.
 */

import { motion } from "framer-motion";

export type Verdict = "Approved" | "Conflicted" | "Rejected";

interface VerdictStampProps {
  verdict: Verdict;
  className?: string;
}

// Map each verdict to its semantic colour (matches tailwind verdict-* tokens).
const VERDICT_COLOR: Record<Verdict, string> = {
  Approved: "#3DDC84",
  Conflicted: "#E8A33D",
  Rejected: "#E5484D",
};

export default function VerdictStamp({ verdict, className = "" }: VerdictStampProps) {
  const color = VERDICT_COLOR[verdict];

  return (
    <motion.div
      // rotate lives in the motion props (NOT CSS transform) so framer-motion's
      // scale animation composes with it instead of clobbering it. scale 1.4->1
      // gives the stamp-down impact; rotate stays fixed at -8deg throughout.
      initial={{ scale: 1.4, opacity: 0, rotate: -8 }}
      animate={{ scale: 1, opacity: 1, rotate: -8 }}
      transition={{ duration: 0.15, ease: "easeOut" }}
      style={{
        color,
        borderColor: color,
        // Asymmetric corners (TL TR BR BL) -> hand-stamped, not a clean badge.
        borderRadius: "2px 8px 3px 6px",
        // Slightly inky, uneven impression.
        boxShadow: `inset 0 0 0 1px ${color}33`,
      }}
      className={`inline-flex select-none items-center border-[5px] px-5 py-2 font-mono text-xl font-bold uppercase tracking-[0.2em] ${className}`}
    >
      {verdict}
    </motion.div>
  );
}
