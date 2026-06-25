/**
 * EvidenceCard — the signature component.
 *
 * A physical index card: cream stock, a real soft *directional* drop shadow
 * (not a glow), a thin dark border, a folded dog-ear corner, and a subtle,
 * deterministic rotation (-2deg..2deg, seeded by `id` so it stays put across
 * re-renders rather than jittering).
 *
 * `confidenceLevel` makes the card *look* as certain as it is:
 *   - strong   -> full opacity, crisp.
 *   - moderate -> slightly faded (0.85).
 *   - low      -> faded (0.6) AND its edges blur/feather away via a mask, so
 *                 weak evidence literally appears to be dissolving, not merely
 *                 labelled "low".
 */

import type { CSSProperties, ReactNode } from "react";

export type ConfidenceLevel = "strong" | "moderate" | "low";

interface EvidenceCardProps {
  id: string;
  confidenceLevel: ConfidenceLevel;
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
  /** Smaller padding/type and no auto micro-label — for dense rows of cards. */
  compact?: boolean;
  /** Override the micro-label. Pass "" to hide it; omit for "evidence · {level}". */
  label?: string;
}

const OPACITY: Record<ConfidenceLevel, number> = {
  strong: 1,
  moderate: 0.85,
  low: 0.6,
};

// Deterministic [-3, 3] degree rotation from the id (stable across renders).
function seededRotation(id: string): number {
  let h = 2166136261;
  for (let i = 0; i < id.length; i += 1) {
    h ^= id.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  const unit = ((h >>> 0) % 1000) / 1000; // 0..1
  return Number((unit * 6 - 3).toFixed(2)); // -3..3
}

export default function EvidenceCard({
  id,
  confidenceLevel,
  children,
  className = "",
  style,
  compact = false,
  label,
}: EvidenceCardProps) {
  const rotation = seededRotation(id);
  const isLow = confidenceLevel === "low";

  // Feather the edges for low confidence: opaque in the middle, transparent at
  // the rim, so the card reads as uncertain / fading.
  const edgeMask =
    "radial-gradient(115% 115% at 50% 45%, #000 55%, rgba(0,0,0,0.35) 80%, transparent 100%)";

  const cardStyle: CSSProperties = {
    transform: `rotate(${rotation}deg)`,
    opacity: OPACITY[confidenceLevel],
    ...(isLow
      ? {
          maskImage: edgeMask,
          WebkitMaskImage: edgeMask,
          filter: "blur(0.4px)",
        }
      : {}),
    ...style,
  };

  return (
    <div
      style={cardStyle}
      className={`dog-ear shadow-card relative rounded-[3px] border border-black/15 bg-paper text-ink ${
        compact ? "px-4 py-3" : "px-5 py-4"
      } ${className}`}
    >
      <div
        className={`font-sans leading-relaxed text-ink/90 ${compact ? "text-[13px]" : "text-[15px]"}`}
      >
        {children}
      </div>
      {!compact && label !== "" && (
        <div className="mt-3 font-mono text-[10px] uppercase tracking-[0.16em] text-ink/45">
          {label ?? `evidence · ${confidenceLevel}`}
        </div>
      )}
    </div>
  );
}
