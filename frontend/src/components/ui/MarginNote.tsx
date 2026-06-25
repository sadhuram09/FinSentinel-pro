/**
 * MarginNote — a small, handwritten-feeling annotation.
 *
 * Set in a casual Fraunces italic (the `note-hand` voice) and positioned
 * absolutely against a `relative` parent (pass placement via `className`). A
 * thin, slightly curved SVG line — hand-drawn, never straight — bridges the gap
 * from the note to whatever it annotates.
 *
 * `side` is the direction of the referent relative to the note, i.e. which edge
 * the connector leaves from: "left"/"right"/"top"/"bottom". For a note placed
 * BELOW a card, use side="top" so the line reaches up to it.
 */

import type { CSSProperties, ReactNode } from "react";

type Side = "left" | "right" | "top" | "bottom";

interface MarginNoteProps {
  children: ReactNode;
  className?: string;
  side?: Side;
}

// Per-side connector geometry: where the SVG sits relative to the note, its
// box, the curved path, and the referent anchor dot at the far end.
const CONNECTORS: Record<
  Side,
  { box: [number, number]; pos: CSSProperties; path: string; dot: [number, number] }
> = {
  left: {
    box: [64, 48],
    pos: { left: "-64px", top: "50%", transform: "translateY(-50%)" },
    path: "M60 12 C 44 14, 22 20, 4 38",
    dot: [4, 38],
  },
  right: {
    box: [64, 48],
    pos: { right: "-64px", top: "50%", transform: "translateY(-50%)" },
    path: "M4 12 C 20 14, 42 20, 60 38",
    dot: [60, 38],
  },
  top: {
    box: [48, 64],
    pos: { top: "-64px", left: "50%", transform: "translateX(-50%)" },
    path: "M12 60 C 14 44, 20 22, 38 4",
    dot: [38, 4],
  },
  bottom: {
    box: [48, 64],
    pos: { bottom: "-64px", left: "50%", transform: "translateX(-50%)" },
    path: "M12 4 C 14 20, 20 42, 38 60",
    dot: [38, 60],
  },
};

export default function MarginNote({ children, className = "", side = "left" }: MarginNoteProps) {
  const c = CONNECTORS[side];

  return (
    <div className={`pointer-events-none absolute z-40 text-paper-muted ${className}`}>
      <div className="relative">
        <p className="note-hand max-w-[12rem] text-[15px] leading-snug">{children}</p>

        <svg
          width={c.box[0]}
          height={c.box[1]}
          viewBox={`0 0 ${c.box[0]} ${c.box[1]}`}
          fill="none"
          className="absolute"
          style={{ ...c.pos, overflow: "visible" }}
          aria-hidden
        >
          <path d={c.path} stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" opacity="0.65" />
          <circle cx={c.dot[0]} cy={c.dot[1]} r="1.6" fill="currentColor" opacity="0.65" />
        </svg>
      </div>
    </div>
  );
}
