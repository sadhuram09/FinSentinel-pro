/**
 * ShaderButton — the primary call-to-action.
 *
 * Tactile, paper-physical rather than neon: a filled cream (primary) or
 * outlined (secondary) control that lifts slightly on hover and presses on
 * click, with a faint grain "sheen" overlay for surface texture. Stays on the
 * editorial brief — no gradients, no glow.
 *
 * Polymorphic: pass `to` for an in-app route (renders a router Link) or `href`
 * for an external destination (renders an anchor with safe rel/target).
 */

import type { ReactNode } from "react";
import { Link } from "react-router-dom";

type Variant = "primary" | "secondary";

interface ShaderButtonProps {
  children: ReactNode;
  to?: string;
  href?: string;
  variant?: Variant;
  className?: string;
}

const GRAIN =
  "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")";

const VARIANTS: Record<Variant, string> = {
  primary: "bg-paper text-ink border-transparent hover:shadow-card",
  secondary: "bg-transparent text-paper border-paper/30 hover:border-paper/60",
};

export default function ShaderButton({
  children,
  to,
  href,
  variant = "primary",
  className = "",
}: ShaderButtonProps) {
  const classes =
    "group relative inline-flex select-none items-center justify-center overflow-hidden rounded-[4px] border px-6 py-3 font-sans text-sm font-medium tracking-wide " +
    "transition-all duration-200 ease-out hover:-translate-y-0.5 active:translate-y-0 active:scale-[0.98] " +
    `${VARIANTS[variant]} ${className}`;

  const inner = (
    <>
      {/* Faint grain sheen, a touch stronger on hover — the "shader" texture. */}
      <span
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.04] transition-opacity duration-200 group-hover:opacity-[0.08]"
        style={{ backgroundImage: GRAIN, backgroundSize: "120px 120px", mixBlendMode: "overlay" }}
      />
      <span className="relative">{children}</span>
    </>
  );

  if (href) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer" className={classes}>
        {inner}
      </a>
    );
  }

  return (
    <Link to={to ?? "#"} className={classes}>
      {inner}
    </Link>
  );
}
