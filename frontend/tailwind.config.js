/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Surfaces. Use as bg-ink / bg-paper / text-paper / text-paper-muted.
        ink: "#0D0E10", // near-black, warm
        paper: {
          DEFAULT: "#F2EEE4", // warm cream
          muted: "#B8B2A3", // muted cream for secondary text
        },
        // Verdict semantics — used everywhere a confidence/verdict appears.
        verdict: {
          approved: "#3DDC84",
          conflicted: "#E8A33D",
          rejected: "#E5484D",
        },
      },
      // border-ink is intentionally a faint paper line on the dark ground, not
      // the solid ink colour — so it overrides the colour-derived border-ink.
      borderColor: {
        ink: "rgba(242,238,228,0.12)",
      },
      fontFamily: {
        display: ['Fraunces', 'ui-serif', 'Georgia', 'serif'],
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      // Sizes set font-size/line-height/spacing/weight; pair with `font-display`
      // for the Fraunces family.
      fontSize: {
        hero: ['clamp(40px,7vw,84px)', { lineHeight: '1.0', letterSpacing: '-0.02em', fontWeight: '500' }],
        section: ['clamp(28px,4vw,44px)', { lineHeight: '1.1', letterSpacing: '-0.015em', fontWeight: '500' }],
      },
    },
  },
  plugins: [],
};
