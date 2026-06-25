/**
 * AnalysisScrollReveal — reveal-on-scroll wrapper.
 *
 * Children fade and slide up gently (opacity 0->1, y 40->0) once 30% of the
 * element is in view. The travel is deliberately short (40, not 80) so it feels
 * like reading calmly down a page, not content bouncing into place. Reusable
 * around any block; reveals once.
 */

import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import type { ReactNode } from "react";

interface AnalysisScrollRevealProps {
  children: ReactNode;
  className?: string;
  delay?: number;
}

export default function AnalysisScrollReveal({
  children,
  className = "",
  delay = 0,
}: AnalysisScrollRevealProps) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { amount: 0.3, once: true });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 40 }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 40 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1], delay }}
      className={className}
    >
      {children}
    </motion.div>
  );
}
