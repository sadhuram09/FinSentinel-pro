/**
 * Auth — a single page handling both sign-in and create-account via a tab
 * toggle (no separate components). The tab is driven by the route (/login vs
 * /signup) so the URL stays meaningful; switching clears fields + errors.
 *
 * The form sits on a paper-stock panel (the EvidenceCard visual language —
 * cream, soft directional shadow, a barely-there -0.5deg tilt — minus the
 * dog-ear, since this is a functional form). Errors surface as small italic
 * MarginNotes beneath the relevant field, connected by the hand-drawn line,
 * rather than as a banner or toast.
 */

import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import MarginNote from "../components/ui/MarginNote";
import ShaderButton from "../components/ui/ShaderButton";

const API_BASE = "http://localhost:8000";

type Mode = "signin" | "signup";
type FieldError = { field: "name" | "email" | "password"; message: string };

export default function Auth() {
  const location = useLocation();
  const navigate = useNavigate();
  const mode: Mode = location.pathname === "/signup" ? "signup" : "signin";

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<FieldError | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // Switching tabs (i.e. the route changing) clears all form + error state.
  useEffect(() => {
    setName("");
    setEmail("");
    setPassword("");
    setError(null);
  }, [mode]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    if (mode === "signup" && password.length < 8) {
      setError({ field: "password", message: "Password must be at least 8 characters." });
      return;
    }

    setSubmitting(true);
    try {
      const path = mode === "signup" ? "/auth/signup" : "/auth/login";
      const body =
        mode === "signup" ? { email, password, name } : { email, password };

      const res = await fetch(`${API_BASE}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include", // send/receive the httpOnly session cookie
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        let detail = "Something went wrong. Please try again.";
        try {
          const data = await res.json();
          if (typeof data.detail === "string") detail = data.detail;
        } catch {
          /* non-JSON error body */
        }
        // Attach the message to the most relevant field.
        const field: FieldError["field"] = res.status === 409 ? "email" : "password";
        setError({ field, message: detail });
        return;
      }

      navigate("/dashboard");
    } catch {
      setError({ field: "password", message: "Network error — is the backend running?" });
    } finally {
      setSubmitting(false);
    }
  }

  const tab = (target: Mode, label: string) => {
    const active = mode === target;
    return (
      <button
        type="button"
        onClick={() => navigate(target === "signup" ? "/signup" : "/login")}
        className={`pb-1.5 font-sans text-sm transition-colors ${
          active
            ? "border-b-2 border-ink font-semibold text-ink"
            : "border-b-2 border-transparent text-ink/45 hover:text-ink/70"
        }`}
      >
        {label}
      </button>
    );
  };

  const field = (
    key: FieldError["field"],
    label: string,
    input: React.ReactNode,
    hint?: string,
  ) => {
    const hasError = error?.field === key;
    return (
      <div className={`relative ${hasError ? "mb-28" : "mb-5"}`}>
        <label className="mb-1.5 block font-sans text-[13px] font-medium text-ink/70">{label}</label>
        {input}
        {hint && <p className="mt-1.5 font-mono text-[11px] text-ink/45">{hint}</p>}
        {hasError && (
          <MarginNote className="left-3 top-[calc(100%+2.5rem)]" side="top" tone="error">
            {error!.message}
          </MarginNote>
        )}
      </div>
    );
  };

  const inputClass =
    "w-full rounded-[4px] border border-black/15 bg-white/60 px-3 py-2 font-sans text-[15px] text-ink " +
    "outline-none transition-colors focus:border-ink/40 focus:bg-white/80";

  return (
    <main className="flex min-h-screen items-center justify-center px-6 py-16">
        <div
          style={{ transform: "rotate(-0.5deg)" }}
          className="shadow-card w-full max-w-md rounded-[4px] border border-black/15 bg-paper px-7 py-8 text-ink"
        >
          {/* Toggle */}
          <div className="mb-7 flex items-center gap-6 border-b border-black/10">
            {tab("signin", "Sign in")}
            {tab("signup", "Create account")}
          </div>

          <form onSubmit={handleSubmit} noValidate>
            {mode === "signup" &&
              field(
                "name",
                "Name",
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  autoComplete="name"
                  className={inputClass}
                />,
              )}

            {field(
              "email",
              "Email",
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
                required
                className={inputClass}
              />,
            )}

            {field(
              "password",
              "Password",
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete={mode === "signup" ? "new-password" : "current-password"}
                required
                className={inputClass}
              />,
              mode === "signup" ? "minimum 8 characters" : undefined,
            )}

            <div className="mt-2 flex items-center gap-4">
              <ShaderButton type="submit" disabled={submitting}>
                {mode === "signup" ? "Create account" : "Sign in"}
              </ShaderButton>
              {submitting && (
                <span className="font-mono text-[12px] text-ink/50">checking...</span>
              )}
            </div>
          </form>
        </div>
    </main>
  );
}
