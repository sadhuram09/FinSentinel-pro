/**
 * Settings — read-only account view (/settings).
 *
 * Displays the logged-in user's profile from GET /auth/me on a paper-stock
 * panel (the Auth page's form-panel styling). Editing endpoints don't exist on
 * the backend yet, so fields are display-only and a muted "coming soon" note
 * stands in for edit/change-password rather than faking functionality.
 */

import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import AppHeader, { type AppUser } from "../components/ui/AppHeader";
import ShaderButton from "../components/ui/ShaderButton";

const API_BASE = "http://localhost:8000";

interface MeResponse extends AppUser {
  created_at?: string;
}

function fmtDate(iso?: string): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? iso
    : d.toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });
}

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="border-b border-black/10 py-4 last:border-b-0">
      <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink/45">{label}</p>
      <p className="mt-1 font-display text-lg text-ink">{value}</p>
    </div>
  );
}

export default function Settings() {
  const navigate = useNavigate();
  const [user, setUser] = useState<MeResponse | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/auth/me`, { credentials: "include" });
        if (!active) return;
        if (res.status === 401) return navigate("/login");
        setUser((await res.json()) as MeResponse);
        setReady(true);
      } catch {
        if (active) navigate("/login");
      }
    })();
    return () => {
      active = false;
    };
  }, [navigate]);

  async function handleLogout() {
    try {
      await fetch(`${API_BASE}/auth/logout`, { method: "POST", credentials: "include" });
    } finally {
      navigate("/login");
    }
  }

  if (!ready || !user) {
    return (
      <div className="flex min-h-screen items-center justify-center font-mono text-sm text-paper-muted">
        loading...
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      <AppHeader user={user} showBack />

      <main className="mx-auto max-w-md px-6 pb-24 pt-20 sm:px-0">
        <h1 className="mb-8 font-display text-section text-paper">Account</h1>

        <div
          style={{ transform: "rotate(-0.5deg)" }}
          className="shadow-card rounded-[4px] border border-black/15 bg-paper px-7 py-6 text-ink"
        >
          <Field label="name" value={user.name?.trim() || "—"} />
          <Field label="email" value={user.email} />
          <Field label="member since" value={fmtDate(user.created_at)} />

          <p className="mt-5 font-mono text-[11px] text-ink/35">Edit profile (coming soon)</p>
        </div>

        <div className="mt-8">
          <ShaderButton type="button" onClick={handleLogout}>
            Log out
          </ShaderButton>
        </div>
      </main>
    </div>
  );
}
