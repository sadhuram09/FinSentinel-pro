/**
 * AppHeader — the shared authenticated nav bar.
 *
 * Wordmark (left, optionally with a "back to dashboard" link), and on the right
 * a Settings link, the logged-in user's name, and Log out. Thin border, no
 * background fill so the page texture shows through. Used by Dashboard,
 * History, Settings and Analysis so the nav is identical across the app.
 */

import { Link, useNavigate } from "react-router-dom";

const API_BASE = "http://localhost:8000";

export interface AppUser {
  email: string;
  name: string | null;
}

interface AppHeaderProps {
  user: AppUser | null;
  showBack?: boolean;
}

export default function AppHeader({ user, showBack = false }: AppHeaderProps) {
  const navigate = useNavigate();

  async function handleLogout() {
    try {
      await fetch(`${API_BASE}/auth/logout`, { method: "POST", credentials: "include" });
    } finally {
      navigate("/login");
    }
  }

  const displayName = user?.name?.trim() || user?.email || "";

  return (
    <header className="flex items-center justify-between border-b border-ink px-6 py-4 sm:px-10">
      <div className="flex items-center gap-5">
        <Link to="/" className="font-display text-xl text-paper">
          FinSentinel Pro
        </Link>
        {showBack && (
          <Link to="/dashboard" className="font-mono text-xs text-paper-muted hover:text-paper">
            ← back to dashboard
          </Link>
        )}
      </div>
      <div className="flex items-center gap-5 font-sans text-sm text-paper-muted">
        <Link to="/settings" className="transition-colors hover:text-paper">
          Settings
        </Link>
        {displayName && <span>{displayName}</span>}
        <button type="button" onClick={handleLogout} className="transition-colors hover:text-paper">
          Log out
        </button>
      </div>
    </header>
  );
}
