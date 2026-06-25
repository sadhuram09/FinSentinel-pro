/**
 * App shell: global reading-room background + client-side routing.
 * "/" renders the Landing page; further routes (e.g. /signup) come later.
 */

import { BrowserRouter, Route, Routes } from "react-router-dom";

import DocumentTexture from "./components/ui/DocumentTexture";
import Analysis from "./pages/Analysis";
import Auth from "./pages/Auth";
import Dashboard from "./pages/Dashboard";
import History from "./pages/History";
import Landing from "./pages/Landing";
import Settings from "./pages/Settings";

function App() {
  return (
    <BrowserRouter>
      <DocumentTexture />
      <Routes>
        <Route path="/" element={<Landing />} />
        {/* Both render Auth; the tab defaults from the path. */}
        <Route path="/login" element={<Auth />} />
        <Route path="/signup" element={<Auth />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/history" element={<History />} />
        <Route path="/analysis/:id" element={<Analysis />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
