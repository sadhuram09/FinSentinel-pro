/**
 * App shell: global reading-room background + client-side routing.
 * "/" renders the Landing page; further routes (e.g. /signup) come later.
 */

import { BrowserRouter, Route, Routes } from "react-router-dom";

import DocumentTexture from "./components/ui/DocumentTexture";
import Auth from "./pages/Auth";
import Landing from "./pages/Landing";

function App() {
  return (
    <BrowserRouter>
      <DocumentTexture />
      <Routes>
        <Route path="/" element={<Landing />} />
        {/* Both render Auth; the tab defaults from the path. */}
        <Route path="/login" element={<Auth />} />
        <Route path="/signup" element={<Auth />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
