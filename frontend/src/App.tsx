/**
 * App shell: global reading-room background + client-side routing.
 * "/" renders the Landing page; further routes (e.g. /signup) come later.
 */

import { BrowserRouter, Route, Routes } from "react-router-dom";

import DocumentTexture from "./components/ui/DocumentTexture";
import Landing from "./pages/Landing";

function App() {
  return (
    <BrowserRouter>
      <DocumentTexture />
      <Routes>
        <Route path="/" element={<Landing />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
