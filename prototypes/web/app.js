/* NextTrack demo — page bootstrap.
   Two responsibilities only: (1) start the recommendation UI (engine.js owns
   all the data + rendering), (2) manage the dark/light theme toggle. Keeping
   this thin is deliberate — page concerns here, reusable UI logic in engine.js.
   NOTE: the initial theme is applied pre-paint by the inline <head> script in
   index.html; this file only keeps the label in sync and handles the toggle. */

(function () {
  "use strict";

  // ---- 1. recommendation UI (data + search + results live in engine.js) ----
  NextTrackUI.mount({ rank: i => String(i + 1).padStart(2, "0") });

  // ---- 2. theme toggle -----------------------------------------------------
  const THEMES = { dark: "Dark", light: "Light" };
  const root = document.documentElement;
  const btn = document.getElementById("theme");
  const label = document.getElementById("theme-label");

  function setTheme(name) {
    root.setAttribute("data-theme", name);
    label.textContent = THEMES[name];
    try { localStorage.setItem("nexttrack-theme", name); } catch (e) {}
  }

  // The <head> script already applied saved-or-OS theme; just sync the label.
  label.textContent = THEMES[root.getAttribute("data-theme")] || THEMES.dark;

  btn.addEventListener("click", () => {
    const next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
    setTheme(next);
  });
})();

// --- header stats -----------------------------------------------------------
// Keep the stat chips honest: read catalogue size and tag coverage from the
// live /health endpoint so the page always describes the model actually
// serving it. The hardcoded defaults match the built-in mock lists, so mock
// mode (file:// or API down) stays self-consistent without any request.
(async () => {
  const base = window.NEXTTRACK_API ||
    (location.protocol.startsWith("http") ? location.origin : null);
  if (!base) return;
  try {
    const r = await fetch(`${base}/health`);
    if (!r.ok) return;
    const h = await r.json();
    if (h.catalogue_tracks) {
      const el = document.getElementById("stat-tracks");
      if (el) el.textContent = h.catalogue_tracks.toLocaleString("en-US");
    }
    if (h.catalogue_tracks && h.tagged_tracks) {
      const el = document.getElementById("stat-tagged");
      if (el) el.textContent =
        ((h.tagged_tracks / h.catalogue_tracks) * 100).toFixed(1) + "%";
    }
  } catch (e) {} // mock mode: defaults stand
})();
