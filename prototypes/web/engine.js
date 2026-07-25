/* NextTrack demo — recommendation UI engine.

   Two data backends behind one interface:
     - LIVE: the page is served over http(s) (normally by the FastAPI app
       itself, see api.py /app mount) -> fetch /search and /recommend.
     - MOCK: opened from file:// or the API is unreachable -> built-in data,
       so the design demos anywhere with zero setup (and this is the honest
       fallback if the hosted API is asleep).
   The active mode is shown in the .mock-banner so nobody mistakes mock
   results for real ones.

   Spotify is playback only (open.spotify.com search links / embeds for the
   few mock tracks with hardcoded IDs). No Spotify data enters the model. */
(function (global) {
  "use strict";

  // ------------------------------------------------------------- mock data --
  const CATALOGUE = [
    { track_id: "41454059", artist: "Pink Floyd", title: "Time" },
    { track_id: "7741243",  artist: "Pink Floyd", title: "Breathe (In The Air)" },
    { track_id: "26878430", artist: "Pink Floyd", title: "Money" },
    { track_id: "43041154", artist: "Pink Floyd", title: "Us and Them" },
    { track_id: "7631440",  artist: "Pink Floyd", title: "Brain Damage" },
    { track_id: "13271747", artist: "Pink Floyd", title: "Eclipse" },
    { track_id: "r1", artist: "Radiohead", title: "Karma Police" },
    { track_id: "r2", artist: "Radiohead", title: "No Surprises" },
    { track_id: "r3", artist: "Radiohead", title: "Paranoid Android" },
    { track_id: "d1", artist: "Daft Punk", title: "One More Time" },
    { track_id: "d2", artist: "Daft Punk", title: "Get Lucky" },
    { track_id: "d3", artist: "Daft Punk", title: "Around the World" },
    { track_id: "u1", artist: "Dua Lipa", title: "Don't Start Now" },
    { track_id: "u2", artist: "Dua Lipa", title: "Levitating" },
    { track_id: "f1", artist: "Fleetwood Mac", title: "Dreams" },
    { track_id: "w1", artist: "The Weeknd", title: "Blinding Lights" },
  ];

  // Both Pink Floyd lists are REAL outputs from the trained model (25 Jul),
  // so the Diversity toggle demos the actual measured behaviour:
  //   off -> pure CF, 3 unique artists (the monoculture problem)
  //   on  -> hybrid re-ranker, 5 unique artists (NDCG cost -4.9%)
  const PF_CF = [
    { title: "On the Run", artist: "Pink Floyd", score: 0.927, why: "Because you listened to progressive rock, psychedelic rock and space rock", shared_tags: ["progressive rock","psychedelic rock","space rock"] },
    { title: "The Great Gig in the Sky", artist: "Pink Floyd", score: 0.902, why: "Because you listened to rock, psychedelic and classic rock", shared_tags: ["rock","psychedelic","classic rock"] },
    { title: "Speak to Me", artist: "Pink Floyd", score: 0.901, why: "Because you listened to progressive rock and psychedelic rock", shared_tags: ["progressive rock","psychedelic rock"] },
    { title: "Hey You", artist: "Pink Floyd", score: 0.803, why: "Because you listened to rock and progressive rock", shared_tags: ["rock","progressive rock"] },
    { title: "Comfortably Numb", artist: "Pink Floyd", score: 0.800, why: "Because you listened to rock, progressive rock and classic rock", shared_tags: ["rock","progressive rock","classic rock"] },
    { title: "Another Brick in the Wall, Pt. 2", artist: "Pink Floyd", score: 0.792, why: "Because you listened to rock and classic rock", shared_tags: ["rock","classic rock"] },
    { title: "Epitaph", artist: "King Crimson", score: 0.781, why: "Because you listened to progressive rock, classic rock and rock", shared_tags: ["progressive rock","classic rock","rock"] },
    { title: "Shine On You Crazy Diamond", artist: "Pink Floyd", score: 0.771, why: "Because you listened to rock, progressive rock and psychedelic", shared_tags: ["rock","progressive rock","psychedelic"] },
    { title: "Have a Cigar", artist: "Pink Floyd", score: 0.761, why: "Because you listened to rock, prog and 70s progressive rock", shared_tags: ["rock","prog","70s progressive rock"] },
    { title: "School", artist: "Supertramp", score: 0.754, why: "Because you listened to classic rock, rock and 70s", shared_tags: ["classic rock","rock","70s"] },
  ];
  const PF_RERANKED = [
    PF_CF[0], PF_CF[1], PF_CF[2],
    PF_CF[6], // King Crimson rises
    PF_CF[3], PF_CF[4],
    PF_CF[9], // Supertramp rises
    PF_CF[5],
    { title: "Going to California", artist: "Led Zeppelin", score: 0.748, why: "Because you listened to classic rock and rock", shared_tags: ["classic rock","rock"] },
    { title: "A Whiter Shade of Pale", artist: "Procol Harum", score: 0.743, why: "Because you listened to classic rock and progressive rock", shared_tags: ["classic rock","progressive rock"] },
  ];

  function mockRecs(session, rerank) {
    // The authored pair above belongs to the Pink Floyd demo session; other
    // mock sessions reuse the CF list so the page never renders empty.
    return (rerank ? PF_RERANKED : PF_CF).slice(0, 10);
  }
  function mockSearch(q, session) {
    const n = q.trim().toLowerCase();
    if (n.length < 2) return [];
    return CATALOGUE
      .filter(t => t.artist.toLowerCase().includes(n) || t.title.toLowerCase().includes(n))
      .filter(t => !session.some(s => s.track_id === t.track_id))
      .slice(0, 6);
  }

  // ------------------------------------------------------------- live API ---
  // Same-origin by default (the FastAPI app serves this page at /app);
  // override with window.NEXTTRACK_API for a split static-page + API deploy.
  const API_BASE = global.NEXTTRACK_API ||
    (location.protocol.startsWith("http") ? location.origin : null);

  async function apiSearch(q, session) {
    const r = await fetch(`${API_BASE}/search?q=${encodeURIComponent(q)}&limit=6`);
    if (!r.ok) throw new Error(`search ${r.status}`);
    const hits = await r.json();
    return hits.filter(t => !session.some(s => s.track_id === t.track_id));
  }
  async function apiRecommend(session, rerank) {
    const r = await fetch(`${API_BASE}/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        recent_track_ids: session.map(s => s.track_id),
        k: 10,
        params: { rerank },
      }),
    });
    if (!r.ok) throw new Error(`recommend ${r.status}`);
    return r.json(); // same shape as mock recs (no spotify_id -> search links)
  }

  // ------------------------------------------------------------- rendering --
  function esc(s) {
    return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }
  function spotifySearch(t) {
    return "https://open.spotify.com/search/" + encodeURIComponent(t.artist + " " + t.title);
  }
  function mount(opts) {
    opts = opts || {};
    const rank = opts.rank || (i => String(i + 1));
    const $ = id => document.getElementById(id);

    // Start in live mode whenever the page is served over http(s); drop to
    // mock (with a visible banner change) on the first failed request.
    let live = Boolean(API_BASE);
    const banner = document.querySelector(".mock-banner");
    function setBanner() {
      if (banner) banner.textContent = live
        ? "NextTrack · demo · live model"
        : "NextTrack · demo · built-in sample data (API unreachable)";
    }
    function fallback(err) {
      console.warn("NextTrack: falling back to mock data —", err);
      live = false;
      setBanner();
    }

    const session = CATALOGUE.slice(0, 5); // pre-seeded so the design shows on load

    function renderHits(list) {
      const el = $("hits");
      el.innerHTML = list.map(t =>
        `<li data-id="${t.track_id}" data-artist="${esc(t.artist)}" data-title="${esc(t.title)}"><span class="h-title">${esc(t.title)}</span><span class="h-artist">${esc(t.artist)}</span></li>`
      ).join("");
      el.classList.toggle("open", list.length > 0);
    }
    function renderSession() {
      $("session").innerHTML = session.map(t =>
        `<span class="chip">${esc(t.title)} <span class="c-artist">· ${esc(t.artist)}</span><button type="button" data-id="${t.track_id}" aria-label="remove ${esc(t.title)}">×</button></span>`
      ).join("");
      $("go").disabled = session.length < 1;
    }
    function renderRecs(recs, rerank) {
      $("results").hidden = false;
      const note = $("resnote");
      if (note) note.textContent = `${recs.length} tracks · ${rerank ? "diversity re-ranked" : "ranked by taste similarity"}`;
      $("recs").innerHTML = recs.map((r, i) => `
        <div class="rec" style="animation-delay:${i * 45}ms">
          <div class="rank">${rank(i)}</div>
          <div class="body">
            <div class="title">${esc(r.title)}</div>
            <div class="artist">${esc(r.artist)}</div>
            <div class="why">${esc(r.why || "")}</div>
            <div class="tags">${(r.shared_tags || []).map(t => `<span class="tag">${esc(t)}</span>`).join("")}</div>
          </div>
          <div class="aside">
            <div class="score">${Number(r.score).toFixed(3)}</div>
            <div class="meter"><i style="width:${Math.round(r.score * 100)}%"></i></div>
            <button type="button" class="play" data-artist="${esc(r.artist)}" data-title="${esc(r.title)}">▶ Play in place</button>
          </div>
          <div class="embed" hidden></div>
        </div>`).join("");
    }

    let searchSeq = 0;
    async function doSearch(q) {
      const seq = ++searchSeq;
      let list = [];
      if (live) {
        try { list = await apiSearch(q, session); }
        catch (e) { fallback(e); list = mockSearch(q, session); }
      } else {
        list = mockSearch(q, session);
      }
      if (seq === searchSeq) renderHits(list); // drop stale async responses
    }
    async function run() {
      const rerank = $("rerank") ? $("rerank").checked : true;
      const go = $("go");
      go.disabled = true;
      const prev = go.textContent;
      go.textContent = "…";
      try {
        let recs;
        if (live) {
          try { recs = await apiRecommend(session, rerank); }
          catch (e) { fallback(e); recs = mockRecs(session, rerank); }
        } else {
          recs = mockRecs(session, rerank);
        }
        renderRecs(recs, rerank);
      } finally {
        go.textContent = prev;
        go.disabled = session.length < 1;
      }
    }

    let searchTimer = null;
    $("q").addEventListener("input", e => {
      const q = e.target.value;
      clearTimeout(searchTimer);
      if (q.trim().length < 2) { renderHits([]); return; }
      // 150ms debounce: a burst of typing collapses to one request. The
      // searchSeq stale-response guard in doSearch handles any that overlap.
      searchTimer = setTimeout(() => doSearch(q), 150);
    });
    $("hits").addEventListener("click", e => {
      const li = e.target.closest("li"); if (!li) return;
      const t = { track_id: li.dataset.id, artist: li.dataset.artist, title: li.dataset.title };
      if (!session.some(s => s.track_id === t.track_id)) session.push(t);
      $("q").value = ""; renderHits([]); renderSession();
    });
    $("session").addEventListener("click", e => {
      const btn = e.target.closest("button"); if (!btn) return;
      const i = session.findIndex(s => s.track_id === btn.dataset.id);
      if (i > -1) session.splice(i, 1);
      renderSession();
    });
    $("recs").addEventListener("click", async e => {
      const play = e.target.closest(".play"); if (!play) return;
      // .embed is a sibling row of .aside (both children of .rec), so the score
      // and meter stay pinned right when the player opens.
      const box = play.closest(".rec").querySelector(".embed");
      if (!box.dataset.loaded) {
        const artist = play.dataset.artist, title = play.dataset.title;
        // Loading state: the /spotify lookup can take a second or two (retry on
        // flaky Spotify), so show a spinner and lock the button meanwhile.
        play.disabled = true;
        play.textContent = "⏳ Loading…";
        box.innerHTML = `<div class="embed-loading">Loading player…</div>`;
        box.hidden = false;
        let id = null;
        if (live) {
          try {
            const r = await fetch(`${API_BASE}/spotify?artist=${encodeURIComponent(artist)}&title=${encodeURIComponent(title)}`);
            if (r.ok) id = (await r.json()).spotify_id;
          } catch (err) { /* fall through to search link */ }
        }
        box.innerHTML = id
          ? `<iframe style="border-radius:10px" src="https://open.spotify.com/embed/track/${id}?utm_source=nextrack" width="100%" height="80" frameborder="0" allow="autoplay;encrypted-media" loading="lazy"></iframe>`
          : `<a class="spotify" href="${spotifySearch({ artist, title })}" target="_blank" rel="noopener">▶ Open in Spotify</a>`;
        box.dataset.loaded = "1";
        play.disabled = false;
        play.textContent = "▾ Hide player";
        return; // already shown
      }
      const show = box.hidden;
      box.hidden = !show;
      play.textContent = show ? "▾ Hide player" : "▶ Play in place";
    });
    $("go").addEventListener("click", () => {
      run().then(() => $("results").scrollIntoView({ behavior: "smooth", block: "start" }));
    });
    if ($("rerank")) $("rerank").addEventListener("change", () => { if (!$("results").hidden) run(); });
    document.addEventListener("click", e => {
      if (!e.target.closest(".search") && !e.target.closest(".hits")) $("hits").classList.remove("open");
    });

    setBanner();
    renderSession();
    run(); // show results immediately with the seeded session
  }

  global.NextTrackUI = { mount };
})(window);
