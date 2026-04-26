/**
 * Layer 3 — Multi-page SSVEP-BCI communicator
 *
 * Two stimulus frequencies drive everything:
 *   6 Hz  →  SELECT  (confirm current option / execute left action)
 *   15 Hz  →  NEXT    (advance selector / execute right action)
 *
 * Pages
 * ─────
 *  home        5-option selector (Wheelchair / Food / Water / Caregiver / Spell)
 *  wheelchair  6 Hz = backward  ·  15 Hz = forward
 *  food        6 Hz = ← home    ·  15 Hz = "I need food"  (TTS)
 *  water       6 Hz = ← home    ·  15 Hz = "I need water" (TTS)
 *  caregiver   6 Hz = ← home    ·  15 Hz = "I need a caregiver" (TTS)
 *  letters     6 Hz = SELECT letter / backspace / done
 *              15 Hz = NEXT letter  (A → Z → ⌫ → ✓ Done → A)
 *
 * Confirmation: the Python bridge counts 5 consecutive SELECT detections at
 * the same frequency and sends a WS "confirmed" event.  The JS reacts to that
 * event (not to individual SELECT messages) so timing is relaxed and accurate.
 */

(function () {
  "use strict";

  // ── Configuration ────────────────────────────────────────────────────────
  const STREAK_REQUIRED  = 5;
  const TTS_COOLDOWN_MS  = 3000;

  // Alphabet + two special items at the end
  const ALPHABET    = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
  const LETTER_ITEMS = [...ALPHABET, "⌫", "✓ Done"];

  // Home-page option order (matches data-idx in HTML)
  const HOME_PAGES = ["wheelchair", "food", "water", "caregiver", "letters"];

  // ── DOM refs ─────────────────────────────────────────────────────────────
  const btn6      = document.getElementById("btn-6");
  const btn15     = document.getElementById("btn-15");
  const label6    = document.getElementById("label-6");
  const label15   = document.getElementById("label-15");
  const badge6    = document.getElementById("streak-6");
  const badge15   = document.getElementById("streak-15");
  const statusBar = document.getElementById("status-bar");
  const statusTxt = document.getElementById("status-text");
  const confirmEl = document.getElementById("confirm-line");
  const dot       = document.getElementById("dot");

  // Home page
  const optionEls = Array.from(document.querySelectorAll(".option"));

  // Letters page
  const letterDisplay = document.getElementById("letter-display");
  const letterBuffer  = document.getElementById("letter-buffer");

  // ── State ────────────────────────────────────────────────────────────────
  let currentPage   = "home";
  let homeIndex     = 0;          // 0 – 4
  let letterIndex   = 0;          // 0 – LETTER_ITEMS.length - 1
  let spelledWord   = [];         // accumulated letters
  let streak6Count = 0;
  let streak15Count = 0;
  let lastStreakKey  = null;
  let ttsBusy       = false;
  let ttsCooldownMs = 0;
  let activeAudio   = null;

  // ── Per-page button labels & colours ────────────────────────────────────
  const PAGE_CFG = {
    home: {
      l6: "SELECT ✓",     l15: "NEXT →",
      c6: "green",        c15: "blue",
    },
    wheelchair: {
      l6: "◀ BACKWARD",  l15: "FORWARD ▶",
      c6: "orange",       c15: "orange",
    },
    food: {
      l6: "← HOME",      l15: "🍽️  I NEED FOOD",
      c6: "grey",         c15: "amber",
    },
    water: {
      l6: "← HOME",      l15: "💧  I NEED WATER",
      c6: "grey",         c15: "blue",
    },
    caregiver: {
      l6: "← HOME",      l15: "🏥  CALL CAREGIVER",
      c6: "grey",         c15: "pink",
    },
    letters: {
      l6: "SELECT ✓",    l15: "NEXT →",
      c6: "green",        c15: "blue",
    },
  };

  // CSS colour classes that can appear on .btn-tile
  const ALL_COLORS = ["green", "blue", "orange", "amber", "grey", "pink"];

  // ── Page navigation ──────────────────────────────────────────────────────
  function showPage(page) {
    document.querySelectorAll(".page").forEach(function (el) {
      el.classList.remove("active");
    });
    document.getElementById("page-" + page).classList.add("active");
    currentPage = page;
    resetStreak();
    updateButtons();

    // Reset letter state when entering
    if (page === "letters") {
      letterIndex  = 0;
      renderLetters();
    }
    // Re-render home highlight when returning
    if (page === "home") {
      renderHome();
    }
  }

  // ── Render helpers ───────────────────────────────────────────────────────
  function renderHome() {
    optionEls.forEach(function (el, idx) {
      el.classList.toggle("highlighted", idx === homeIndex);
    });
  }

  function renderLetters() {
    letterDisplay.textContent = LETTER_ITEMS[letterIndex];
    letterBuffer.textContent  =
      spelledWord.length > 0 ? spelledWord.join("") : "\u00a0";
  }

  function updateButtons() {
    var cfg = PAGE_CFG[currentPage] || PAGE_CFG.home;

    label6.textContent = cfg.l6;
    label15.textContent = cfg.l15;

    // Swap colour class on btn-6
    ALL_COLORS.forEach(function (c) { btn6.classList.remove(c); });
    btn6.classList.add(cfg.c6);

    // Swap colour class on btn-15
    ALL_COLORS.forEach(function (c) { btn15.classList.remove(c); });
    btn15.classList.add(cfg.c15);
  }

  // ── Action handlers ──────────────────────────────────────────────────────
  function executeSelect() {
    switch (currentPage) {
      case "home":
        showPage(HOME_PAGES[homeIndex]);
        break;
      case "wheelchair":
        speakText("Move backward");
        break;
      case "food":
        showPage("home");
        break;
      case "water":
        showPage("home");
        break;
      case "caregiver":
        showPage("home");
        break;
      case "letters":
        selectLetter();
        break;
    }
  }

  function executeNext() {
    switch (currentPage) {
      case "home":
        homeIndex = (homeIndex + 1) % HOME_PAGES.length;
        renderHome();
        confirmEl.textContent = "";
        break;
      case "wheelchair":
        speakText("Move forward");
        break;
      case "food":
        speakText("I need food, please.");
        break;
      case "water":
        speakText("I need water, please.");
        break;
      case "caregiver":
        speakText("I need a caregiver, please.");
        break;
      case "letters":
        letterIndex = (letterIndex + 1) % LETTER_ITEMS.length;
        renderLetters();
        confirmEl.textContent = "";
        break;
    }
  }

  function selectLetter() {
    var item = LETTER_ITEMS[letterIndex];
    if (item === "⌫") {
      // Backspace
      spelledWord.pop();
      renderLetters();
      speakText("Backspace");
    } else if (item === "✓ Done") {
      // Speak the whole word and return home
      var word = spelledWord.join("");
      if (word) {
        speakText(word);
      }
      spelledWord  = [];
      letterIndex  = 0;
      renderLetters();
      // Short delay so TTS can start before page switches
      setTimeout(function () { showPage("home"); }, 1200);
    } else {
      // Regular letter
      spelledWord.push(item);
      renderLetters();
      speakText(item);
    }
  }

  // ── TTS via /api/speak-text ──────────────────────────────────────────────
  function speakText(text) {
    var now = Date.now();
    if (ttsBusy || now < ttsCooldownMs) {
      var rem = Math.max(0, ttsCooldownMs - now);
      confirmEl.textContent = "Cooldown: " + (rem / 1000).toFixed(1) + " s";
      return;
    }
    ttsBusy       = true;
    ttsCooldownMs = now + TTS_COOLDOWN_MS;
    statusBar.classList.add("speaking");
    confirmEl.textContent = "Speaking: " + text;

    fetch("/api/speak-text", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text: text }),
    })
      .then(function (resp) {
        if (!resp.ok) {
          return resp.text().then(function (t) { throw new Error(t || resp.statusText); });
        }
        return resp.blob();
      })
      .then(function (blob) {
        var url   = URL.createObjectURL(blob);
        if (activeAudio) {
          activeAudio.pause();
          activeAudio = null;
        }
        var audio  = new Audio(url);
        activeAudio = audio;

        function finish() {
          URL.revokeObjectURL(url);
          if (activeAudio === audio) { activeAudio = null; }
          ttsBusy = false;
          statusBar.classList.remove("speaking");
          confirmEl.textContent = "✓ Said: " + text;
        }

        audio.addEventListener("ended", finish);
        audio.addEventListener("error", finish);
        audio.play().catch(finish);
      })
      .catch(function () {
        ttsBusy = false;
        statusBar.classList.remove("speaking");
        confirmEl.textContent = "TTS unavailable — check ELEVENLABS_API_KEY";
      });
  }

  // ── Streak display (client-side progress tracking) ───────────────────────
  function resetStreak() {
    streak6Count = 0;
    streak15Count = 0;
    lastStreakKey  = null;
    badge6.textContent = "0 / " + STREAK_REQUIRED;
    badge15.textContent = "0 / " + STREAK_REQUIRED;
    badge6.classList.remove("active");
    badge15.classList.remove("active");
  }

  function feedStreak(key) {
    // If frequency changes, reset the OTHER button's counter
    if (lastStreakKey !== null && lastStreakKey !== key) {
      if (key === "6.0") {
        streak15Count = 0;
        badge15.textContent = "0 / " + STREAK_REQUIRED;
        badge15.classList.remove("active");
      } else {
        streak6Count = 0;
        badge6.textContent = "0 / " + STREAK_REQUIRED;
        badge6.classList.remove("active");
      }
    }
    lastStreakKey = key;

    if (key === "6.0") {
      streak6Count = Math.min(streak6Count + 1, STREAK_REQUIRED);
      badge6.textContent = streak6Count + " / " + STREAK_REQUIRED;
      badge6.classList.add("active");
      confirmEl.textContent =
        label6.textContent + "  " + streak6Count + " / " + STREAK_REQUIRED;
    } else {
      streak15Count = Math.min(streak15Count + 1, STREAK_REQUIRED);
      badge15.textContent = streak15Count + " / " + STREAK_REQUIRED;
      badge15.classList.add("active");
      confirmEl.textContent =
        label15.textContent + "  " + streak15Count + " / " + STREAK_REQUIRED;
    }
  }

  function flashConfirmed(btn) {
    btn.classList.add("confirmed");
    setTimeout(function () { btn.classList.remove("confirmed"); }, 700);
  }

  // ── WebSocket payload handler ────────────────────────────────────────────
  function onPayload(payload) {
    // Per-SELECT updates: update streak display
    if (payload.command === "SELECT") {
      var key = Number(payload.frequency).toFixed(1);
      if (key === "6.0" || key === "15.0") {
        feedStreak(key);
      }
      return;
    }

    // Confirmed event: fire the action for the current page
    if (payload.type === "confirmed") {
      var fkey = Number(payload.frequency_hz).toFixed(1);
      resetStreak();
      if (fkey === "6.0") {
        flashConfirmed(btn6);
        executeSelect();
      } else if (fkey === "15.0") {
        flashConfirmed(btn15);
        executeNext();
      }
      return;
    }
  }

  // ── Flicker animation (requestAnimationFrame) ────────────────────────────
  var flickerTargets = [
    { el: btn6, freq: 6.0 },
    { el: btn15, freq: 15.0 },
  ];

  function flickerTick(timestamp) {
    var t = timestamp / 1000;
    for (var i = 0; i < flickerTargets.length; i++) {
      var target = flickerTargets[i];
      var on = Math.floor(t * target.freq * 2) % 2 === 0;
      target.el.classList.toggle("on", on);
    }
    requestAnimationFrame(flickerTick);
  }

  // ── WebSocket connection ─────────────────────────────────────────────────
  var ws             = null;
  var reconnectDelay = 500;

  function connectWs() {
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(proto + "//" + location.host + "/ws");

    ws.onopen = function () {
      dot.classList.add("live");
      statusTxt.textContent = "Connected";
      reconnectDelay = 500;
    };

    ws.onmessage = function (ev) {
      try { onPayload(JSON.parse(ev.data)); } catch (_) {}
    };

    ws.onclose = function () {
      dot.classList.remove("live");
      statusTxt.textContent = "Reconnecting…";
      setTimeout(function () {
        reconnectDelay = Math.min(reconnectDelay * 2, 5000);
        connectWs();
      }, reconnectDelay);
    };

    ws.onerror = function () { ws.close(); };
  }

  // ── Keyboard shortcut (dev/demo: press 1 = 6Hz confirm, 2 = 15Hz confirm)
  document.addEventListener("keydown", function (ev) {
    if (ev.key === "1") { flashConfirmed(btn6); executeSelect(); }
    if (ev.key === "2") { flashConfirmed(btn15); executeNext(); }
  });

  // ── Bootstrap ────────────────────────────────────────────────────────────
  showPage("home");
  renderHome();
  renderLetters();
  requestAnimationFrame(flickerTick);
  connectWs();

})();
