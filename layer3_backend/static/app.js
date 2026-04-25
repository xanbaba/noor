/**
 * Layer 4 MVP — SSVEP Flicker Frontend
 *
 * Two responsibilities:
 *   1. Drive frame-synchronised square-wave flickering at the configured
 *      stimulus frequencies (using requestAnimationFrame + wall-clock time).
 *   2. Connect to the Layer 3 WebSocket, display every received SELECT
 *      payload in the status panel, and highlight the matched tile.
 */

(function () {
  "use strict";

  // ── Tile references ──────────────────────────────────────────────────
  const tiles = [
    { freq: 12, el: document.getElementById("tile-12") },
    { freq: 15, el: document.getElementById("tile-15") },
  ];

  // ── rAF flicker loop ─────────────────────────────────────────────────
  // Uses the high-resolution timestamp from rAF to compute a 50%-duty
  // square wave.  Because rAF only fires on display vsync, the phase
  // self-corrects after any dropped frame (no drift).

  function tick(timestamp) {
    const t = timestamp / 1000; // seconds
    for (const tile of tiles) {
      // half-period count — toggles every 1/(2·freq) seconds
      const on = Math.floor(t * tile.freq * 2) % 2 === 0;
      tile.el.classList.toggle("on", on);
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  // ── Status panel ─────────────────────────────────────────────────────
  const statusEl = document.getElementById("status");
  const detectedFreqEl = document.getElementById("detected-freq");
  const snrEl = document.getElementById("snr-value");
  const confEl = document.getElementById("conf-value");
  const epochEl = document.getElementById("epoch-value");

  function updateStatus(payload) {
    detectedFreqEl.textContent = payload.frequency.toFixed(1) + " Hz";
    snrEl.textContent = payload.snr_db.toFixed(1) + " dB";
    confEl.textContent = (payload.confidence * 100).toFixed(0) + "%";
    epochEl.textContent = payload.epoch_ms + " ms";

    // Highlight the matching tile for 400 ms
    const matchId = "tile-" + payload.frequency;
    const matchEl = document.getElementById(matchId);
    if (matchEl) {
      matchEl.classList.add("selected");
      setTimeout(() => matchEl.classList.remove("selected"), 400);
    }
  }

  // ── WebSocket ────────────────────────────────────────────────────────
  let ws = null;
  let reconnectDelay = 500;

  function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(proto + "//" + location.host + "/ws");

    ws.onopen = function () {
      statusEl.classList.add("connected");
      reconnectDelay = 500;
    };

    ws.onmessage = function (ev) {
      try {
        const payload = JSON.parse(ev.data);
        updateStatus(payload);
      } catch (_) {
        // ignore non-JSON frames
      }
    };

    ws.onclose = function () {
      statusEl.classList.remove("connected");
      scheduleReconnect();
    };

    ws.onerror = function () {
      ws.close();
    };
  }

  function scheduleReconnect() {
    setTimeout(function () {
      reconnectDelay = Math.min(reconnectDelay * 2, 5000);
      connect();
    }, reconnectDelay);
  }

  connect();
})();
