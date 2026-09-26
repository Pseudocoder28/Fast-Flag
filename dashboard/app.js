"use strict";

// Pit wall dashboard. No build step, no CDN. Connects to ws://<host>/stream,
// draws the track map + car dots on a canvas, lists rec envelopes as an alert feed.

const WS_URL = `ws://${location.host}/stream`;
const RESET_JUMP_S = 2.0; // must match engine.RESET_JUMP_S
const MARGIN_PX = 40;
const FEED_MAX = 50;
const DOT_RADIUS_PX = 5;

const FLAG_COLORS = {
  CLEAR: "#898781",
  YELLOW: "#fab219",
  DOUBLE_YELLOW: "#ec835a",
  VSC: "#3987e5",
  SC: "#d03b3b",
  RED: "#e66767",
};

const FLAG_WIDTH = {
  CLEAR: 2,
  YELLOW: 3,
  DOUBLE_YELLOW: 4,
  VSC: 4,
  SC: 5,
  RED: 6,
};

const DEFAULT_DOT_COLOR = "#898781";

// green -> yellow -> orange -> red, same steps as the design system's status ramp
const RISK_STOPS = [
  [0.0, [0x0c, 0xa3, 0x0c]],
  [0.33, [0xfa, 0xb2, 0x19]],
  [0.66, [0xec, 0x83, 0x5a]],
  [1.0, [0xd0, 0x3b, 0x3b]],
];

function riskColor(risk) {
  const r = Math.max(0, Math.min(1, risk));
  for (let i = 0; i < RISK_STOPS.length - 1; i++) {
    const [t0, c0] = RISK_STOPS[i];
    const [t1, c1] = RISK_STOPS[i + 1];
    if (r >= t0 && r <= t1) {
      const f = t1 === t0 ? 0 : (r - t0) / (t1 - t0);
      const mix = c0.map((v, i2) => Math.round(v + (c1[i2] - v) * f));
      return `rgb(${mix[0]},${mix[1]},${mix[2]})`;
    }
  }
  return DEFAULT_DOT_COLOR;
}

function dist(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return Math.sqrt(dx * dx + dy * dy);
}

function findSector(d, msectors) {
  for (const s of msectors) {
    if (s.start_dist <= s.end_dist) {
      if (d >= s.start_dist && d < s.end_dist) return s.id;
    } else {
      // sector wraps across the finish line (start_dist > end_dist)
      if (d >= s.start_dist || d < s.end_dist) return s.id;
    }
  }
  return null;
}

function buildSegments(track) {
  const pts = track.ref_line;
  const n = pts.length;
  const cum = new Array(n).fill(0);
  for (let i = 1; i < n; i++) cum[i] = cum[i - 1] + dist(pts[i - 1], pts[i]);
  const total = cum[n - 1] + dist(pts[n - 1], pts[0]);

  const segSector = new Array(n);
  for (let i = 0; i < n; i++) {
    const segStart = cum[i];
    const segEnd = i === n - 1 ? total : cum[i + 1];
    let mid = (segStart + segEnd) / 2;
    if (mid >= total) mid -= total;
    segSector[i] = findSector(mid, track.msectors);
  }
  return { pts, segSector, total };
}

function computeTransform(pts, canvasW, canvasH, marginPx) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const [x, y] of pts) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const w = maxX - minX || 1;
  const h = maxY - minY || 1;
  const availW = Math.max(canvasW - 2 * marginPx, 1);
  const availH = Math.max(canvasH - 2 * marginPx, 1);
  const scale = Math.min(availW / w, availH / h);
  const offsetX = (canvasW - w * scale) / 2;
  const offsetY = (canvasH - h * scale) / 2;
  return { minX, minY, scale, offsetX, offsetY, canvasH };
}

function toCanvas(t, x, y) {
  const cx = t.offsetX + (x - t.minX) * t.scale;
  const cy = t.canvasH - (t.offsetY + (y - t.minY) * t.scale); // track is Y-up, canvas is Y-down
  return [cx, cy];
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// --- state -------------------------------------------------------------

const canvas = document.getElementById("track");
const ctx = canvas.getContext("2d");
const feedEl = document.getElementById("feed");
const statusEl = document.getElementById("conn-status");

let segments = null;
let transform = null;
const cars = new Map(); // drv -> {x, y}
const risk = new Map(); // drv -> risk_30s
const sectorFlags = new Map(); // msector -> flag
let lastTickT = null;
let feedCount = 0;

// --- legend + alert feed (DOM, updated directly, not through the rAF loop) ---

function buildLegend() {
  const el = document.getElementById("legend");
  el.innerHTML = "";
  for (const flag of Object.keys(FLAG_COLORS)) {
    const row = document.createElement("div");
    row.className = "legend-row";
    const sw = document.createElement("span");
    sw.className = "legend-swatch";
    sw.style.background = FLAG_COLORS[flag];
    const label = document.createElement("span");
    label.textContent = flag.replace("_", " ");
    row.appendChild(sw);
    row.appendChild(label);
    el.appendChild(row);
  }
}

function pushFeedRow(rec) {
  const li = document.createElement("li");
  li.className = "feed-row";
  li.style.borderLeftColor = FLAG_COLORS[rec.flag] || FLAG_COLORS.CLEAR;
  const t = Number(rec.t).toFixed(2);
  const conf = Number(rec.confidence).toFixed(2);
  li.innerHTML =
    `<div class="feed-main">` +
    `<span class="feed-t">t=${t}</span> sector ${rec.msector} ` +
    `<span class="feed-flag">${escapeHtml(rec.flag)}</span> (${conf})<br>` +
    `<span class="feed-reason">${escapeHtml(rec.reason)}</span>` +
    `</div>`;
  feedEl.insertBefore(li, feedEl.firstChild);
  feedCount++;
  while (feedCount > FEED_MAX) {
    feedEl.removeChild(feedEl.lastChild);
    feedCount--;
  }
}

function clearFeed() {
  feedEl.innerHTML = "";
  feedCount = 0;
}

// --- canvas sizing -------------------------------------------------------

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (segments) {
    transform = computeTransform(segments.pts, rect.width, rect.height, MARGIN_PX);
  }
}
window.addEventListener("resize", resizeCanvas);

// --- draw loop, decoupled from websocket message rate ---------------------

function draw() {
  requestAnimationFrame(draw);
  if (!segments || !transform) return;

  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);

  const pts = segments.pts;
  const n = pts.length;
  for (let i = 0; i < n; i++) {
    const a = pts[i];
    const b = pts[(i + 1) % n];
    const flag = sectorFlags.get(segments.segSector[i]) || "CLEAR";
    ctx.strokeStyle = FLAG_COLORS[flag] || FLAG_COLORS.CLEAR;
    ctx.lineWidth = FLAG_WIDTH[flag] || FLAG_WIDTH.CLEAR;
    const [ax, ay] = toCanvas(transform, a[0], a[1]);
    const [bx, by] = toCanvas(transform, b[0], b[1]);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }

  for (const [drv, pos] of cars) {
    const [cx, cy] = toCanvas(transform, pos.x, pos.y);
    const r = risk.get(drv);
    ctx.beginPath();
    ctx.arc(cx, cy, DOT_RADIUS_PX, 0, Math.PI * 2);
    ctx.fillStyle = r === undefined ? DEFAULT_DOT_COLOR : riskColor(r);
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#0d0d0d";
    ctx.stroke();
  }
}

// --- websocket -----------------------------------------------------------

function handleTick(tick) {
  if (lastTickT !== null && tick.t < lastTickT - RESET_JUMP_S) {
    // matches engine.RaceControl's own reset rule: the engine emits no
    // synthetic CLEAR recs on reset, so the dashboard clears itself here
    sectorFlags.clear();
    clearFeed();
  }
  lastTickT = tick.t;
  for (const car of tick.cars) {
    cars.set(car.drv, { x: car.x, y: car.y });
  }
}

function connect() {
  const ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    statusEl.textContent = "connected";
  };

  ws.onclose = () => {
    statusEl.textContent = "disconnected, retrying...";
    setTimeout(connect, 2000);
  };

  ws.onerror = () => {
    ws.close();
  };

  ws.onmessage = (event) => {
    let env;
    try {
      env = JSON.parse(event.data);
    } catch (e) {
      return;
    }
    if (!env || typeof env !== "object") return;
    const data = env.data;
    if (!data || typeof data !== "object") return;

    if (env.kind === "tick") {
      handleTick(data);
    } else if (env.kind === "risk") {
      risk.set(data.drv, data.risk_30s);
    } else if (env.kind === "rec") {
      sectorFlags.set(data.msector, data.flag);
      pushFeedRow(data);
    }
    // detection, official: not used by this view
  };
}

// --- init ------------------------------------------------------------------

async function init() {
  buildLegend();
  resizeCanvas();
  const res = await fetch("/track");
  const track = await res.json();
  segments = buildSegments(track);
  resizeCanvas();
  connect();
  requestAnimationFrame(draw);
}

init();
