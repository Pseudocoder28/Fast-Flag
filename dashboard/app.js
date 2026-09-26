"use strict";

// Pit wall dashboard. No build step, no CDN. Connects to ws://<host>/stream, draws the
// track map and the cars on a canvas (only inside the requestAnimationFrame loop) and
// keeps a feed of detections, recs and official race control messages (DOM, updated
// only when one of those envelopes arrives).

const WS_URL = `ws://${location.host}/stream`;
const RESET_JUMP_S = 2.0;      // same rule as engine.RESET_JUMP_S: a bigger jump is a seek or loop
const MARGIN_PX = 40;
const FEED_MAX = 50;

// car movement: ticks come every 0.25 s of replay time. At 1x that is one tick every
// ~256 ms of wall time, at 10x the server sends bursts of 2 ticks every ~57 ms, at 50x
// bursts of 10 ticks every ~74 ms. FastF1 position samples are irregular, so about 1 tick
// in 6 repeats a moving car's previous position and the next one jumps twice as far:
// gliding from the previous to the latest tick would stutter. Each car keeps its recent
// distinct positions with their replay times instead, and every frame is drawn at a
// display clock that runs one burst plus LAG_EXTRA_S behind the newest tick, advancing
// at the measured rate of replay time per wall-clock ms.
const BURST_MS = 8;            // ticks closer together than this came in one server step
const LAG_EXTRA_S = 0.3;       // display lag beyond one burst: covers one repeated sample (1x)
const MOVING_KMH = 10;         // a repeated position at this speed or more is a missed sample
const HISTORY_S = 5;           // positions kept per car, replay seconds
const SNAP_JUMP_M = 200;       // a bigger move between two ticks is drawn as a jump

// stopped and missing cars, all in replay time
const STILL_MOVE_M = 3;        // a car that stays within this distance of one spot ...
const STOPPED_AFTER_S = 3;     // ... for this long is marked STOPPED
const OUT_AFTER_S = 60;        // ... and after this long OUT (retired: its data freezes)
const FIELD_STILL_MIN = 5;     // this many still cars at once is a grid or a restart, not a stop
const FADE_AFTER_S = 5;        // no data for this long: fade the car out ...
const FADE_S = 2;              // ... over this long

const DOT_RADIUS_PX = 15;      // 3x the first version, room for the car number
const PIT_DOT_RADIUS_PX = 6;

// risk_30s is a real probability: the median car is about 0.001. RISK_HIGH is the model's
// alert line (threshold_p30 in data/models/risk_meta.json, crossed by 0.1% of normal ticks).
const RISK_HIGH = 0.027;
const RISK_ELEVATED = RISK_HIGH / 3;

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

const TRACK_FLAGS = new Set(["VSC", "SC", "RED"]);   // track-wide recs (PROJECT_BRIEF.md 7.4)
const CAR_FILL = { low: "#3a4150", elevated: "#a86f00", high: "#c62828", stopped: "#4a4a4a", pit: "#5b6270" };
const RING_STOPPED = "#9aa0a6";

// --- geometry ------------------------------------------------------------------

function dist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
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

function fmtTime(t) {
  // SessionTime in seconds as h:mm:ss.s
  const s = Math.max(0, Number(t));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = (s % 60).toFixed(1).padStart(4, "0");
  return `${h}:${String(m).padStart(2, "0")}:${sec}`;
}

// --- state -------------------------------------------------------------------

const canvas = document.getElementById("track");
const ctx = canvas.getContext("2d");
const feedEl = document.getElementById("feed");
const statusEl = document.getElementById("conn-status");
const trackFlagEl = document.getElementById("track-flag");

let raceId = null;
let segments = null;
let transform = null;
let canvasW = 0;
let canvasH = 0;

const cars = new Map();         // drv -> car (see newCar)
const risk = new Map();         // drv -> risk_30s
const sectorFlags = new Map();  // msector -> sector flag (CLEAR, YELLOW, DOUBLE_YELLOW)
let trackFlag = "CLEAR";        // track-wide flag: CLEAR, VSC, SC or RED
let replayT = null;             // SessionTime of the latest tick
let lastTickWall = -Infinity;   // performance.now() of the latest tick
let burstT = null;              // SessionTime and wall time of the first tick of the latest burst
let burstWall = null;
let rate = 0.25 / 256;          // replay seconds per wall-clock ms, measured
let burstSpan = 0.25;           // replay seconds per server step, measured
let dispT = null;               // replay time the map is drawn at
let lastFrameWall = null;
let feedCount = 0;

// --- cars ------------------------------------------------------------------

function newCar(c, t) {
  return {
    hist: [[t, c.x, c.y]],       // recent distinct positions: [replay t, x, y], oldest first
    x: c.x, y: c.y,              // latest tick position
    inPit: c.in_pit,
    seenT: t,                    // replay time of the latest tick with this car
    anchorX: c.x, anchorY: c.y,  // where the car was when it last moved STILL_MOVE_M
    movedT: t,
  };
}

function drawnPos(car, at) {
  // position at replay time `at`, interpolated between the two positions around it
  const h = car.hist;
  let i = h.length - 1;
  while (i > 0 && h[i][0] > at) i--;
  if (i === h.length - 1 || h[i][0] > at) return [h[i][1], h[i][2]];
  const [t0, x0, y0] = h[i];
  const [t1, x1, y1] = h[i + 1];
  const f = (at - t0) / (t1 - t0);
  return [x0 + (x1 - x0) * f, y0 + (y1 - y0) * f];
}

function updateCar(c, t) {
  const car = cars.get(c.drv);
  if (!car) {
    cars.set(c.drv, newCar(c, t));
    return;
  }
  const moved = dist([c.x, c.y], [car.x, car.y]);
  if (moved > SNAP_JUMP_M || c.in_pit !== car.inPit) {
    car.hist = [[t, c.x, c.y]];              // snap: never glide across a jump or the pit line
  } else if (moved > 0.05 || c.speed < MOVING_KMH) {
    car.hist.push([t, c.x, c.y]);            // a repeat at racing speed is a missed sample: skip it
    while (car.hist.length > 2 && car.hist[1][0] < t - HISTORY_S) car.hist.shift();
  }
  car.x = c.x;
  car.y = c.y;
  car.inPit = c.in_pit;
  car.seenT = t;
  if (dist([c.x, c.y], [car.anchorX, car.anchorY]) > STILL_MOVE_M) {
    car.anchorX = c.x;
    car.anchorY = c.y;
    car.movedT = t;
  }
}

function riskLevel(r) {
  if (r === undefined) return "low";
  if (r >= RISK_HIGH) return "high";
  if (r >= RISK_ELEVATED) return "elevated";
  return "low";
}

// --- feed (DOM, only touched when a detection, rec or official arrives) --------

function feedRow(kind, flag, title, meta, t) {
  const li = document.createElement("li");
  li.className = `feed-row kind-${kind}`;
  li.dataset.flag = flag;
  li.style.borderLeftColor = FLAG_COLORS[flag] || FLAG_COLORS.CLEAR;
  const source = { rec: "Fast Flag", det: "Detection", official: "Race control" }[kind];
  li.innerHTML =
    `<div class="feed-top"><span class="badge">${escapeHtml(flag.replace("_", " "))}</span>` +
    `<span class="feed-source">${source}</span><span class="feed-t">${fmtTime(t)}</span></div>` +
    `<div class="feed-title">${escapeHtml(title)}</div>` +
    (meta ? `<div class="feed-meta">${escapeHtml(meta)}</div>` : "");
  feedEl.insertBefore(li, feedEl.firstChild);
  feedCount++;
  while (feedCount > FEED_MAX) {
    feedEl.removeChild(feedEl.lastChild);
    feedCount--;
  }
}

function onRec(rec) {
  const conf = Number(rec.confidence).toFixed(2);
  if (TRACK_FLAGS.has(rec.flag) || (rec.flag === "CLEAR" && rec.message === "TRACK CLEAR")) {
    trackFlag = rec.flag;
    renderTrackFlag();
    feedRow("rec", rec.flag, rec.message, `${rec.reason}, from sector ${rec.msector}, confidence ${conf}`, rec.t);
  } else {
    sectorFlags.set(rec.msector, rec.flag);
    feedRow("rec", rec.flag, rec.message, `${rec.reason}, confidence ${conf}`, rec.t);
  }
}

function onDetection(det) {
  const who = det.drivers.length > 1 ? `Cars ${det.drivers.join(", ")}` : `Car ${det.drivers[0]}`;
  feedRow("det", det.type, `${who}, sector ${det.msector}`,
          `${det.evidence}, severity ${Number(det.severity).toFixed(2)}`, det.t);
}

function onOfficial(ev) {
  feedRow("official", ev.flag, ev.message, ev.scope === "Track" ? "Track-wide" : `Sector ${ev.msector}`, ev.t);
}

function clearFeed() {
  feedEl.innerHTML = "";
  feedCount = 0;
}

function renderTrackFlag() {
  trackFlagEl.textContent = trackFlag === "CLEAR" ? "GREEN" : trackFlag;
  trackFlagEl.dataset.flag = trackFlag;
}

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

// --- seek, loop and race change ------------------------------------------------

function resetForJump() {
  // the engine and the dashboard both wipe state on a jump over RESET_JUMP_S
  sectorFlags.clear();
  trackFlag = "CLEAR";
  renderTrackFlag();
  clearFeed();
  risk.clear();
  cars.clear();                 // every car is placed fresh: a seek snaps, never glides
  burstT = null;
  dispT = null;
  checkRace();                  // POST /replay can also load another race
}

async function loadTrack() {
  const track = await (await fetch("/track")).json();
  raceId = track.race;
  segments = buildSegments(track);
  resizeCanvas();
}

async function checkRace() {
  try {
    const status = await (await fetch("/status")).json();
    if (status.race !== raceId) await loadTrack();
  } catch (e) {
    // the next jump or reconnect checks again
  }
}

// --- canvas ------------------------------------------------------------------

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvasW = rect.width;
  canvasH = rect.height;
  canvas.width = Math.round(canvasW * dpr);
  canvas.height = Math.round(canvasH * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (segments) transform = computeTransform(segments.pts, canvasW, canvasH, MARGIN_PX);
}
window.addEventListener("resize", resizeCanvas);

function drawTrack() {
  const pts = segments.pts;
  const n = pts.length;
  ctx.lineCap = "round";
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
}

function carStates() {
  // which cars to draw, how visible, and whether they are stopped, out or in the pit lane
  const out = [];
  let onTrack = 0;
  let still = 0;
  for (const [drv, car] of cars) {
    const age = replayT - car.seenT;
    if (age >= FADE_AFTER_S + FADE_S) continue;
    const alpha = age <= FADE_AFTER_S ? 1 : 1 - (age - FADE_AFTER_S) / FADE_S;
    const fresh = age < 1;      // stopped means fresh ticks keep showing the same spot, not missing data
    const stillFor = replayT - car.movedT;
    if (!car.inPit && fresh) {
      onTrack++;
      if (stillFor >= STOPPED_AFTER_S) still++;
    }
    out.push({ drv, car, alpha, fresh, stillFor });
  }
  const fieldStill = still >= FIELD_STILL_MIN && still >= onTrack / 2;
  for (const s of out) {
    if (s.car.inPit) s.status = "PIT";
    else if (fieldStill || !s.fresh) s.status = null;
    else if (s.stillFor >= OUT_AFTER_S) s.status = "OUT";
    else if (s.stillFor >= STOPPED_AFTER_S) s.status = "STOPPED";
    else s.status = null;
    s.level = s.status ? null : riskLevel(risk.get(s.drv));
  }
  // pit cars at the bottom, the highest risk on top
  const order = { PIT: 0, OUT: 1, STOPPED: 2 };
  const rank = { low: 3, elevated: 4, high: 5 };
  out.sort((p, q) => (order[p.status] ?? rank[p.level]) - (order[q.status] ?? rank[q.level]));
  return out;
}

function drawCar(s) {
  const [x, y] = drawnPos(s.car, dispT);
  const [cx, cy] = toCanvas(transform, x, y);
  ctx.globalAlpha = s.alpha;
  if (s.status === "PIT") {
    ctx.globalAlpha = s.alpha * 0.55;
    ctx.beginPath();
    ctx.arc(cx, cy, PIT_DOT_RADIUS_PX, 0, Math.PI * 2);
    ctx.fillStyle = CAR_FILL.pit;
    ctx.fill();
    ctx.globalAlpha = 1;
    return;
  }
  const stopped = s.status === "STOPPED" || s.status === "OUT";
  ctx.beginPath();
  ctx.arc(cx, cy, DOT_RADIUS_PX, 0, Math.PI * 2);
  ctx.fillStyle = stopped ? CAR_FILL.stopped : CAR_FILL[s.level];
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "#0d0d0d";
  ctx.stroke();
  if (stopped) {
    ctx.beginPath();
    ctx.arc(cx, cy, DOT_RADIUS_PX + 4, 0, Math.PI * 2);
    ctx.lineWidth = 3;
    ctx.strokeStyle = RING_STOPPED;
    ctx.stroke();
  }
  ctx.fillStyle = "#ffffff";
  ctx.font = "700 13px system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(s.drv, cx, cy + 0.5);
  if (stopped) drawTag(s.status, cx + DOT_RADIUS_PX + 8, cy);
  ctx.globalAlpha = 1;
}

function drawTag(text, x, y) {
  ctx.font = "700 11px system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
  const w = ctx.measureText(text).width + 10;
  ctx.fillStyle = "rgba(13, 13, 13, 0.85)";
  ctx.beginPath();
  ctx.roundRect(x, y - 9, w, 18, 4);
  ctx.fill();
  ctx.strokeStyle = RING_STOPPED;
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.fillStyle = "#e8e8e8";
  ctx.textAlign = "left";
  ctx.fillText(text, x + 5, y + 0.5);
}

function advanceDisplayClock(now) {
  // runs at the measured replay rate and is pulled gently towards one and a bit bursts
  // behind the replay time estimated for this moment, never past the data
  const dt = lastFrameWall === null ? 0 : Math.min(now - lastFrameWall, 100);
  lastFrameWall = now;
  const targetLag = burstSpan + Math.max(LAG_EXTRA_S, burstSpan / 2);
  if (dispT === null) dispT = replayT - targetLag;
  const estNow = replayT + Math.min((now - lastTickWall) * rate, burstSpan);
  dispT += dt * rate + (estNow - dispT - targetLag) * Math.min(1, dt / 300);
  dispT = Math.min(dispT, replayT);
}

function draw(now) {
  requestAnimationFrame(draw);
  if (!segments || !transform) return;
  ctx.clearRect(0, 0, canvasW, canvasH);
  drawTrack();
  if (replayT === null) return;
  advanceDisplayClock(now);
  for (const s of carStates()) drawCar(s);
}

// --- websocket ---------------------------------------------------------------

function handleTick(tick) {
  const now = performance.now();
  if (replayT !== null && Math.abs(tick.t - replayT) > RESET_JUMP_S) resetForJump();
  if (now - lastTickWall > BURST_MS) {
    // first tick of a new server step: measure replay time per wall ms since the last
    // step (a pause or a seek in between is not a sample)
    if (burstT !== null && now - burstWall < 1000 && tick.t > burstT) {
      rate = 0.7 * rate + 0.3 * (tick.t - burstT) / (now - burstWall);
      burstSpan = 0.7 * burstSpan + 0.3 * (tick.t - burstT);
    }
    burstT = tick.t;
    burstWall = now;
  }
  lastTickWall = now;
  replayT = tick.t;
  for (const c of tick.cars) updateCar(c, tick.t);
}

function connect() {
  const ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    statusEl.textContent = "connected";
    checkRace();
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

    if (env.kind === "tick") handleTick(data);
    else if (env.kind === "risk") risk.set(data.drv, data.risk_30s);
    else if (env.kind === "rec") onRec(data);
    else if (env.kind === "detection") onDetection(data);
    else if (env.kind === "official") onOfficial(data);
  };
}

// --- init --------------------------------------------------------------------

async function init() {
  buildLegend();
  renderTrackFlag();
  resizeCanvas();
  await loadTrack();
  connect();
  requestAnimationFrame(draw);
}

init();
