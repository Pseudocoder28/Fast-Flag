"use strict";

// Pit wall dashboard. No build step, no CDN. Connects to ws://<host>/stream, draws the
// track map and the cars on a canvas (only inside the requestAnimationFrame loop) and
// keeps a feed of detections, recs and official race control messages (DOM, updated
// only when one of those envelopes arrives).

const WS_URL = `ws://${location.host}/stream`;
const RESET_JUMP_S = 2.0;      // same rule as engine.RESET_JUMP_S: a bigger jump is a seek or loop
const MARGIN_PX = 48;
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
  CLEAR: "#2fbf71",
  YELLOW: "#ffd400",
  DOUBLE_YELLOW: "#ffb000",
  VSC: "#00a3e0",
  SC: "#ff6b00",
  RED: "#e10600",
};
const FLAG_LABEL = {
  CLEAR: "CLEAR", YELLOW: "YELLOW", DOUBLE_YELLOW: "DOUBLE YELLOW", VSC: "VSC", SC: "SC", RED: "RED",
};
const TRACK_STATE_LABEL = { CLEAR: "GREEN", VSC: "VSC", SC: "SAFETY CAR", RED: "RED FLAG" };

// track line widths: CLEAR 2.5x the first version, flagged sectors wider, and a halo
// under the whole track for a track-wide flag
const TRACK_LINE = "#8d949e";
const SECTOR_WIDTH = { CLEAR: 5, YELLOW: 8, DOUBLE_YELLOW: 10 };
const HALO_WIDTH = { VSC: 12, SC: 13, RED: 14 };
const HALO_ALPHA = 0.55;
const MAP_BG = "#101318";      // matches the map panel, used for the gap in the double stripe

const TRACK_FLAGS = new Set(["VSC", "SC", "RED"]);   // track-wide recs (PROJECT_BRIEF.md 7.4)
const CAR_FILL = { low: "#343c49", elevated: "#a56200", high: "#c2185b", stopped: "#4a4d52", pit: "#6b7480" };
const RING_STOPPED = "#9aa0a6";
const LABEL_FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
const MONO_FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

// --- icons (inline SVG, no images, no emoji) -----------------------------------------

function flagIcon(flag, color) {
  const pole = `<path d="M3 1.5v13" stroke="${color}" stroke-width="1.6" stroke-linecap="round"/>`;
  const cloth = (dx, dy) => `<path d="M${4 + dx} ${2 + dy}h8l-1.7 2.6 1.7 2.6h-8z" fill="${color}"/>`;
  const body = flag === "DOUBLE_YELLOW"
    ? `${cloth(0, 0)}<path d="M5 6.2h8.5l-1.7 2.6 1.7 2.6H5z" fill="${color}" stroke="#0b0d10" stroke-width="1"/>`
    : cloth(0, 0);
  return `<svg class="ico" viewBox="0 0 16 16" aria-hidden="true">${pole}${body}</svg>`;
}

function detIcon(color) {
  return `<svg class="ico" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.8 15 14H1z" fill="none" ` +
    `stroke="${color}" stroke-width="1.6" stroke-linejoin="round"/><path d="M8 6v3.6" stroke="${color}" ` +
    `stroke-width="1.6" stroke-linecap="round"/><circle cx="8" cy="11.8" r="1" fill="${color}"/></svg>`;
}

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
  return { pts, segSector, total, labels: sectorLabels(pts, segSector) };
}

function sectorLabels(pts, segSector) {
  // one label per marshal sector at its middle segment, pushed away from the track centre
  const n = pts.length;
  const cx = pts.reduce((s, p) => s + p[0], 0) / n;
  const cy = pts.reduce((s, p) => s + p[1], 0) / n;
  const bySector = new Map();
  segSector.forEach((id, i) => {
    if (id === null) return;
    if (!bySector.has(id)) bySector.set(id, []);
    bySector.get(id).push(i);
  });
  const labels = [];
  for (const [id, idx] of bySector) {
    const i = idx[Math.floor(idx.length / 2)];
    const a = pts[i];
    const b = pts[(i + 1) % n];
    const len = dist(a, b) || 1;
    let nx = -(b[1] - a[1]) / len;
    let ny = (b[0] - a[0]) / len;
    if ((a[0] - cx) * nx + (a[1] - cy) * ny < 0) {
      nx = -nx;
      ny = -ny;
    }
    labels.push({ id, x: a[0], y: a[1], nx, ny });
  }
  return labels;
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

function raceName(id) {
  return `${String(id).replace(/_/g, " ")} GP`;
}

// --- state -------------------------------------------------------------------

const canvas = document.getElementById("track");
const ctx = canvas.getContext("2d");
const feedEl = document.getElementById("feed");
const statusEl = document.getElementById("conn-status");
const trackFlagEl = document.getElementById("track-flag");
const officialFlagEl = document.getElementById("official-flag");
const lapEl = document.getElementById("lap");
const clockEl = document.getElementById("clock");
const speedEl = document.getElementById("speed");
const raceEl = document.getElementById("race-name");

let raceId = null;
let segments = null;
let transform = null;
let canvasW = 0;
let canvasH = 0;
let officialEvents = [];        // GET /official, used only for the state at or before the replay time

const cars = new Map();         // drv -> car (see newCar)
const risk = new Map();         // drv -> risk_30s
const sectorFlags = new Map();  // msector -> sector flag (CLEAR, YELLOW, DOUBLE_YELLOW)
let trackFlag = null;           // our track-wide flag: CLEAR, VSC, SC or RED. null: not known yet,
                                // recs only arrive on changes, so a page opened mid-incident has
                                // to wait for the next track-wide rec or a seek (reset)
let officialFlag = null;        // official track-wide flag at the replay time (null before the first tick)
let replayT = null;             // SessionTime of the latest tick
let replayLap = null;
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
  const isFlag = flag in FLAG_COLORS;
  const icon = isFlag ? flagIcon(flag, "#0b0d10") : detIcon("#d7dce2");
  const badgeIcon = isFlag && flag === "RED" ? flagIcon(flag, "#ffffff") : icon;
  const source = { rec: "Fast Flag", det: "Detection", official: "Race control" }[kind];
  li.innerHTML =
    `<div class="feed-top"><span class="badge" data-flag="${escapeHtml(flag)}">${badgeIcon}` +
    `${escapeHtml(FLAG_LABEL[flag] || flag)}</span><span class="feed-source">${source}</span>` +
    `<span class="feed-t">${fmtTime(t)}</span></div>` +
    `<div class="feed-title">${escapeHtml(title)}</div>` +
    (meta ? `<div class="feed-meta">${meta}</div>` : "");
  feedEl.insertBefore(li, feedEl.firstChild);
  feedCount++;
  while (feedCount > FEED_MAX) {
    feedEl.removeChild(feedEl.lastChild);
    feedCount--;
  }
}

function num(v, digits = 2) {
  return `<span class="mono">${Number(v).toFixed(digits)}</span>`;
}

function onRec(rec) {
  const conf = `confidence ${num(rec.confidence)}`;
  if (TRACK_FLAGS.has(rec.flag) || (rec.flag === "CLEAR" && rec.message === "TRACK CLEAR")) {
    trackFlag = rec.flag;
    renderTrackState(trackFlagEl, trackFlag);
    feedRow("rec", rec.flag, rec.message,
            `${escapeHtml(rec.reason)}, from sector <span class="mono">${rec.msector}</span>, ${conf}`, rec.t);
  } else {
    sectorFlags.set(rec.msector, rec.flag);
    feedRow("rec", rec.flag, rec.message, `${escapeHtml(rec.reason)}, ${conf}`, rec.t);
  }
}

function onDetection(det) {
  const who = det.drivers.length > 1 ? `Cars ${det.drivers.join(", ")}` : `Car ${det.drivers[0]}`;
  feedRow("det", det.type, `${who}, sector ${det.msector}`,
          `${escapeHtml(det.evidence)}, severity ${num(det.severity)}`, det.t);
}

function onOfficial(ev) {
  if (ev.msector === null) {
    officialFlag = ev.flag;
    renderTrackState(officialFlagEl, officialFlag);
  }
  feedRow("official", ev.flag, ev.message, ev.msector === null ? "Track-wide" : `Sector ${ev.msector}`, ev.t);
}

function officialStateAt(t) {
  // the official track-wide state at time t, from events at or before t only
  let state = "CLEAR";
  for (const ev of officialEvents) {
    if (ev.t > t) break;
    if (ev.msector === null) state = ev.flag;
  }
  return state;
}

function clearFeed() {
  feedEl.innerHTML = "";
  feedCount = 0;
}

function renderTrackState(el, flag) {
  if (flag === null) {
    el.dataset.flag = "UNKNOWN";
    el.textContent = "WAITING";
    el.title = "Known after the next seek or track-wide recommendation";
    return;
  }
  const state = TRACK_STATE_LABEL[flag] ? flag : "CLEAR";
  el.dataset.flag = state;
  el.title = "";
  el.innerHTML = flagIcon(state, state === "RED" ? "#ffffff" : "#0b0d10") + TRACK_STATE_LABEL[state];
}

function buildLegend() {
  const el = document.getElementById("legend");
  el.innerHTML = Object.keys(FLAG_COLORS).map((flag) => {
    const line = flag === "DOUBLE_YELLOW" ? `<span class="legend-line double"></span>`
      : `<span class="legend-line" style="background:${TRACK_FLAGS.has(flag) ? FLAG_COLORS[flag] + "66" :
        (flag === "CLEAR" ? TRACK_LINE : FLAG_COLORS[flag])}"></span>`;
    return `<div class="legend-row">${flagIcon(flag, FLAG_COLORS[flag])}${line}<span>${FLAG_LABEL[flag]}</span></div>`;
  }).join("");
  const dot = (fill, cls = "") => `<span class="legend-dot ${cls}" style="background:${fill}"></span>`;
  document.getElementById("car-legend").innerHTML = [
    [dot(CAR_FILL.low), "Risk normal"],
    [dot(CAR_FILL.elevated), "Risk elevated"],
    [dot(CAR_FILL.high), "Risk high"],
    [dot(CAR_FILL.stopped, "ring"), "Stopped or out"],
    [dot(CAR_FILL.pit, "pit"), "In pit lane"],
  ].map(([d, text]) => `<div class="legend-row">${d}<span>${text}</span></div>`).join("");
}

// --- seek, loop and race change ------------------------------------------------

function resetForJump() {
  // the engine and the dashboard both wipe state on a jump over RESET_JUMP_S
  sectorFlags.clear();
  trackFlag = "CLEAR";
  renderTrackState(trackFlagEl, trackFlag);
  clearFeed();
  risk.clear();
  cars.clear();                 // every car is placed fresh: a seek snaps, never glides
  burstT = null;
  dispT = null;
  checkRace();                  // POST /replay can also load another race
}

async function loadTrack() {
  const [track, official] = await Promise.all([
    fetch("/track").then((r) => r.json()),
    fetch("/official").then((r) => r.json()),
  ]);
  raceId = track.race;
  raceEl.textContent = raceName(raceId);
  officialEvents = official.slice().sort((a, b) => a.t - b.t);
  segments = buildSegments(track);
  resizeCanvas();
  if (replayT !== null) {
    officialFlag = officialStateAt(replayT);
    renderTrackState(officialFlagEl, officialFlag);
  }
}

async function checkRace() {
  try {
    const status = await (await fetch("/status")).json();
    speedEl.textContent = status.speed > 0 ? `${status.speed}x` : "paused";
    if (status.race !== raceId) await loadTrack();
  } catch (e) {
    // the next poll, jump or reconnect checks again
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

function tracePath(from, to) {
  // path along the reference line from segment `from` to segment `to` (inclusive)
  const pts = segments.pts;
  const n = pts.length;
  ctx.beginPath();
  const [x0, y0] = toCanvas(transform, pts[from][0], pts[from][1]);
  ctx.moveTo(x0, y0);
  for (let i = from; i <= to; i++) {
    const p = pts[(i + 1) % n];
    const [x, y] = toCanvas(transform, p[0], p[1]);
    ctx.lineTo(x, y);
  }
}

function drawTrack() {
  const n = segments.pts.length;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  if (TRACK_FLAGS.has(trackFlag)) {
    // our track-wide flag: a halo under the whole track
    tracePath(0, n - 1);
    ctx.globalAlpha = HALO_ALPHA;
    ctx.strokeStyle = FLAG_COLORS[trackFlag];
    ctx.lineWidth = HALO_WIDTH[trackFlag];
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  tracePath(0, n - 1);
  ctx.strokeStyle = TRACK_LINE;
  ctx.lineWidth = SECTOR_WIDTH.CLEAR;
  ctx.stroke();

  // flagged sectors: each run of segments in one flagged sector as one stroke
  let i = 0;
  while (i < n) {
    const id = segments.segSector[i];
    const flag = sectorFlags.get(id) || "CLEAR";
    let j = i;
    while (j + 1 < n && segments.segSector[j + 1] === id) j++;
    if (flag !== "CLEAR") {
      tracePath(i, j);
      ctx.strokeStyle = FLAG_COLORS[flag];
      ctx.lineWidth = SECTOR_WIDTH[flag];
      ctx.stroke();
      if (flag === "DOUBLE_YELLOW") {
        ctx.strokeStyle = MAP_BG;       // gap down the middle: a double stripe
        ctx.lineWidth = 2.5;
        ctx.stroke();
      }
    }
    i = j + 1;
  }

  drawStartLine();
  drawSectorLabels();
}

function drawStartLine() {
  const pts = segments.pts;
  const [ax, ay] = toCanvas(transform, pts[0][0], pts[0][1]);
  const [bx, by] = toCanvas(transform, pts[1][0], pts[1][1]);
  const len = Math.hypot(bx - ax, by - ay) || 1;
  const nx = -(by - ay) / len;
  const ny = (bx - ax) / len;
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 3;
  ctx.lineCap = "butt";
  ctx.beginPath();
  ctx.moveTo(ax - nx * 9, ay - ny * 9);
  ctx.lineTo(ax + nx * 9, ay + ny * 9);
  ctx.stroke();
  ctx.lineCap = "round";
}

function drawSectorLabels() {
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (const lab of segments.labels) {
    const [px, py] = toCanvas(transform, lab.x, lab.y);
    const x = px + lab.nx * 17;
    const y = py - lab.ny * 17;       // canvas Y points down
    const flag = sectorFlags.get(lab.id);
    const flagged = flag && flag !== "CLEAR";
    ctx.font = `${flagged ? 800 : 600} 10.5px ${MONO_FONT}`;
    ctx.fillStyle = flagged ? FLAG_COLORS[flag] : "#5f6873";
    ctx.fillText(String(lab.id), x, y);
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
  ctx.strokeStyle = "#05070a";
  ctx.stroke();
  if (stopped) {
    ctx.beginPath();
    ctx.arc(cx, cy, DOT_RADIUS_PX + 4, 0, Math.PI * 2);
    ctx.lineWidth = 3;
    ctx.strokeStyle = RING_STOPPED;
    ctx.stroke();
  }
  ctx.fillStyle = "#ffffff";
  ctx.font = `700 13px ${LABEL_FONT}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(s.drv, cx, cy + 0.5);
  ctx.globalAlpha = 1;
  if (stopped) s.tagAt = [cx + DOT_RADIUS_PX + 8, cy];   // tags go on top of every car, see draw()
}

function drawTag(text, x, y) {
  ctx.font = `800 10.5px ${LABEL_FONT}`;
  const w = ctx.measureText(text).width + 12;
  ctx.fillStyle = "rgba(10, 12, 15, 0.9)";
  ctx.beginPath();
  ctx.roundRect(x, y - 9, w, 18, 4);
  ctx.fill();
  ctx.strokeStyle = RING_STOPPED;
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.fillStyle = "#eef0f2";
  ctx.textAlign = "left";
  ctx.fillText(text, x + 6, y + 0.5);
}

function renderHeader() {
  const lap = replayLap === null ? "-" : String(replayLap);
  const clock = replayT === null ? "-" : fmtTime(replayT);
  if (lapEl.textContent !== lap) lapEl.textContent = lap;
  if (clockEl.textContent !== clock) clockEl.textContent = clock;
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
  renderHeader();
  advanceDisplayClock(now);
  const states = carStates();
  for (const s of states) drawCar(s);
  for (const s of states) {
    if (!s.tagAt) continue;
    ctx.globalAlpha = s.alpha;
    drawTag(s.status, s.tagAt[0], s.tagAt[1]);
    ctx.globalAlpha = 1;
  }
}

// --- websocket ---------------------------------------------------------------

function handleTick(tick) {
  const now = performance.now();
  const jumped = replayT !== null && Math.abs(tick.t - replayT) > RESET_JUMP_S;
  if (jumped) resetForJump();
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
  const first = replayT === null;
  replayT = tick.t;
  replayLap = tick.lap;
  if (first || jumped) {
    officialFlag = officialStateAt(replayT);
    renderTrackState(officialFlagEl, officialFlag);
  }
  for (const c of tick.cars) updateCar(c, tick.t);
}

function setConn(state, text) {
  statusEl.dataset.state = state;
  statusEl.textContent = text;
}

function connect() {
  const ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    setConn("connected", "live stream");
    checkRace();
  };

  ws.onclose = () => {
    setConn("disconnected", "reconnecting");
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
  renderTrackState(trackFlagEl, trackFlag);
  renderTrackState(officialFlagEl, officialFlag);
  resizeCanvas();
  await loadTrack();
  connect();
  setInterval(checkRace, 2000);   // replay speed and race, set by POST /replay from anywhere
  requestAnimationFrame(draw);
}

init();
