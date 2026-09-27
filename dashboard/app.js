"use strict";

// Pit wall dashboard. No build step, no CDN. Connects to ws://<host>/stream, draws the
// track map and the cars on a canvas (only inside the requestAnimationFrame loop) and
// keeps a feed of detections, recs and official race control messages (DOM, updated
// only when one of those envelopes arrives).

const WS_URL = `ws://${location.host}/stream`;
const RESET_JUMP_S = 2.0;      // same rule as engine.RESET_JUMP_S: a bigger jump is a seek or loop
const MARGIN_PX = 48;
const FEED_MAX = 50;

// lead-time timeline and jump-to-incident list
const MATCH_BEFORE_S = 60;     // our rec pairs with an official flag from 60 s before it ...
const MATCH_AFTER_S = 10;      // ... to 10 s after it (PROJECT_BRIEF.md 6.7)
const TL_WINDOW_S = 90;        // the timeline shows the last 90 s of replay time
const INCIDENT_GAP_S = 60;     // official flags further apart than this are separate incidents
const JUMP_BEFORE_S = 30;

// open the page with ?debug to see frame rate, display lag and message rate on the map
const DEBUG = new URLSearchParams(location.search).has("debug");
const dbg = { fps: 0, frameMs: 0, last: 0, msgs: 0, msgRate: 0, windowStart: 0 };      // a jump lands this long before race control's first message

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
const RETIRED_AFTER_S = 120;   // still this long and no longer part of a flagged incident: out of the race ...
const RETIRE_FADE_S = 3;       // ... it fades off the map over this long (it comes back if it moves again)
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
const CAR_FILL = { low: "#343c49", stricken: "#d62839", stopped: "#4a4d52", pit: "#6b7480" };
const RISK_HALO = { elevated: "#e0a800", high: "#ff4fa3" };   // risk is a secondary cue: a halo, never the fill
const RING_STOPPED = "#9aa0a6";
const WATCH_COLOR = "#a99cff";  // ANOMALY: advisory only, never a flag, kept apart from flag and risk colours
const WATCH_S = 10;             // a watched car keeps its dashed ring this long (replay time)
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
const tlCanvas = document.getElementById("timeline");
const tctx = tlCanvas.getContext("2d");
const leadSummaryEl = document.getElementById("lead-summary");
const playEl = document.getElementById("play");
const speedButtons = [...document.querySelectorAll("[data-speed]")];
const seekbarEl = document.getElementById("seekbar");
const seekTicksEl = document.getElementById("seek-ticks");
const seekHeadEl = document.getElementById("seek-head");
const seekEndEl = document.getElementById("seek-end");
const incidentsEl = document.getElementById("incidents");
const tabNoteEl = document.getElementById("tab-note");

let raceId = null;
let segments = null;
let transform = null;
let canvasW = 0;
let canvasH = 0;
let officialEvents = [];        // GET /official, used only for the state at or before the replay time

const cars = new Map();         // drv -> car (see newCar)
const risk = new Map();         // drv -> risk_30s
const watchUntil = new Map();
const involved = new Map();     // drv -> sector of its latest physical detection (stricken while that sector is flagged)   // drv -> replay time until which an ANOMALY watch ring shows
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

let nSectors = 0;
let ourEvents = [];             // our recs above CLEAR since the last jump: {t, flag, msector, trackWide}
let officialSeen = [];          // official envelopes above CLEAR since the last jump, same shape
let replay = { t_start: null, t_end: null, speed: null };   // from GET /status and POST /replay
let lastSpeed = 1;              // speed to resume at after a pause
let incidents = [];             // from GET /official, for the jump list
let currentIncident = -1;
let tlW = 0;
let tlH = 0;
let seekHeadPct = -1;

// --- cars ------------------------------------------------------------------

function newCar(c, t) {
  return {
    hist: [[t, c.x, c.y]],       // recent distinct positions: [replay t, x, y], oldest first
    x: c.x, y: c.y,              // latest tick position
    inPit: c.in_pit,
    msector: c.msector,
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
  car.msector = c.msector;
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
  const icon = isFlag ? flagIcon(flag, "#0b0d10") : detIcon(kind === "watch" ? WATCH_COLOR : "#d7dce2");
  const badgeIcon = isFlag && flag === "RED" ? flagIcon(flag, "#ffffff") : icon;
  const source = { rec: "Fast Flag", det: "Detection", official: "Race control", watch: "Anomaly model" }[kind];
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
  if (rec.flag !== "CLEAR") {
    ourEvents.push({ t: rec.t, flag: rec.flag, msector: rec.msector, trackWide: TRACK_FLAGS.has(rec.flag) });
    renderLeadSummary();
  }
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
  if (det.type === "ANOMALY") {
    // advisory: the anomaly model flags unusual driving, race control raises no flag for it
    for (const d of det.drivers) watchUntil.set(d, det.t + WATCH_S);
    feedRow("watch", "WATCH", `${who}, sector ${det.msector}: unusual driving, watch`,
            escapeHtml(det.evidence), det.t);
    return;
  }
  for (const d of det.drivers) involved.set(d, det.msector);
  feedRow("det", det.type, `${who}, sector ${det.msector}`,
          `${escapeHtml(det.evidence)}, severity ${num(det.severity)}`, det.t);
}

function onOfficial(ev) {
  if (ev.msector === null) {
    officialFlag = ev.flag;
    renderTrackState(officialFlagEl, officialFlag);
  }
  let meta = ev.msector === null ? "Track-wide" : `Sector ${ev.msector}`;
  if (ev.flag !== "CLEAR") {
    const off = { t: ev.t, flag: ev.flag, msector: ev.msector, trackWide: ev.msector === null };
    officialSeen.push(off);
    const ours = matchFor(off);
    if (ours && ours.t < off.t) {
      meta += `, <span class="lead-good">Fast Flag <span class="mono">${(off.t - ours.t).toFixed(1)} s</span> earlier</span>`;
    }
  }
  feedRow("official", ev.flag, ev.message, meta, ev.t);
  renderLeadSummary();
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
  const dot = (fill, cls = "", extra = "") => `<span class="legend-dot ${cls}" style="background:${fill};${extra}"></span>`;
  document.getElementById("car-legend").innerHTML = [
    [dot(CAR_FILL.stricken, "stricken"), "In an incident"],
    [dot(CAR_FILL.low, "", `box-shadow:0 0 0 2px var(--panel),0 0 0 4px ${FLAG_COLORS.YELLOW}`), "In a yellow sector"],
    [dot(CAR_FILL.low, "", `box-shadow:0 0 0 2px var(--panel),0 0 0 3.5px ${FLAG_COLORS.SC}`), "Under our SC or VSC"],
    [dot(CAR_FILL.low, "", `box-shadow:0 0 0 5px ${RISK_HALO.elevated}73`), "Elevated risk"],
    [dot(CAR_FILL.low, "", `box-shadow:0 0 0 5px ${RISK_HALO.high}73`), "High risk"],
    [dot(CAR_FILL.stopped, "ring"), "Stopped or out"],
    [dot(CAR_FILL.low, "watch"), "Watch (anomaly)"],
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
  watchUntil.clear();
  involved.clear();
  cars.clear();                 // every car is placed fresh: a seek snaps, never glides
  burstT = null;
  dispT = null;
  ourEvents = [];
  officialSeen = [];
  renderLeadSummary();
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
  nSectors = track.msectors.length;
  segments = buildSegments(track);
  buildIncidents();
  buildSeekTicks();
  resizeCanvas();
  if (replayT !== null) {
    officialFlag = officialStateAt(replayT);
    renderTrackState(officialFlagEl, officialFlag);
  }
}

async function checkRace() {
  try {
    const status = await (await fetch("/status")).json();
    applyStatus(status);
    if (status.race !== raceId) await loadTrack();
  } catch (e) {
    // the next poll, jump or reconnect checks again
  }
}

function applyStatus(status) {
  const range = status.t_start !== replay.t_start || status.t_end !== replay.t_end;
  replay = { t_start: status.t_start, t_end: status.t_end, speed: status.speed };
  if (status.speed > 0) lastSpeed = status.speed;
  speedEl.textContent = status.speed > 0 ? `${status.speed}x` : "paused";
  playEl.textContent = status.speed > 0 ? "Pause" : "Play";
  for (const b of speedButtons) b.classList.toggle("active", Number(b.dataset.speed) === status.speed);
  seekEndEl.textContent = fmtTime(status.t_end);
  if (range) {
    buildSeekTicks();
    buildIncidents();
  }
}

// --- lead time: our recs next to the race control feed -------------------------

function nearSector(a, b) {
  // same or adjacent marshal sector, wrapping at the finish line
  if (a === null || b === null || a === undefined || b === undefined) return false;
  const d = Math.abs(a - b);
  return d <= 1 || (nSectors > 0 && d === nSectors - 1);
}

function matchFor(off) {
  // our first rec for an official flag, as the eval matches them (PROJECT_BRIEF.md 6.7):
  // from MATCH_BEFORE_S before to MATCH_AFTER_S after it, track-wide with track-wide,
  // sector flags with a sector flag in the same or an adjacent sector
  for (const r of ourEvents) {
    if (r.t < off.t - MATCH_BEFORE_S) continue;
    if (r.t > off.t + MATCH_AFTER_S) break;
    if (off.trackWide ? r.trackWide : !r.trackWide && nearSector(r.msector, off.msector)) return r;
  }
  return null;
}

function leadSummary() {
  const n = officialSeen.length;
  if (n === 0) return "No official flag since the last seek";
  const leads = officialSeen.map(matchFor).map((r, i) => (r ? officialSeen[i].t - r.t : null));
  const earlier = leads.filter((l) => l !== null && l > 0).sort((a, b) => a - b);
  if (!earlier.length) return `Fast Flag first on <span class="mono">0</span> of <span class="mono">${n}</span> official flags`;
  const mid = earlier.length / 2;
  const median = earlier.length % 2 ? earlier[Math.floor(mid)] : (earlier[mid - 1] + earlier[mid]) / 2;
  return `Fast Flag first on <span class="mono">${earlier.length}</span> of <span class="mono">${n}</span> official ` +
    `flags since the last seek, median lead <span class="mono">${median.toFixed(1)} s</span>`;
}

function renderLeadSummary() {
  leadSummaryEl.innerHTML = leadSummary();
}

function drawMarker(x, y, flag, lane) {
  // lane.labelEnd: where the previous label in this lane ends, so labels never overlap
  const c = FLAG_COLORS[flag] || FLAG_COLORS.CLEAR;
  tctx.fillStyle = c;
  if (flag === "DOUBLE_YELLOW") {
    tctx.fillRect(x - 4, y - 8, 3, 16);
    tctx.fillRect(x + 1, y - 8, 3, 16);
  } else {
    tctx.fillRect(x - 3, y - 8, 6, 16);
  }
  const text = { YELLOW: "Y", DOUBLE_YELLOW: "DY" }[flag] || flag;
  tctx.font = `800 9.5px ${LABEL_FONT}`;
  const w = tctx.measureText(text).width;
  if (x + 6 < lane.labelEnd) return;
  tctx.textAlign = "left";
  tctx.textBaseline = "middle";
  tctx.fillText(text, x + 6, y);
  lane.labelEnd = x + 6 + w + 4;
}

function drawTimeline() {
  tctx.clearRect(0, 0, tlW, tlH);
  if (replayT === null || tlW === 0) return;
  const left = 104;
  const right = tlW - 10;
  const t1 = replayT + 3;
  const t0 = replayT - TL_WINDOW_S;
  const x = (t) => left + ((t - t0) / (t1 - t0)) * (right - left);
  const yOurs = 20;
  const yOff = 56;

  tctx.font = `800 9.5px ${LABEL_FONT}`;
  tctx.textAlign = "left";
  tctx.textBaseline = "middle";
  tctx.fillStyle = "#a3abb6";
  tctx.fillText("FAST FLAG", 0, yOurs);
  tctx.fillText("RACE CONTROL", 0, yOff);
  tctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
  tctx.lineWidth = 1;
  for (const y of [yOurs, yOff]) {
    tctx.beginPath();
    tctx.moveTo(left, y + 0.5);
    tctx.lineTo(right, y + 0.5);
    tctx.stroke();
  }
  tctx.font = `600 9.5px ${MONO_FONT}`;
  tctx.textAlign = "center";
  tctx.fillStyle = "#6f7883";
  for (let k = 15; k <= TL_WINDOW_S; k += 15) {
    const gx = x(replayT - k);
    tctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    tctx.beginPath();
    tctx.moveTo(gx + 0.5, 8);
    tctx.lineTo(gx + 0.5, yOff + 10);
    tctx.stroke();
    tctx.fillText(`-${k} s`, gx, tlH - 6);
  }
  const nx = x(replayT);
  tctx.strokeStyle = "#ffffff";
  tctx.beginPath();
  tctx.moveTo(nx + 0.5, 6);
  tctx.lineTo(nx + 0.5, yOff + 12);
  tctx.stroke();
  tctx.fillStyle = "#e8eaed";
  tctx.fillText("now", nx, tlH - 6);

  // pairs first, so the markers sit on top of the lines
  for (const off of officialSeen) {
    const ours = matchFor(off);
    if (!ours || Math.max(off.t, ours.t) < t0) continue;
    const ax = Math.max(x(ours.t), left);
    const bx = x(off.t);
    const lead = off.t - ours.t;
    const good = lead > 0;
    tctx.strokeStyle = good ? "rgba(47, 191, 113, 0.9)" : "rgba(163, 171, 182, 0.7)";
    tctx.lineWidth = 1.5;
    tctx.beginPath();
    tctx.moveTo(ax, yOurs + 9);
    tctx.lineTo(bx, yOff - 9);
    tctx.stroke();
    const label = good ? `+${lead.toFixed(1)} s` : `${lead.toFixed(1)} s`;
    const lx = (ax + bx) / 2;
    const ly = (yOurs + yOff) / 2;
    tctx.font = `700 10px ${MONO_FONT}`;
    const lw = tctx.measureText(label).width + 8;
    tctx.fillStyle = "rgba(10, 12, 15, 0.9)";
    tctx.beginPath();
    tctx.roundRect(lx - lw / 2, ly - 7, lw, 14, 3);
    tctx.fill();
    tctx.textAlign = "center";
    tctx.textBaseline = "middle";
    tctx.fillStyle = good ? "#2fbf71" : "#a3abb6";
    tctx.fillText(label, lx, ly + 0.5);
  }
  const oursLane = { labelEnd: -Infinity };
  const offLane = { labelEnd: -Infinity };
  for (const r of ourEvents) if (r.t >= t0) drawMarker(x(r.t), yOurs, r.flag, oursLane);
  for (const o of officialSeen) if (o.t >= t0) drawMarker(x(o.t), yOff, o.flag, offLane);
}

// --- replay controls (POST /replay, for every viewer of this server) -----------

async function postReplay(body) {
  try {
    const res = await fetch("/replay", {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
    });
    if (res.ok) applyStatus(await res.json());
  } catch (e) {
    // the status poll shows the real state
  }
}

function seekTo(t) {
  if (replay.t_start === null) return;
  postReplay({ seek_t: Math.min(Math.max(t, replay.t_start), replay.t_end) });
}

function togglePlay() {
  postReplay({ speed: replay.speed > 0 ? 0 : lastSpeed || 1 });
}

function buildSeekTicks() {
  seekTicksEl.innerHTML = "";
  const { t_start: a, t_end: b } = replay;
  if (a === null || b <= a) return;
  for (const ev of officialEvents) {
    if (ev.flag === "CLEAR" || ev.t < a || ev.t > b) continue;
    const tick = document.createElement("span");
    tick.className = "seek-tick";
    tick.style.left = `${((ev.t - a) / (b - a)) * 100}%`;
    tick.style.background = FLAG_COLORS[ev.flag] || FLAG_COLORS.CLEAR;
    seekTicksEl.appendChild(tick);
  }
}

function renderSeekHead() {
  const { t_start: a, t_end: b } = replay;
  if (a === null || replayT === null || b <= a) return;
  const pct = Math.round(((replayT - a) / (b - a)) * 1000) / 10;
  if (pct !== seekHeadPct) {
    seekHeadPct = pct;
    seekHeadEl.style.left = `${Math.min(Math.max(pct, 0), 100)}%`;
  }
}

// --- jump to incident (GET /official) --------------------------------------------

function buildIncidents() {
  incidents = [];
  const { t_start: a, t_end: b } = replay;
  for (const ev of officialEvents) {
    if (ev.flag === "CLEAR") continue;
    if (a !== null && (ev.t < a || ev.t > b)) continue;   // before the start or after the end of the replay
    const last = incidents[incidents.length - 1];
    if (last && ev.t - last.tLast <= INCIDENT_GAP_S) {
      last.tLast = ev.t;
      last.events.push(ev);
    } else {
      incidents.push({ t: ev.t, tLast: ev.t, events: [ev] });
    }
  }
  incidentsEl.innerHTML = "";
  incidents.forEach((inc, i) => {
    const flags = [...new Set(inc.events.map((e) => e.flag))];
    const sectors = [...new Set(inc.events.map((e) => e.msector).filter((m) => m !== null))].sort((a, b) => a - b);
    const where = sectors.length ? `Sector${sectors.length > 1 ? "s" : ""} ${sectors.join(", ")}` : "Track-wide";
    const li = document.createElement("li");
    li.className = "incident";
    li.dataset.index = String(i);
    li.innerHTML =
      `<div class="incident-main"><div class="incident-flags">` +
      flags.map((f) => `<span class="badge" data-flag="${f}">${flagIcon(f, f === "RED" ? "#ffffff" : "#0b0d10")}` +
        `${FLAG_LABEL[f] || f}</span>`).join("") +
      `</div><div class="incident-meta">${where}, ${inc.events.length} official message` +
      `${inc.events.length > 1 ? "s" : ""}</div></div>` +
      `<span class="incident-t">${fmtTime(inc.t)}</span><button type="button" class="ctl jump">Jump</button>`;
    li.addEventListener("click", () => {
      postReplay({ seek_t: Math.max(inc.t - JUMP_BEFORE_S, replay.t_start ?? inc.t - JUMP_BEFORE_S), speed: 1 });
    });
    incidentsEl.appendChild(li);
  });
  currentIncident = -1;
}

function renderCurrentIncident() {
  // the incident whose window holds the replay time, highlighted in the list
  let cur = -1;
  if (replayT !== null) {
    cur = incidents.findIndex((inc) => replayT >= inc.t - JUMP_BEFORE_S && replayT <= inc.tLast + INCIDENT_GAP_S);
  }
  if (cur === currentIncident) return;
  incidentsEl.querySelector(".incident.current")?.classList.remove("current");
  if (cur >= 0) incidentsEl.children[cur]?.classList.add("current");
  currentIncident = cur;
}

function wireControls() {
  playEl.addEventListener("click", togglePlay);
  for (const b of speedButtons) b.addEventListener("click", () => postReplay({ speed: Number(b.dataset.speed) }));
  document.getElementById("back30").addEventListener("click", () => replayT !== null && seekTo(replayT - 30));
  document.getElementById("fwd30").addEventListener("click", () => replayT !== null && seekTo(replayT + 30));
  seekbarEl.addEventListener("click", (e) => {
    const r = seekbarEl.getBoundingClientRect();
    const f = Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1);
    seekTo(replay.t_start + f * (replay.t_end - replay.t_start));
  });
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && e.target === document.body) {
      e.preventDefault();
      togglePlay();
    }
  });
  for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => {
      const which = tab.dataset.tab;
      for (const t of document.querySelectorAll(".tab")) {
        t.classList.toggle("active", t === tab);
        t.setAttribute("aria-selected", String(t === tab));
      }
      feedEl.hidden = which !== "feed";
      incidentsEl.hidden = which !== "incidents";
      tabNoteEl.textContent = which === "feed" ? "newest first, last 50"
        : `jumps to ${JUMP_BEFORE_S} s before race control`;
    });
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
  const tr = tlCanvas.getBoundingClientRect();
  tlW = tr.width;
  tlH = tr.height;
  tlCanvas.width = Math.round(tlW * dpr);
  tlCanvas.height = Math.round(tlH * dpr);
  tctx.setTransform(dpr, 0, 0, dpr, 0, 0);
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
  // which cars to draw, how visible, stopped, out or in the pit lane, and their situation:
  // stricken (named in a detection in a sector that is still flagged), inside a flagged
  // sector, or under a track-wide flag. Risk is only a secondary cue.
  const out = [];
  let onTrack = 0;
  let still = 0;
  let freshAll = 0;
  let stillAll = 0;
  for (const [drv, car] of cars) {
    const age = replayT - car.seenT;
    if (age >= FADE_AFTER_S + FADE_S) continue;
    const alpha = age <= FADE_AFTER_S ? 1 : 1 - (age - FADE_AFTER_S) / FADE_S;
    const fresh = age < 1;      // stopped means fresh ticks keep showing the same spot, not missing data
    const stillFor = replayT - car.movedT;
    if (fresh) {
      freshAll++;
      if (stillFor >= STOPPED_AFTER_S) stillAll++;
    }
    if (!car.inPit && fresh) {
      onTrack++;
      if (stillFor >= STOPPED_AFTER_S) still++;
    }
    out.push({ drv, car, alpha, fresh, stillFor });
  }
  const fieldStill = still >= FIELD_STILL_MIN && still >= onTrack / 2;
  // the whole field stopped (a red flag in the pit lane, the grid): nobody is out of the race
  const fieldStopped = stillAll >= FIELD_STILL_MIN && stillAll >= freshAll / 2;
  for (const s of out) {
    if (s.car.inPit) s.status = "PIT";
    else if (fieldStill || !s.fresh) s.status = null;
    else if (s.stillFor >= OUT_AFTER_S) s.status = "OUT";
    else if (s.stillFor >= STOPPED_AFTER_S) s.status = "STOPPED";
    else s.status = null;
    const inc = involved.get(s.drv);
    s.stricken = s.status !== "PIT" && inc !== undefined && isFlagged(sectorFlags.get(inc));
    const zone = sectorFlags.get(s.car.msector);
    s.zone = s.status !== "PIT" && isFlagged(zone) ? zone : null;
    s.level = riskLevel(risk.get(s.drv));
    // out of the race: still this long, on track or in its garage, no longer part of a flagged
    // incident, while the rest of the field keeps moving. It fades off the map.
    if (!fieldStopped && s.fresh && s.stillFor >= RETIRED_AFTER_S && !s.stricken) {
      s.alpha *= Math.max(0, 1 - (s.stillFor - RETIRED_AFTER_S) / RETIRE_FADE_S);
    }
  }
  // pit cars at the bottom, then running cars, then cars in a flagged sector, stopped and
  // out cars, and the stricken car on top: a car passing the scene never hides it
  const layer = (s) => (s.status === "PIT" ? 0 : s.stricken ? 5 : s.status ? 4 : s.zone ? 3 : 1);
  out.sort((p, q) => layer(p) - layer(q));
  return out.filter((s) => s.alpha > 0.02);
}

function isFlagged(flag) {
  return flag === "YELLOW" || flag === "DOUBLE_YELLOW";
}

function ring(cx, cy, r, width, color, dash) {
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.lineWidth = width;
  ctx.strokeStyle = color;
  if (dash) ctx.setLineDash(dash);
  ctx.stroke();
  if (dash) ctx.setLineDash([]);
}

function drawCar(s, now) {
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

  // secondary cue: a soft risk halo behind the dot
  if (!stopped && !s.stricken && (s.level === "elevated" || s.level === "high")) {
    ctx.globalAlpha = s.alpha * 0.45;
    ctx.beginPath();
    ctx.arc(cx, cy, DOT_RADIUS_PX + 9, 0, Math.PI * 2);
    ctx.fillStyle = RISK_HALO[s.level];
    ctx.fill();
    ctx.globalAlpha = s.alpha;
  }

  ctx.beginPath();
  ctx.arc(cx, cy, DOT_RADIUS_PX, 0, Math.PI * 2);
  ctx.fillStyle = s.stricken ? CAR_FILL.stricken : stopped ? CAR_FILL.stopped : CAR_FILL.low;
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "#05070a";
  ctx.stroke();

  if (s.stricken) {
    // pulsing ring, about once a second
    const p = (Math.sin((now / 1000) * Math.PI * 2 * 1.1) + 1) / 2;
    ctx.globalAlpha = s.alpha * (0.35 + 0.55 * (1 - p));
    ring(cx, cy, DOT_RADIUS_PX + 4 + 6 * p, 3, CAR_FILL.stricken);
    ctx.globalAlpha = s.alpha;
  } else if (stopped) {
    ring(cx, cy, DOT_RADIUS_PX + 4, 3, RING_STOPPED);
  } else if (s.zone) {
    ring(cx, cy, DOT_RADIUS_PX + 3.5, 3.5, FLAG_COLORS[s.zone]);          // driving through a flagged sector
  } else if (TRACK_FLAGS.has(trackFlag)) {
    ring(cx, cy, DOT_RADIUS_PX + 3, 2, FLAG_COLORS[trackFlag]);           // under our VSC, SC or red flag
  }
  if (!stopped && !s.stricken && (watchUntil.get(s.drv) ?? -Infinity) >= replayT) {
    ring(cx, cy, DOT_RADIUS_PX + 8, 2, WATCH_COLOR, [4, 3]);
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

function drawDebug(now) {
  const dt = dbg.last ? now - dbg.last : 16;
  dbg.last = now;
  dbg.frameMs = dbg.frameMs ? 0.9 * dbg.frameMs + 0.1 * dt : dt;
  dbg.fps = 1000 / Math.max(dbg.frameMs, 1);
  if (now - dbg.windowStart >= 1000) {
    dbg.msgRate = (dbg.msgs * 1000) / (now - dbg.windowStart || 1000);
    dbg.msgs = 0;
    dbg.windowStart = now;
  }
  const lines = [
    `fps ${dbg.fps.toFixed(0)}  (frame ${dbg.frameMs.toFixed(1)} ms)`,
    `display lag ${replayT === null || dispT === null ? "-" : (replayT - dispT).toFixed(2)} s replay`,
    `replay rate ${(rate * 1000).toFixed(1)}x, burst ${burstSpan.toFixed(2)} s`,
    `messages ${dbg.msgRate.toFixed(0)}/s, cars ${cars.size}`,
  ];
  ctx.font = `600 11px ${MONO_FONT}`;
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillStyle = "rgba(10, 12, 15, 0.85)";
  ctx.fillRect(10, 10, 270, 16 * lines.length + 10);
  ctx.fillStyle = "#e8eaed";
  lines.forEach((l, i) => ctx.fillText(l, 16, 15 + 16 * i));
}

function draw(now) {
  requestAnimationFrame(draw);
  if (!segments || !transform) return;
  ctx.clearRect(0, 0, canvasW, canvasH);
  drawTrack();
  drawTimeline();
  if (replayT === null) return;
  renderHeader();
  renderSeekHead();
  renderCurrentIncident();
  advanceDisplayClock(now);
  const states = carStates();
  for (const s of states) drawCar(s, now);
  for (const s of states) {
    if (!s.tagAt) continue;
    ctx.globalAlpha = s.alpha;
    drawTag(s.status, s.tagAt[0], s.tagAt[1]);
    ctx.globalAlpha = 1;
  }
  if (DEBUG) drawDebug(now);
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
    dbg.msgs++;
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
  wireControls();
  resizeCanvas();
  await loadTrack();
  connect();
  setInterval(checkRace, 2000);   // replay speed and race, set by POST /replay from anywhere
  requestAnimationFrame(draw);
}

init();
