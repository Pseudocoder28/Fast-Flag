"use strict";

// Fast Flag broadcast overlay. No build step, no CDN. Subscribes to ws://<host>/stream
// and consumes the existing envelopes only (tick, detection, rec, official).
//
// Strictly causal: everything on screen comes from envelopes already received,
// and every clock runs on replay time from the ticks, never on the wall clock.
//
// - Every rec at YELLOW or above slides in a lower-third banner.
// - Exposure Clock: when our rec reaches VSC, SC or RED before the official message
//   of that level, it counts replay seconds and the cars passing the stricken car
//   at racing speed. The official message freezes it.
// - After the official message: FAST FLAG AHEAD BY X.X s, or RACE CONTROL FIRST BY X.X s.

const WS_URL = `ws://${location.host}/stream`;
const RESET_JUMP_S = 2.0;          // same rule as the race control engine and the dashboard
const CONFIRM_WINDOW_S = 120.0;    // an official message confirms a flag we raised within this long before it
const RACING_SHARE = 0.8;          // racing speed: at least this share of the car's own speed here last lap
const RACING_FALLBACK_KMH = 120;   // ... or, with no previous lap yet, at least this fast
const BIN_M = 50;                  // per-car speed memory along the lap
const MAX_STEP_M = 150;            // a bigger jump between two ticks is not a pass (seek, data gap)
const REST_KMH = 5;                // the stricken car has come to rest below this
const REST_WITHIN_S = 30;          // stop following the stricken car after this long
const FIELD_MIN_KMH = 60;          // field reference: samples slower than this are not racing
const FIELD_MAX = 40;              // samples kept per bin for the field reference
const TICKER_MAX = 6;
const BANNER_MAX = 3;
const HIDE_AFTER_CLEAR_S = 12;     // frozen clock stays this long (replay time) after the track clears

const RANK = { CLEAR: 0, YELLOW: 1, DOUBLE_YELLOW: 2, VSC: 3, SC: 4, RED: 5 };
const GLOBAL_FLAGS = new Set(["VSC", "SC", "RED"]);
const FLAG_TEXT = { YELLOW: "YELLOW FLAG", DOUBLE_YELLOW: "DOUBLE YELLOW", VSC: "VIRTUAL SAFETY CAR",
  SC: "SAFETY CAR", RED: "RED FLAG" };
const TYPE_TEXT = { IMPACT: "IMPACT", STOPPED: "STOPPED", SPIN: "SPIN", DROPOUT: "NO DATA",
  MULTI: "MULTI-CAR", ANOMALY: "ANOMALY" };
const TRACK = "track";

// --- helpers ---------------------------------------------------------------------

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function circ(d, length) {
  // signed circular difference along the lap, in (-length / 2, length / 2]
  return ((d + length / 2) % length + length) % length - length / 2;
}

function sectorOffset(car, flagged, n) {
  const h = Math.floor(n / 2);
  return (((car - flagged + h) % n) + n) % n - h;
}

function sectorMatches(car, flagged, n) {
  // same rule as src/ingest/sectors.py: the flagged sector, 2 downstream or 1 upstream
  const off = sectorOffset(car, flagged, n);
  return off >= -1 && off <= 2;
}

function carNum(drv) {
  const n = parseInt(drv, 10);
  return Number.isFinite(n) ? n : 1e6;
}

function carsText(cars) {
  if (!cars.length) return "";
  const nums = cars.map((c) => `#${c}`);
  return (cars.length === 1 ? "CAR " : "CARS ") + nums.join(", ");
}

function flagIcon(flag) {
  // inline SVG flag icons, black on the coloured badge (white on red)
  const ink = flag === "RED" ? "#fff" : "#000";
  const pole = `<rect x="5" y="4" width="3" height="32" rx="1.5" fill="${ink}"/>`;
  const wave = (x, y, fill) =>
    `<path d="M${x} ${y} q8 -4 16 0 q7 3 14 -1 v14 q-7 4 -14 1 q-8 -4 -16 0z" fill="${fill}" stroke="${ink}" stroke-width="2" stroke-linejoin="round"/>`;
  if (flag === "DOUBLE_YELLOW") {
    return `<svg viewBox="0 0 44 40" aria-hidden="true">${pole}${wave(9, 6, "#ffd400")}${wave(9, 18, "#ffd400")}</svg>`;
  }
  if (flag === "VSC" || flag === "SC") {
    return `<svg viewBox="0 0 44 40" aria-hidden="true">${pole}<rect x="9" y="6" width="30" height="20" rx="2" fill="${ink}"/>` +
      `<text x="24" y="21" text-anchor="middle" font-family="system-ui, sans-serif" font-size="${flag === "VSC" ? 10 : 12}" font-weight="900" fill="${flag === "RED" ? "#e10600" : "#fff"}">${flag}</text></svg>`;
  }
  const fill = flag === "RED" ? "#e10600" : "#ffd400";
  return `<svg viewBox="0 0 44 40" aria-hidden="true">${pole}${wave(9, 8, fill)}</svg>`;
}

// --- state ------------------------------------------------------------------------

const S = {
  t: null,                    // latest replay time from the ticks
  lapLength: null,            // from GET /track, else the largest dist seen
  nSectors: 1000,             // from GET /track, else no wrap
  detections: new Map(),      // id -> detection
  cars: new Map(),            // drv -> {t, dist, speed, inPit}
  hist: new Map(),            // drv -> {bins: Map(bin -> speed), prev: Map, lastDist}
  field: new Map(),           // bin -> recent speeds of every car there (field reference)
  level: new Map(),           // scope (sector number or TRACK) -> current rank
  episode: new Map(),         // scope -> [{t, rank}] raises of the current episode
  official: new Map(),        // scope -> [{t, rank}] official messages seen
  cause: new Map(),           // sector -> {cars, type} last known cause
  banners: new Map(),         // scope -> {el, rec, rank}
  clock: null,
};

const el = (id) => document.getElementById(id);
const ui = { conn: el("conn"), replayT: el("replay-t"), banners: el("banners"), clock: el("clockbox"),
  clockFlag: el("clock-flag"), clockValue: el("clock-value"), clockCars: el("clock-cars"),
  clockResult: el("clock-result"), ticker: el("ticker") };

function resetAll() {
  S.t = null;
  S.detections.clear();
  S.cars.clear();
  S.hist.clear();
  S.field.clear();
  S.level.clear();
  S.episode.clear();
  S.official.clear();
  S.cause.clear();
  for (const b of S.banners.values()) b.el.remove();
  S.banners.clear();
  S.clock = null;
  hideClock();
}

// --- episodes and lead times (same rules as src/lab/voice.py) ------------------------

function raiseScope(scope, t, rank) {
  const cur = S.level.get(scope) || 0;
  if (cur === 0) S.episode.set(scope, []);
  if (rank > cur) {
    S.level.set(scope, rank);
    S.episode.get(scope).push({ t, rank });
  }
}

function scopesMatching(scope) {
  if (scope === TRACK) return [TRACK];
  const out = [];
  for (const s of S.episode.keys()) if (s !== TRACK && sectorMatches(s, scope, S.nSectors)) out.push(s);
  return out;
}

function ourFirst(rank, scope, tOfficial) {
  // when we first raised at least this level for a matching scope, within the confirm window
  let best = null;
  for (const s of scopesMatching(scope)) {
    for (const r of S.episode.get(s) || []) {
      if (r.rank >= rank && r.t >= tOfficial - CONFIRM_WINDOW_S && (best === null || r.t < best)) best = r.t;
    }
  }
  return best;
}

function officialScopes(scope) {
  // official messages that cover this scope: track-wide ones cover every sector
  if (scope === TRACK) return [TRACK];
  return [...S.official.keys()].filter((s) => s === TRACK || sectorMatches(scope, s, S.nSectors));
}

function officialFirst(rank, scope, tRec) {
  // the earliest official message of at least this level for this scope in the window before our rec
  let best = null;
  for (const s of officialScopes(scope)) {
    for (const o of S.official.get(s) || []) {
      if (o.rank >= rank && o.t <= tRec && o.t >= tRec - CONFIRM_WINDOW_S && (best === null || o.t < best)) best = o.t;
    }
  }
  return best;
}

function causeOf(rec, msector) {
  const dets = (rec.source_detections || []).map((id) => S.detections.get(id)).filter(Boolean);
  if (dets.length) {
    const cars = [...new Set(dets.flatMap((d) => (d.drivers || []).map(String)))].sort((a, b) => carNum(a) - carNum(b));
    const last = dets[dets.length - 1];
    const type = TYPE_TEXT[last.type] ? last.type : null;
    S.cause.set(msector, { cars, type });
    return { cars, type };
  }
  return S.cause.get(msector) || { cars: [], type: null };
}

// --- banners ---------------------------------------------------------------------

function leadHtml(rec, cause) {
  const scope = GLOBAL_FLAGS.has(rec.flag) ? TRACK : rec.msector;
  const rank = RANK[rec.flag];
  const rc = officialFirst(rank, scope, rec.t);
  if (rc !== null) return { cls: "behind", text: `RACE CONTROL FIRST BY ${(rec.t - rc).toFixed(1)} s` };
  const ours = ourFirst(rank, scope, rec.t);
  const off = latestOfficialAfter(rank, scope, ours);
  if (off !== null) return { cls: "ahead", text: `FAST FLAG AHEAD BY ${(off - ours).toFixed(1)} s` };
  return { cls: "wait", text: "WAITING FOR RACE CONTROL" };
}

function latestOfficialAfter(rank, scope, ours) {
  // an official message of at least this level that arrived after our first raise
  if (ours === null) return null;
  let best = null;
  for (const s of officialScopes(scope)) {
    for (const o of S.official.get(s) || []) if (o.rank >= rank && o.t >= ours && (best === null || o.t < best)) best = o.t;
  }
  return best;
}

function bannerHtml(rec, cause) {
  const parts = [];
  if (cause.cars.length) parts.push(escapeHtml(carsText(cause.cars)));
  if (cause.type) parts.push(TYPE_TEXT[cause.type]);
  parts.push(`SECTOR ${escapeHtml(rec.msector)}`);
  const lead = leadHtml(rec, cause);
  return `<div class="badge">${flagIcon(rec.flag)}<span>${FLAG_TEXT[rec.flag]}</span></div>` +
    `<div class="body">` +
    `<div class="line1">${parts.join('<span class="sep">|</span>')}</div>` +
    `<div class="line2">CONFIDENCE <span class="conf">${Number(rec.confidence).toFixed(2)}</span>` +
    `<span class="sep"> · </span>${escapeHtml(rec.reason || rec.message || "")}</div>` +
    `<div class="lead ${lead.cls}">${lead.text}</div>` +
    `</div><div class="mark">FAST FLAG</div>`;
}

function showBanner(rec, cause) {
  const scope = GLOBAL_FLAGS.has(rec.flag) ? TRACK : rec.msector;
  const old = S.banners.get(scope);
  if (old) old.el.remove();
  const div = document.createElement("div");
  div.className = `banner flag-${rec.flag}`;
  div.dataset.scope = String(scope);
  div.innerHTML = bannerHtml(rec, cause);
  ui.banners.appendChild(div);
  S.banners.set(scope, { el: div, rec, cause, rank: RANK[rec.flag] });
  while (S.banners.size > BANNER_MAX) {
    const [k, b] = S.banners.entries().next().value;   // the oldest
    b.el.remove();
    S.banners.delete(k);
  }
}

function refreshLeads() {
  for (const b of S.banners.values()) {
    const leadEl = b.el.querySelector(".lead");
    const lead = leadHtml(b.rec, b.cause);
    leadEl.className = `lead ${lead.cls}`;
    leadEl.textContent = lead.text;
  }
}

function dropBanner(scope) {
  const b = S.banners.get(scope);
  if (!b) return;
  S.banners.delete(scope);
  b.el.classList.add("out");
  setTimeout(() => b.el.remove(), 450);
}

// --- exposure clock -------------------------------------------------------------------

function startClock(rec, cause) {
  S.clock = { t0: rec.t, rank: RANK[rec.flag], flag: rec.flag, stricken: new Set(cause.cars),
    crashDist: null, resting: false, exposed: [], frozen: false, official: null, clearedAt: null };
  for (const drv of cause.cars) {
    const c = S.cars.get(drv);
    if (c && c.dist !== null) S.clock.crashDist = c.dist;
  }
  ui.clock.className = `running flag-${rec.flag}`;
  ui.clockFlag.textContent = FLAG_TEXT[rec.flag];
  ui.clockResult.textContent = "";
  ui.clockResult.className = "clock-result";
  ui.ticker.innerHTML = "";
  updateClockText();
}

function freezeClock(tOfficial) {
  const c = S.clock;
  if (!c || c.frozen) return;
  c.frozen = true;
  c.official = tOfficial;
  ui.clock.classList.remove("running");
  ui.clock.classList.add("frozen");
  const lead = tOfficial - c.t0;
  ui.clockResult.className = "clock-result ahead";
  ui.clockResult.textContent = `Race control +${lead.toFixed(1)} s | Cars exposed ${c.exposed.length}`;
  updateClockText();
}

function endClockWithoutOfficial() {
  const c = S.clock;
  if (!c || c.frozen) return;
  c.frozen = true;
  ui.clock.classList.remove("running");
  ui.clock.classList.add("frozen");
  ui.clockResult.className = "clock-result";
  ui.clockResult.textContent = `No official ${FLAG_TEXT[c.flag]} | Cars exposed ${c.exposed.length}`;
}

function hideClock() {
  ui.clock.className = "hidden";
}

function updateClockText() {
  const c = S.clock;
  if (!c || S.t === null) return;
  const end = c.frozen && c.official !== null ? c.official : S.t;
  const secs = Math.max(0, end - c.t0);
  ui.clockValue.innerHTML = `${secs.toFixed(1)}<span class="unit">s</span>`;
  ui.clockCars.innerHTML = `CARS EXPOSED <b>${c.exposed.length}</b>`;
}

function pushTicker(p) {
  const li = document.createElement("li");
  li.className = "new";
  const pct = p.pct === null ? `<span class="pct">no reference yet</span>`
    : `<span class="pct">${p.pct.toFixed(0)}% of ${p.ref}</span>`;
  li.innerHTML = `<span class="car">#${escapeHtml(p.drv)}</span><span>${p.speed.toFixed(0)} km/h</span>${pct}`;
  ui.ticker.insertBefore(li, ui.ticker.firstChild);
  while (ui.ticker.children.length > TICKER_MAX) ui.ticker.removeChild(ui.ticker.lastChild);
}

function followStricken(t) {
  // the crash location is where the stricken car comes to rest (or is now, while still moving)
  const c = S.clock;
  if (c.resting || t > c.t0 + REST_WITHIN_S) return;
  for (const drv of c.stricken) {
    const car = S.cars.get(drv);
    if (!car || car.dist === null) continue;
    c.crashDist = car.dist;
    if (car.speed < REST_KMH) c.resting = true;
  }
}

function fieldMedian(bin) {
  // median speed of every car seen at this point so far (any lap), once 3 samples exist
  const v = (S.field.get(bin) || []).slice().sort((a, b) => a - b);
  return v.length >= 3 ? v[Math.floor(v.length / 2)] : undefined;
}

function racingSpeed(drv, dist, speed) {
  // racing speed: at least 80% of the car's own speed here on its previous lap; before it has
  // one, 80% of the field's median speed here; before anyone has passed, a fixed 120 km/h
  const h = S.hist.get(drv);
  const bin = Math.floor(dist / BIN_M);
  let ref = h && h.prev ? (h.prev.get(bin) ?? h.prev.get(bin - 1) ?? h.prev.get(bin + 1)) : undefined;
  let label = "own pace";
  if (!(ref > 0)) {
    ref = fieldMedian(bin) ?? fieldMedian(bin - 1) ?? fieldMedian(bin + 1);
    label = "field pace";
  }
  if (!(ref > 0)) return { racing: speed >= RACING_FALLBACK_KMH, pct: null, ref: null };
  return { racing: speed >= RACING_SHARE * ref, pct: 100 * speed / ref, ref: label };
}

function checkPasses(prev, drv, car) {
  // did this car cross the crash location between its previous sample and this one?
  const c = S.clock;
  if (!c || c.frozen || c.crashDist === null || c.stricken.has(drv)) return;
  if (!prev || prev.dist === null || car.dist === null || car.inPit || prev.inPit) return;
  const L = S.lapLength;
  if (!L) return;
  const step = circ(car.dist - prev.dist, L);
  const toCrash = circ(c.crashDist - prev.dist, L);
  if (!(step > 0 && step < MAX_STEP_M && toCrash >= 0 && toCrash < step)) return;
  const frac = toCrash / step;
  const tp = prev.t + frac * (car.t - prev.t);
  if (tp < c.t0) return;
  const speed = prev.speed + frac * (car.speed - prev.speed);
  const { racing, pct, ref } = racingSpeed(drv, c.crashDist, speed);
  if (!racing) return;
  const pass = { drv, t: tp, speed, pct, ref };
  c.exposed.push(pass);
  pushTicker(pass);
}

// --- per-car speed memory (previous lap) ------------------------------------------------

function remember(drv, dist, speed) {
  let h = S.hist.get(drv);
  if (!h) {
    h = { bins: new Map(), prev: null, lastDist: null };
    S.hist.set(drv, h);
  }
  if (S.lapLength && h.lastDist !== null && dist < h.lastDist - S.lapLength / 2) {
    h.prev = h.bins;          // crossed the line: last lap becomes the reference
    h.bins = new Map();
  }
  const bin = Math.floor(dist / BIN_M);
  h.bins.set(bin, speed);
  h.lastDist = dist;
  if (speed >= FIELD_MIN_KMH) {               // a crawling or stopped car is not a reference for the field
    let f = S.field.get(bin);
    if (!f) S.field.set(bin, (f = []));
    if (f.length >= FIELD_MAX) f.shift();
    f.push(speed);
  }
}

// --- envelope handlers ------------------------------------------------------------------

function onTick(tick) {
  const t = Number(tick.t);
  if (S.t !== null && t < S.t - RESET_JUMP_S) resetAll();
  S.t = t;
  ui.replayT.textContent = `REPLAY t ${t.toFixed(1)} s · LAP ${tick.lap}`;
  let maxDist = 0;
  for (const car of tick.cars) maxDist = Math.max(maxDist, Number(car.dist) || 0);
  if (maxDist > 0 && !(S.lapLength >= maxDist)) S.lapLength = maxDist + (S.lapLength ? 0.5 : 100);   // the track's last sector ends before the line
  if (S.clock && !S.clock.frozen) followStricken(t);
  for (const car of tick.cars) {
    const drv = String(car.drv);
    const dist = Number.isFinite(car.dist) ? car.dist : null;
    const cur = { t, dist, speed: Number(car.speed) || 0, inPit: Boolean(car.in_pit) };
    const prev = S.cars.get(drv);
    if (dist !== null && !cur.inPit) remember(drv, dist, cur.speed);
    S.cars.set(drv, cur);
    checkPasses(prev, drv, cur);
  }
  if (S.clock) {
    if (!S.clock.frozen) followStricken(t);
    updateClockText();
    if (S.clock.clearedAt !== null && t - S.clock.clearedAt > HIDE_AFTER_CLEAR_S) {
      S.clock = null;
      hideClock();
    }
  }
}

function onDetection(det) {
  S.detections.set(String(det.id), det);
}

function onRec(rec) {
  const flag = String(rec.flag);
  if (!(flag in RANK)) return;
  const msector = Number(rec.msector);
  const t = Number(rec.t);
  if (flag === "CLEAR") return onClear(rec, msector, t);
  const rank = RANK[flag];
  const cause = causeOf(rec, msector);
  const scope = GLOBAL_FLAGS.has(flag) ? TRACK : msector;
  const levelUp = rank > (S.level.get(scope) || 0);
  raiseScope(scope, t, rank);
  if (scope === TRACK) raiseScope(msector, t, rank);     // the cause sector is covered by the track-wide flag too
  showBanner(rec, cause);
  if (scope === TRACK && levelUp) {
    const rcFirst = officialFirst(rank, TRACK, t);
    if (rcFirst === null && (!S.clock || S.clock.frozen || rank > S.clock.rank)) startClock(rec, cause);
  }
}

function onClear(rec, msector, t) {
  const msg = String(rec.message || "").toUpperCase();
  const sectorClear = msg.includes("SECTOR") || ((S.level.get(TRACK) || 0) === 0 && (S.level.get(msector) || 0) > 0);
  const scope = sectorClear ? msector : TRACK;
  if ((S.level.get(scope) || 0) === 0) return;
  S.level.set(scope, 0);
  if (sectorClear) S.cause.delete(msector);
  dropBanner(scope);
  if (!sectorClear && S.clock) {
    if (!S.clock.frozen) endClockWithoutOfficial();
    S.clock.clearedAt = t;
  }
}

function onOfficial(off) {
  const flag = String(off.flag);
  if (!(flag in RANK) || flag === "CLEAR") return;
  const rank = RANK[flag];
  const t = Number(off.t);
  const scope = off.msector === null || off.msector === undefined ? TRACK : Number(off.msector);
  if (!S.official.has(scope)) S.official.set(scope, []);
  S.official.get(scope).push({ t, rank });
  refreshLeads();
  if (scope === TRACK && S.clock && !S.clock.frozen && rank >= S.clock.rank) freezeClock(t);
}

// --- websocket ---------------------------------------------------------------------------

function connect() {
  const ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    ui.conn.textContent = "LIVE REPLAY";
    ui.conn.classList.add("on");
  };
  ws.onclose = () => {
    ui.conn.textContent = "RECONNECTING";
    ui.conn.classList.remove("on");
    setTimeout(connect, 2000);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (event) => {
    let env;
    try {
      env = JSON.parse(event.data);
    } catch (e) {
      return;
    }
    if (!env || typeof env !== "object" || !env.data || typeof env.data !== "object") return;
    try {
      if (env.kind === "tick") onTick(env.data);
      else if (env.kind === "detection") onDetection(env.data);
      else if (env.kind === "rec") onRec(env.data);
      else if (env.kind === "official") onOfficial(env.data);
    } catch (e) {
      console.warn("overlay: skipped envelope", env.kind, e);
    }
  };
}

async function init() {
  try {
    const track = await (await fetch("/track")).json();
    let length = 0;
    for (const s of track.msectors || []) length = Math.max(length, Number(s.start_dist) || 0, Number(s.end_dist) || 0);
    if (length > 0) S.lapLength = length;
    if ((track.msectors || []).length) S.nSectors = track.msectors.length;
  } catch (e) {
    console.warn("overlay: GET /track failed, lap length from ticks", e);
  }
  connect();
}

init();
