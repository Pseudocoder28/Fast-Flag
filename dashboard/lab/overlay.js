"use strict";

// Fast Flag broadcast overlay. No build step, no CDN. Subscribes to ws://<host>/stream
// and consumes the existing envelopes only (tick, detection, rec, official).
//
// Strictly causal: everything on screen comes from envelopes already received,
// and every clock runs on replay time from the ticks, never on the wall clock. The only
// wall-clock timers are the websocket reconnect, the banner slide-out, a GET /status
// poll that resets the overlay when another race is loaded, and the standby check that
// notices a paused replay; none of them is displayed as a time.
//
// While no flag is out, a standby panel shows the race, the lap and how many cars are
// being watched (or that the replay is paused), so the overlay never looks dead.
//
// - Every rec at YELLOW or above slides in a lower-third banner.
// - Exposure Clock: when our rec reaches VSC, SC or RED before the official message
//   of that level, it counts replay seconds and the cars passing the stricken car
//   at racing speed. The official message freezes it.
// - After the official message: FAST FLAG AHEAD BY X.X s, or RACE CONTROL FIRST BY X.X s.
// - When we lift a flag, its banner turns into a green CLEAR banner for a few seconds.
// - Race control panel: the official messages as they arrive, clears included, each raise
//   with how much earlier Fast Flag called it.

const WS_URL = `ws://${location.host}/stream`;
const RESET_JUMP_S = 2.0;          // same rule as the race control engine and the dashboard
const CONFIRM_WINDOW_S = 120.0;    // an official message and our raise this far apart are about different things
const MIN_LEAD_S = 0.05;          // below the displayed resolution: not a lead
const STATUS_POLL_MS = 3000;       // how often GET /status is checked for a race switch
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
const CLEAR_BANNER_MS = 5000;      // a green CLEAR banner stays this long (wall time, like the slide-out)
const RC_MAX = 4;                  // race control messages shown

const RANK = { CLEAR: 0, YELLOW: 1, DOUBLE_YELLOW: 2, VSC: 3, SC: 4, RED: 5 };
const GLOBAL_FLAGS = new Set(["VSC", "SC", "RED"]);
const FLAG_TEXT = { YELLOW: "YELLOW FLAG", DOUBLE_YELLOW: "DOUBLE YELLOW", VSC: "VIRTUAL SAFETY CAR",
  SC: "SAFETY CAR", RED: "RED FLAG" };
const RC_TEXT = { YELLOW: "YELLOW", DOUBLE_YELLOW: "DOUBLE YELLOW", VSC: "VSC", SC: "SAFETY CAR", RED: "RED FLAG",
  CLEAR: "CLEAR" };
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
  const fill = flag === "RED" ? "#e10600" : flag === "CLEAR" ? "#4cc36a" : "#ffd400";
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
  official: new Map(),        // scope -> [{t, rank}] official flags still in force
  cause: new Map(),           // sector -> {cars, type} last known cause
  banners: new Map(),         // scope -> {el, rec, cause, scope}, oldest first
  race: null,                 // race id and start time from GET /status, to reset on a race switch
  clock: null,
};

const el = (id) => document.getElementById(id);
const ui = { conn: el("conn"), replayT: el("replay-t"), banners: el("banners"), clock: el("clockbox"),
  clockFlag: el("clock-flag"), clockValue: el("clock-value"), clockCars: el("clock-cars"),
  clockResult: el("clock-result"), ticker: el("ticker"), standby: el("standby"), sbRace: el("sb-race"),
  sbState: el("sb-state"), rcList: el("rc-list"), rcEmpty: el("rc-empty") };

// --- standby: shown while no flag is out, so the overlay never looks dead ----------------

const PAUSED_AFTER_MS = 1500;      // no tick for this long (wall time): the replay is paused or stopped
const view = { raceName: null, lap: null, watching: 0, connected: false, lastTickWall: null };

function raceTitle(id) {
  return id ? String(id).split("|")[0].replace(/_/g, " ").toUpperCase() : null;
}

function updateStandby() {
  const paused = view.lastTickWall === null || performance.now() - view.lastTickWall > PAUSED_AFTER_MS;
  ui.standby.classList.toggle("hidden", S.banners.size > 0);
  ui.standby.classList.toggle("paused", paused);
  ui.standby.querySelector(".sb-badge").textContent = !view.connected ? "OFFLINE" : paused ? "PAUSED" : "MONITORING";
  const race = view.raceName ? `${view.raceName} GP` : "FAST FLAG";
  ui.sbRace.textContent = view.lap === null ? race : `${race} · LAP ${view.lap}`;
  const at = S.t === null ? "" : ` at t ${S.t.toFixed(1)} s`;
  ui.sbState.textContent = !view.connected ? "Connecting to the replay server"
    : S.t === null ? "Connected. Start the replay to see the cars"
    : paused ? `Replay paused${at}. Press play on the pit wall`
    : `Watching ${view.watching} cars · no flag from Fast Flag${at}`;
  ui.conn.textContent = !view.connected ? "RECONNECTING" : paused ? "REPLAY PAUSED" : "REPLAY RUNNING";
  ui.conn.classList.toggle("on", view.connected && !paused);
  ui.conn.classList.toggle("paused", view.connected && paused);
}

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
  ui.rcList.innerHTML = "";
  ui.rcEmpty.classList.remove("hidden");
  S.clock = null;
  hideClock();
  updateStandby();
}

// --- our episodes and the official feed ------------------------------------------------
//
// Ours: per scope (a sector, or TRACK for VSC, SC and RED) the current level and the raises
// of the open episode, from CLEAR up and back. Official: per scope, the official flags still in
// force. An official CLEAR for a sector closes that sector, an official TRACK CLEAR closes
// everything, so an earlier incident's messages never speak for a new one.
//
// Lead texts compare like with like: a sector banner against official messages for a matching
// sector (the same one, 2 downstream or 1 upstream), a track banner against track-wide ones,
// each at least the banner's level, measured from the first time this episode reached that
// level. A re-sent rec (the engine re-sends a flag when the ANOMALY detector corroborates it)
// never moves that time.

function raiseScope(scope, t, rank) {
  const cur = S.level.get(scope) || 0;
  if (cur === 0) S.episode.set(scope, []);
  if (rank > cur) {
    S.level.set(scope, rank);
    S.episode.get(scope).push({ t, rank });
  }
}

function ourFirstAt(scope, rank) {
  const raise = (S.episode.get(scope) || []).find((r) => r.rank >= rank);
  return raise ? raise.t : null;
}

function officialsFor(scope, rank) {
  // official messages still in force for this scope, at least this level
  const keys = scope === TRACK ? [TRACK]
    : [...S.official.keys()].filter((s) => s !== TRACK && sectorMatches(scope, s, S.nSectors));
  return keys.flatMap((s) => S.official.get(s) || []).filter((o) => o.rank >= rank);
}

function leadFor(scope, rank) {
  const ours = ourFirstAt(scope, rank);
  if (ours === null) return { cls: "wait", text: "WAITING FOR RACE CONTROL" };
  const offs = officialsFor(scope, rank).filter((o) => Math.abs(o.t - ours) <= CONFIRM_WINDOW_S);
  if (!offs.length) return { cls: "wait", text: "WAITING FOR RACE CONTROL" };
  const d = Math.min(...offs.map((o) => o.t)) - ours;
  if (d <= -MIN_LEAD_S) return { cls: "behind", text: `RACE CONTROL FIRST BY ${(-d).toFixed(1)} s` };
  if (d < MIN_LEAD_S) return { cls: "wait", text: "LEVEL WITH RACE CONTROL" };
  return { cls: "ahead", text: `FAST FLAG AHEAD BY ${d.toFixed(1)} s` };
}

function causeOf(rec, msector) {
  // cars and type behind a rec: the type of its last physical detection (ANOMALY only when
  // nothing else corroborates); a rec without sources keeps the sector's last known cause
  const dets = (rec.source_detections || []).map((id) => S.detections.get(id)).filter(Boolean);
  if (dets.length) {
    const cars = [...new Set(dets.flatMap((d) => (d.drivers || []).map(String)))].sort((a, b) => carNum(a) - carNum(b));
    const types = dets.map((d) => d.type).filter((x) => Object.hasOwn(TYPE_TEXT, x));
    const physical = types.filter((x) => x !== "ANOMALY");
    const type = (physical.length ? physical : types).at(-1) ?? null;
    S.cause.set(msector, { cars, type });
    return { cars, type };
  }
  return S.cause.get(msector) || { cars: [], type: null };
}

// --- banners ---------------------------------------------------------------------

function bannerHtml(rec, cause, scope) {
  const parts = [];
  if (cause.cars.length) parts.push(escapeHtml(carsText(cause.cars)));
  if (cause.type) parts.push(TYPE_TEXT[cause.type]);
  parts.push(`SECTOR ${escapeHtml(rec.msector)}`);
  const lead = leadFor(scope, RANK[rec.flag]);
  return `<div class="badge">${flagIcon(rec.flag)}<span>${FLAG_TEXT[rec.flag]}</span></div>` +
    `<div class="body">` +
    `<div class="line1">${parts.join('<span class="sep">|</span>')}</div>` +
    `<div class="line2">CONFIDENCE <span class="conf">${Number(rec.confidence).toFixed(2)}</span>` +
    `<span class="sep"> · </span>${escapeHtml(rec.reason || rec.message || "")}</div>` +
    `<div class="lead ${lead.cls}">${lead.text}</div>` +
    `</div><div class="mark">FAST FLAG</div>`;
}

function showBanner(rec, cause, scope) {
  const old = S.banners.get(scope);
  if (old) {
    old.el.remove();
    S.banners.delete(scope);            // re-inserted below, so Map order stays oldest first
  }
  const div = document.createElement("div");
  div.className = `banner flag-${rec.flag}`;
  div.dataset.scope = String(scope);
  div.innerHTML = bannerHtml(rec, cause, scope);
  ui.banners.appendChild(div);
  updateStandby();
  S.banners.set(scope, { el: div, rec, cause, scope });
  while (S.banners.size > BANNER_MAX) {
    const [k, b] = S.banners.entries().next().value;   // the oldest
    b.el.remove();
    S.banners.delete(k);
  }
}

function refreshLeads() {
  for (const b of S.banners.values()) {
    if (b.clear) continue;                // a CLEAR banner has no lead
    const leadEl = b.el.querySelector(".lead");
    const lead = leadFor(b.scope, RANK[b.rec.flag]);
    leadEl.className = `lead ${lead.cls}`;
    leadEl.textContent = lead.text;
  }
}

function showClearBanner(rec, scope) {
  // our flag for this scope was lifted: a green banner for a few seconds, then it slides out
  const old = S.banners.get(scope);
  if (old) {
    old.el.remove();
    S.banners.delete(scope);
  }
  const track = scope === TRACK;
  const div = document.createElement("div");
  div.className = "banner flag-CLEAR";
  div.dataset.scope = String(scope);
  div.innerHTML = `<div class="badge">${flagIcon("CLEAR")}<span>${track ? "TRACK CLEAR" : "CLEAR"}</span></div>` +
    `<div class="body"><div class="line1">${track ? "GREEN FLAG, RACING RESUMES" : `SECTOR ${escapeHtml(rec.msector)}`}</div>` +
    `<div class="line2">${escapeHtml(rec.reason || rec.message || "")}</div></div><div class="mark">FAST FLAG</div>`;
  ui.banners.appendChild(div);
  const entry = { el: div, rec, cause: { cars: [], type: null }, scope, clear: true };
  S.banners.set(scope, entry);
  updateStandby();
  while (S.banners.size > BANNER_MAX) {
    const [k, b] = S.banners.entries().next().value;
    b.el.remove();
    S.banners.delete(k);
  }
  setTimeout(() => { if (S.banners.get(scope) === entry) dropBanner(scope); }, CLEAR_BANNER_MS);
}

function dropBanner(scope) {
  const b = S.banners.get(scope);
  if (!b) return;
  S.banners.delete(scope);
  b.el.classList.add("out");
  updateStandby();
  setTimeout(() => b.el.remove(), 450);
}

// --- exposure clock -------------------------------------------------------------------

function startClock(rec, cause) {
  S.clock = { t0: Number(rec.t), rank: RANK[rec.flag], flag: rec.flag, stricken: new Set(cause.cars),
    crashDist: null, resting: false, exposed: [], counted: new Set(), frozen: false, end: null, clearedAt: null };
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
  // an official message of at least the clock's level: the exposure window ends there
  const c = S.clock;
  if (!c || c.frozen) return;
  c.frozen = true;
  c.end = Math.max(tOfficial, c.t0);
  ui.clock.classList.remove("running");
  ui.clock.classList.add("frozen");
  ui.clockResult.className = "clock-result ahead";
  ui.clockResult.textContent = `Race control +${(tOfficial - c.t0).toFixed(1)} s | Cars exposed ${c.exposed.length}`;
  updateClockText();
}

function endClockWithoutOfficial(t) {
  // our flag cleared before race control called that level: the window ends at our clear
  const c = S.clock;
  if (!c || c.frozen) return;
  c.frozen = true;
  c.end = Math.max(t, c.t0);
  ui.clock.classList.remove("running");
  ui.clock.classList.add("frozen");
  ui.clockResult.className = "clock-result";
  ui.clockResult.textContent = `No official ${FLAG_TEXT[c.flag]} | Cars exposed ${c.exposed.length}`;
  updateClockText();
}

function hideClock() {
  ui.clock.className = "hidden";
}

function updateClockText() {
  const c = S.clock;
  if (!c || S.t === null) return;
  const secs = Math.max(0, (c.end ?? S.t) - c.t0);
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
  // median green-flag speed of every car seen at this point so far, once 3 samples exist
  const v = (S.field.get(bin) || []).slice().sort((a, b) => a - b);
  return v.length >= 3 ? v[Math.floor(v.length / 2)] : undefined;
}

function racingSpeed(drv, dist, speed) {
  // racing speed: at least 80% of the car's own speed here on its last green lap; before it has
  // one, 80% of the field's median green speed here; before anyone has passed, a fixed 120 km/h
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
  // did this car cross the crash location between its previous sample and this one? Each car
  // counts once per clock, even while the stricken car is still creeping forward
  const c = S.clock;
  if (!c || c.frozen || c.crashDist === null || c.stricken.has(drv) || c.counted.has(drv)) return;
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
  c.counted.add(drv);
  c.exposed.push(pass);
  pushTicker(pass);
}

// --- per-car speed memory (last green lap) ------------------------------------------------

function neutralised(tick) {
  // no reference samples under yellow, VSC, SC or red: a lap at neutralised pace is not racing speed
  return String(tick.track_status) !== "1" || (S.official.get(TRACK) || []).length > 0;
}

function remember(drv, dist, speed, green) {
  let h = S.hist.get(drv);
  if (!h) {
    h = { bins: new Map(), prev: null, lastDist: null, green: true };
    S.hist.set(drv, h);
  }
  if (S.lapLength && h.lastDist !== null && dist < h.lastDist - S.lapLength / 2) {
    if (h.green) h.prev = h.bins;   // crossed the line: a fully green lap becomes the reference
    h.bins = new Map();
    h.green = true;
  }
  h.lastDist = dist;
  if (!green) {
    h.green = false;
    return;
  }
  const bin = Math.floor(dist / BIN_M);
  h.bins.set(bin, speed);
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
  if (S.t !== null && Math.abs(t - S.t) > RESET_JUMP_S) resetAll();     // a seek or loop, back or forward
  S.t = t;
  ui.replayT.textContent = `REPLAY t ${t.toFixed(1)} s · LAP ${tick.lap}`;
  view.lastTickWall = performance.now();
  view.lap = tick.lap;
  view.watching = tick.cars.filter((c) => !c.in_pit).length;
  updateStandby();
  let maxDist = 0;
  for (const car of tick.cars) maxDist = Math.max(maxDist, Number(car.dist) || 0);
  if (maxDist > 0 && !(S.lapLength >= maxDist)) S.lapLength = maxDist + (S.lapLength ? 0.5 : 100);   // the track's last sector ends before the line
  if (S.clock && !S.clock.frozen) followStricken(t);
  const green = !neutralised(tick);
  for (const car of tick.cars) {
    const drv = String(car.drv);
    const dist = Number.isFinite(car.dist) ? car.dist : null;
    const cur = { t, dist, speed: Number(car.speed) || 0, inPit: Boolean(car.in_pit) };
    const prev = S.cars.get(drv);
    if (dist !== null && !cur.inPit) remember(drv, dist, cur.speed, green);
    S.cars.set(drv, cur);
    checkPasses(prev, drv, cur);
  }
  if (S.clock) {
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
  if (!Object.hasOwn(RANK, flag)) return;
  const msector = Number(rec.msector);
  const t = Number(rec.t);
  if (!Number.isFinite(msector) || !Number.isFinite(t)) return;
  if (flag === "CLEAR") return onClear(rec, msector, t);
  const rank = RANK[flag];
  const cause = causeOf(rec, msector);
  const scope = GLOBAL_FLAGS.has(flag) ? TRACK : msector;
  const levelUp = rank > (S.level.get(scope) || 0);
  raiseScope(scope, t, rank);
  showBanner(rec, cause, scope);
  if (scope === TRACK && levelUp && (!S.clock || S.clock.frozen)) maybeStartClock(rec, cause, rank, t);
}

function maybeStartClock(rec, cause, rank, t) {
  // run the clock only when race control has not called this level yet; an official that was
  // received before our rec but is stamped later still means we were first, so freeze at once
  const offs = officialsFor(TRACK, rank).map((o) => o.t);
  if (offs.some((ot) => ot <= t)) return;
  startClock(rec, cause);
  if (offs.length) freezeClock(Math.min(...offs));
}

function onClear(rec, msector, t) {
  const msg = String(rec.message || "").toUpperCase();
  const sectorClear = msg.trim() !== "TRACK CLEAR";     // contract: only TRACK CLEAR ends VSC, SC and RED
  const scope = sectorClear ? msector : TRACK;
  if ((S.level.get(scope) || 0) === 0) return;
  S.level.set(scope, 0);
  if (sectorClear) S.cause.delete(msector);
  showClearBanner(rec, scope);
  if (!sectorClear && S.clock) {
    if (!S.clock.frozen) endClockWithoutOfficial(t);
    S.clock.clearedAt = t;
  }
}

function pushOfficial(off, flag, scope, t) {
  // the race control panel: newest first, each raise with how much earlier Fast Flag called it
  let lead = "";
  if (flag !== "CLEAR") {
    const ours = ourFirstAt(scope, RANK[flag]);
    if (ours !== null && Math.abs(t - ours) <= CONFIRM_WINDOW_S) {
      const d = t - ours;
      lead = d >= MIN_LEAD_S ? `<span class="rc-lead ahead">FAST FLAG ${d.toFixed(1)} s EARLIER</span>`
        : d <= -MIN_LEAD_S ? `<span class="rc-lead behind">RACE CONTROL FIRST</span>` : "";
    }
  }
  const li = document.createElement("li");
  li.className = `rc-row flag-${flag}`;
  li.innerHTML = `<span class="rc-badge">${RC_TEXT[flag]}</span>` +
    `<span class="rc-msg">${escapeHtml(off.message || "")}</span>` +
    `<span class="rc-t">t ${t.toFixed(1)} s</span>${lead}`;
  ui.rcList.insertBefore(li, ui.rcList.firstChild);
  while (ui.rcList.children.length > RC_MAX) ui.rcList.removeChild(ui.rcList.lastChild);
  ui.rcEmpty.classList.add("hidden");
}

function onOfficial(off) {
  const flag = String(off.flag);
  if (!Object.hasOwn(RANK, flag)) return;
  const t = Number(off.t);
  if (!Number.isFinite(t)) return;
  const scope = off.msector === null || off.msector === undefined ? TRACK : Number(off.msector);
  pushOfficial(off, flag, scope, t);
  if (flag === "CLEAR") {
    if (scope === TRACK) S.official.clear();       // TRACK CLEAR ends every official flag
    else S.official.delete(scope);
    refreshLeads();
    return;
  }
  const rank = RANK[flag];
  if (!S.official.has(scope)) S.official.set(scope, []);
  S.official.get(scope).push({ t, rank });
  refreshLeads();
  if (scope === TRACK && S.clock && !S.clock.frozen && rank >= S.clock.rank) freezeClock(t);
}

// --- websocket ---------------------------------------------------------------------------

function connect() {
  const ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    view.connected = true;
    updateStandby();
  };
  ws.onclose = () => {
    view.connected = false;
    updateStandby();
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

async function loadTrack() {
  try {
    const track = await (await fetch("/track")).json();
    let length = 0;
    for (const s of track.msectors || []) length = Math.max(length, Number(s.start_dist) || 0, Number(s.end_dist) || 0);
    S.lapLength = length > 0 ? length : null;
    S.nSectors = (track.msectors || []).length || 1000;
  } catch (e) {
    console.warn("overlay: GET /track failed, lap length from ticks", e);
  }
}

async function watchRace() {
  // a race switch (POST /replay with another race) can move time forwards, so the tick-based
  // reset never fires: nothing from the previous race may feed this race's claims
  try {
    const st = await (await fetch("/status")).json();
    const id = `${st.race}|${st.t_start}`;
    if (S.race !== null && id !== S.race) {
      resetAll();
      await loadTrack();
    }
    S.race = id;
    view.raceName = raceTitle(id);
    updateStandby();
  } catch (e) {
    // the server is restarting: the websocket reconnect handles it
  }
}

async function init() {
  await loadTrack();
  await watchRace();
  setInterval(watchRace, STATUS_POLL_MS);
  setInterval(updateStandby, 500);          // notices a paused replay (no ticks) within a second
  connect();
}

init();
