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
// It connects with ?catchup=1: the server first sends everything since the last seek
// (src/replay/hub.py), so the flags already out and race control's messages show at once
// when the page opens, or reopens after a switch from the pit wall. In a precomputed race
// the server does the same after every seek: the tick at the new time, then everything
// before it. An envelope older than the latest tick by more than RESET_JUMP_S (or before the
// first tick) is catch-up: a call made before the page saw it gets its banner and lead, but
// no exposure clock (the cars that passed before then were not seen), and an old clear shows
// no green banner. Cars out of the race (GET /cars, the same list as the pit wall) leave the
// map and are listed beside it.
//
// Replay controls (top): seek back and forward, play or pause, speed. They send the same
// POST /replay as the pit wall, so every open page follows.
//
// - Every rec at YELLOW or above slides in a lower-third banner.
// - Exposure Clock: when our rec reaches VSC, SC or RED before the official message
//   of that level, and race control has no track-wide flag out yet, it counts replay
//   seconds and the cars passing the stricken car at racing speed. The official message
//   freezes it; race control's TRACK CLEAR or our downgrade ends it.
// - After the official message: FAST FLAG AHEAD BY X.X s, or RACE CONTROL FIRST BY X.X s.
// - When we lift a flag, its banner turns into a green CLEAR banner for a few seconds.
// - Race control panel: the official messages as they arrive, clears included. The message
//   that first confirms one of our banners shows that banner's lead; repeats show none.
// - Track map: GET /track's reference line with every car from the ticks, eased between ticks
//   (display only, never ahead of the last tick received). Big in the middle while no flag is
//   out, a smaller top-centre card once one is. Our flagged sectors are coloured, a track-wide flag tints
//   the whole line, and the cars behind a banner are red. Chips: our flag and race control's
//   track status (from the tick).

const WS_URL = `ws://${location.host}/stream?catchup=1`;
const RESET_JUMP_S = 2.0;          // same rule as the race control engine and the dashboard
const CONFIRM_WINDOW_S = 120.0;    // an official message this long before our raise, or after it once our flag is
                                   // down, is about something else (the voice's rule, src/lab/voice.py)
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
const BANNER_MAX = 3;              // CLEAR banners make way first, a live flag banner never goes for one
const HIDE_AFTER_CLEAR_S = 12;     // frozen clock stays this long (replay time) after the track clears
const CLEAR_BANNER_MS = 5000;      // a green CLEAR banner stays this long (wall time, like the slide-out)
const RC_MAX = 4;                  // race control messages shown
const SEEK_SMALL_S = 10;           // arrow keys; shift for SEEK_BIG_S
const SEEK_BIG_S = 30;

const RANK = { CLEAR: 0, YELLOW: 1, DOUBLE_YELLOW: 2, VSC: 3, SC: 4, RED: 5 };
const GLOBAL_FLAGS = new Set(["VSC", "SC", "RED"]);
const FLAG_TEXT = { YELLOW: "YELLOW FLAG", DOUBLE_YELLOW: "DOUBLE YELLOW", VSC: "VIRTUAL SAFETY CAR",
  SC: "SAFETY CAR", RED: "RED FLAG" };
const RC_TEXT = { YELLOW: "YELLOW", DOUBLE_YELLOW: "DOUBLE YELLOW", VSC: "VSC", SC: "SAFETY CAR", RED: "RED FLAG",
  CLEAR: "CLEAR" };
const TYPE_TEXT = { IMPACT: "IMPACT", STOPPED: "STOPPED", SPIN: "SPIN", DROPOUT: "NO DATA",
  MULTI: "MULTI-CAR", ANOMALY: "ANOMALY" };
const TRACK = "track";
const DOWNGRADE_RE = /a downgrade, not a new call/;   // src/racecontrol/engine.py, when our red by time ends

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
  resetT: null,               // the first tick's time since the page connected or jumped (catch-up is older)
  quietTimer: null,
  lapLength: null,            // from GET /track, else the largest dist seen
  nSectors: 1000,             // from GET /track, else no wrap
  detections: new Map(),      // id -> detection
  cars: new Map(),            // drv -> {t, dist, speed, inPit}
  hist: new Map(),            // drv -> {bins: Map(bin -> speed), prev: Map, lastDist}
  field: new Map(),           // bin -> recent speeds of every car there (field reference)
  level: new Map(),           // scope (sector number or TRACK) -> current rank
  episode: new Map(),         // scope -> [{t, rank}] raises of the current episode
  official: new Map(),        // scope -> [{t, rank}] official flags still in force
  offLog: [],                 // every official raise since the last reset: {t, rank, scope, clearedAt}
  cause: new Map(),           // sector -> {cars, type} last known cause
  banners: new Map(),         // scope -> {el, rec, cause, scope}, oldest first
  race: null,                 // race id and start time from GET /status, to reset on a race switch
  clock: null,
};

const el = (id) => document.getElementById(id);
const ui = { stage: el("stage"), conn: el("conn"), replayT: el("replay-t"), banners: el("banners"), clock: el("clockbox"),
  clockFlag: el("clock-flag"), clockValue: el("clock-value"), clockCars: el("clock-cars"),
  clockResult: el("clock-result"), ticker: el("ticker"), standby: el("standby"), sbRace: el("sb-race"),
  sbState: el("sb-state"), rcList: el("rc-list"), rcEmpty: el("rc-empty"), mapbox: el("mapbox"), map: el("map"), mapOut: el("map-out"),
  mapRace: el("map-race"), mapFoot: el("map-foot"), chipFF: el("chip-ff"), chipRC: el("chip-rc") };

// --- standby: shown while no flag is out, so the overlay never looks dead ----------------

const PAUSED_AFTER_MS = 1500;      // no tick for this long (wall time): the replay is paused or stopped
const view = { raceName: null, lap: null, watching: 0, connected: false, lastTickWall: null,
  speed: null, lastSpeed: 1, tStatus: null };

function raceTitle(id) {
  return id ? String(id).split("|")[0].replace(/_/g, " ").toUpperCase() : null;
}

function updateStandby() {
  // paused: the server says so (a paused seek still sends one tick), or no tick for a while
  const paused = view.speed === 0 || view.lastTickWall === null || performance.now() - view.lastTickWall > PAUSED_AFTER_MS;
  ui.standby.classList.toggle("hidden", S.banners.size > 0);
  ui.standby.classList.toggle("paused", paused);
  ui.standby.querySelector(".sb-badge").textContent = !view.connected ? "OFFLINE" : paused ? "PAUSED" : "MONITORING";
  const race = view.raceName ? `${view.raceName} GP` : "FAST FLAG";
  ui.sbRace.textContent = view.lap === null ? race : `${race} · LAP ${view.lap}`;
  const at = S.t === null ? "" : ` at t ${S.t.toFixed(1)} s`;
  ui.sbState.textContent = !view.connected ? "Connecting to the replay server"
    : S.t === null ? "Connected. Start the replay to see the cars"
    : paused ? `Replay paused${at}. Press play`
    : `Watching ${view.watching} cars · no flag from Fast Flag${at}`;
  ui.conn.textContent = !view.connected ? "RECONNECTING" : paused ? "REPLAY PAUSED" : "REPLAY RUNNING";
  ui.conn.classList.toggle("on", view.connected && !paused);
  ui.conn.classList.toggle("paused", view.connected && paused);
  updateMapPanel(paused);
}

function resetAll() {
  quiet();
  S.t = null;
  S.resetT = null;
  S.detections.clear();
  S.cars.clear();
  S.hist.clear();
  S.field.clear();
  S.level.clear();
  S.episode.clear();
  S.official.clear();
  S.offLog = [];
  S.cause.clear();
  for (const b of S.banners.values()) b.el.remove();
  S.banners.clear();
  ui.rcList.innerHTML = "";
  ui.rcEmpty.classList.remove("hidden");
  S.clock = null;
  hideClock();
  M.cars.clear();
  M.top = null;
  M.rcStatus = null;
  M.out = new Map();            // asked again at the next tick
  M.outReady = false;
  M.gen++;
  M.crashed.clear();
  renderOut();
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
// never moves that time, and neither does race control clearing and re-issuing its flag while
// ours stays out. A downgrade (our red by time ended, our SC or VSC stays out) lowers the
// level and drops the higher raises, so a later red is a new call.

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

function lowerTrack(rank, t) {
  // a downgrade: not a new call. The higher raises leave the episode and their clock ends
  S.level.set(TRACK, rank);
  S.episode.set(TRACK, (S.episode.get(TRACK) || []).filter((r) => r.rank <= rank));
  if (S.clock && !S.clock.frozen && S.clock.rank > rank) endClockWithoutOfficial(t);
}

function inWindow(scope, ours, t) {
  // can an official message at t be about our raise of this scope at `ours`? Before it, within
  // the confirm window; after it, any time while our flag is still out, else within the window
  return t <= ours ? ours - t <= CONFIRM_WINDOW_S : (S.level.get(scope) || 0) > 0 || t - ours <= CONFIRM_WINDOW_S;
}

function officialsFor(scope, rank, ours) {
  // race control's messages for this scope, at least this level, about our raise at `ours`:
  // still in force at that time, or sent since
  return S.offLog.filter((o) => o.rank >= rank && inWindow(scope, ours, o.t)
    && (o.clearedAt === null || o.clearedAt > ours)
    && (scope === TRACK ? o.scope === TRACK : o.scope !== TRACK && sectorMatches(scope, o.scope, S.nSectors)));
}

function leadFor(scope, rank) {
  const ours = ourFirstAt(scope, rank);
  if (ours === null) return { cls: "wait", text: "" };     // a downgrade whose call came before this page opened
  const offs = officialsFor(scope, rank, ours);
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
  div.className = `banner flag-${rec.flag}${ui.stage.classList.contains("catchup") ? " instant" : ""}`;
  div.dataset.scope = String(scope);
  div.innerHTML = bannerHtml(rec, cause, scope);
  ui.banners.appendChild(div);
  S.banners.set(scope, { el: div, rec, cause, scope });
  trimBanners();
  updateStandby();
}

function trimBanners() {
  // over BANNER_MAX: CLEAR banners go first, oldest first, then the oldest flag banner
  while (S.banners.size > BANNER_MAX) {
    const all = [...S.banners.entries()];
    const [k, b] = all.find(([, x]) => x.clear) || all[0];
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
  // our track clear while race control's own SC, VSC or red is still out: racing does not resume yet
  const rcOut = Math.max(0, ...(S.official.get(TRACK) || []).map((o) => o.rank));
  const rcFlag = Object.keys(RANK).find((f) => RANK[f] === rcOut);
  // ... or the field is still stopped (a red flag stoppage, GET /cars): the restart is still to come
  const stopped = M.stoppages.some(([a, b]) => S.t !== null && S.t >= a && S.t <= b + 2);
  const line1 = !track ? `SECTOR ${escapeHtml(rec.msector)}`
    : rcOut > 0 ? `RACE CONTROL'S ${FLAG_TEXT[rcFlag]} STILL OUT`
    : stopped ? "RACE STILL STOPPED · RESTART TO COME" : "GREEN FLAG, RACING RESUMES";
  const div = document.createElement("div");
  div.className = "banner flag-CLEAR";
  div.dataset.scope = String(scope);
  div.innerHTML = `<div class="badge">${flagIcon("CLEAR")}<span>${track ? "TRACK CLEAR" : "CLEAR"}</span></div>` +
    `<div class="body"><div class="line1">${line1}</div>` +
    `<div class="line2">${escapeHtml(rec.reason || rec.message || "")}</div></div><div class="mark">FAST FLAG</div>`;
  ui.banners.appendChild(div);
  const entry = { el: div, rec, cause: { cars: [], type: null }, scope, clear: true };
  S.banners.set(scope, entry);
  trimBanners();
  updateStandby();
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

// --- track map --------------------------------------------------------------------------
//
// Display only: it draws what the ticks already said. Between two ticks a car eases from the
// position it was drawn at to its latest position, never towards a future one.

const TRACK_STATUS = { 1: ["GREEN", "CLEAR"], 2: ["YELLOW", "YELLOW"], 4: ["SAFETY CAR", "SC"], 5: ["RED FLAG", "RED"],
  6: ["VSC", "VSC"], 7: ["VSC ENDING", "VSC"] };
const FLAG_COLOUR = { YELLOW: "#ffd400", DOUBLE_YELLOW: "#ffb000", VSC: "#00a3e0", SC: "#ff6b00", RED: "#e10600" };
const RANK_FLAG = Object.fromEntries(Object.entries(RANK).map(([f, r]) => [r, f]));
const CAR_STALE_S = 5;             // a car missing from the ticks this long leaves the map
const EASE_MAX_MS = 400;           // longest ease between two ticks (1x replay sends one every 250 ms)
const MODE_MOVE_MS = 700;          // redraw every frame this long after the map changes size

const M = { track: null, cars: new Map(), top: null, rcStatus: null, tickMs: 250, lastTickWall: null,
  active: null, busyUntil: 0, dirty: true,
  out: new Map(), outReady: false, outKey: "", crashed: new Set(), gen: 0,   // out of the race (GET /cars)
  stoppages: [] };                 // [start, end] field stops (red flag, grid), from GET /cars
const CARS_POLL_MS = 1000;
const CATCHUP_QUIET_MS = 250;      // animations stay off this long after the last catch-up envelope

function isCatchup(t) {
  // from before this page saw the replay: sent before the first tick since the page connected
  // or jumped (the server's history is strictly before that tick; live envelopes never are)
  return S.resetT === null || t < S.resetT;
}

function quiet() {
  // a seek or a connect rebuilds the page from history: a clean cut, no banner slide-ins and no
  // map resize while it does, until the history has stopped arriving
  ui.stage.classList.add("catchup");
  clearTimeout(S.quietTimer);
  S.quietTimer = setTimeout(() => ui.stage.classList.remove("catchup"), CATCHUP_QUIET_MS);
}

async function fetchCars() {
  // until the first answer after a connect or a seek no car is drawn, so a car out of the race
  // never flashes back on; an answer to a question asked before the last seek is dropped
  const gen = M.gen;
  try {
    const r = await (await fetch("/cars")).json();
    if (gen !== M.gen) return;    // asked before the last seek: the next tick asks again
    M.out = new Map((r.out || []).map((o) => [String(o.drv), o]));
    M.stoppages = Array.isArray(r.stoppages) ? r.stoppages : [];
    if (M.top && M.out.has(String(M.top.drv))) M.top = null;   // the next tick finds the fastest car still racing
  } catch (e) {
    if (gen !== M.gen) return;    // an older server without /cars: nothing is taken off the map
  }
  M.outReady = true;
  renderOut();
  M.dirty = true;
}

function renderOut() {
  const out = [...M.out.values()];
  const key = out.map((o) => `${o.drv}:${o.why}:${M.crashed.has(String(o.drv))}`).join(",");
  if (key === M.outKey) return;
  M.outKey = key;
  ui.mapOut.innerHTML = !out.length ? "" : `<li class="out-head">OUT OF THE RACE</li>` + out.map((o) => {
    const d = String(o.drv);
    const crash = M.crashed.has(d) && o.why === "stopped";
    const why = o.why === "no data" ? "NO DATA" : o.why === "in the pit lane" ? "GARAGE"
      : `${crash ? "CRASHED" : "STOPPED"}${o.msector === null || o.msector === undefined ? "" : ` · S${o.msector}`}`;
    return `<li class="out-item${crash ? " crashed" : ""}"><b>#${escapeHtml(d)}</b><span>${escapeHtml(why)}</span></li>`;
  }).join("");
}

function buildTrack(track) {
  // reference line, cumulative distance and each segment's marshal sector (as the pit wall map)
  const pts = (track.ref_line || []).map((p) => [Number(p[0]), Number(p[1])]).filter((p) => p.every(Number.isFinite));
  if (pts.length < 3) return null;
  const n = pts.length;
  const cum = [0];
  for (let i = 1; i < n; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  const total = cum[n - 1] + Math.hypot(pts[0][0] - pts[n - 1][0], pts[0][1] - pts[n - 1][1]);
  const msectors = track.msectors || [];
  const sectorAt = (d) => {
    for (const s of msectors) {
      const a = Number(s.start_dist);
      const b = Number(s.end_dist);
      if (a <= b ? d >= a && d < b : d >= a || d < b) return Number(s.id);
    }
    return null;
  };
  const seg = pts.map((_, i) => sectorAt(((cum[i] + (i === n - 1 ? total : cum[i + 1])) / 2) % total));
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const box = { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
  const corners = (track.corners || []).map((c) => ({ n: c.number, x: Number(c.x), y: Number(c.y) }))
    .filter((c) => Number.isFinite(c.x) && Number.isFinite(c.y));
  return { pts, seg, box, corners, cx: xs.reduce((a, b) => a + b, 0) / n, cy: ys.reduce((a, b) => a + b, 0) / n };
}

function mapTick(tick, t) {
  const now = performance.now();
  if (M.lastTickWall !== null) {
    const gap = now - M.lastTickWall;
    if (gap > 0 && gap < 2000) M.tickMs = 0.8 * M.tickMs + 0.2 * gap;
  }
  M.lastTickWall = now;
  M.rcStatus = String(tick.track_status ?? "");
  for (const car of tick.cars) {
    const x = Number(car.x);
    const y = Number(car.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const drv = String(car.drv);
    const m = M.cars.get(drv);
    const from = m ? carAt(m, now) : [x, y];
    M.cars.set(drv, { fx: from[0], fy: from[1], x, y, wall: now, t, inPit: Boolean(car.in_pit) });
    const speed = Number(car.speed) || 0;
    if (!car.in_pit && !M.out.has(drv) && (!M.top || speed > M.top.speed)) M.top = { drv, speed };
  }
  for (const [drv, m] of M.cars) if (t - m.t > CAR_STALE_S) M.cars.delete(drv);
  M.dirty = true;
}

function carAt(m, now) {
  const a = Math.min(1, Math.max(0, (now - m.wall) / Math.min(M.tickMs, EASE_MAX_MS)));
  const e = a * (2 - a);                                        // ease out
  return [m.fx + (m.x - m.fx) * e, m.fy + (m.y - m.fy) * e];
}

function ourFlag() {
  // the highest flag we have out: track-wide first, else the highest sector flag and its sector
  const g = S.level.get(TRACK) || 0;
  if (g > 0) return { flag: RANK_FLAG[g], text: FLAG_TEXT[RANK_FLAG[g]] };
  let best = null;
  for (const [scope, rank] of S.level) {
    if (scope !== TRACK && rank > 0 && (!best || rank > best.rank)) best = { rank, scope };
  }
  return best ? { flag: RANK_FLAG[best.rank], text: `${FLAG_TEXT[RANK_FLAG[best.rank]]} · S${best.scope}` }
    : { flag: "CLEAR", text: "NO FLAG" };
}

function updateMapPanel(paused) {
  const active = S.banners.size > 0 || !ui.clock.classList.contains("hidden");
  if (active !== M.active) {
    M.active = active;
    ui.mapbox.classList.toggle("active", active);
    M.busyUntil = performance.now() + MODE_MOVE_MS;
  }
  ui.mapRace.textContent = [view.raceName ? `${view.raceName} GP` : null, view.lap !== null ? `LAP ${view.lap}` : null]
    .filter(Boolean).join(" · ");
  const ours = ourFlag();
  ui.chipFF.className = `chip lvl-${ours.flag}`;
  ui.chipFF.innerHTML = `FAST FLAG <b>${escapeHtml(ours.text)}</b>`;
  const rc = TRACK_STATUS[M.rcStatus];
  ui.chipRC.className = `chip lvl-${rc ? rc[1] : "NONE"}`;
  ui.chipRC.innerHTML = `RACE CONTROL <b>${rc ? rc[0] : "WAITING"}</b>`;
  const racing = [...M.cars.entries()].filter(([d]) => !M.out.has(d)).map(([, m]) => m);
  const onTrack = racing.filter((m) => !m.inPit).length;
  const inPit = racing.length - onTrack;
  ui.mapFoot.innerHTML = !M.cars.size ? (view.connected ? "Press play to start the replay" : "Waiting for the replay")
    : `ON TRACK <b>${onTrack}</b> · IN PIT <b>${inPit}</b>` + (M.out.size ? ` · OUT <b>${M.out.size}</b>` : "") +
      (M.top && M.top.speed > 0 ? ` · TOP SPEED <b>#${escapeHtml(M.top.drv)} ${M.top.speed.toFixed(0)} km/h</b>` : "") +
      (paused ? " · PAUSED" : "");
  M.dirty = true;
}

function strickenCars() {
  // the cars behind a live flag banner
  const out = new Set();
  for (const b of S.banners.values()) if (!b.clear) for (const c of b.cause.cars) out.add(String(c));
  return out;
}

function drawMap(now) {
  const cv = ui.map;
  const w = cv.clientWidth;
  const h = cv.clientHeight;
  if (!w || !h) return;
  const dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const tr = M.track;
  if (!tr) return;
  const big = Math.min(w, h) > 320;
  const r = Math.max(8, Math.min(12, Math.min(w, h) / 38));      // 8 px: a two-digit number still reads
  const pad = r * 2 + 4;
  const bw = tr.box.maxX - tr.box.minX || 1;
  const bh = tr.box.maxY - tr.box.minY || 1;
  const k = Math.min((w - 2 * pad) / bw, (h - 2 * pad) / bh);
  const ox = (w - bw * k) / 2;
  const oy = (h - bh * k) / 2;
  const P = (x, y) => [ox + (x - tr.box.minX) * k, h - (oy + (y - tr.box.minY) * k)];   // track is y-up
  const lw = big ? 5 : 3.5;
  const line = (idx) => {
    ctx.beginPath();
    let open = false;
    for (let i = 0; i < tr.pts.length; i++) {
      if (!idx(i)) { open = false; continue; }
      const a = P(...tr.pts[i]);
      const b = P(...tr.pts[(i + 1) % tr.pts.length]);
      if (!open) ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      open = true;
    }
  };
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  // the track: a soft casing, then the line; a track-wide flag of ours tints all of it
  const g = S.level.get(TRACK) || 0;
  line(() => true);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.07)";
  ctx.lineWidth = lw * 3.2;
  ctx.stroke();
  ctx.strokeStyle = g > 0 ? FLAG_COLOUR[RANK_FLAG[g]] : "#62625d";
  ctx.lineWidth = lw;
  ctx.globalAlpha = g > 0 ? 0.85 : 1;
  ctx.stroke();
  ctx.globalAlpha = 1;
  // our sector flags on top
  for (const [scope, rank] of S.level) {
    if (scope === TRACK || rank <= 0) continue;
    line((i) => tr.seg[i] === scope);
    ctx.strokeStyle = FLAG_COLOUR[RANK_FLAG[rank]];
    ctx.lineWidth = lw * 1.9;
    ctx.stroke();
  }
  // start and finish line
  const [s0, s1] = [P(...tr.pts[0]), P(...tr.pts[1])];
  const len = Math.hypot(s1[0] - s0[0], s1[1] - s0[1]) || 1;
  const [nx, ny] = [-(s1[1] - s0[1]) / len, (s1[0] - s0[0]) / len];
  ctx.beginPath();
  ctx.moveTo(s0[0] - nx * lw * 2.2, s0[1] - ny * lw * 2.2);
  ctx.lineTo(s0[0] + nx * lw * 2.2, s0[1] + ny * lw * 2.2);
  ctx.strokeStyle = "#f4f4f2";
  ctx.lineWidth = 2.5;
  ctx.lineCap = "butt";
  ctx.stroke();
  // corner numbers, big map only
  if (big) {
    ctx.fillStyle = "rgba(244, 244, 242, 0.28)";
    ctx.font = `600 ${Math.round(r * 0.95)}px ui-monospace, Menlo, monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const c of tr.corners) {
      const [px, py] = P(c.x, c.y);
      const [cx, cy] = P(tr.cx, tr.cy);
      const d = Math.hypot(px - cx, py - cy) || 1;
      ctx.fillText(String(c.n), px + (px - cx) / d * r * 2.2, py + (py - cy) / d * r * 2.2);
    }
  }
  // a car out of the race is off the map; while it is behind a live flag banner, a hazard
  // marker stays where it stopped
  const hit = strickenCars();
  for (const [drv, o] of M.out) {
    if (!hit.has(drv) || o.x === null || o.x === undefined) continue;
    const [px, py] = P(o.x, o.y);
    const pulse = 0.5 + 0.5 * Math.sin(now / 260);
    ctx.beginPath();
    ctx.arc(px, py, r * (1.5 + 0.6 * pulse), 0, 2 * Math.PI);
    ctx.fillStyle = `rgba(225, 6, 0, ${0.16 + 0.18 * (1 - pulse)})`;
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(px, py - r * 1.05);
    ctx.lineTo(px + r, py + r * 0.75);
    ctx.lineTo(px - r, py + r * 0.75);
    ctx.closePath();
    ctx.fillStyle = "#e10600";
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgba(0, 0, 0, 0.8)";
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.font = `900 ${Math.round(r * 1.05)}px system-ui, -apple-system, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("!", px, py + r * 0.12);
  }
  // cars: the field, then the cars behind a flag on top, pulsing (none until GET /cars answered)
  const order = [...M.cars.entries()].filter(([d]) => M.outReady && !M.out.has(d))
    .sort(([a], [b]) => (hit.has(a) ? 1 : 0) - (hit.has(b) ? 1 : 0));
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = `800 ${Math.round(r * 1.1)}px system-ui, -apple-system, sans-serif`;
  for (const [drv, m] of order) {
    const [px, py] = P(...carAt(m, now));
    const struck = hit.has(drv);
    if (struck) {
      const pulse = 0.5 + 0.5 * Math.sin(now / 180);
      ctx.beginPath();
      ctx.arc(px, py, r * (1.5 + 0.7 * pulse), 0, 2 * Math.PI);
      ctx.fillStyle = `rgba(225, 6, 0, ${0.18 + 0.2 * (1 - pulse)})`;
      ctx.fill();
    }
    ctx.globalAlpha = m.inPit ? 0.35 : 1;
    ctx.beginPath();
    ctx.arc(px, py, r, 0, 2 * Math.PI);
    ctx.fillStyle = struck ? "#e10600" : "#f4f4f2";
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgba(0, 0, 0, 0.8)";
    ctx.stroke();
    ctx.fillStyle = struck ? "#fff" : "#050505";
    ctx.fillText(drv, px, py + 0.5);
    ctx.globalAlpha = 1;
  }
}

function mapFrame(now) {
  // redraw while cars are easing, the map is resizing, a car pulses, or something changed
  const easing = M.lastTickWall !== null && now - M.lastTickWall < Math.min(M.tickMs, EASE_MAX_MS) + 50;
  if (M.dirty || easing || now < M.busyUntil || strickenCars().size) {
    M.dirty = false;
    drawMap(now);
  }
  requestAnimationFrame(mapFrame);
}

// --- envelope handlers ------------------------------------------------------------------

function onTick(tick) {
  const t = Number(tick.t);
  if (S.t !== null && Math.abs(t - S.t) > RESET_JUMP_S) resetAll();     // a seek or loop, back or forward
  const first = S.t === null;
  S.t = t;
  if (first) {
    S.resetT = t;
    fetchCars();
  }
  ui.replayT.textContent = `REPLAY t ${t.toFixed(1)} s · LAP ${tick.lap}`;
  view.lastTickWall = performance.now();
  view.lap = tick.lap;
  view.watching = tick.cars.filter((c) => !c.in_pit && !M.out.has(String(c.drv))).length;
  mapTick(tick, t);
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
  if (isCatchup(Number(det.t))) quiet();
  if (det.type === "IMPACT" || det.type === "MULTI") for (const d of det.drivers || []) M.crashed.add(String(d));
}

function onRec(rec) {
  const flag = String(rec.flag);
  if (!Object.hasOwn(RANK, flag)) return;
  const msector = Number(rec.msector);
  const t = Number(rec.t);
  if (!Number.isFinite(msector) || !Number.isFinite(t)) return;
  if (isCatchup(t)) quiet();
  if (flag === "CLEAR") return onClear(rec, msector, t);
  const rank = RANK[flag];
  const cause = causeOf(rec, msector);
  const scope = GLOBAL_FLAGS.has(flag) ? TRACK : msector;
  const cur = S.level.get(scope) || 0;
  if (scope === TRACK && (rank < cur || DOWNGRADE_RE.test(String(rec.reason || "")))) {
    lowerTrack(rank, t);
    showBanner(rec, cause, scope);
    return;
  }
  raiseScope(scope, t, rank);
  showBanner(rec, cause, scope);
  // catch-up (from before the page saw this time): no exposure clock
  if (scope === TRACK && rank > cur && !isCatchup(t) && (!S.clock || S.clock.frozen)) maybeStartClock(rec, cause, rank, t);
}

function maybeStartClock(rec, cause, rank, t) {
  // run the clock only while race control has no track-wide flag out (the field is not
  // neutralised yet; our red by time only comes under race control's own SC) and has not
  // called this level; an official that was received before our rec but is stamped later
  // still means we were first, so freeze at once
  if ((S.official.get(TRACK) || []).some((o) => o.t <= t)) return;
  const offs = officialsFor(TRACK, rank, t).map((o) => o.t);
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
  if (isCatchup(t)) {                                   // catch-up: an old clear, no green banner
    const b = S.banners.get(scope);
    if (b) b.el.remove();
    S.banners.delete(scope);
    updateStandby();
    return;
  }
  showClearBanner(rec, scope);
  if (!sectorClear && S.clock) {
    if (!S.clock.frozen) endClockWithoutOfficial(t);
    S.clock.clearedAt = t;
  }
}

function panelLead(scope, rank, t) {
  // the lead of the banner this message confirms (leadFor's rules), only when it is race
  // control's first message about it: a repeat, or a message after race control had already
  // called it at this level, gets none. Called before the message joins the log.
  const ourScopes = scope === TRACK ? [TRACK]
    : [...S.episode.keys()].filter((m) => m !== TRACK && sectorMatches(m, scope, S.nSectors));
  let best = null;
  for (const m of ourScopes) {
    const ours = ourFirstAt(m, rank);
    if (ours === null || !inWindow(m, ours, t) || officialsFor(m, rank, ours).length) continue;
    if (best === null || t - ours > best) best = t - ours;
  }
  return best;
}

function pushOfficial(off, flag, scope, t) {
  // the race control panel: newest first; the message that first confirms one of our banners
  // shows how much earlier Fast Flag called it
  let lead = "";
  const d = flag === "CLEAR" ? null : panelLead(scope, RANK[flag], t);
  if (d !== null) {
    lead = d >= MIN_LEAD_S ? `<span class="rc-lead ahead">FAST FLAG ${d.toFixed(1)} s EARLIER</span>`
      : d <= -MIN_LEAD_S ? `<span class="rc-lead behind">RACE CONTROL FIRST</span>` : "";
  }
  const li = document.createElement("li");
  li.className = `rc-row flag-${flag}${ui.stage.classList.contains("catchup") ? " instant" : ""}`;
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
  if (isCatchup(t)) quiet();
  const scope = off.msector === null || off.msector === undefined ? TRACK : Number(off.msector);
  pushOfficial(off, flag, scope, t);
  if (flag === "CLEAR") {
    if (scope === TRACK) S.official.clear();       // TRACK CLEAR ends every official flag
    else S.official.delete(scope);
    for (const o of S.offLog) if (o.clearedAt === null && (scope === TRACK || o.scope === scope)) o.clearedAt = t;
    // race control's green ends a clock it never froze: passes after it are racing, not exposure
    if (scope === TRACK && S.clock && !S.clock.frozen) endClockWithoutOfficial(t);
    refreshLeads();
    return;
  }
  const rank = RANK[flag];
  if (!S.official.has(scope)) S.official.set(scope, []);
  S.official.get(scope).push({ t, rank });
  S.offLog.push({ t, rank, scope, clearedAt: null });
  refreshLeads();
  if (scope === TRACK && S.clock && !S.clock.frozen && rank >= S.clock.rank) freezeClock(t);
}

// --- websocket ---------------------------------------------------------------------------

function connect() {
  const ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    resetAll();                 // the catch-up replays everything since the last seek: never twice
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
    M.track = buildTrack(track);
    M.dirty = true;
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
    applyStatus(st);
    updateStandby();
  } catch (e) {
    // the server is restarting: the websocket reconnect handles it
  }
}

// --- replay controls ----------------------------------------------------------------------

function applyStatus(st) {
  view.speed = Number(st.speed);
  if (view.speed > 0) view.lastSpeed = view.speed;
  view.tStatus = Number(st.t);
  el("play").textContent = view.speed > 0 ? "PAUSE" : "PLAY";
  for (const b of document.querySelectorAll("[data-speed]")) {
    b.classList.toggle("active", Number(b.dataset.speed) === view.speed);
  }
}

async function postReplay(body) {
  try {
    const res = await fetch("/replay", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(body) });
    applyStatus(await res.json());
  } catch (e) {
    console.warn("overlay: POST /replay failed", e);
  }
}

function seekBy(ds) {
  // a jump over RESET_JUMP_S wipes every page and race control: flags start fresh from there
  const t = S.t ?? view.tStatus;
  if (t !== null && Number.isFinite(t)) postReplay({ seek_t: t + ds });
}

function togglePlay() {
  postReplay({ speed: view.speed > 0 ? 0 : view.lastSpeed || 1 });
}

function wireControls() {
  // a clicked button loses focus, so the space bar is play or pause, never that button again
  for (const b of document.querySelectorAll(".controls .ctl")) b.addEventListener("click", () => b.blur());
  el("play").addEventListener("click", togglePlay);
  for (const b of document.querySelectorAll("[data-seek]")) {
    b.addEventListener("click", () => seekBy(Number(b.dataset.seek)));
  }
  for (const b of document.querySelectorAll("[data-speed]")) {
    b.addEventListener("click", () => postReplay({ speed: Number(b.dataset.speed) }));
  }
  document.addEventListener("keydown", (e) => {
    if (e.target !== document.body) return;
    if (e.code === "Space") {
      e.preventDefault();
      togglePlay();
    } else if (e.code === "ArrowLeft" || e.code === "ArrowRight") {
      e.preventDefault();
      const ds = e.shiftKey ? SEEK_BIG_S : SEEK_SMALL_S;
      seekBy(e.code === "ArrowLeft" ? -ds : ds);
    }
  });
}

async function init() {
  wireControls();
  await loadTrack();
  await watchRace();
  setInterval(watchRace, STATUS_POLL_MS);
  setInterval(updateStandby, 500);          // notices a paused replay (no ticks) within a second
  setInterval(() => { if (S.t !== null && M.outReady) fetchCars(); }, CARS_POLL_MS);
  window.addEventListener("resize", () => { M.dirty = true; });
  requestAnimationFrame(mapFrame);
  connect();
}

init();
