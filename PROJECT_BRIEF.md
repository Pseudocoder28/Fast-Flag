# PROJECT BRIEF: Fast Flag (Predictive Race Control)
FormulaTech Hacks, University of Waterloo
Main track: Track 1 (Safety Diagnosis) | Sub-track: Ampere (AI for Motorsport Safety)
Build window: Saturday 11:30am to Sunday 11:30am (24h, all-nighter)
Team: 2 people, team We Are So Back. A = Naman (Claude Max account, Mac, main builder). B = Ishaan (Claude Pro account, separate laptop).

---

## 0. Instructions for Claude (read first)

- This file is the single source of truth for project context. CLAUDE.md holds the short rules + contracts and points here.
- Repo root is the `fast-flag` folder. Other files: `docs/KICKOFF.md` (A's first session, step by step), `docs/session_a.md` (A's ordered tasks), `docs/session_ishaan.md` (B's ordered tasks).
- Respect the ownership split in Section 9. Do not edit B's directories unless explicitly asked.
- Never change a contract (Section 7) silently. Propose the change, get agreement, update CLAUDE.md and this file.
- Free tools only. No paid APIs at runtime (this includes the Claude API).
- No future leakage anywhere. Every component only sees data with SessionTime <= current tick.
- The holdout race (Section 4) must never be loaded by training or tuning code.
- Style: Python 3.11, type hints, small functions, every module runnable with `python -m`, minimal dependencies.
- Communication style for B: minimal explanations, direct bullets, step-by-step commands, no em-dashes.

### First tasks for Claude on A's account (expanded step by step in docs/KICKOFF.md)
1. Start the FastF1 race scan + cache job (Section 4) in the background.
2. Scaffold the repo (Section 8), requirements.txt, .gitignore, CLAUDE.md.
3. Build fixtures + mock server (Section 7.6) from one training race so B can start immediately.
4. Check `docs/session_ishaan.md` (already written in the Section 13 format) against what was actually scaffolded, fix any mismatched command, path or fixture name, and push it.
5. Then proceed with A's work plan (Section 10).

---

## 1. Hackathon context

Main tracks (only ONE allowed):
- Track 1, Safety Diagnosis: detect or identify a safety risk in real time, or before it becomes critical. **(CHOSEN)**
- Track 2, Safety Testing: test or simulate a safety system under stress and failure.
- Track 3, Safety Fixing: directly address an identified safety issue.

Sub-track: **Ampere only**. "Develop an AI-powered solution that detects, predicts, or prevents safety risks in motorsport."

Implication: AI must be the core of the product, not decoration. Detection + prediction + recommendation all need a clear ML/AI component, and we need numbers to prove it works.

---

## 2. The project in one paragraph

An AI race control assistant that watches every car's telemetry simultaneously during a replayed F1 race. It (1) detects incidents the moment the data shows them (stopped car, impact, spin, data dropout, multi-car), (2) predicts which cars and marshal sectors are at elevated incident risk 10 to 30 seconds ahead, and (3) recommends a response (local yellow, double yellow, VSC, SC, red) with a confidence and reason. We benchmark against official FIA race control messages and report: "our system raised the alert X seconds before official race control." Final proof: a holdout race that finished the morning of the hackathon.

Pitch line: trained on 2023 to 2026 races, tested on a race that finished this morning.

---

## 3. Constraints

- Free versions of everything: FastF1, OpenF1 (historical only, free), pandas, numpy, scipy, scikit-learn, LightGBM, FastAPI, uvicorn, Ollama (local LLM), pyserial, Arduino IDE.
- The Claude API is NOT free and must not be used at runtime. Narration uses a local model via Ollama or templates.
- Hackathon WiFi will degrade overnight. Cache everything early, then work offline.
- Software first. Hardware is integrated only after the software MVP works end to end.
- Hardware available (Section 11): Arduino Uno R3, Arduino Nano, sensors, servo, stepper, 1602A LCD, buzzers, LEDs, buttons. No WiFi boards. No temperature sensors. Laptop is the gateway.

---

## 4. Data

### 4.1 Source: FastF1
- Data is usually available 30 to 120 minutes after a session ends.
- Per session, `session.load(...)` gives:
  - `session.car_data`: Speed, RPM, nGear, Throttle, Brake (bool), DRS. Roughly 4 Hz.
  - `session.pos_data`: X, Y, Z. Roughly 4 Hz. Check units (FastF1 position is typically 1/10 m).
  - `session.laps`: lap times, Compound, TyreLife, PitInTime/PitOutTime, per-lap TrackStatus.
  - `session.race_control_messages`: Time, Category, Message, Status, Flag, Scope, Sector, RacingNumber. **These are our labels and our benchmark.**
  - `session.track_status`: timeline of track state codes.
  - `session.weather_data`: AirTemp, TrackTemp, Rainfall, etc.
  - `session.get_circuit_info()`: corners and marshal sector positions. Use this to map car X/Y to the same marshal sector numbers race control uses.
- Telemetry helpers: `add_distance()`, `add_driver_ahead()` (driver ahead + distance to driver ahead).
- Track status codes: "1" green, "2" yellow, "4" SC, "5" red, "6" VSC deployed, "7" VSC ending.
- Enable the cache first thing: `fastf1.Cache.enable_cache("data/fastf1_cache")`. Without it, each session re-downloads roughly 200 to 500 MB.

### 4.2 2026 data caveats
- DRS channel is all zeros in 2026 (active aero replaced DRS, no replacement channel). Do not build features on DRS.
- ERS / energy deployment data is not in the feed. Do not plan for it.
- Mixing 2023 to 2025 with 2026 data: watch for schema quirks per season.

### 4.3 Race selection strategy (important: storage + WiFi)
Downloading every race since 2023 is tens of GB. Do this instead:
1. **Cheap scan:** for every race session 2023 to 2026, load messages only:
   `session.load(laps=False, telemetry=False, weather=False, messages=True)`.
2. Rank races by number of yellow / double yellow / SC / VSC / red events.
3. **Full download** of the top ~15 to 25 incident-heavy races (street circuits and wet races tend to rank high).
4. Save engineered features to Parquet in `data/features/` so nobody reloads raw data again.
5. Copy the cache to B's laptop via USB instead of downloading twice.

### 4.4 Holdout
- **Primary holdout: 2026 Azerbaijan GP race (Baku, Saturday 26 Sept 2026).** Never loaded by training or tuning code.
- Secondary holdout if Baku is incident-light: the 2026 Madrid round (the race before Baku).
- At kickoff: verify Baku loads and list every official event from race control messages with its SessionTime here. News reports from race day (verify all of it against the race control messages, lap numbers differ between reports):
  - Albon crashed hard at Turn 5 around lap 29 to 31, first Safety Car.
  - Multi-car collision at Turn 1 on the Safety Car restart (Colapinto into Gasly, Gasly into Norris), second Safety Car.
  - Bottas crashed late in the race and was reportedly covered by only a single yellow. If the messages confirm this, it is a strong demo case for our severity-based recommendation.
  - Both Aston Martins retired with technical issues (possible DROPOUT or STOPPED cases).
- Verified official events (checked at kickoff from FastF1 race control messages, SessionTime in seconds; FastF1 loads Baku as "2026 Azerbaijan Grand Prix", Madrid as "2026 Spanish Grand Prix"):
  - Pre-race and formation noise: yellows in sector 1 (t=267, 380) and DY sector 3 (t=2547). Eval must only count events between race start and the chequered flag (t=9294.5).
  - t=4300.5, lap 9: brief YELLOW sectors 14 and 15, cleared within 8 s.
  - **Albon (23), retired after 29 laps:** t=6648.5 lap 30 DOUBLE YELLOW sector 7 + YELLOW sector 6, **SC deployed t=6667.5** (lap 31), recovery vehicle at Turn 6 (news said Turn 5). Also YELLOW sectors 1 and 2 at t=6580.5, 68 s earlier.
  - t=6891.5 lap 32: DOUBLE YELLOW sector 10 (under SC). t=7023.5: DOUBLE YELLOW sector 11.
  - **Turn 1 restart collision (cars 1 NOR, 10 GAS, 43 COL, all retired laps 35 to 36):** t=7478.5 lap 36 DOUBLE YELLOW sector 2 + YELLOW sector 1, YELLOW sectors 3 and 4 at t=7501.5, **second SC deployed t=7542.5**. COL got a 10 s penalty for causing a collision.
  - t=8043.5 lap 40 and t=9153.5 lap 50: brief YELLOW sectors 1 and 2 (Turn 1 incidents, e.g. LIN / LAW noted).
  - **Late incident, final lap (likely Bottas, car 77, T15 incident noted, retired):** t=9234.5 DOUBLE YELLOW sector 15 + YELLOW sector 14, no SC or VSC. The news claim "only a single yellow" is **not** confirmed: the sector got a double yellow. Still a useful severity case (check telemetry at A7).
  - Aston Martins: STR retired after 7 laps, ALO after 20, no flags near either (likely pit-lane retirements, not track incidents).
  - Events after t=9294.5 (cool-down lap) are ignored.
  - Verdict: Baku is not incident-light. It stays the primary holdout.
- Check races (27 Sept): the next 20 races in data/race_ranking.csv after the 20 training races, skipping the holdout, built into data/case_studies and never used for fitting or tuning (2021 Azerbaijan is kept apart: the late-race red was checked on it and the demo uses it). They test the rules and models tuned on the training races, run once after the freeze like the holdout (python -m src.eval.check_races, logged in docs/charts/escalation_check_runs.json), with the production models. Result: docs/charts/escalation_check.md, quoted in docs/charts/NUMBERS.md.

### 4.5 OpenF1 (backup only)
- Historical data from 2023 onwards is free, no auth. Real-time requires paid subscription. Not needed (no live F1 session during the hackathon).

---

## 5. Architecture

Six layers, all connected through one WebSocket stream. Hardware is just another subscriber, so it plugs in later without touching the core.

```
FastF1 cache
   |
[1] ingest: load, merge, reference profiles, features -> Parquet
   |
[2] replay engine: tick-by-tick, ~4 Hz, speed control, seek
   |
   +--> [3] detect: stopped / impact / spin / dropout / multi
   +--> [4] predict: per-car risk_10s, risk_30s; SC probability
   |
[5] racecontrol: rules + flag state machine + narration (Ollama/templates)
   |
WebSocket ws://localhost:8000/stream
   |
   +--> [6a] dashboard (pit wall view)
   +--> [6b] serial bridge -> Uno marshal panel + Nano in-car dash (after MVP)
   +--> eval logger (lead time, recall, false alarms)
```

---

## 6. Layer details

### 6.1 Ingest (A)
- Merge car_data + pos_data per driver onto a common 250 ms grid (interpolate, do not extrapolate).
- **Reference line + speed profile per circuit:**
  - Take clean laps: TrackStatus all "1", not pit in/out laps, reasonably quick.
  - Reference line: X/Y of the fastest clean lap, resampled every ~5 to 10 m of distance.
  - Speed profile: median speed per distance bin (10 m) across clean laps.
  - Build a scipy KD-tree on the reference line to map any X/Y to (track distance, lateral offset).
- Map each position to a marshal sector using circuit info.
- Output per race: Parquet of per-driver time series with engineered features.

### 6.2 Replay engine (A)
- Iterates over the merged time grid and emits a tick with every car's latest state.
- Strict causality: at time t, nothing from t' > t is available to any consumer.
- Controls: speed (1x, 5x, 10x, 50x), seek to time, pause.
- Owns the FastAPI WebSocket server and envelope (Section 7.5).
- Precomputed race (27 Sept, the server's default; `--live` keeps the old path): our detectors, risk model and race control run once over the whole race at startup, strictly tick by tick, cached in data/timeline/ (src/replay/timeline.py), and the server plays the result back and sends our recs itself (a separately started race control client is ignored). After a seek over 2 s it sends the tick at the new time, then every official message, detection and rec before it to pages that joined with `?catchup=1`, so a page shows what a continuous run shows at that time: no wiped flags, no cold detectors. Proposed optional contract additions, pending Ishaan's OK: `/stream?catchup=1`, `GET /cars` (cars out of the race at the replay time: no data for 60 s or never any, or not driving at 20 km/h or more for 60 s while the field moves; a wreck on track also goes out through a red flag) and `cars` and `mode` in `GET /status`. Nothing existing changes for clients that do not use them.

### 6.3 Detection (A)
Reactive but fast. Each detector outputs the Detection contract.
- **STOPPED / SLOW:** speed far below reference speed at that track position, outside pit lane, sustained > 1 to 2 s.
- **IMPACT:** large speed loss with Brake = False (e.g. > 80 km/h lost within ~1 s, tune it). Strongest single signal: nobody loses that much speed without braking unless they hit something.
- **SPIN / OFF:** lateral offset from reference line jumps beyond threshold, or heading diverges sharply from track direction.
- **DROPOUT:** a car's data stops mid-lap outside the pit lane.
- **MULTI:** 2+ cars flagged in the same or adjacent marshal sector within a few seconds.
- Severity 0 to 1: from speed lost, deceleration magnitude, position on track (on racing line vs run-off), cars involved.
- **ANOMALY (ML, must-have, never cut):** IsolationForest trained only on normal racing windows from training races (clean laps, not in pit, green track status). Score each car's rolling 2 to 5 s window of features (speed deviation from reference, lateral offset, throttle/brake pattern, deceleration). This is the AI floor of the product: if prediction is cut at M2, the product still has a learned model at its core, which Ampere requires.
- **Safety Car and VSC periods (track status 4 or 6):** every car is slow, so compare speed against the current field median instead of the reference profile, otherwise STOPPED and ANOMALY fire for every car. Track status is known at time t, so this is causal.
- Baseline to beat: a naive "speed < X km/h" threshold. We must show our detectors beat it.
- Fixes of 27 Sept, found on the 2021 Azerbaijan replay and checked on the 20 training races (false alarms 94 to 66, incidents 83 to 82 of 146, docs/charts/detect_eval.md): STOPPED fires only below 60 km/h (a car at the pit limiter on a 325 km/h straight is 25% of reference but moving); IMPACT "without braking" means no brake sample in the whole second of the drop; after a suspension of 60 s or more the detectors stay in grid mode until the field forms up (at most 240 s: the lap to the grid is not racing); nothing fires after race control's CHEQUERED FLAG. 2021 Azerbaijan is therefore no longer out of sample for the detector rules; the ML models never saw it.

### 6.4 Prediction (A)
**Targets:**
- Car level: does this car have an incident within the next H seconds (H = 10, 20, 30)?
- Sector level: does this marshal sector get a yellow within the next H seconds?

**Labels (training races only):**
- Incident time per car = earliest detector-confirmed IMPACT / STOPPED / SPIN for that car, cross-checked with an official yellow/SC in the same or adjacent sector within a window.
- Positive samples: ticks within H seconds before the incident. Drop all ticks after the incident for that car.

**Features (rolling windows of 2 to 5 s):**
- Speed deviation from reference at current distance
- Throttle oscillation, throttle/brake overlap
- Lateral offset from reference line + its variance
- Distance to car ahead + closing rate
- Cars within 1 s (battle density)
- Lap time delta vs driver's own recent laps
- Compound, tyre life, laps since pit (out-lap cold tyres)
- Lap number (lap 1), restart flag (first laps after SC/VSC)
- Track temp, rainfall

**Models:**
- Primary: LightGBM classifier. Fast, strong on tabular, gives feature importance for the pitch.
- Baselines: the ANOMALY detector's IsolationForest score (Section 6.3) and the naive speed threshold. LightGBM must beat both to earn its place in the pitch.
- Stretch: small 1D-CNN or LSTM on raw windows.
- SC probability model: classifier on incident features (location, severity, cars involved, lap) predicting whether the incident led to SC/VSC.

**Training rules:**
- Class imbalance: class weights + negative downsampling. Report PR-AUC, not accuracy.
- Leave-one-race-out cross validation. Then one single run on the holdout.
- Frame honestly: some incidents have no precursor. This is risk forecasting, not a crystal ball.

### 6.5 Race control engine (A, taken over from B on 26 Sept)
Consumes detections + risk, maintains a per-sector flag state machine, emits Recommendations.
- Interface (`src/racecontrol/engine.py`, class `RaceControl`): `on_tick(tick)` returns `(recs, did_reset)`, where `did_reset` is True when the tick jumped more than 2 s back or forward in time (seek or loop) and all flag state was wiped. `on_detection(det)` and `on_risk(risk)` return a list of recs.
- Single car stopped in run-off, low severity: YELLOW in that sector.
- Car stopped on racing line, or high severity: DOUBLE_YELLOW, consider VSC.
- Multi-car incident or high SC probability: SC.
- RED only late in the race (27 Sept, see below): a multi-car incident alone calls the SC.
- VSC or SC for a stopped car (as built): SC after 3 s when our IMPACT or MULTI detection involved the car (from 30 s before its stop onward), VSC after 10 s when it stopped without one. A later IMPACT or MULTI for that car upgrades the VSC to an SC. Every stopped car in a sector is tracked, and the highest call wins. A stopped car that rolls on into another sector at 80 km/h or less (a car losing power crawls over a sector boundary before it stops) takes its stop and its sector flag along, so its VSC or SC still comes (27 Sept; before, its old sector cleared behind it and 3 of race control's neutralisations on the training races were missed that way). Lateral offset cannot make this call, because FastF1 positions of stopped cars sit on the racing line even in run-off (99.97% within 1 m). How often our flag matches race control's: docs/charts/escalation.md (same first flag), quoted in docs/charts/NUMBERS.md.
- Clearing: sector returns to CLEAR once no flagged car remains in it for N seconds (N = 5). A flagged car stops counting as in the sector once it drives out, enters the pit lane, sends no data for 120 s, or has not moved more than 3 m for 180 s. The last rule is needed because a retired car keeps reporting its last position for the rest of the session (2023 Australia: cars 16 and 23). Except a crash site (a car stopped after an impact, added 26 Sept): it holds its sector until the car is moved 20 m away, its data stops, or race control's own track status (in every tick) has been green for 20 s after race control reacted to the crash (180 s if race control never reacts). Before this, we recommended green 90 s before race control's red flag for Verstappen (2021 Azerbaijan). A track-wide flag (VSC, SC, RED) clears 5 s after every sector that caused or supported it is CLEAR, so the next incident can escalate again, but never while race control's own track status (in every tick) is SC, VSC or red (added 27 Sept): lifting a neutralisation is race control's call, only raising one is ours. Before this, a car stopped with no impact (2023 Australia, Magnussen: TRACK CLEAR 5 s before race control's red flag) or a crashed car moved a few metres by a crane let us recommend green under race control's Safety Car; on the training races none of our TRACK CLEARs now comes while race control's status shows SC, VSC or red (docs/charts/escalation.md).
- Red flag, late in the race, which never blocks (27 Sept, replacing the red by time at a crash site and the multi-car red): while our VSC or SC is out for an incident that is still there and 2 to 4 laps are left, a RED rec (confidence 0.75), because race control red-flags a late neutralisation so the race can restart and finish racing. The race length comes from the lap length, known before the race (the fewest laps that cover 305 km, 260 km at Monaco, from GET /track; it comes out 1 to 3 laps longer than the real race on most circuits), so no contract changed. It is a soft overlay: every other escalation still compares against our VSC or SC, so a new incident elsewhere escalates on its own, and it ends with our TRACK CLEAR. Why (docs/lab/red_flag.md): race control's reds in this data follow either a late neutralisation or damage car data cannot see (barrier, gravel, debris, rain). The red by time at a crash site (120 s) caught 2 mid-race reds but, with the multi-car red, sent 18 reds race control never called on the training races, and more on replays of 2021 Azerbaijan (Stroll) and the 2026 Azerbaijan holdout (Albon and the Turn 1 pile-up), where race control kept its Safety Car. Now a crash site still stopped 120 s after its stop, once race control reacted, only gets an advisory: its sector flag re-sent unchanged with "recovery taking long, race control may need a red flag". On the training races (docs/charts/escalation.md) race control showed a red in 5 of its escalated incidents and we recommended red in 2 of them (Magnussen at 2023 Australia, our red out 121 s before race control's; the crash at that race's last restart, 7 s); we send 3 reds, race control showed a red around 2 of them and never called 1. Tuned on the training races, with 2021 Azerbaijan as a check (red 142 s before race control's for Verstappen, 1 red race control never called). On the 20 check races (Section 4.4, never used for tuning) it caught neither of race control's 2 reds (rain at 2023 Netherlands, a first-lap crash at 2023 São Paulo: neither is late in a race) and sent 3 reds race control never called. Seven reds in 40 races is little data and the data cannot see barrier damage, debris or medical needs, so do not claim red flag accuracy. RED_BY_TIME and RED_FROM_MULTI in engine.py bring the old reds back for comparison.
- Hysteresis so flags do not flicker: flags escalate at once and only come down after a hold, at least 2 s for a sector flag and 60 s for a track-wide flag. A stopped car's SC or VSC timer pauses while the car moves and restarts when it stops again, so one noisy speed sample cannot cancel it.
- Narration: template message immediately ("YELLOW IN TRACK SECTOR 7"), optional Ollama-generated steward summary generated asynchronously. The LLM never decides, it only explains.

### 6.6 Dashboard (A, taken over from B on 26 Sept)
- Served by FastAPI as static HTML + JS, subscribes to the WebSocket. No build step, no CDN dependencies (WiFi).
- Canvas track map: reference line, marshal sectors colored by current flag, car dots colored by risk.
- Alert feed: time, car(s), sector, recommendation, confidence, reason.
- Lead-time timeline: our alerts vs official race control messages.
- Replay controls: speed, seek, jump-to-incident list.
- Cars out of the race (27 Sept): the cars in GET /cars leave the map on the pit wall and the overlay and are listed beside it (crashed, stopped, in the garage, no data); while a crashed car's incident sector is flagged, a red triangle marks where it stopped. After a seek or reconnect the pages rebuild from the server's history, so colours, flags and the out list are the continuous-run state.

### 6.7 Evaluation (A builds, B visualizes)
- **Official events** = race control messages with Flag in {YELLOW, DOUBLE YELLOW, RED} or SC/VSC deployment messages. Ignore BLUE, track limits, DRS, investigations.
- **Match:** our first detection/recommendation in the same or adjacent marshal sector within [t_official - 60 s, t_official + 10 s].
- **Lead time** = t_official - t_ours (positive = we were earlier).
- **Recall** = matched official events / all official events.
- **False alarms** = our alerts with no matching official event, per race hour.
- **Prediction:** PR-AUC per horizon, and % of incidents with risk above threshold at least N seconds before.
- Output charts: lead-time distribution, per-incident timeline, holdout summary.

---

## 7. Contracts (all JSON)

### 7.1 Tick (replay -> all), ~4 Hz
```json
{"t": 3601.25, "lap": 12, "track_status": "1",
 "cars": [{"drv": "44", "x": 0.0, "y": 0.0, "dist": 0.0, "lat_off": 0.0,
           "speed": 0.0, "throttle": 0.0, "brake": false, "gear": 7, "rpm": 0,
           "msector": 7, "gap_ahead_m": 0.0, "in_pit": false}]}
```
- Units: `t` is SessionTime in seconds. `x`, `y`, `dist`, `lat_off`, `gap_ahead_m` are metres (convert FastF1 position from 1/10 m). `speed` is km/h. `drv` is the car number as a string.

### 7.2 Detection (detect -> racecontrol)
```json
{"id": "det-000123", "t": 3601.25, "drivers": ["44"], "msector": 7,
 "type": "STOPPED|IMPACT|SPIN|DROPOUT|MULTI|ANOMALY", "severity": 0.0,
 "evidence": "speed 12 km/h vs ref 210 km/h"}
```

### 7.3 Risk (predict -> racecontrol, dashboard)
```json
{"t": 3601.25, "drv": "44", "risk_10s": 0.12, "risk_30s": 0.31,
 "top_features": ["closing_rate", "tyre_life"]}
```

### 7.4 Recommendation (racecontrol -> dashboard, bridge, eval)
```json
{"id": "rec-000045", "t": 3601.50, "msector": 7,
 "flag": "CLEAR|YELLOW|DOUBLE_YELLOW|VSC|SC|RED", "confidence": 0.0,
 "reason": "car 44 impact signature, stopped on racing line",
 "message": "YELLOW IN TRACK SECTOR 7", "source_detections": ["det-000123"]}
```
- A rec with flag VSC, SC or RED is track-wide, and its `msector` is the sector that caused it. A CLEAR rec with message `TRACK CLEAR` ends the track-wide flag. A CLEAR rec with message `CLEAR IN TRACK SECTOR 7` clears only that sector.

### 7.5 Transport
- WebSocket `ws://localhost:8000/stream` (owned by A).
- Envelope: `{"kind": "tick|detection|risk|rec|official", "data": {...}}`
  - `official` = official race control message at its real timestamp (for the dashboard comparison, never fed to detect/predict).
- Control: `POST /replay {"speed": 1, "seek_t": 3500.0, "race": "2024_Singapore"}`
- racecontrol runs as a subscriber that publishes `rec` messages back to the server (or in-process, A's choice, but the envelope stays the same).
- Publishing: any client may send an envelope with `kind` = `rec` on the same socket. The server rebroadcasts it to every subscriber. The mock server does the same.
- `GET /track` returns the map for the loaded race: `{"race": "2024_Singapore", "ref_line": [[x, y], ...], "msectors": [{"id": 7, "start_dist": 0.0, "end_dist": 0.0}], "corners": [{"number": 1, "x": 0.0, "y": 0.0}]}` (metres).
- `GET /official` returns all official events for the loaded race, for the dashboard timeline and jump-to-incident list only. Never fed to detect or predict.
- `GET /status` returns the replay state: `{"race": "2023_Australian", "t": 4390.25, "speed": 1.0, "t_start": 4240.0, "t_end": 4569.75, "clients": 2}`. The real server adds `"finished": false`. `POST /replay` returns the same object.
- `GET /races` returns the race ids that can be loaded with `POST /replay`, e.g. `["2023_Australian", "2024_Canadian"]`. The real server lists built training races (plus the holdout only when started with `--holdout`), the mock server lists only the fixture race.
- Parked until after M2: the proposed `human` envelope kind. It is not part of the contract, so nothing sends or handles it yet.
- Official event shape (the `data` of an `official` envelope and each item of `GET /official`), added at kickoff:
  `{"t": 4390.18, "category": "Flag", "message": "YELLOW IN TRACK SECTOR 9", "flag": "YELLOW|DOUBLE_YELLOW|RED|SC|VSC|CLEAR", "scope": "Sector|Track", "msector": 9, "drivers": []}`. `msector` is `null` for track-wide events (SC, VSC, RED, TRACK CLEAR).
- Tick details, added at kickoff: a car with no fresh data (latest sample older than 1 s) is left out of that tick (that gap is a DROPOUT signal). `gap_ahead_m` is `-1.0` when unknown (for example in the pit lane). Values at tick t are the latest sample at or before t, never interpolated towards a future sample.
- The server also serves the `dashboard/` folder at `/`, so the dashboard opens at http://localhost:8000.

### 7.6 Fixtures + mock server (A, by M0)
- `fixtures/ticks_sample.jsonl`, `detections_sample.jsonl`, `risk_sample.jsonl`, `recs_sample.jsonl`, `official_sample.jsonl`
- Cut from a real training race around at least one real incident.
- Also `fixtures/track_sample.json` (same shape as `GET /track`). Total fixture size under 20 MB, committed to git.
- `python -m src.replay.mock_server` plays fixtures over the exact same socket + envelope. B builds entirely against this until M1.
- The mock server supports the same endpoints as the real server (`/stream`, `POST /replay`, `GET /track`, `GET /official`, dashboard at `/`) and a `--no-recs` flag so B's racecontrol replaces the fixture recs.
- `tests/test_contracts.py` validates every fixture line against the shapes in this section (keys, types, enum values). Both people run `pytest` before every merge into `main`.

### 7.7 Serial (bridge -> Arduino, after MVP, owned by B)
- 115200 baud, ASCII, newline terminated.
- Laptop -> Uno: `Z,<zone 1-3>,<CLEAR|YELLOW|DOUBLE_YELLOW>` (bridge maps marshal sectors to 3 panel zones)
- Laptop -> Uno: `G,<GREEN|VSC|SC|RED>`
- Laptop -> Nano: `M,<text max 16 chars>` and `F,<flag>`
- Nano -> Laptop: `ACK,<ms since last message>`

---

## 8. Repo layout

```
fast-flag/
  CLAUDE.md                # short rules + contracts, points to PROJECT_BRIEF.md
  PROJECT_BRIEF.md         # this file
  README.md
  requirements.txt         # shared pins from kickoff + "-r requirements-a.txt" + "-r requirements-b.txt"
  requirements-a.txt       # A's later additions
  requirements-b.txt       # B's later additions
  .gitignore               # data/, fastf1 cache, __pycache__, .venv
  data/                    # gitignored: fastf1_cache/, features/, models/
  fixtures/
  tests/
    test_contracts.py      # shared
  docs/
    KICKOFF.md
    session_a.md
    session_ishaan.md
    charts/                # eval charts for the pitch (A writes, B uses)
  src/
    ingest/                # A: scan, cache, merge, reference profiles, features
    replay/                # A: engine, server, mock_server
    detect/                # A
    predict/               # A: risk model, SC model
    eval/                  # A: matching, metrics, chart data
    racecontrol/           # A (from B, 26 Sept): rules, flag state machine, narration
    bridge/                # B: serial bridge (after MVP)
  dashboard/               # A (from B, 26 Sept): static HTML/JS served by FastAPI
  firmware/                # B (after MVP)
    uno_marshal/
    nano_dash/
```

Git: each person commits only in owned directories. `main` only receives working code. Commit small and often. Full workflow (branches `a-work` and `b-work`, diff check, merge order) is in CLAUDE.md.

---

## 9. Ownership

**A (Naman, this account):** src/ingest, src/replay (server and mock server), src/detect, src/predict, src/eval, src/racecontrol, dashboard/, fixtures/, docs/charts/.
**B (Ishaan):** src/bridge, firmware/, pitch slides, demo script. Tests the dashboard in his browser after every push and reports issues to A.

Handover on 26 Sept (agreed by both): src/racecontrol, dashboard/ and tests/test_racecontrol.py moved from B to A when B's Claude limit ran out. B uses a Codex account while the limit resets.

Tests: tests/test_<area>.py belongs to the owner of that area. tests/test_contracts.py stays shared.

Shared: CLAUDE.md, PROJECT_BRIEF.md, docs/ (except docs/charts/), requirements.txt, tests/test_contracts.py. New packages go in your own requirements-a.txt or requirements-b.txt, pinned, and you announce them.

---

## 10. Timeline and milestones

Since the 26 Sept handover (Section 9), the racecontrol and dashboard items below marked B are done by A.

**M0, 1:00pm: Kickoff done**
- Cache scan running, repo scaffolded, CLAUDE.md written, contracts agreed
- Fixtures + mock server working
- Baku verified loadable
- 15-min hardware smoke test (blink both boards, LCD works, check for I2C backpack)
- Cache copied to B's laptop
- GitHub repo pushed with branches `a-work` and `b-work`, `pytest` passes on both laptops

**1:00pm to 5:00pm**
- A: full downloads, merge, reference profiles, replay engine on real data
- B: racecontrol rules + state machine against fixtures, dashboard skeleton (map + feed)

**5:00pm to 9:00pm**
- A: all detectors, eval matching + lead-time metric on training races
- B: dashboard lead-time timeline, replay controls, template narration

**M1, 9:00pm:** real replay + detectors streaming, racecontrol + dashboard consuming live.

**9:00pm to midnight:** integration, fix contract mismatches, tune detector thresholds.

**M2, midnight: MVP CHECKPOINT**
- End to end on a training race incident: replay -> detect -> recommend -> dashboard.
- If not working: drop prediction entirely, polish detection. Detection + lead time alone is a valid project.

**Midnight to 5:00am**
- A: risk model, SC model, feature importance, leave-one-race-out results
- B: serial bridge + Uno marshal panel + Nano dash firmware, Ollama narration

**M3, 5:00am:** prediction + narration + hardware working.

**5:00am to 8:00am**
- A: run on holdout (Baku), generate all charts
- B: dashboard polish, slides, demo script
- Stretch only if everything above is solid

**M4, 8:00am: CODE FREEZE.** Bug fixes only after this.

**8:00am to 10:30am:** rehearse demo 3+ times, record backup demo video, finalize slides.
**10:30am to 11:30am:** buffer + submission.

---

## 11. Hardware plan (after MVP only)

Available: Arduino Uno R3, Arduino Nano, 830 + mini breadboards, MPU-6500, HC-SR501 PIR, HC-SR04, TCRT5000, XY joystick, MAX30102, BH1750, 28BYJ-48 + ULN2003, SG90 servo, 1602A LCD, 5V active buzzers, resistor kit, jumpers, BJTs, LEDs, push buttons, USB cables.

**Uno = marshal panel**
- 3 zones x (green + yellow LED) = 6 LEDs
- 1 red LED (red flag), 1 LED for SC/VSC (or blink pattern)
- SG90 servo waves a small flag on YELLOW or above
- Active buzzer on new alerts, driven through a BJT (do not drive directly from the pin)
- Roughly 10 pins total, fits the Uno

**Nano = in-car dash**
- 1602A LCD in 4-bit parallel mode (RS, E, D4 to D7 = 6 pins) unless it has an I2C backpack
- No potentiometer in the kit: set LCD contrast (V0) with a fixed resistor from the resistor kit (try ~1k to 2.2k to GND)
- Joystick: VRx A0, VRy A1, SW on a digital pin. Press = driver acknowledges, send ACK with reaction time
- Second buzzer via BJT

**Stretch: live hardware mode**
- MPU-6500 on I2C (A4/A5) on a toy car or handheld "car". Knock it and a live IMPACT detection flows through the same pipeline as a synthetic car in the replay.
- Proves the system runs on live sensor data, not only replays.

---

## 12. Demo, pitch, judging

### 3-minute demo
1. Hook: "This morning in Baku, [incident]. Official race control took X seconds. We took Y."
2. Replay that incident live: car dot turns red, sector lights up, marshal panel LEDs + servo, in-car LCD updates, driver acknowledges.
3. Lead-time chart across all test incidents.
4. Prediction case: a car flagged high risk before it happened, with top features.
5. Limitations + what live integration would need.

### Judge questions to prepare
- "Isn't this using future data?" Strict tick-by-tick causality, holdout never touched in training.
- "How many false alarms?" Have the per-hour number ready.
- "Would race control use this?" Decision support, human stays in the loop.
- "Why not just a speed threshold?" Show our detectors + model beating that baseline.
- "Is comparing to official timing fair?" Official timing includes human deliberation, which is exactly the delay we cut. Caveat: replay of post-session data is slightly optimistic vs a live feed with transmission latency.
- Marshals wave local flags before a race control message appears, so always claim "earlier than the race control feed", never "earlier than the marshals".

### Honesty rule
- Everything shown to judges says it is a replay of historical FastF1 data, which values are derived and which parts are simulated. Never present a replay as live track data.

---

## 13. Format for docs/session_ishaan.md (Claude on this account generates it)

- Role + owned directories
- Setup commands: clone, venv, `pip install -r requirements.txt`, run mock server, open dashboard
- Tasks in order, each with: goal, input contract, output, "done when" test
- Checkpoint times matching Section 10
- Don't-touch list
- Blocker protocol: message A immediately with the error + what was tried, never sit stuck > 20 min
- Style: minimal explanations, bullets, step-by-step commands, no em-dashes

---

## 14. Risks and mitigations

- **WiFi dies overnight:** everything cached and copied to both laptops by M0.
- **Baku data not available or incident-light:** fall back to Madrid holdout.
- **Contract drift between A and B:** mock server + fixtures enforce the contract, any change goes through CLAUDE.md.
- **Detectors noisy:** tune thresholds on training races only, add hysteresis in racecontrol.
- **Prediction weak:** ship detection + lead time as the core, present prediction honestly as a first result.
- **Local LLM too slow:** templates only, LLM summary optional and async.
- **Hardware part dead:** found in the M0 smoke test, not at 3am.
- **Main builder hits usage limits overnight:** pace heavy sessions, keep the plan in files so work can continue on either account.
- **B's Pro account hits usage limits:** B's Claude Code runs Sonnet by default and switches to Opus only for stubborn bugs.
- **Two Claude sessions edit the same file:** ownership rules plus the `git diff --stat main...HEAD` check before every merge (CLAUDE.md).
- **Exhaustion:** code freeze at 8am is non-negotiable.

---

## 15. Cut list (drop in this order if behind)
1. Stretch: live MPU-6500 mode
2. Ollama narration (keep templates)
3. SC probability model (keep rules)
4. Sector-level prediction (keep car-level)
5. Nano in-car dash (keep Uno panel)
6. Prediction entirely (keep detection + lead time)

Never cut: the ANOMALY detector. Without it the product is threshold rules only and stops qualifying as an Ampere (AI) project.
