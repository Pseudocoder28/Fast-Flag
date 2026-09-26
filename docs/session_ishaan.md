# Session: Ishaan (B)

Start Claude Code in the repo folder and send:
`Read CLAUDE.md, PROJECT_BRIEF.md and docs/session_ishaan.md, then follow docs/session_ishaan.md.`

Claude: minimal explanations, bullets, step-by-step commands, no em dashes.

## Role

- Race control engine, dashboard, serial bridge, firmware, pitch slides and demo script.
- Owned: `src/racecontrol/`, `src/bridge/`, `dashboard/`, `firmware/`, `requirements-b.txt`.
- Branch: `b-work`. Commit messages start with `B:`.

## Claude Code setup (Pro account)

- `claude update`
- `claude --model sonnet` inside the repo folder.
- Stubborn bug after 2 tries: `/model`, pick Opus, press `s` (this session only).

## Setup

1. `git clone <REPO_URL> fast-flag`
2. `cd fast-flag`
3. Mac: `python3.11 -m venv .venv && source .venv/bin/activate`. Windows: `py -3.11 -m venv .venv` then `.venv\Scripts\activate`.
   - No Python 3.11? `python3 -m pip install --user uv`, then `uv python install 3.11` and `uv venv --python 3.11 .venv` (uv is in `~/Library/Python/3.9/bin/` on Mac), then activate as above and use `uv pip install -r requirements.txt` in step 4.
   - LightGBM on Mac needs `libomp` (`brew install libomp`). You do not use LightGBM, so an import error from it is safe to ignore.
4. `pip install -r requirements.txt`
5. `git checkout b-work && git merge main`
6. `pytest` (must pass)
7. `python -m src.replay.mock_server`
8. Open http://localhost:8000 (a placeholder page until you create `dashboard/index.html` in B3).

## Fixture facts (what the mock server plays)

- Race `2023_Australian`, SessionTime 4240 to 4570 s, ~4 Hz ticks, loops forever.
- Incident: car `23` (Albon) crashes at Turn 6, marshal sector 9. IMPACT at t=4387.25, STOPPED at t=4390.5.
- Official: YELLOW sector 9 at t=4390.18, SC at t=4402.18, RED at t=4560.18.
- Jump straight to it: `curl -X POST localhost:8000/replay -H 'content-type: application/json' -d '{"speed": 1, "seek_t": 4370}'`. Speed 0 pauses.
- `GET /status` shows current replay time and speed.
- `official` data: `{t, category, message, flag, scope, msector, drivers}`, `msector` is null for track-wide events (SC, RED).
- Ticks: a car with no fresh data is left out of that tick. `gap_ahead_m` is -1.0 when unknown.
- Detections, risk and recs in the fixtures are hand-built. Ticks and official events are real data.

## Tasks

**B0. Hardware smoke test (15 min, before 1:00pm)**
- Goal: find dead parts now, not at 3am.
- Blink the Uno and the Nano with the Arduino IDE Blink example.
- 1602A LCD prints text. Check for an I2C backpack. If there is none, use 4-bit mode and a fixed resistor (about 1k to 2.2k to GND) on V0 for contrast.
- SG90 servo sweeps. Buzzer sounds through a BJT (never driven straight from a pin).
- Done when: you message Naman a list of working and dead parts.

**B1. Race control engine core (1:00pm to 5:00pm)**
- Goal: per-sector flag state machine (PROJECT_BRIEF.md Section 6.5).
- Input: `tick`, `detection`, `risk` (Sections 7.1 to 7.3).
- Output: `rec` (Section 7.4).
- Pure logic in `src/racecontrol/engine.py`: class `RaceControl` with `on_tick`, `on_detection` and `on_risk`, each returning a list of recs. Rules from 6.5, hysteresis and clearing after N seconds.
- Template message on every rec, e.g. `YELLOW IN TRACK SECTOR 7`.
- Put your tests in your own files, e.g. `tests/test_racecontrol.py`.
- Done when: a pytest test feeds `fixtures/detections_sample.jsonl` and `fixtures/ticks_sample.jsonl` and gets YELLOW or stronger for the fixture incident, with no flag flicker.

**B2. Race control client (1:00pm to 5:00pm)**
- Goal: run the engine live on the stream.
- Input: envelopes from `ws://localhost:8000/stream`.
- Output: `rec` envelopes sent back on the same socket (the server rebroadcasts them).
- `python -m src.racecontrol` connects, feeds the engine and sends recs.
- Done when: with `python -m src.replay.mock_server --no-recs` running, your recs appear on the stream.

**B3. Dashboard skeleton (1:00pm to 5:00pm)**
- Goal: pit wall view (Section 6.6).
- Input: `GET /track`, stream envelopes.
- Output: `dashboard/index.html`, `dashboard/app.js`, `dashboard/style.css`. No CDN, no build step.
- Canvas map: reference line, marshal sectors coloured by current flag, car dots coloured by risk.
- Alert feed: time, cars, sector, flag, confidence, reason.
- Done when: the mock replay shows cars moving and the fixture incident appears in the feed.

**B4. Dashboard features (5:00pm to 9:00pm)**
- Lead-time timeline: our recs next to `official` envelopes.
- Replay controls: speed and seek through `POST /replay`, jump-to-incident list from `GET /official`.
- Banner on screen: "Replay of historical FastF1 data" (honesty rule).
- Done when: you can jump to the fixture incident and see our alert time next to the official one.

**M1 at 9:00pm:** Naman merges the real server. Pull `main`, merge it into `b-work` and run the dashboard and racecontrol against the real stream instead of the mock.

**B5. Integration (9:00pm to midnight)**
- Fix your side of any mismatch. Contract mismatches in A's code: message Naman, never patch his files.
- Done when: a training race incident runs end to end on Naman's laptop.

**M2 at midnight: MVP checkpoint.** Replay, detect, recommend and dashboard working end to end.

**B6. Hardware (midnight to 5:00am, only after M2 passes)**
- `src/bridge/`: subscribes to the stream and sends the Section 7.7 serial protocol at 115200 baud. Maps marshal sectors to 3 panel zones.
- `firmware/uno_marshal/`: marshal panel (Section 11). 3 zones of green and yellow LEDs, red LED, SC/VSC LED, servo flag, buzzer through a BJT.
- `firmware/nano_dash/`: in-car dash (Section 11). LCD message and flag, joystick press sends `ACK,<ms>`, second buzzer through a BJT.
- Ollama narration only if everything else works (async, it never decides).
- Done when: a replayed incident lights the right zone, waves the flag and the Nano ACK shows up on the laptop.

**M3 at 5:00am:** prediction, narration and hardware working.

**B7. Pitch (5:00am to 8:00am)**
- Dashboard polish.
- Slides and 3-minute demo script following PROJECT_BRIEF.md Section 12, using Naman's charts in `docs/charts/`.
- Prepare the judge questions in Section 12.

**M4 at 8:00am: code freeze.** Rehearse the demo 3+ times, record a backup video, finalize slides by 10:30am, submit by 11:30am.

## Checkpoints

- M0 1:00pm, M1 9:00pm, M2 midnight, M3 5:00am, M4 8:00am (code freeze), rehearse until 10:30am, submit by 11:30am.

## Don't touch

- `src/ingest/`, `src/replay/`, `src/detect/`, `src/predict/`, `src/eval/`, `fixtures/`, `docs/charts/`
- `CLAUDE.md`, `PROJECT_BRIEF.md`, `requirements.txt`, `tests/test_contracts.py`, `tests/test_mock_server.py`

## Merging

- Before merging into `main`: `pytest` passes and `git diff --stat main...HEAD` shows only your paths.
- Merge one person at a time, following the CLAUDE.md workflow.

## Blocker protocol

- Stuck for more than 20 minutes: message Naman immediately with the error and what you tried.
