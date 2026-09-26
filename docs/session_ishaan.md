# Session: Ishaan (B)

Start Claude Code (or Codex, see below) in the repo folder and send:
`Read CLAUDE.md, PROJECT_BRIEF.md and docs/session_ishaan.md, then follow docs/session_ishaan.md.`

Claude: minimal explanations, bullets, step-by-step commands, no em dashes.

## Role

- Serial bridge, firmware, pitch slides and demo script. Tests the dashboard in the browser after every push.
- Owned: `src/bridge/`, `firmware/`, `requirements-b.txt`.
- Handover (26 Sept, agreed): `src/racecontrol/`, `dashboard/` and `tests/test_racecontrol.py` are Naman's now. B1 to B3 (engine, client, dashboard skeleton) are done and merged in PR #5. Report dashboard and race control problems to Naman, never fix them yourself.
- Branch: `b-work`. Commit messages start with `B:`.
- `b-work` was deleted on GitHub after PR #5 merged. Your local `b-work` is fine: `git push -u origin b-work` recreates it the next time you push.

## Claude Code setup (Pro account)

- `claude update`
- `claude --model sonnet` inside the repo folder.
- Stubborn bug after 2 tries: `/model`, pick Opus, press `s` (this session only).
- Claude limit used up: use your Codex account until it resets. Codex does not read `CLAUDE.md` on its own, so always start it with the first message above. Every `CLAUDE.md` rule applies to Codex too.

## Setup

1. Accept the GitHub collaborator invite email (the repo is private), then `git clone https://github.com/Pseudocoder28/Fast-Flag.git fast-flag`
2. `cd fast-flag`
3. Mac: `python3.11 -m venv .venv && source .venv/bin/activate`. Windows: `py -3.11 -m venv .venv` then `.venv\Scripts\activate`.
   - No Python 3.11? `python3 -m pip install --user uv`, then `uv python install 3.11` and `uv venv --python 3.11 .venv` (uv is in `~/Library/Python/3.9/bin/` on Mac), then activate as above and use `uv pip install -r requirements.txt` in step 4.
   - LightGBM on Mac needs `libomp` (`brew install libomp`). You do not use LightGBM, so an import error from it is safe to ignore.
4. `pip install -r requirements.txt`
5. `git checkout b-work && git merge main`
6. `pytest` (must pass)
7. `python -m src.replay.mock_server`
8. Open http://localhost:8000 (the dashboard).

## Fixture facts (what the mock server plays)

- Race `2023_Australian`, SessionTime 4240 to 4570 s, ~4 Hz ticks, loops forever.
- Incident: car `23` (Albon) crashes at Turn 6, marshal sector 9. IMPACT at t=4387.25, STOPPED at t=4390.5.
- Official: YELLOW sector 9 at t=4390.18, SC at t=4402.18, RED at t=4560.18.
- Jump straight to it: `curl -X POST localhost:8000/replay -H 'content-type: application/json' -d '{"speed": 1, "seek_t": 4370}'`. Speed 0 pauses.
- `GET /status` shows current replay time and speed. `GET /races` lists the races you can load.
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

**Ongoing: test the dashboard in your browser after every push (now until code freeze)**
- Goal: a second laptop and browser catch dashboard problems early. You report, Naman fixes.
- When Naman says he pushed:
  1. Save your own work: commit it, or `git stash`.
  2. `git fetch origin && git checkout --detach origin/a-work`
  3. Terminal 1: `python -m src.replay.mock_server --no-recs`
  4. Terminal 2: `python -m src.racecontrol`
  5. Open http://localhost:8000, click the Incidents tab and click the first incident (it jumps to 30 s before race control's first message). The curl command above still works too.
  6. Try speed 10 and 50 with the 10x and 50x buttons under the map.
  7. If you have `data/features` and `data/models` from Naman: stop terminal 1, run `python -m src.replay.server` instead and repeat steps 4 to 6.
  8. Back to your branch: `git checkout b-work`, then `git stash pop` if you stashed.
- Check:
  - Cars move smoothly at 1x, 10x and 50x.
  - Car 23 stops in sector 9 and is clearly marked as stopped.
  - Sector 9 lights up on the map and a row appears in the alert feed before the official YELLOW at t=4390.18.
  - The lead-time strip under the map pairs our YELLOW with the official one and shows a green +2.9 s.
  - Play/pause, 1x to 50x, -30 s/+30 s and clicking the seek bar all move the replay.
  - The banner says it is a replay of historical FastF1 data.
  - Nothing overlaps or gets cut off at your screen size.
  - No errors in the browser console (F12, Console tab).
- Expected, not bugs: on the mock every car shows "risk high" (the fixture risk values are hand-built and all above the real model's alert line). After a page reload the "Fast Flag recommends" chip says WAITING until the next seek or track-wide rec.
- Report to Naman: what you did, what you saw, a screenshot, browser name and window size. Never edit `dashboard/` or `src/racecontrol/`.
- Done when: every push Naman tells you about gets either "works" or an issue list from you.

**M1:** Naman runs the real server, racecontrol and dashboard end to end and merges into `main`. Pull `main` and merge it into `b-work`.

**M2 at midnight: MVP checkpoint.** Replay, detect, recommend and dashboard working end to end.

**B6. Hardware (midnight to 5:00am, only after M2 passes)**
- `src/bridge/`: subscribes to the stream and sends the Section 7.7 serial protocol at 115200 baud. Maps marshal sectors to 3 panel zones.
- Which rec goes where (PROJECT_BRIEF.md 7.4): flag VSC, SC or RED is track-wide, send `G,<flag>`. A CLEAR rec with message `TRACK CLEAR` ends it, send `G,GREEN`. Every other rec is a sector flag, send `Z,<zone of msector>,<flag>`.
- `firmware/uno_marshal/`: marshal panel (Section 11). 3 zones of green and yellow LEDs, red LED, SC/VSC LED, servo flag, buzzer through a BJT.
- `firmware/nano_dash/`: in-car dash (Section 11). LCD message and flag, joystick press sends `ACK,<ms>`, second buzzer through a BJT.
- Ollama narration only if everything else works (async, it never decides).
- Done when: a replayed incident lights the right zone, waves the flag and the Nano ACK shows up on the laptop.

**M3 at 5:00am:** prediction, narration and hardware working.

**B7. Pitch (5:00am to 8:00am)**
- Tell Naman what the dashboard still needs for the demo (he does the dashboard polish now).
- Slides and 3-minute demo script following PROJECT_BRIEF.md Section 12, using Naman's charts in `docs/charts/`. Every number on a slide comes from `docs/charts/NUMBERS.md` (Naman writes it, ask him if a figure is missing).
- Prepare the judge questions in Section 12.

**M4 at 8:00am: code freeze.** Rehearse the demo 3+ times, record a backup video, finalize slides by 10:30am, submit by 11:30am.

## Checkpoints

- M0 1:00pm, M1 9:00pm, M2 midnight, M3 5:00am, M4 8:00am (code freeze), rehearse until 10:30am, submit by 11:30am.

## Don't touch

- `src/ingest/`, `src/replay/`, `src/detect/`, `src/predict/`, `src/eval/`, `src/racecontrol/`, `dashboard/`, `fixtures/`, `docs/charts/`
- Naman's tests: `tests/test_racecontrol.py`, `tests/test_mock_server.py` and every other `tests/test_<area>.py` for his areas.
- Shared, only by agreement: `CLAUDE.md`, `PROJECT_BRIEF.md`, `requirements.txt`, `tests/test_contracts.py`

## Merging

- Before merging into `main`: `pytest` passes and `git diff --stat main...HEAD` shows only your paths.
- Merge one person at a time, following the CLAUDE.md workflow.

## Blocker protocol

- Stuck for more than 20 minutes: message Naman immediately with the error and what you tried.
