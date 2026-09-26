# Session: Naman (A)

Follow after `docs/KICKOFF.md` is done. Full specs are in `PROJECT_BRIEF.md`, and the rules in `CLAUDE.md` always apply. Work on branch `a-work`. For every task: do it, show me how to check it worked, commit with an `A:` message and then move on.

## 1:00pm to 5:00pm

**A1. Full downloads and ingest (Section 6.1)**
- Fully download the top 15 to 25 races from `data/race_ranking.csv`, never the holdout.
- Merge `car_data` and `pos_data` per driver onto a 250 ms grid (interpolate, never extrapolate).
- Per circuit: reference line, 10 m speed profile, KD-tree mapping X/Y to (dist, lat_off), marshal sector mapping from `get_circuit_info()`.
- Save to `data/features/<race>.parquet`.
- Done when: 3 races are processed and a plot of the reference line with marshal sectors looks right to me.

**A2. Replay engine and real server (Section 6.2)**
- `src/replay/engine.py` and `src/replay/server.py`, with exactly the same endpoints and envelope as the mock server.
- Add a test that proves strict causality: no consumer ever receives data with t' > t.
- Done when: a small test client receives real ticks from a real training race at 1x and 10x speed.

## 5:00pm to 9:00pm

**A3. Detectors (Section 6.3)**
- STOPPED, IMPACT, SPIN, DROPOUT, MULTI, plus the ANOMALY detector: an IsolationForest trained only on normal windows from training races.
- Handle Safety Car and VSC periods by comparing against the field median, as Section 6.3 says.
- Emit `detection` envelopes on the stream.
- Done when: on 3 training races every official incident has a detection, and the false alarms per race hour are printed.

**A4. Evaluation (Section 6.7)**
- Matching, lead time, recall and false alarms per hour, compared with the naive speed-threshold baseline.
- Tune thresholds with leave-one-race-out on training races only.
- Done when: a results table for the training races is printed and saved to `docs/charts/`.

## From 4:30pm on 26 Sept: race control and dashboard (taken over from Ishaan)

Naman now also owns `src/racecontrol/`, `dashboard/` and `tests/test_racecontrol.py` (CLAUDE.md, Who is who). Ishaan finished B1 to B3 (engine, client, dashboard skeleton, merged in PR #5) and now tests the dashboard in his browser after every push. `RaceControl.on_tick` returns `(recs, did_reset)`, `on_detection` and `on_risk` return a list of recs. The `human` envelope stays parked until after M2.

**A8. End to end run (this is M1)**
- Run the real server (`python -m src.replay.server`), `python -m src.racecontrol` and the dashboard together on one training race incident.
- List every problem you see before fixing any of them.
- Done when: the problem list is written down and the incident flows through replay, detect, recommend and dashboard.

**M1:** A8 passes. Merge into `main` following the CLAUDE.md workflow and tell Ishaan to test the dashboard against the real server.

**A9. Race control: clearing and re-escalation**
- Bug: the engine never clears an SC, so later crashes never escalate. Fix it with the clearing and hysteresis rules in PROJECT_BRIEF.md Section 6.5.
- Done when: a test with two separate incidents in one race passes, and the second one escalates after the first clears.

**A10. Dashboard: car movement and the stuck car (`dashboard/` only)**
- Stuck car: find the cause first. Same x/y in every tick = stopped or retired car, ticks stop arriving = data dropout, or `in_pit` is true. A stopped car stays on the map, clearly marked (grey ring plus a STOPPED or OUT label). No data for 5 s of replay time: fade it out. `in_pit` true: hide it or draw it in the pit lane.
- Smooth movement: keep the previous and latest position per car and interpolate in the requestAnimationFrame loop, based on wall-clock time since the latest tick. Use the measured wall-clock gap between ticks, not a fixed 250 ms (it shrinks at 10x and 50x). Snap instead of interpolating on seek, on jumps over 200 m and on pit entry or exit.
- Draw the canvas only inside the rAF loop. Update the feed only when a detection, rec or official envelope arrives, and keep only the last 50 rows.
- Dots about 3x the old radius, car number (`drv`) in bold white text centred on each dot.
- Done when: cars glide at 1x, 10x and 50x, and a stopped car, a car with no data and a car in the pit lane each look right.

**A11. Dashboard: visual overhaul (`dashboard/` only, no CDN, no external fonts or images)**
- Broadcast pit wall look: matte dark panels, subtle carbon-fibre texture from CSS gradients, padded cards, subtle 1 px borders, rounded corners, uppercase section headers.
- Typography: system monospace stack (ui-monospace, SFMono-Regular, Menlo, Consolas) for times and lap numbers, system sans stack (system-ui, -apple-system, Segoe UI, Roboto) for labels.
- Feed rows: a flag-colour bar on the left and a coloured badge with the flag name as text.
- Track: CLEAR `ref_line` stroke about 2 to 3x the old width, flagged segments still wider than CLEAR.
- Flag colours: YELLOW #ffd400, DOUBLE_YELLOW #ffb000 with a double stripe, VSC #00a3e0, SC #ff6b00, RED #e10600.
- Small inline SVG flag icons next to flag names in the legend and the feed. No emoji.
- Keep the "Replay of historical FastF1 data" banner.
- Done when: it looks right in Chrome at 1920x1080 and at laptop size, against both the mock server and the real server.

**A12. Housekeeping (last)**
- Regenerate or delete the four stale `detect_*` files in `docs/charts/` so they match the final detector settings.
- Write `docs/charts/NUMBERS.md`: every pitch figure with its source file. It is the single source of truth for the slides.

**Still open from Ishaan's B4 (not scheduled yet):** lead-time timeline (our recs next to `official` envelopes), replay controls (speed and seek through `POST /replay`) and a jump-to-incident list from `GET /official`.

## 9:00pm to midnight

**A5. Integration and tuning**
- Fix contract mismatches in your own code and the dashboard issues Ishaan reports. Never edit his files (`src/bridge/`, `firmware/`).
- Tune detector thresholds on training races only.

**M2 at midnight: MVP checkpoint.** End to end on a training race incident: replay, detect, recommend and dashboard. If this isn't working, drop prediction and polish detection plus lead time.

## Midnight to 5:00am

**A6. Prediction (Section 6.4)**
- Labels, LightGBM risk model for `risk_10s` and `risk_30s`, leave-one-race-out CV, PR-AUC per horizon and `top_features`.
- Compare against the ANOMALY score and the speed threshold.
- SC probability model if time allows (cut list item 3).
- Emit `risk` envelopes.
- Done when: PR-AUC per horizon is saved to `docs/charts/` and risk streams live.

**M3 at 5:00am:** prediction working. Merge.

## 5:00am to 8:00am

**A7. Holdout and charts**
- Run the full pipeline once on the holdout race. Never tune after seeing the result.
- Charts in `docs/charts/`: lead-time distribution, per-incident timeline and holdout summary, ready for Ishaan's slides.

**M4 at 8:00am: code freeze.** Bug fixes only after this point.
