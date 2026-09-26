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

**M1 at 9:00pm:** the real replay and detectors are streaming. Merge into `main` following the CLAUDE.md workflow and tell Ishaan to switch from the mock server to the real one.

## 9:00pm to midnight

**A5. Integration and tuning**
- Fix contract mismatches that Ishaan reports in your own code. Never edit his files.
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
