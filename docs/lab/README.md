# Fast Flag Lab

Experiments on top of the core stream. Everything here only consumes the existing contract (PROJECT_BRIEF.md Section 7): the WebSocket envelopes and the HTTP endpoints. Nothing sends a new envelope kind and nothing changes the server. Every output says it is a replay of historical FastF1 data; every exposure number is a counterfactual with no model of driver reactions.

Files: `src/lab/`, `dashboard/lab/`, `docs/lab/`, `tests/test_lab_*.py`. Run everything from the repo root with the project venv.

## 1. Race control voice (`src/lab/voice.py`)

Speaks the `rec` envelopes like race control radio, from templates only: flag, car numbers, detection type and sector, for example "Safety Car. Car 23, impact, sector 9." It speaks only when a sector's flag level goes up, says "Sector 9 clear." when it returns to CLEAR, and when the official message for a flag we already raised arrives it says "Race control confirms Safety Car, 8.7 seconds after Fast Flag." When race control was first it says nothing about lead time. macOS `say` runs in a non-blocking subprocess; a higher flag level cuts the current phrase, lower levels wait for the phrase and a 2 s gap. On a non-Mac, or with `--print`, the lines are printed.

```
python -m src.lab.voice                                   # ws://localhost:8000/stream, voice Daniel
python -m src.lab.voice --url ws://127.0.0.1:8765/stream --voice Samantha
python -m src.lab.voice --print                           # lines instead of speech
```

Check: run the mock server on a spare port (`python -m src.replay.mock_server --port 8765`), seek it to just before the fixture crash (`curl -X POST localhost:8765/replay -H 'content-type: application/json' -d '{"speed": 1, "seek_t": 4380}'`) and start the voice with `--print --url ws://127.0.0.1:8765/stream`. Expected lines, in this order: Yellow flag (car 23, impact, sector 9), Race control confirms Yellow flag 2.7 s after Fast Flag, Double yellow, Safety Car, Race control confirms Safety Car 8.7 s after Fast Flag. Tests: `pytest tests/test_lab_voice.py`.

## 2. Broadcast overlay (`dashboard/lab/overlay.html`)

Served by the existing static mount: http://localhost:8000/lab/overlay.html (or the mock server's port). Full-screen lower-third banners for every rec at YELLOW or above (flag badge, cars, sector, detection type, confidence, FAST FLAG wordmark), plus the Exposure Clock: when our rec reaches VSC, SC or RED before the official message of that level, it counts replay seconds and the cars that pass the stricken car at racing speed (80% or more of that car's own speed there on its previous lap, else of the field's median speed there, else 120 km/h), one ticker line per car. The official message freezes it: "Race control +X.X s | Cars exposed N", and the banners show "FAST FLAG AHEAD BY X.X s" or "RACE CONTROL FIRST BY X.X s". Strictly causal: only envelopes already received, clocks on replay time.

Check: with the mock server on 8765 and the replay seeked to 4300 at speed 3, open http://127.0.0.1:8765/lab/overlay.html in Chrome at 1920x1080. At t = 4387.5 the YELLOW banner slides in, at 4393.5 the SAFETY CAR banner and the clock start, at 4402.18 the clock freezes at 8.7 s with the cars that passed car 23. No file is fetched from a CDN, and the label "Replay of historical FastF1 data. Counterfactual." never leaves the screen.

## 3. Delay-cost curve (`src/lab/delay_cost.py`)

For every training race incident with an identifiable onset, plus the 2021 Azerbaijan crashes (never a holdout), how many cars passed the crash site at racing speed within d seconds of the onset, for d from 0 to 60 s in 0.5 s steps, with markers for our first alert, our escalation, the official yellow and the official escalation. The onset rule and the onset-car selection are imported from `src.eval.onset` and `src.eval.latency_by_type`; the crash location and the pass detection from `src.eval.case_study`.

```
python -m src.lab.delay_cost                 # every training race + 2021_Azerbaijan, about 3 minutes
python -m src.lab.delay_cost 2021_Azerbaijan --workers 1
python -m src.lab.delay_cost --replot        # redraw the PNGs from delay_cost.json
```

Outputs: `docs/lab/delay_cost.json`, `docs/lab/delay_cost.md` (one summary line per incident, "each second of delay here averaged X cars") and `docs/lab/delay_cost_<race>_car<N>_<t>.png`. Tests: `pytest tests/test_lab_delay_cost.py`.

## 4. Steward's cards (`src/lab/cards.py`)

One self-contained HTML card (inline SVG, no CDN, system fonts) and one 1280x720 PNG per incident in `docs/lab/cards/`, rendered from `delay_cost.json`: header (race, lap, sector, car), timeline strip, the car's speed against its own normal speed, the detector evidence, the official race control feed, our recommendation and reason, the exposure numbers, a delay-cost mini chart and the honesty label. `docs/lab/cards/index.html` lists them.

```
python -m src.lab.cards
python -m src.lab.cards --only 2021_Azerbaijan
```

Tests: `pytest tests/test_lab_cards.py`.

## Requests to the core

None so far. If the lab ever needs something from the core it goes in `docs/lab/REQUESTS.md`.
