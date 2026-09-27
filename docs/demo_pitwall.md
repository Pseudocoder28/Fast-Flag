# Demo run sheet: the pit wall

Drop-in sections for the main demo script (Ishaan owns the script; the lab overlay segment is `docs/lab/DEMO.md`). Two segments, each about 40 to 60 s:

- **A. 2021 Azerbaijan GP, Verstappen (the demo).** Our Safety Car call comes 83.7 s before race control's.
- **B. 2026 Azerbaijan GP holdout, Albon (backup, or the "tested on a race no model saw" point).** Our Safety Car call comes 23.3 s before race control's.

Both rehearsed on 27 Sept at 1440x900 on main with the precomputed server (PR #33) and dashboard v23, following this sheet. Wall times below are from those runs.

## Honesty lines (say them, word for word)

- Always: "This is a replay of historical FastF1 data."
- Say "earlier than the race control feed". Never "earlier than the marshals": marshals wave local flags before any race control message.
- Segment A: "Our machine-learning models never saw this race." Not "our system never saw it": some detector rule thresholds were refined with 2021 Baku in view.
- Segment B: "Our models never saw this race: the numbers on our slides come from one run on it, after the code was frozen." If asked: race control rules were refined after that frozen run and checked on replays of it; the early calls shown are the same in both.
- Car colours: red pulse is the car in the incident, outlines are the flag around a car, the halo is the risk model. The halo is an honest prediction only on races the model never trained on (both Baku races here).
- Numbers on slides come only from `docs/charts/NUMBERS.md`. The leads in these segments are read off the screen.

## Setup (before going on stage)

One terminal in the repo root, venv active (`source .venv/bin/activate`). The server computes our detections, risk and race control flags itself; no `python -m src.racecontrol` terminal is needed (the server ignores its recs).

```
python -m src.replay.server --race 2021_Azerbaijan --speed 0                 # segment A
python -m src.replay.server --race 2026_Azerbaijan --holdout --speed 0       # segment B (only on Naman's Mac)
```

The first start of a race runs our pipeline over the whole race once (about 1 minute for A, about 40 s for B) and caches it in `data/timeline/`; later starts take seconds. Start it before the judges arrive. Wait until `curl -s localhost:8000/status` answers.

Chrome, full screen (Cmd+Ctrl+F), at http://localhost:8000. Present at 1440x900 or larger.

Pre-flight checklist:
- Top bar: race name, lap, "REPLAY OF HISTORICAL FASTF1 DATA, NOT LIVE", "stream connected".
- After the cue: "Fast Flag recommends GREEN" and "Official race control GREEN".
- Feed tab selected (not Incidents), so rows appear as they arrive.
- Do one full dry run, then send the cue again.

## Segment A: 2021 Azerbaijan GP, Verstappen (car 33, lap 46)

Cue, 8 s before the crash, paused:

```
curl -s -X POST localhost:8000/replay -H 'content-type: application/json' -d '{"speed": 0, "seek_t": 7270}'
```

Start with the **Play** button under the map (or the space bar). After a cue, Play always starts at 1x.

| Replay t (s) | Wall time | On screen | Presenter says |
|---|---|---|---|
| 7270 | 0 s | Car 33 on the main straight, lap 46 | "Baku 2021, lap 46. Watch car 33, Max Verstappen, flat out on the straight." |
| 7279.0 | 9 s | Car 33 turns red with a pulsing ring, sector 21 yellow. Feed: IMPACT "lost 169 km/h in 1 s" | "A tyre failure at over 260 km/h. Our detectors see it in the telemetry." |
| 7280.25 | 10 s | Sector 21 double yellow | |
| 7283.25 | 14 s | **Fast Flag recommends SAFETY CAR**, official still **GREEN**. Orange halo, orange outline on every car | "Four seconds after the crash we recommend the Safety Car. Race control's feed is still green." |
| 7297.99 | 30 s | Race control's DOUBLE YELLOW. Lead strip: green **+19.0 s** | "Race control's first flag arrives, 19 seconds after ours." |
| | 30 s | Click **10x** under the map | |
| 7366.99 | about 37 s | Official **SAFETY CAR**. Lead strip: green **+83.7 s**, feed row "Fast Flag 83.7 s earlier" | "Race control calls the Safety Car, 84 seconds after we did." |
| | | Click **Play/Pause** to freeze | Point at the strip and the two status chips. |

Optional, if there is time (keep 10x running instead of pausing):
- About 2 minutes after the crash (t 7400.25) the feed shows an advisory: "recovery taking long, race control may need a red flag". It is advice; our flag stays the Safety Car.
- At t 7420.75 we recommend a **red flag** ("SC out for a live incident with about 4 laps left"). Race control's red flag came at t 7562.99, 142 s later. Pause and point at the two chips.

Slide to follow, from `docs/charts/NUMBERS.md` (Case studies and Lab sections): our SC +5.0 s after the crash onset, race control's +88.7 s; 16 cars passed between the two calls.

## Segment B: 2026 Azerbaijan GP holdout, Albon (car 23, lap 30)

Cue, 8 s before the crash, paused:

```
curl -s -X POST localhost:8000/replay -H 'content-type: application/json' -d '{"speed": 0, "seek_t": 6632}'
```

| Replay t (s) | Wall time | On screen | Presenter says |
|---|---|---|---|
| 6632 | 0 s | Car 23 at racing speed, lap 30 | "Yesterday in Baku, lap 30. Watch car 23, Alex Albon." |
| 6640.25 | 8 s | Sector 7 yellow, car 23 red with a pulsing ring. Feed: IMPACT, then YELLOW | "Our detectors see the impact in the telemetry." |
| 6644.25 | 13 s | **Fast Flag recommends SAFETY CAR**, official still **GREEN** | "Three seconds after the car stops, we recommend the Safety Car. Race control's feed is still green." |
| 6648.53 | 17 s | Race control's DOUBLE YELLOW. Lead strip: green **+8.3 s** | "Race control's first flag arrives." |
| | 17 s | Click **5x** | |
| 6667.53 | about 22 s | Official **SAFETY CAR**. Lead strip: green **+23.3 s** | "Race control calls the Safety Car. We were 23 seconds earlier than the race control feed." |
| | | Click **Play/Pause** to freeze | |

Slide to follow, from `docs/charts/NUMBERS.md` (Holdout section): on this race our rule alerts caught 7 of 10 official incidents at 0.6 false alarms per race hour; race control called the Safety Car 28.3 s after Albon's crash; 4 cars passed the crash site between our recommendation and the official one.

Do not show in B: the Turn 1 restart pile-up (t about 7478). Our Safety Car from Albon's crash is still held there, so the pile-up gets no new call of its own; it is not a clean story on stage.

## If something goes wrong

- Nothing moves: the replay is paused. Click Play.
- Play runs too fast: click 1x.
- Reloading the page or coming back from the overlay keeps the flags, the feed and the lead strip (the page catches up from the last seek).
- Something looks slow or choppy: open http://localhost:8000/?debug, which shows the frame rate in the corner.
- Server crashed: Ctrl+C, start it again (seconds, the race is cached), cue again.
- Anything else: switch to the backup video.
- No holdout data on the machine (Ishaan's laptop): segment A only needs `data/case_studies/2021_Azerbaijan*`; segment B needs `data/holdout/2026_Azerbaijan*`, which is only on Naman's Mac.

## On Windows

Same steps with `.venv\Scripts\python -m ...` instead of `python -m ...`, port 8000. For the cue, in PowerShell, for example:
`Invoke-RestMethod -Method Post -Uri http://localhost:8000/replay -ContentType 'application/json' -Body '{"speed": 0, "seek_t": 7270}'`
