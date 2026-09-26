# Demo run sheet: the pit wall segment (about 60 s)

A drop-in section for the main demo script (Ishaan owns the script; this is the pit wall part, next to the lab segment in `docs/lab/DEMO.md`). It replays this morning's race, the 2026 Azerbaijan GP holdout, on the dashboard: Alex Albon (car 23) crashes on lap 30 and our race control recommends the Safety Car while the official race control feed is still green. Rehearsed on 26 Sept at 1440x900 following this sheet: wall times matched within about 1 s.

Run it on Naman's Mac: the holdout data (`data/holdout/2026_Azerbaijan*`) is only there.

## Honesty lines (say them, word for word)

- "This is a replay of historical FastF1 data from this morning's Azerbaijan Grand Prix. Our models never saw this race: it was run once, after the code was frozen."
- Say "earlier than the race control feed". Never say "earlier than the marshals": marshals wave local flags before any race control message.
- Car colours (risk heat) are an honest prediction here, because the model never saw this race. On the 20 training races the risk model trained on the very race, so red cars there are memory, not prediction: never present them as a prediction (the fallback below uses the mock, whose risk values come from the leave-one-race-out model).
- Numbers on slides come only from `docs/charts/NUMBERS.md`. The leads in this segment are read off the screen, from the live replay.

## Setup (before going on stage)

Two terminals in the repo root, venv active (`source .venv/bin/activate`):

```
python -m src.replay.server --race 2026_Azerbaijan --holdout --speed 0    # 1: server, paused (loading takes about 10 s)
python -m src.racecontrol                                                 # 2: our race control, sends the recommendations
```

Chrome, full screen (Cmd+Ctrl+F), at http://localhost:8000.

Cue it, 8 s before the crash, paused. It plays 1 s and then pauses: a seek while paused sends no tick, so the page and race control would only reset when you press Play, in front of the room.

```
curl -s -X POST localhost:8000/replay -H 'content-type: application/json' -d '{"speed": 1, "seek_t": 6631}' && sleep 1 && curl -s -X POST localhost:8000/replay -H 'content-type: application/json' -d '{"speed": 0}'
```

Pre-flight checklist:
- Terminal 2 printed `connected to ws://localhost:8000/stream` and `RESET at t=6631.0`.
- Top bar: "2026 Azerbaijan GP", lap 30, "REPLAY OF HISTORICAL FASTF1 DATA, NOT LIVE".
- Track status: "Fast Flag recommends GREEN", "Official race control GREEN". If ours says WAITING, send the cue command again.
- Feed tab selected (not Incidents), so the rows appear as they arrive.
- Do one full dry run, then cue again. A seek back resets everything cleanly.

## Run of show

Start with the Play button under the map (or the space bar).

| Replay t (s) | Wall time | On screen | Presenter says |
|---|---|---|---|
| 6632 | 0 s | Albon's car 23 at racing speed, lap 30 | "This morning in Baku, lap 30. Watch car 23, Alex Albon." |
| 6640.25 | 8 s | Sector 7 turns yellow. Feed: IMPACT, then YELLOW from Fast Flag | "Our detectors see the impact in the telemetry." |
| 6641.25 | 9 s | Sector 7 double yellow (double stripe). At about 11 s car 23 gets a grey ring and STOPPED | |
| 6644.25 | 12 s | Track status: **Fast Flag recommends SAFETY CAR**, official still **GREEN**. Orange halo around the track | "Three seconds after the car stops, we recommend the Safety Car. Race control's feed is still green." |
| 6648.53 | 17 s | Race control's DOUBLE YELLOW arrives. Lead strip: green **+8.3 s** | "Race control's first flag arrives. The strip shows our lead." |
| | 17 s | Click **5x** under the map, so the room does not wait | |
| 6667.53 | about 22 s | Official **SAFETY CAR**. Lead strip: green **+23.3 s**, feed row "Fast Flag 23.3 s earlier" | "Race control calls the Safety Car. We were 23 seconds earlier than the race control feed." |
| | | Click **Play/Pause** to freeze on the lead strip | Point at the strip and the two status chips. |

Slide to follow, numbers from `docs/charts/NUMBERS.md` (Holdout section): on this race our rule alerts caught 7 of 10 official incidents at 0.6 false alarms per race hour; race control called the Safety Car 28.3 s after Albon's crash; 4 cars passed the crash site between our recommendation and the official one.

## Do not show

- The Turn 1 restart pile-up (t about 7478). We recommend RED within 1 s; race control chose a double yellow and then a Safety Car 65 s later. It is a real disagreement, not a clean lead, and invites a debate on stage.

## If something goes wrong

- Nothing moves: the replay is paused. Click Play.
- "Fast Flag recommends WAITING": the page was reloaded after the cue. Send the cue command again.
- Server or race control crashed: Ctrl+C both, start them again, cue again (about 15 s).
- Anything else: switch to the backup video.
- No holdout data on the machine (Ishaan's laptop): the mock server shows the 2023 Australian GP Albon crash instead. `python -m src.replay.mock_server --no-recs` plus `python -m src.racecontrol`, then the Incidents tab, first incident. Say "2023 Australian Grand Prix", not "this morning".

## On Windows

Same steps with `.venv\Scripts\python -m ...` instead of `python -m ...`, port 8000. For the cue, in PowerShell:
`Invoke-RestMethod -Method Post -Uri http://localhost:8000/replay -ContentType 'application/json' -Body '{"speed": 1, "seek_t": 6631}'; Start-Sleep 1; Invoke-RestMethod -Method Post -Uri http://localhost:8000/replay -ContentType 'application/json' -Body '{"speed": 0}'`
