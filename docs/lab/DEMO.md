# Demo script: the lab segment (about 45 s)

A drop-in section for the main demo script (Ishaan's). It shows one crash through the broadcast overlay and the race control voice, then the cost of the delay. Every number below comes from a live run of this exact setup (26 Sept, real server, real detectors, real race control engine) and from `docs/lab/delay_cost.json`.

**The crash:** 2021 Azerbaijan GP, lap 31. Lance Stroll (car 18) has a tyre failure at full speed on the main straight, marshal sector 20. Crash onset: SessionTime 5274.25 s.

Why this race: it is out of sample twice. Our models never saw it, and 2021 cars ran under different technical rules from our 2023 to 2026 training data. It is not a holdout race, so using it breaks no rule.

## Honesty lines (say them, word for word)

- "This is a replay of historical FastF1 data from the 2021 Azerbaijan Grand Prix."
- Say "earlier than the race control feed". Never say "earlier than the marshals": marshals wave local flags before any race control message.
- The cars-exposed numbers are a counterfactual: we count how cars actually drove, with no model of how they would have reacted to an earlier flag.

## Setup (before going on stage)

Four terminals, all in the repo root with the venv active (`source .venv/bin/activate`). Until PR #8 is merged, run them from `.claude/worktrees/lab` with `/Users/namanshah/projects/fast-flag/.venv/bin/python` instead of `python`.

```
python -m src.replay.server --race 2021_Azerbaijan --speed 0      # 1: server, paused (loading takes about 10 s)
python -m src.racecontrol                                          # 2: race control engine, sends our recs
python -m src.lab.voice                                            # 3: the voice (Daniel)
```

4. Chrome, full screen (Cmd+Ctrl+F), at `http://localhost:8000/lab/overlay.html`. Optionally, a second window with the pit wall dashboard at `http://localhost:8000`.

Cue it, 6 s before the crash, paused:

```
curl -s -X POST localhost:8000/replay -H 'content-type: application/json' -d '{"speed": 0, "seek_t": 5268}'
```

Pre-flight checklist:
- Mac volume up, output to the room speakers. Test with `say -v Daniel "Fast Flag check"`.
- The overlay's top right reads LIVE REPLAY. The bottom left label reads "Replay of historical FastF1 data. Counterfactual."
- Terminal 2 printed `connected`, terminal 3 printed `voice: connected`.
- Do a full dry run once, then re-cue with the command above. Seeking back resets everything cleanly.

### Race names and switching races

- The real server: `python -m src.replay.server --race <race id>`, on port 8000. Other options: `--speed 0` starts paused, `--port 8001` uses another port, `--no-detect` and `--no-predict` turn off detections and risk.
- A race id is `<year>_<event name without "Grand Prix", spaces as underscores>`: `2023_Australian`, `2024_Canadian`, `2023_Mexico_City`, `2024_São_Paulo`. `GET /races` lists every id the running server can switch to.
- `POST /replay {"race": "2024_Canadian", "seek_t": 4980, "speed": 1}` switches race while the server runs, but only to the 20 training races in `GET /races`.
- **`2021_Azerbaijan` is a case study, not a training race, so `POST /replay` answers 404 for it.** Start the server with `--race 2021_Azerbaijan`. If you switch away during rehearsal, restart the server to get back.
- The holdout races (`2026_Azerbaijan`, `2026_Spanish`) need `--holdout`, and only for the A7 demo.

### On Windows (Ishaan's laptop)

- Same four steps, with `.venv\Scripts\python -m ...` instead of `python -m ...`. Everything runs on port 8000. Nothing extra to install, no Ollama.
- The server needs `data\case_studies\2021_Azerbaijan*` (5 files, 75 MB), copied from Naman's `data` folder.
- **The voice only speaks on a Mac** (it uses the built-in macOS `say`). On Windows it prints the lines instead. For a spoken demo, run the voice on Naman's Mac, pointed at the demo machine: `python -m src.lab.voice --url ws://<demo machine IP>:8000/stream`. The server must then be started with `--host 0.0.0.0` so the Mac can reach it.
- The `curl` lines below are for macOS. In PowerShell, use for example:
  `Invoke-RestMethod -Method Post -Uri http://localhost:8000/replay -ContentType 'application/json' -Body '{"speed": 0, "seek_t": 5268}'`

## Run of show

Start: `curl -s -X POST localhost:8000/replay -H 'content-type: application/json' -d '{"speed": 1}'`

| Replay t (s) | Wall time | On screen | Voice | Presenter says |
|---|---|---|---|---|
| 5268 | 0 s | Big track map, every car moving, both chips green | | "Baku 2021, lap 31. Watch car 18, Lance Stroll, flat out on the main straight." || 5275.25 | 7 s | YELLOW banner: CAR #18, IMPACT, SECTOR 20 | "Yellow flag. Car 18, impact, sector 20." | (let the voice speak) |
| 5276.25 | 8 s | Banner becomes DOUBLE YELLOW: STOPPED | "Double yellow. Car 18, stopped, sector 20." | |
| 5277.99 | 10 s | Double yellow banner: FAST FLAG AHEAD BY 1.7 s | | |
| 5279.25 | 11 s | SAFETY CAR banner. Exposure Clock starts in the top right. The map shrinks to the top centre: the track turns orange, car 18 pulses red, chips read FAST FLAG SAFETY CAR and RACE CONTROL YELLOW | "Safety Car. Car 18, stopped, sector 20." | "Five seconds after the crash, we recommend the Safety Car. The clock counts how long the track stays live, and every car that passes the wreck at racing speed." |
| about 5281 | 13 s | Ticker lines appear: #16, #5, #22, ... | "Race control confirms Double yellow, 2.7 seconds after Fast Flag." | |

At about 13 s wall time, speed it up so the room does not wait 30 s:
`curl -s -X POST localhost:8000/replay -H 'content-type: application/json' -d '{"speed": 4}'`

| Replay t (s) | Wall time | On screen | Voice | Presenter says |
|---|---|---|---|---|
| 5282 to 5311 | 13 to 21 s | Clock runs, CARS EXPOSED climbs to 6 | | "Four times speed now. Race control's feed is still quiet." |
| 5311.99 | 21 s | Clock freezes: "Race control +32.7 s, Cars exposed 6". SC banner: FAST FLAG AHEAD BY 32.7 s | "Race control confirms Safety Car, 32.7 seconds after Fast Flag." | "Race control called the Safety Car 32.7 seconds after us. In that gap, 6 cars passed the stricken car at racing speed." |

Pause: `curl -s -X POST localhost:8000/replay -H 'content-type: application/json' -d '{"speed": 0}'`

If the replay keeps running: at 5396.25 (2 minutes after the crash) the double-yellow banner's reason changes to "recovery taking long, race control may need a red flag (advisory, flag unchanged)". That is not a red flag, and the voice stays silent. Race control never red-flagged this crash, and neither do we. If asked, say: "We only call a red where the data can support it: late in the race. A long recovery mid-race is an advisory, because the reasons for a red (barrier damage, gravel, debris) aren't visible in car data."

Close on the slide with `docs/lab/delay_cost_2021_Azerbaijan_car18_5274.png`, or the steward card `docs/lab/cards/card_2021_Azerbaijan_car18_5274.png`:

> "This is the cost of a delay: cars past a crash at racing speed, for every second a flag waits. Across 62 crashes in our training races, each second of delay averaged about 0.12 cars. It's a counterfactual: we count what cars actually did, not how they would have reacted to an earlier flag."

## The numbers, and why two of them differ

| Number | What it is | Source |
|---|---|---|
| 2.7 s | Race control's double yellow (5277.99, sector 21) after our first yellow (5275.25, sector 20). Sector 21 is the next sector, where race control flags the approach. | voice, live run |
| 1.7 s | The double-yellow banner compares like with like: our double yellow (5276.25) against race control's double yellow (5277.99). | overlay, live run |
| 32.7 s | Race control's Safety Car (5311.99) after ours (5279.25). The same gap as W3 in `docs/charts/case_studies.json`. | voice, overlay, case study |
| 6 cars | Passed at racing speed between our Safety Car call and race control's (overlay clock). | overlay, live run |
| 7 cars | Passed at racing speed between the crash onset and race control's Safety Car (delay-cost chart). One more car, Gasly, passed 0.3 s before our call, so it is not in the overlay's count. | `delay_cost.json` |
| 0.12 cars per second | Mean over the 62 crash curves, first 60 s after onset. | `delay_cost.json` |

The slide numbers come from `docs/charts/NUMBERS.md`, section "Lab: cost of delay". The overlay's live clock can differ from them by a car. The chart compares each car with its own previous 3 clean laps. The overlay can only compare with what it has seen since the page loaded: the car's last green lap, or the field's median speed at that point. For Stroll both give 6. For Verstappen the chart gives 11 from our call and the live overlay showed 12.

If a judge asks why the clock says 6 and the chart says 7: the chart counts from the crash, the clock counts from our call.

## Backup clip: Verstappen, same race (if there is time, or if the Stroll run misbehaves)

Car 33, lap 46, onset 7278.25 s, again a tyre failure on the straight. Cue with `seek_t` 7272 and play at speed 1, then speed 4 from about 7286.

- Voice: "Yellow flag. Car 33, impact, sector 21." (7279.0), then Double yellow, then "Safety Car. Car 33, stopped, sector 21." (7283.25).
- Race control's double yellow comes at 7297.99: "Race control confirms Double yellow, 19.0 seconds after Fast Flag."
- Race control's Safety Car comes at 7366.99: "Race control confirms Safety Car, 83.7 seconds after Fast Flag." The clock freezes at 83.7 s. In the live test it showed 12 cars exposed.
- Say: "83.7 seconds between our Safety Car call and race control's. Watch the counter: every one of those cars went past the wreck at racing speed."
- Don't say a car count for this clip. NUMBERS.md says 11 cars from our call, and the slides use that number. The overlay showed 12 because its live speed reference differs, as explained in the table above. A spoken 12 next to a slide saying 11 would look like an error.
- Keep it running for the red flag. At 7400.25 the double-yellow banner gets the "recovery taking long" advisory. At 7420.75 (lap 48, about 4 laps left) we call **RED**: "SC out for a live incident with about 4 laps left: a red flag lets the race restart and finish racing, not behind the SC." The voice says "Red flag. Car 33, stopped, sector 21." Race control's real red came at 7562.99, **142 s later**. Say: "With four laps left, we recommend the red flag so the race can finish racing, not behind the Safety Car. Race control made the same call 142 seconds later."

## If something goes wrong

- **No sound:** the overlay still shows everything. Keep going and read the banner lines yourself. Check the volume afterwards.
- **No banners:** terminal 2 (race control) is not connected. Restart it, re-cue with the seek command and start again.
- **Wrong state after a mistake:** any seek resets the voice, the overlay and the race control engine. Re-cue and restart.
- **Nothing works:** show the steward card PNG and the delay-cost chart, and tell it as a story from the timeline on the card.

## Likely judge questions

- **"Is this live?"** No. It's a replay of historical FastF1 data, tick by tick. No component ever sees data from after the current tick.
- **"Did you train on this race?"** No. 2021 is out of sample twice: never seen by our models, and different car rules from our 2023 to 2026 training data.
- **"Would 6 cars really have slowed down?"** We don't claim that. It's a counterfactual count of how they actually drove, with no model of how drivers react.
- **"Were you earlier than the marshals?"** We only claim earlier than the race control feed. Marshals wave local flags before race control's messages appear.
