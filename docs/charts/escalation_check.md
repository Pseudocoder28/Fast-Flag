# Escalation scorecard (check races)

Replay of historical FastF1 data: 20 check races, 30.1 race hours, the next 20 in data/race_ranking.csv after the training races (the holdout skipped). Our side is the race control engine (src/racecontrol) fed by the production detectors and ANOMALY model, trained on the training races. Definitions: src/eval/escalation.py.

Out of sample: run 2, at 2026-09-27 04:30, on commit 141f173. No rule or model was fitted or tuned on these races.
- Earlier run 1: 2026-09-27T03:09:49, commit 013bd25
- This run's reason: 27 Sept detector and race control fixes found on the 2021 Azerbaijan replay and checked on the training races (never on these races)

Counterfactual: assumes race control acted on our recommendation at once. Race control also has marshal reports and CCTV, and picks VSC or SC by recovery work we cannot see. Claim earlier than the race control feed, never earlier than the marshals.

## Official escalations: 33

- We recommended a VSC, SC or red for 22 of 33 (67%): 22 earlier than race control, 0 later. Median lead 32.6 s (middle half 21.7 to 48.4 s).
- Already out (our flag was up before, no new recommendation): 1. Missed: 10, of which 3 had a car collapse our onset rule found and 7 had none.
- Same first flag as race control: 14 of 22 (official Safety car and ours Safety car: 6; official Safety car and ours VSC: 4; official VSC and ours Safety car: 4; official VSC and ours VSC: 8).
- Red flags: race control showed a red in 2 of its escalated incidents and we recommended red in 0 of them. In all we sent 3 red flags (3 late in the race); race control showed a red around 0 of them and never called 3. The data cannot see barrier damage, debris or medical needs: do not claim red flag accuracy.
- Our TRACK CLEARs while race control's track status showed its own SC, VSC or red: 0.

| official flag | escalations | earlier | later | missed | median lead (s) |
|---|---|---|---|---|---|
| VSC | 18 | 12 | 0 | 5 | 32.6 |
| Safety car | 15 | 10 | 0 | 5 | 33.7 |
| Red flag | 0 | 0 | 0 | 0 | n/a |

Missed official escalations:

| race | official time (s) | flag | sectors | kind | detail |
|---|---|---|---|---|---|
| 2023_Las_Vegas | 3681.8 | SC | track-wide | no car collapse | car already slow or stopped before the window (re-flag of an earlier stoppage) |
| 2026_Monaco | 7929.7 | SC | 17 18 | no car collapse | no data at the official time |
| 2026_Monaco | 8605.7 | SC | 18 | no car collapse | no data at the official time |
| 2026_Austrian | 5199.9 | VSC | 1 16 | no car collapse | no car collapsed below 50% (debris, weather, or a car that went off and kept going) |
| 2026_Austrian | 7248.9 | VSC | track-wide | no car collapse | no car collapsed below 50% (debris, weather, or a car that went off and kept going) |
| 2023_Japanese | 3856.6 | SC | 4 | no car collapse | lap 1 or 2: no clean reference laps yet |
| 2023_Japanese | 5312.6 | VSC | track-wide | car collapse seen | car 11 collapsed 9 s before the first official message; we made no recommendation |
| 2026_Barcelona | 8405.2 | VSC | 7 | no car collapse | no car collapsed below 50% (debris, weather, or a car that went off and kept going) |
| 2023_Canadian | 4664.0 | SC | 9 | car collapse seen | car 63 collapsed 10 s before the first official message; we showed double yellow but did not escalate |
| 2025_Las_Vegas | 4993.1 | VSC | track-wide | car collapse seen | car 23 collapsed 70 s before the first official message; we made no recommendation |

## Our escalations: 47

- Matched an official escalation: 23. During an official neutralisation (not extra): 1.
- Extra, race control never escalated: 23, 0.76 per race hour (16 where race control kept yellows only, 0 near an escalated incident but outside its match window, 7 with no official flag, 0 red flags race control never called while its own SC or VSC was out). By flag: VSC 9, Safety car 12, Red flag 2.
- Race control engine resets (tick jumps over 2 s): 0.

## Flag choice: VSC or SC

- Our first flag matched race control's in 14 of 22 matched escalations (official Safety car and ours Safety car: 6; official Safety car and ours VSC: 4; official VSC and ours Safety car: 4; official VSC and ours VSC: 8).
- Our 47 recommendations by flag: VSC 21, Safety car 23, Red flag 3.
- The lateral offset cannot choose between them. FastF1 positions of stopped cars sit on the racing line, even for retired cars parked in run-off: of 466,715 stopped-car rows (below 5 km/h, outside the pit lane), 96.11% are within 1 m of it and 0 are more than 8 m away (short episodes: none).
- An impact separates them better. On the 17 official escalations with an onset car, 2 of 7 SC or red followed an IMPACT or MULTI detection involving the car and 8 of 10 VSC did not: "SC after an impact, VSC otherwise" separates race control's VSCs from its SCs and reds correctly 10 times, "always SC" 7 times. This check does not tell SC from red; exact agreement is the first line. In-sample, and a small sample.

## Per race

| race | race hours | official escalations | earlier | later | missed | our escalations | extra |
|---|---|---|---|---|---|---|---|
| 2023_Las_Vegas | 1.49 | 3 | 2 | 0 | 1 | 4 | 2 |
| 2023_Singapore | 1.78 | 2 | 2 | 0 | 0 | 8 | 6 |
| 2026_Monaco | 0.95 | 2 | 0 | 0 | 2 | 0 | 0 |
| 2024_Azerbaijan | 1.55 | 1 | 1 | 0 | 0 | 2 | 0 |
| 2026_Austrian | 1.45 | 2 | 0 | 0 | 2 | 0 | 0 |
| 2023_Dutch | 1.70 | 2 | 2 | 0 | 0 | 2 | 0 |
| 2024_Monaco | 1.71 | 0 | 0 | 0 | 0 | 1 | 1 |
| 2024_Australian | 1.34 | 2 | 2 | 0 | 0 | 2 | 0 |
| 2024_Chinese | 1.68 | 2 | 2 | 0 | 0 | 2 | 0 |
| 2026_Chinese | 1.55 | 1 | 1 | 0 | 0 | 8 | 7 |
| 2023_Japanese | 1.52 | 2 | 0 | 0 | 2 | 2 | 2 |
| 2026_Barcelona | 1.54 | 2 | 1 | 0 | 1 | 1 | 0 |
| 2025_São_Paulo | 1.54 | 2 | 2 | 0 | 0 | 2 | 0 |
| 2023_Azerbaijan | 1.55 | 1 | 1 | 0 | 0 | 1 | 0 |
| 2024_Abu_Dhabi | 1.44 | 1 | 1 | 0 | 0 | 3 | 2 |
| 2023_Canadian | 1.57 | 2 | 1 | 0 | 1 | 4 | 2 |
| 2023_São_Paulo | 1.52 | 1 | 1 | 0 | 0 | 1 | 0 |
| 2025_Las_Vegas | 1.36 | 2 | 0 | 0 | 1 | 1 | 1 |
| 2025_Saudi_Arabian | 1.35 | 1 | 1 | 0 | 0 | 1 | 0 |
| 2025_Emilia_Romagna | 1.53 | 2 | 2 | 0 | 0 | 2 | 0 |

Chart: escalation_check.png. Every official escalation: escalation_check_official.csv. Every recommendation of ours: escalation_check_ours.csv.
