# Escalation scorecard (training races)

Replay of historical FastF1 data: 20 training races, 31.2 race hours. Our side is the race control engine (src/racecontrol) fed by the production detector settings with an ANOMALY model trained on the other training races. Definitions: src/eval/escalation.py.

In-sample: the detector settings were tuned and the race control rules were set on these races; the holdout run (A7) gives the out-of-sample version.

Counterfactual: assumes race control acted on our recommendation at once. Race control also has marshal reports and CCTV, and picks VSC or SC by recovery work we cannot see. Claim earlier than the race control feed, never earlier than the marshals.

## Official escalations: 51

- We recommended a VSC, SC or red for 29 of 51 (57%): 27 earlier than race control, 2 later. Median lead 20.0 s (middle half 9.1 to 33.9 s).
- Already out (our flag was up before, no new recommendation): 1. Missed: 21, of which 7 had a car collapse our onset rule found and 14 had none.
- Same first flag as race control: 22 of 29 (official Red flag and ours Red flag: 1; official Safety car and ours Red flag: 2; official Safety car and ours Safety car: 13; official Safety car and ours VSC: 3; official VSC and ours Safety car: 2; official VSC and ours VSC: 8).
- Red flags, per official escalation until race control's TRACK CLEAR: race control called 5, we called 3 of them, median 51 s earlier, and 13 that race control handled without a red flag (mostly our red for a crashed car still at its crash site after 2 minutes: the data cannot see barrier damage, debris or medical needs).

| official flag | escalations | earlier | later | missed | median lead (s) |
|---|---|---|---|---|---|
| VSC | 23 | 8 | 2 | 13 | 14.6 |
| Safety car | 25 | 18 | 0 | 6 | 21.3 |
| Red flag | 3 | 1 | 0 | 2 | 7.2 |

Missed official escalations:

| race | official time (s) | flag | sectors | kind | detail |
|---|---|---|---|---|---|
| 2023_Australian | 6454.2 | VSC | track-wide | car collapse seen | car 63 collapsed 31 s before the first official message; we showed double yellow but did not escalate |
| 2023_Australian | 9605.2 | RED | track-wide | no car collapse | car already slow or stopped before the window (re-flag of an earlier stoppage) |
| 2023_Mexico_City | 4084.9 | VSC | track-wide | no car collapse | field not racing (SC, VSC, restart or procession) |
| 2024_Qatar | 6468.8 | SC | track-wide | car collapse seen | car 44 collapsed 20 s before the first official message; we made no recommendation |
| 2024_São_Paulo | 6890.2 | SC | track-wide | no car collapse | car already slow or stopped before the window (re-flag of an earlier stoppage) |
| 2024_São_Paulo | 7159.2 | RED | 14 | no car collapse | car already slow or stopped before the window (re-flag of an earlier stoppage) |
| 2025_British | 3487.4 | VSC | 6 | no car collapse | lap 1 or 2: no clean reference laps yet |
| 2025_British | 4893.4 | SC | 12 | no car collapse | no car collapsed below 50% (debris, weather, or a car that went off and kept going) |
| 2025_Dutch | 5115.0 | SC | 4 5 | car collapse seen | car 44 collapsed 2 s before the first official message; we showed double yellow but did not escalate |
| 2025_Dutch | 5868.0 | VSC | track-wide | no car collapse | car already slow or stopped before the window (re-flag of an earlier stoppage) |
| 2025_Dutch | 7532.0 | SC | 4 5 | no car collapse | car already slow or stopped before the window (re-flag of an earlier stoppage) |
| 2025_Miami | 6098.2 | VSC | track-wide | car collapse seen | car 87 collapsed 48 s before the first official message; we showed double yellow but did not escalate |
| 2026_Australian | 6673.4 | VSC | track-wide | car collapse seen | car 63 collapsed 59 s before the first official message; we showed yellow but did not escalate |
| 2026_Belgian | 5476.9 | VSC | track-wide | no car collapse | car already slow or stopped before the window (re-flag of an earlier stoppage) |
| 2026_Belgian | 5721.9 | VSC | track-wide | no car collapse | car already slow or stopped before the window (re-flag of an earlier stoppage) |
| 2026_British | 5360.5 | VSC | track-wide | no car collapse | no car collapsed below 50% (debris, weather, or a car that went off and kept going) |
| 2026_Canadian | 6069.5 | VSC | 9 10 | no car collapse | no car collapsed below 50% (debris, weather, or a car that went off and kept going) |
| 2026_Canadian | 7293.5 | VSC | track-wide | car collapse seen | car 77 collapsed 23 s before the first official message; we showed double yellow but did not escalate |
| 2026_Canadian | 7812.5 | VSC | track-wide | no car collapse | car already slow or stopped before the window (re-flag of an earlier stoppage) |
| 2026_Dutch | 10659.2 | VSC | track-wide | car collapse seen | car 23 collapsed 99 s before the first official message; we made no recommendation |
| 2026_Italian | 3613.8 | SC | 14 15 16 | no car collapse | no car collapsed below 50% (debris, weather, or a car that went off and kept going) |

## Our escalations: 74

- Matched an official escalation: 29. During an official neutralisation (not extra): 21.
- Extra, race control never escalated: 24, 0.77 per race hour (20 where race control kept yellows only, 0 near an escalated incident but outside its match window, 4 with no official flag). By flag: VSC 6, Safety car 14, Red flag 4.
- Race control engine resets (tick jumps over 2 s): 0.

## Flag choice: VSC or SC

- Our first flag matched race control's in 22 of 29 matched escalations (official Red flag and ours Red flag: 1; official Safety car and ours Red flag: 2; official Safety car and ours Safety car: 13; official Safety car and ours VSC: 3; official VSC and ours Safety car: 2; official VSC and ours VSC: 8).
- Our 74 recommendations by flag: VSC 17, Safety car 33, Red flag 24.
- The lateral offset cannot choose between them. FastF1 positions of stopped cars sit on the racing line, even for retired cars parked in run-off: of 616,792 stopped-car rows (below 5 km/h, outside the pit lane), 99.97% are within 1 m of it and 118 are more than 8 m away (short episodes: 2026_Italian car 16, 2026_Miami car 6).
- An impact separates them better. On the 26 official escalations with an onset car, 8 of 13 SC or red followed an IMPACT or MULTI detection involving the car and 10 of 13 VSC did not: "SC after an impact, VSC otherwise" separates race control's VSCs from its SCs and reds correctly 18 times, "always SC" 13 times. This check does not tell SC from red; exact agreement is the first line. In-sample, and a small sample.

## Per race

| race | race hours | official escalations | earlier | later | missed | our escalations | extra |
|---|---|---|---|---|---|---|---|
| 2023_Australian | 1.46 | 6 | 4 | 0 | 2 | 6 | 0 |
| 2023_Mexico_City | 1.68 | 2 | 1 | 0 | 1 | 4 | 1 |
| 2023_Monaco | 1.82 | 0 | 0 | 0 | 0 | 6 | 6 |
| 2024_Canadian | 1.77 | 2 | 2 | 0 | 0 | 6 | 3 |
| 2024_Mexico_City | 1.68 | 1 | 1 | 0 | 0 | 2 | 0 |
| 2024_Qatar | 1.51 | 3 | 1 | 1 | 1 | 5 | 1 |
| 2024_São_Paulo | 1.69 | 4 | 2 | 0 | 2 | 3 | 1 |
| 2025_Australian | 1.68 | 3 | 3 | 0 | 0 | 9 | 2 |
| 2025_Azerbaijan | 1.56 | 1 | 1 | 0 | 0 | 3 | 1 |
| 2025_Belgian | 1.41 | 0 | 0 | 0 | 0 | 2 | 0 |
| 2025_British | 1.62 | 4 | 2 | 0 | 2 | 5 | 2 |
| 2025_Dutch | 1.64 | 4 | 1 | 0 | 3 | 1 | 0 |
| 2025_Miami | 1.48 | 3 | 1 | 1 | 1 | 2 | 0 |
| 2026_Australian | 1.39 | 3 | 2 | 0 | 1 | 2 | 0 |
| 2026_Belgian | 1.41 | 3 | 1 | 0 | 2 | 2 | 0 |
| 2026_British | 1.46 | 4 | 2 | 0 | 1 | 4 | 1 |
| 2026_Canadian | 1.47 | 3 | 0 | 0 | 3 | 1 | 1 |
| 2026_Dutch | 1.61 | 2 | 1 | 0 | 1 | 3 | 2 |
| 2026_Italian | 1.32 | 2 | 1 | 0 | 1 | 5 | 2 |
| 2026_Miami | 1.56 | 1 | 1 | 0 | 0 | 3 | 1 |

Chart: escalation.png. Every official escalation: escalation_official.csv. Every recommendation of ours: escalation_ours.csv.
