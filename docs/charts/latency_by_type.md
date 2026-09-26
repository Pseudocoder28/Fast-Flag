# Latency from crash onset: race control vs our system (training races)

Replay of historical FastF1 data, 20 training races, no holdout race. Latencies are measured from the crash onset.

**Onset rule** (fixed, independent of the detectors): speed below 50% of the car's own median speed at that point (10 m bins) on its previous 3 clean laps, for at least 1 s, right after a tick at 50% or more, while green or under a local yellow, on track, not in the pit lane, and while the field is racing (median car on track at 80% or more of its own reference).

The onset car of an official incident is the car whose collapse race control reacted to: among onsets from 120 s before the incident's first official message to 5 s after it, in a matching marshal sector, the first car of the last chain of onsets less than 10 s apart that starts up to that message. Changed on 26 Sept 2026: the onset car used to be the earliest collapse in the window. A review found incidents anchored on an earlier, unrelated slowdown up to two minutes before race control's first message (7 of 63: 3 a different car, 4 the same car slowing twice), so it is now the first car of the last chain of collapses (less than 10 s apart) up to that message. The change can only shorten race control's measured delay. Race control latency = first official message of that flag type - onset. Our latency = our first alert (any detection from 10 s before the onset to 180 s after, involving an onset car or in a matching sector) - onset. Detections: production settings, ANOMALY model trained on the other races.

**What this shows and what it does not.** This compares latency on crashes where a car clearly collapsed, which is also what our detectors are best at, so the matched share here is not a recall figure (recall per incident is in detect_eval.md and tune_results.md). An incident contributes one event per flag type it reached, so SC and red flag events are usually escalations of an incident that also had a yellow. Our detection thresholds were tuned on these training races; the ANOMALY model was not trained on the race it scores.

## By official flag type

| flag          |   events |   matched |   rc_median_s |   rc_p25_s |   rc_p75_s |   ours_median_s |   ours_p25_s |   ours_p75_s |   earlier_share |
|:--------------|---------:|----------:|--------------:|-----------:|-----------:|----------------:|-------------:|-------------:|----------------:|
| YELLOW        |       45 |        43 |          2.77 |       1.01 |       5.32 |            1.5  |         0.5  |         2.88 |            0.74 |
| DOUBLE_YELLOW |       36 |        33 |          6.44 |       2.07 |      37.99 |            1.5  |         0.5  |         2.75 |            0.85 |
| VSC           |       13 |        12 |         35.01 |      23.03 |      48.16 |            2.62 |         1.81 |         5    |            1    |
| SC            |       14 |        13 |         22.52 |      18.93 |      29.16 |            1    |         0.5  |         2.25 |            1    |
| RED           |        2 |         2 |        173.05 |     172.99 |     173.11 |            0.5  |         0.5  |         0.5  |            1    |

## By our alert type

| our_alert_type   |   events |   matched |   rc_median_s |   rc_p25_s |   rc_p75_s |   ours_median_s |   ours_p25_s |   ours_p75_s |   earlier_share |
|:-----------------|---------:|----------:|--------------:|-----------:|-----------:|----------------:|-------------:|-------------:|----------------:|
| IMPACT           |       51 |        51 |          5    |       2.41 |      20.76 |            0.5  |         0.5  |         0.75 |            1    |
| STOPPED          |       47 |        47 |          9.52 |       1.64 |      30.12 |            2.75 |         2.25 |         4.75 |            0.68 |
| ANOMALY          |        5 |         5 |         43.01 |      35.01 |      46.02 |            2    |         2    |         2    |            0.8  |
| none (missed)    |        7 |         0 |         19.81 |      -0.5  |      62.52 |          nan    |       nan    |       nan    |          nan    |

## Events without an identifiable onset car (115 events)

| reason                                                                              |   DOUBLE_YELLOW |   RED |   SC |   VSC |   YELLOW |
|:------------------------------------------------------------------------------------|----------------:|------:|-----:|------:|---------:|
| car already slow or stopped before the window (re-flag of an earlier stoppage)      |              20 |     3 |    4 |     4 |       11 |
| field not racing (SC, VSC, restart or procession)                                   |               8 |     0 |    0 |     1 |        3 |
| lap 1 or 2: no clean reference laps yet                                             |              15 |     0 |    6 |     2 |       15 |
| no car collapsed below 50% (debris, weather, or a car that went off and kept going) |               8 |     0 |    2 |     3 |       10 |

Every one is listed in latency_by_type_no_onset.csv.

**Check:** in 2 of 58 incidents the onset car entered the pit lane within 20 s (a damaged car heading in, or a car braking for the pit entry before the pit-lane geometry covers it). They are kept and marked in latency_by_type.csv (onset_car_pitted_20s).

![latency by flag type](latency_by_type.png)
