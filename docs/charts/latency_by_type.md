# Latency from crash onset: race control vs our system (training races)

Replay of historical FastF1 data, 20 training races, no holdout race. Latencies are measured from the crash onset.

**Onset rule** (fixed, independent of the detectors): speed below 50% of the car's own median speed at that point (10 m bins) on its previous 3 clean laps, for at least 1 s, right after a tick at 50% or more, while green or under a local yellow, on track, not in the pit lane, and while the field is racing (median car on track at 80% or more of its own reference).

The onset car of an official incident is the car with the earliest onset in the 120 s before the incident's first official message, in a matching marshal sector. Race control latency = first official message of that flag type - onset. Our latency = our first alert (any detection from 10 s before the onset to 180 s after, involving an onset car or in a matching sector) - onset. Detections: production settings, ANOMALY model trained on the other races.

**What this shows and what it does not.** This compares latency on crashes where a car clearly collapsed, which is also what our detectors are best at, so the matched share here is not a recall figure (recall per incident is in detect_eval.md and tune_results.md). An incident contributes one event per flag type it reached, so SC and red flag events are usually escalations of an incident that also had a yellow. Our detection thresholds were tuned on these training races; the ANOMALY model was not trained on the race it scores.

## By official flag type

| flag          |   events |   matched |   rc_median_s |   rc_p25_s |   rc_p75_s |   ours_median_s |   ours_p25_s |   ours_p75_s |   earlier_share |
|:--------------|---------:|----------:|--------------:|-----------:|-----------:|----------------:|-------------:|-------------:|----------------:|
| YELLOW        |       45 |        43 |          2.93 |       1.01 |       8.77 |            1.5  |         0.5  |         2.75 |            0.79 |
| DOUBLE_YELLOW |       36 |        33 |         12.76 |       2.43 |      60.13 |            1.5  |         0.5  |         2.75 |            0.85 |
| VSC           |        5 |         5 |         48.16 |      31.43 |     108.41 |            2.5  |         1.5  |         9    |            1    |
| SC            |       22 |        21 |         24.65 |      21.16 |      56.66 |            1.25 |         0.75 |         2.75 |            1    |
| RED           |        2 |         2 |        173.05 |     172.99 |     173.11 |            0.5  |         0.5  |         0.5  |            1    |

## By our alert type

| our_alert_type   |   events |   matched |   rc_median_s |   rc_p25_s |   rc_p75_s |   ours_median_s |   ours_p25_s |   ours_p75_s |   earlier_share |
|:-----------------|---------:|----------:|--------------:|-----------:|-----------:|----------------:|-------------:|-------------:|----------------:|
| IMPACT           |       52 |        52 |          8.38 |       2.82 |      49.07 |            0.5  |         0.5  |         0.75 |             1   |
| STOPPED          |       47 |        47 |         14.22 |       2.01 |      41.77 |            2.75 |         2.12 |         4.75 |             0.7 |
| ANOMALY          |        5 |         5 |         43.01 |      35.01 |      60.56 |            2    |         2    |         2    |             1   |
| none (missed)    |        6 |         0 |         30.64 |      -0.62 |      63.02 |          nan    |       nan    |       nan    |           nan   |

## Events without an identifiable onset car (115 events)

| reason                                                                              |   DOUBLE_YELLOW |   RED |   SC |   VSC |   YELLOW |
|:------------------------------------------------------------------------------------|----------------:|------:|-----:|------:|---------:|
| car already slow or stopped before the window (re-flag of an earlier stoppage)      |              20 |     3 |    7 |     1 |       11 |
| field not racing (SC, VSC, restart or procession)                                   |               8 |     0 |    0 |     1 |        3 |
| lap 1 or 2: no clean reference laps yet                                             |              15 |     0 |    6 |     2 |       15 |
| no car collapsed below 50% (debris, weather, or a car that went off and kept going) |               8 |     0 |    4 |     1 |       10 |

Every one is listed in latency_by_type_no_onset.csv.

![latency by flag type](latency_by_type.png)
