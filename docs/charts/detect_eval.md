# Detection eval (training races)

Replay of historical FastF1 data. Matching and metric definitions: src/eval/incidents.py and src/eval/run.py.

**Settings.** Detectors: production settings (data/models/detector_config.json, chosen by leave-one-race-out tuning at 3 false alarms per race hour; see tune_results.md). Those thresholds were selected on these same 20 races, so for strictly out-of-sample detection numbers use tune_results.md. ANOMALY is an advisory, not an alert: its detections are scored as their own row (anomaly_advisory) and are not counted in the detectors' recall or false alarms. Model trained on the other training races (leave-one-race-out), firing above the 99.995th percentile of scores on normal racing and only for cars slower than the field or their own last lap. Baseline: the untuned naive speed threshold (any car below 50 km/h outside the pit lane for 1 s); for a like-for-like comparison at the same false-alarm rate see tune_results.md.

| system           |   races |   incidents |   matched |   held |   recall |   recall_window |   median_lead_s |   earlier_than_official |   false_alarms |   fa_per_hour |
|:-----------------|--------:|------------:|----------:|-------:|---------:|----------------:|----------------:|------------------------:|---------------:|--------------:|
| anomaly_advisory |      20 |         146 |         3 |      0 |    0.021 |           0.021 |           32.31 |                   1     |             87 |         2.786 |
| baseline         |      20 |         146 |       100 |      2 |    0.685 |           0.671 |            2.25 |                   0.745 |           1925 |        61.638 |
| detectors        |      20 |         146 |        83 |      4 |    0.568 |           0.541 |            1.67 |                   0.772 |             94 |         3.01  |

## Detector alerts and ANOMALY advisories by type

| type    |   alerts |   false_alarms |
|:--------|---------:|---------------:|
| ANOMALY |       91 |             87 |
| IMPACT  |       80 |             37 |
| MULTI   |       17 |              8 |
| SPIN    |        5 |              5 |
| STOPPED |       90 |             44 |

## Per race

| race             | system           |   incidents |   matched |   held |   recall |   recall_window |   median_lead_s |   alerts |   false_alarms |   fa_per_hour |   hours |
|:-----------------|:-----------------|------------:|----------:|-------:|---------:|----------------:|----------------:|---------:|---------------:|--------------:|--------:|
| 2023_Australian  | anomaly_advisory |           6 |         0 |      0 |    0     |           0     |         nan     |        2 |              2 |         1.366 |   1.464 |
| 2023_Australian  | detectors        |           6 |         4 |      0 |    0.667 |           0.667 |           2.43  |       10 |              0 |         0     |   1.464 |
| 2023_Australian  | baseline         |           6 |         5 |      0 |    0.833 |           0.833 |           1.93  |      113 |             91 |        62.16  |   1.464 |
| 2023_Mexico_City | anomaly_advisory |           6 |         0 |      0 |    0     |           0     |         nan     |        5 |              4 |         2.382 |   1.679 |
| 2023_Mexico_City | detectors        |           6 |         4 |      0 |    0.667 |           0.667 |           2.045 |       11 |              7 |         4.169 |   1.679 |
| 2023_Mexico_City | baseline         |           6 |         4 |      0 |    0.667 |           0.667 |           0.67  |       22 |             18 |        10.72  |   1.679 |
| 2023_Monaco      | anomaly_advisory |          13 |         0 |      0 |    0     |           0     |         nan     |       33 |             33 |        18.154 |   1.818 |
| 2023_Monaco      | detectors        |          13 |         9 |      1 |    0.692 |           0.615 |          -0.605 |       10 |              1 |         0.55  |   1.818 |
| 2023_Monaco      | baseline         |          13 |        11 |      0 |    0.846 |           0.846 |          49.02  |     1089 |           1008 |       554.523 |   1.818 |
| 2024_Canadian    | anomaly_advisory |           9 |         0 |      0 |    0     |           0     |         nan     |        0 |              0 |         0     |   1.766 |
| 2024_Canadian    | detectors        |           9 |         5 |      0 |    0.556 |           0.556 |           2.5   |       10 |              3 |         1.699 |   1.766 |
| 2024_Canadian    | baseline         |           9 |         6 |      0 |    0.667 |           0.667 |           1.375 |      180 |            167 |        94.554 |   1.766 |
| 2024_Mexico_City | anomaly_advisory |           1 |         0 |      0 |    0     |           0     |         nan     |        3 |              3 |         1.781 |   1.685 |
| 2024_Mexico_City | detectors        |           1 |         1 |      0 |    1     |           1     |           0.85  |        3 |              1 |         0.594 |   1.685 |
| 2024_Mexico_City | baseline         |           1 |         1 |      0 |    1     |           1     |           1.1   |       90 |             71 |        42.143 |   1.685 |
| 2024_Qatar       | anomaly_advisory |           7 |         1 |      0 |    0.143 |           0.143 |          32.31  |        7 |              6 |         3.976 |   1.509 |
| 2024_Qatar       | detectors        |           7 |         3 |      0 |    0.429 |           0.429 |           1.56  |       12 |              6 |         3.976 |   1.509 |
| 2024_Qatar       | baseline         |           7 |         3 |      0 |    0.429 |           0.429 |          -0.69  |       62 |             57 |        37.767 |   1.509 |
| 2024_São_Paulo   | anomaly_advisory |          13 |         0 |      0 |    0     |           0     |         nan     |        0 |              0 |         0     |   1.689 |
| 2024_São_Paulo   | detectors        |          13 |         7 |      0 |    0.538 |           0.538 |          -0.25  |        8 |              1 |         0.592 |   1.689 |
| 2024_São_Paulo   | baseline         |          13 |        11 |      0 |    0.846 |           0.846 |           0     |       50 |             32 |        18.942 |   1.689 |
| 2025_Australian  | anomaly_advisory |           7 |         0 |      0 |    0     |           0     |         nan     |        0 |              0 |         0     |   1.68  |
| 2025_Australian  | detectors        |           7 |         6 |      0 |    0.857 |           0.857 |          11.005 |       15 |              7 |         4.166 |   1.68  |
| 2025_Australian  | baseline         |           7 |         6 |      0 |    0.857 |           0.857 |          10.63  |      107 |             91 |        54.162 |   1.68  |
| 2025_Azerbaijan  | anomaly_advisory |           3 |         0 |      0 |    0     |           0     |         nan     |        6 |              6 |         3.847 |   1.56  |
| 2025_Azerbaijan  | detectors        |           3 |         2 |      0 |    0.667 |           0.667 |           1.045 |        3 |              1 |         0.641 |   1.56  |
| 2025_Azerbaijan  | baseline         |           3 |         2 |      0 |    0.667 |           0.667 |           0.67  |       66 |             64 |        41.033 |   1.56  |
| 2025_Belgian     | anomaly_advisory |           0 |         0 |      0 |  nan     |         nan     |         nan     |       13 |             13 |         9.24  |   1.407 |
| 2025_Belgian     | detectors        |           0 |         0 |      0 |  nan     |         nan     |         nan     |       10 |             10 |         7.108 |   1.407 |
| 2025_Belgian     | baseline         |           0 |         0 |      0 |  nan     |         nan     |         nan     |       33 |             33 |        23.456 |   1.407 |
| 2025_British     | anomaly_advisory |          12 |         0 |      0 |    0     |           0     |         nan     |        2 |              2 |         1.232 |   1.624 |
| 2025_British     | detectors        |          12 |        10 |      0 |    0.833 |           0.833 |           0.64  |       12 |              1 |         0.616 |   1.624 |
| 2025_British     | baseline         |          12 |         8 |      0 |    0.667 |           0.667 |           0.64  |       58 |             50 |        30.796 |   1.624 |
| 2025_Dutch       | anomaly_advisory |           6 |         0 |      0 |    0     |           0     |         nan     |        0 |              0 |         0     |   1.644 |
| 2025_Dutch       | detectors        |           6 |         2 |      0 |    0.333 |           0.333 |           0.615 |        2 |              0 |         0     |   1.644 |
| 2025_Dutch       | baseline         |           6 |         3 |      0 |    0.5   |           0.5   |           0.74  |       68 |             59 |        35.887 |   1.644 |
| 2025_Miami       | anomaly_advisory |           7 |         0 |      0 |    0     |           0     |         nan     |        0 |              0 |         0     |   1.483 |
| 2025_Miami       | detectors        |           7 |         6 |      1 |    0.857 |           0.714 |          24.41  |        6 |              3 |         2.022 |   1.483 |
| 2025_Miami       | baseline         |           7 |         6 |      0 |    0.857 |           0.857 |          26.785 |       50 |             10 |         6.741 |   1.483 |
| 2026_Australian  | anomaly_advisory |           5 |         0 |      0 |    0     |           0     |         nan     |        2 |              2 |         1.441 |   1.388 |
| 2026_Australian  | detectors        |           5 |         4 |      0 |    0.8   |           0.8   |          22.13  |        4 |              2 |         1.441 |   1.388 |
| 2026_Australian  | baseline         |           5 |         3 |      0 |    0.6   |           0.6   |          13.38  |        5 |              4 |         2.882 |   1.388 |
| 2026_Belgian     | anomaly_advisory |          12 |         1 |      0 |    0.083 |           0.083 |          32.4   |       14 |             13 |         9.188 |   1.415 |
| 2026_Belgian     | detectors        |          12 |         4 |      2 |    0.333 |           0.167 |           1.775 |        3 |              1 |         0.707 |   1.415 |
| 2026_Belgian     | baseline         |          12 |         5 |      2 |    0.417 |           0.25  |           1.4   |       10 |              6 |         4.24  |   1.415 |
| 2026_British     | anomaly_advisory |           9 |         1 |      0 |    0.111 |           0.111 |           6.01  |        2 |              1 |         0.687 |   1.456 |
| 2026_British     | detectors        |           9 |         4 |      0 |    0.444 |           0.444 |           2.385 |        7 |              1 |         0.687 |   1.456 |
| 2026_British     | baseline         |           9 |         5 |      0 |    0.556 |           0.556 |           0.51  |        8 |              2 |         1.374 |   1.456 |
| 2026_Canadian    | anomaly_advisory |           8 |         0 |      0 |    0     |           0     |         nan     |        0 |              0 |         0     |   1.474 |
| 2026_Canadian    | detectors        |           8 |         3 |      0 |    0.375 |           0.375 |          -1.47  |        7 |              4 |         2.714 |   1.474 |
| 2026_Canadian    | baseline         |           8 |         6 |      0 |    0.75  |           0.75  |          -0.345 |       29 |             17 |        11.536 |   1.474 |
| 2026_Dutch       | anomaly_advisory |           6 |         0 |      0 |    0     |           0     |         nan     |        0 |              0 |         0     |   1.608 |
| 2026_Dutch       | detectors        |           6 |         2 |      0 |    0.333 |           0.333 |           3.22  |       16 |             12 |         7.461 |   1.608 |
| 2026_Dutch       | baseline         |           6 |         3 |      0 |    0.5   |           0.5   |           7.47  |       44 |             38 |        23.627 |   1.608 |
| 2026_Italian     | anomaly_advisory |           3 |         0 |      0 |    0     |           0     |         nan     |        0 |              0 |         0     |   1.324 |
| 2026_Italian     | detectors        |           3 |         3 |      0 |    1     |           1     |           8.57  |       33 |             30 |        22.662 |   1.324 |
| 2026_Italian     | baseline         |           3 |         2 |      0 |    0.667 |           0.667 |           6.32  |       55 |             48 |        36.259 |   1.324 |
| 2026_Miami       | anomaly_advisory |          13 |         0 |      0 |    0     |           0     |         nan     |        2 |              2 |         1.284 |   1.558 |
| 2026_Miami       | detectors        |          13 |         4 |      0 |    0.308 |           0.308 |          -0.12  |       10 |              3 |         1.926 |   1.558 |
| 2026_Miami       | baseline         |          13 |        10 |      0 |    0.769 |           0.769 |          48.885 |      109 |             59 |        37.871 |   1.558 |
