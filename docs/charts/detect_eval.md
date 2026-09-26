# Detection eval (training races)

Replay of historical FastF1 data. Matching and metric definitions: src/eval/incidents.py and src/eval/run.py.

**Settings.** Detectors: production settings (data/models/detector_config.json, chosen by leave-one-race-out tuning at 3 false alarms per race hour; see tune_results.md). Those thresholds were selected on these same 20 races, so for strictly out-of-sample detection numbers use tune_results.md. ANOMALY: model trained on the other training races (leave-one-race-out), alerting above the 99.995th percentile of scores on normal racing and only for cars slower than the field or their own last lap. Baseline: the untuned naive speed threshold (any car below 50 km/h outside the pit lane for 1 s); for a like-for-like comparison at the same false-alarm rate see tune_results.md.

| system    |   races |   incidents |   matched |   held |   recall |   recall_window |   median_lead_s |   earlier_than_official |   false_alarms |   fa_per_hour |
|:----------|--------:|------------:|----------:|-------:|---------:|----------------:|----------------:|------------------------:|---------------:|--------------:|
| baseline  |      20 |         146 |       100 |      2 |    0.685 |           0.671 |            2.25 |                   0.745 |           1925 |        61.638 |
| detectors |      20 |         146 |        85 |      4 |    0.582 |           0.555 |            1.81 |                   0.778 |            179 |         5.732 |

## Detector alerts by type

| type    |   alerts |   false_alarms |
|:--------|---------:|---------------:|
| ANOMALY |       89 |             86 |
| IMPACT  |       80 |             37 |
| MULTI   |       17 |              8 |
| SPIN    |        5 |              5 |
| STOPPED |       87 |             43 |

## Per race

| race             | system    |   incidents |   matched |   held |   recall |   recall_window |   median_lead_s |   alerts |   false_alarms |   fa_per_hour |   hours |
|:-----------------|:----------|------------:|----------:|-------:|---------:|----------------:|----------------:|---------:|---------------:|--------------:|--------:|
| 2023_Australian  | detectors |           6 |         4 |      0 |    0.667 |           0.667 |           2.43  |       12 |              2 |         1.366 |   1.464 |
| 2023_Australian  | baseline  |           6 |         5 |      0 |    0.833 |           0.833 |           1.93  |      113 |             91 |        62.16  |   1.464 |
| 2023_Mexico_City | detectors |           6 |         4 |      0 |    0.667 |           0.667 |           2.045 |       15 |             11 |         6.551 |   1.679 |
| 2023_Mexico_City | baseline  |           6 |         4 |      0 |    0.667 |           0.667 |           0.67  |       22 |             18 |        10.72  |   1.679 |
| 2023_Monaco      | detectors |          13 |         9 |      1 |    0.692 |           0.615 |          -0.605 |       42 |             34 |        18.704 |   1.818 |
| 2023_Monaco      | baseline  |          13 |        11 |      0 |    0.846 |           0.846 |          49.02  |     1089 |           1008 |       554.523 |   1.818 |
| 2024_Canadian    | detectors |           9 |         5 |      0 |    0.556 |           0.556 |           2.5   |       10 |              3 |         1.699 |   1.766 |
| 2024_Canadian    | baseline  |           9 |         6 |      0 |    0.667 |           0.667 |           1.375 |      180 |            167 |        94.554 |   1.766 |
| 2024_Mexico_City | detectors |           1 |         1 |      0 |    1     |           1     |           0.85  |        6 |              4 |         2.374 |   1.685 |
| 2024_Mexico_City | baseline  |           1 |         1 |      0 |    1     |           1     |           1.1   |       90 |             71 |        42.143 |   1.685 |
| 2024_Qatar       | detectors |           7 |         4 |      0 |    0.571 |           0.571 |           1.685 |       18 |             11 |         7.288 |   1.509 |
| 2024_Qatar       | baseline  |           7 |         3 |      0 |    0.429 |           0.429 |          -0.69  |       62 |             57 |        37.767 |   1.509 |
| 2024_São_Paulo   | detectors |          13 |         7 |      0 |    0.538 |           0.538 |          -0.25  |        8 |              1 |         0.592 |   1.689 |
| 2024_São_Paulo   | baseline  |          13 |        11 |      0 |    0.846 |           0.846 |           0     |       50 |             32 |        18.942 |   1.689 |
| 2025_Australian  | detectors |           7 |         6 |      0 |    0.857 |           0.857 |          11.005 |       15 |              7 |         4.166 |   1.68  |
| 2025_Australian  | baseline  |           7 |         6 |      0 |    0.857 |           0.857 |          10.63  |      107 |             91 |        54.162 |   1.68  |
| 2025_Azerbaijan  | detectors |           3 |         2 |      0 |    0.667 |           0.667 |           1.045 |        9 |              7 |         4.488 |   1.56  |
| 2025_Azerbaijan  | baseline  |           3 |         2 |      0 |    0.667 |           0.667 |           0.67  |       66 |             64 |        41.033 |   1.56  |
| 2025_Belgian     | detectors |           0 |         0 |      0 |  nan     |         nan     |         nan     |       23 |             23 |        16.348 |   1.407 |
| 2025_Belgian     | baseline  |           0 |         0 |      0 |  nan     |         nan     |         nan     |       33 |             33 |        23.456 |   1.407 |
| 2025_British     | detectors |          12 |        10 |      0 |    0.833 |           0.833 |           0.64  |       14 |              3 |         1.848 |   1.624 |
| 2025_British     | baseline  |          12 |         8 |      0 |    0.667 |           0.667 |           0.64  |       58 |             50 |        30.796 |   1.624 |
| 2025_Dutch       | detectors |           6 |         2 |      0 |    0.333 |           0.333 |           0.615 |        2 |              0 |         0     |   1.644 |
| 2025_Dutch       | baseline  |           6 |         3 |      0 |    0.5   |           0.5   |           0.74  |       68 |             59 |        35.887 |   1.644 |
| 2025_Miami       | detectors |           7 |         6 |      1 |    0.857 |           0.714 |          24.41  |        6 |              3 |         2.022 |   1.483 |
| 2025_Miami       | baseline  |           7 |         6 |      0 |    0.857 |           0.857 |          26.785 |       50 |             10 |         6.741 |   1.483 |
| 2026_Australian  | detectors |           5 |         4 |      0 |    0.8   |           0.8   |          22.13  |        6 |              4 |         2.882 |   1.388 |
| 2026_Australian  | baseline  |           5 |         3 |      0 |    0.6   |           0.6   |          13.38  |        5 |              4 |         2.882 |   1.388 |
| 2026_Belgian     | detectors |          12 |         5 |      2 |    0.417 |           0.25  |           3.15  |       17 |             14 |         9.894 |   1.415 |
| 2026_Belgian     | baseline  |          12 |         5 |      2 |    0.417 |           0.25  |           1.4   |       10 |              6 |         4.24  |   1.415 |
| 2026_British     | detectors |           9 |         4 |      0 |    0.444 |           0.444 |           2.885 |        8 |              2 |         1.374 |   1.456 |
| 2026_British     | baseline  |           9 |         5 |      0 |    0.556 |           0.556 |           0.51  |        8 |              2 |         1.374 |   1.456 |
| 2026_Canadian    | detectors |           8 |         3 |      0 |    0.375 |           0.375 |          -1.47  |        7 |              4 |         2.714 |   1.474 |
| 2026_Canadian    | baseline  |           8 |         6 |      0 |    0.75  |           0.75  |          -0.345 |       29 |             17 |        11.536 |   1.474 |
| 2026_Dutch       | detectors |           6 |         2 |      0 |    0.333 |           0.333 |           3.22  |       16 |             12 |         7.461 |   1.608 |
| 2026_Dutch       | baseline  |           6 |         3 |      0 |    0.5   |           0.5   |           7.47  |       44 |             38 |        23.627 |   1.608 |
| 2026_Italian     | detectors |           3 |         3 |      0 |    1     |           1     |           8.57  |       33 |             30 |        22.662 |   1.324 |
| 2026_Italian     | baseline  |           3 |         2 |      0 |    0.667 |           0.667 |           6.32  |       55 |             48 |        36.259 |   1.324 |
| 2026_Miami       | detectors |          13 |         4 |      0 |    0.308 |           0.308 |          -0.12  |       11 |              4 |         2.568 |   1.558 |
| 2026_Miami       | baseline  |          13 |        10 |      0 |    0.769 |           0.769 |          48.885 |      109 |             59 |        37.871 |   1.558 |
