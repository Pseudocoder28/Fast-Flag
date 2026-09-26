# Detector tuning, leave-one-race-out (training races only)

Replay of historical FastF1 data. For each race the setting is chosen on the other training races and scored on that race; the naive speed-threshold baseline gets the same procedure and budget.

| system    |   budget_fa_h |   incidents |   matched |   recall |   recall_window |   median_lead_s |   earlier_than_official |   fa_h |
|:----------|--------------:|------------:|----------:|---------:|----------------:|----------------:|------------------------:|-------:|
| detectors |             2 |         146 |        83 |    0.568 |           0.541 |           1.67  |                   0.772 |  3.01  |
| baseline  |             2 |         146 |        73 |    0.5   |           0.473 |           0.75  |                   0.609 |  2.722 |
| detectors |             3 |         146 |        83 |    0.568 |           0.541 |           1.67  |                   0.772 |  3.01  |
| baseline  |             3 |         146 |        74 |    0.507 |           0.479 |           0.935 |                   0.614 |  2.978 |
| detectors |             5 |         146 |        83 |    0.568 |           0.541 |           1.67  |                   0.785 |  4.483 |
| baseline  |             5 |         146 |        78 |    0.534 |           0.507 |           1.175 |                   0.649 |  3.458 |

Settings chosen on all training races:

- budget 2.0/h: detectors {'stop_ratio': 0.3, 'impact_drop_kmh': 80.0, 'slowdown_lap_ratio': None}, baseline 20 km/h
- budget 3.0/h: detectors {'stop_ratio': 0.3, 'impact_drop_kmh': 80.0, 'slowdown_lap_ratio': None}, baseline 20 km/h
- budget 5.0/h: detectors {'stop_ratio': 0.3, 'impact_drop_kmh': 80.0, 'slowdown_lap_ratio': 0.5, 'slowdown_rel_ratio': 0.4, 'slowdown_sustain_s': 2.0}, baseline 30 km/h
