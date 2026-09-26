# Risk model, leave-one-race-out (training races only)

Replay of historical FastF1 data. For each race, LightGBM is trained on the other training races and scores every racing tick of that race. PR-AUC is pooled over all held-out ticks; pr_auc_precursor leaves out each incident's last 3 s before detection. Baselines: ANOMALY score (IsolationForest, also leave-one-race-out) and the naive speed threshold (lower speed = higher risk).

|   horizon_s | model           |   pr_auc |   pr_auc_precursor |   pr_auc_mean_per_race |   base_rate |
|------------:|:----------------|---------:|-------------------:|-----------------------:|------------:|
|          10 | lightgbm        |   0.1105 |             0.0165 |                 0.1504 |      0.0004 |
|          10 | anomaly_score   |   0.002  |             0.0005 |                 0.0067 |      0.0004 |
|          10 | speed_threshold |   0.002  |             0.0005 |                 0.0581 |      0.0004 |
|          20 | lightgbm        |   0.0448 |             0.0103 |                 0.0696 |      0.0008 |
|          20 | anomaly_score   |   0.002  |             0.0009 |                 0.0062 |      0.0008 |
|          20 | speed_threshold |   0.0019 |             0.0009 |                 0.0316 |      0.0008 |
|          30 | lightgbm        |   0.0271 |             0.0077 |                 0.0524 |      0.0012 |
|          30 | anomaly_score   |   0.0023 |             0.0014 |                 0.0057 |      0.0012 |
|          30 | speed_threshold |   0.0021 |             0.0013 |                 0.0227 |      0.0012 |

Early warning for 79 car incidents (risk_30s, flagged = crossed the threshold at least 3 s before detection; false episodes = a car above the threshold with no incident of its own in the next 30 s):

|   neg_tick_rate |   threshold_p30 |   flagged_3s_before |   median_s_before_when_flagged |   false_episodes_per_hour |
|----------------:|----------------:|--------------------:|-------------------------------:|--------------------------:|
|           0.005 |          0.0106 |              0.3924 |                           8    |                   70.205  |
|           0.002 |          0.0193 |              0.3418 |                           7.25 |                   53.753  |
|           0.001 |          0.0271 |              0.2911 |                           6.5  |                   35.1025 |

Some incidents have no precursor in the data: this is risk forecasting, not a crystal ball.
