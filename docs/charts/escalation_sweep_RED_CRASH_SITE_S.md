# Escalation scorecard vs RED_CRASH_SITE_S (training races)

Replay of historical FastF1 data, 20 training races, 31.2 race hours. Each row reruns python -m src.eval.escalation with src.racecontrol.engine.RED_CRASH_SITE_S set to that value (in the worker processes only). In-sample: choose before the freeze, the holdout run tests the choice.

| RED_CRASH_SITE_S | official | matched | earlier | later | missed | median_lead_s | lead_p25_s | lead_p75_s | same_first_flag | our_vsc | our_sc | our_red | red_both | red_race_control_only | red_ours_only | red_median_lead_s | extra | extra_per_hour | extra_yellows_only | extra_no_official_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 51 | 29 | 27 | 2 | 21 | 20 | 9.1 | 33.9 | 22 | 17 | 33 | 24 | 3 | 2 | 14 | 111.4 | 24 | 0.77 | 20 | 4 |
| 120 | 51 | 29 | 27 | 2 | 21 | 20 | 9.1 | 33.9 | 22 | 17 | 33 | 23 | 3 | 2 | 13 | 51.4 | 24 | 0.77 | 20 | 4 |
| 180 | 51 | 29 | 27 | 2 | 21 | 20 | 9.1 | 33.9 | 22 | 17 | 33 | 23 | 3 | 2 | 13 | -8.3 | 24 | 0.77 | 20 | 4 |
| 240 | 51 | 29 | 27 | 2 | 21 | 20 | 9.1 | 33.9 | 22 | 17 | 33 | 23 | 3 | 2 | 13 | -68.3 | 24 | 0.77 | 20 | 4 |
| 100000 | 51 | 29 | 27 | 2 | 21 | 20 | 9.1 | 33.9 | 22 | 17 | 33 | 8 | 1 | 4 | 2 | 7.2 | 24 | 0.77 | 20 | 4 |
