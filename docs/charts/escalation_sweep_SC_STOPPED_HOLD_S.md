# Escalation scorecard vs SC_STOPPED_HOLD_S (training races)

Replay of historical FastF1 data, 20 training races, 31.2 race hours. Each row reruns python -m src.eval.escalation with src.racecontrol.engine.SC_STOPPED_HOLD_S set to that value (in the worker processes only). In-sample. The A7 holdout run tested the engine frozen before the 26 Sept race control changes; those changes were checked on replays of the holdout, so they have no out-of-sample test.

| SC_STOPPED_HOLD_S | official | matched | earlier | later | missed | median_lead_s | lead_p25_s | lead_p75_s | same_first_flag | our_vsc | our_sc | our_red | red_both | red_race_control_only | red_ours_only | red_median_lead_s | our_reds_uncalled | extra | extra_per_hour | extra_yellows_only | extra_no_official_flag | extra_outside_window | extra_red_uncalled | track_clears_under_neutral |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 51 | 29 | 27 | 2 | 20 | 20 | 9.1 | 33.9 | 22 | 17 | 33 | 23 | 3 | 2 | 13 | 51.4 | 18 | 36 | 1.15 | 20 | 4 | 0 | 12 | 0 |
| 5 | 51 | 29 | 27 | 2 | 20 | 18.1 | 7.6 | 31.9 | 22 | 17 | 31 | 23 | 3 | 2 | 13 | 51.4 | 18 | 34 | 1.09 | 18 | 4 | 0 | 12 | 0 |
| 8 | 51 | 29 | 26 | 3 | 20 | 15.1 | 7.1 | 28.9 | 22 | 17 | 29 | 23 | 3 | 2 | 13 | 51.4 | 18 | 32 | 1.02 | 16 | 4 | 0 | 12 | 0 |
| 10 | 51 | 29 | 26 | 3 | 20 | 13.1 | 7 | 26.9 | 22 | 17 | 27 | 23 | 3 | 2 | 13 | 51.4 | 18 | 31 | 0.99 | 15 | 4 | 0 | 12 | 0 |
