# Escalation scorecard vs SC_STOPPED_HOLD_S (training races)

Replay of historical FastF1 data, 20 training races, 31.2 race hours. Each row reruns python -m src.eval.escalation with src.racecontrol.engine.SC_STOPPED_HOLD_S set to that value (in the worker processes only). In-sample. The A7 holdout run tested the engine frozen before the race control changes of 26 and 27 Sept; those changes were checked on replays of the holdout, so they have no out-of-sample test.

| SC_STOPPED_HOLD_S | official | matched | earlier | later | missed | median_lead_s | lead_p25_s | lead_p75_s | same_first_flag | our_vsc | our_sc | our_red | red_both | red_race_control_only | red_median_lead_s | our_reds_uncalled | extra | extra_per_hour | extra_yellows_only | extra_no_official_flag | extra_outside_window | extra_red_uncalled | track_clears_under_neutral |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 51 | 31 | 30 | 1 | 16 | 20.1 | 8.4 | 34.3 | 24 | 20 | 36 | 3 | 2 | 3 | 64.1 | 1 | 23 | 0.74 | 20 | 2 | 0 | 1 | 0 |
| 5 | 51 | 31 | 30 | 1 | 16 | 20 | 7.5 | 32.9 | 24 | 20 | 35 | 3 | 2 | 3 | 64.1 | 1 | 22 | 0.7 | 19 | 2 | 0 | 1 | 0 |
| 8 | 51 | 31 | 29 | 2 | 16 | 20 | 7.1 | 29.9 | 24 | 20 | 33 | 3 | 2 | 3 | 64.1 | 1 | 20 | 0.64 | 17 | 2 | 0 | 1 | 0 |
| 10 | 51 | 31 | 29 | 2 | 16 | 18.4 | 6.1 | 27.9 | 24 | 20 | 32 | 3 | 2 | 3 | 64.1 | 1 | 19 | 0.61 | 16 | 2 | 0 | 1 | 0 |
