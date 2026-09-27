# Escalation scorecard vs VSC_STOPPED_HOLD_S (training races)

Replay of historical FastF1 data, 20 training races, 31.2 race hours. Each row reruns python -m src.eval.escalation with src.racecontrol.engine.VSC_STOPPED_HOLD_S set to that value (in the worker processes only). In-sample. The A7 holdout run tested the engine frozen before the race control changes of 26 and 27 Sept; those changes were checked on replays of the holdout, so they have no out-of-sample test.

| VSC_STOPPED_HOLD_S | official | matched | earlier | later | missed | median_lead_s | lead_p25_s | lead_p75_s | same_first_flag | our_vsc | our_sc | our_red | red_both | red_race_control_only | red_median_lead_s | our_reds_uncalled | extra | extra_per_hour | extra_yellows_only | extra_no_official_flag | extra_outside_window | extra_red_uncalled | track_clears_under_neutral |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 51 | 31 | 31 | 0 | 16 | 25 | 11.2 | 34.9 | 24 | 23 | 39 | 3 | 2 | 3 | 64.1 | 1 | 28 | 0.9 | 23 | 4 | 0 | 1 | 0 |
| 10 | 51 | 31 | 30 | 1 | 16 | 20.1 | 8.4 | 34.3 | 24 | 20 | 39 | 3 | 2 | 3 | 64.1 | 1 | 25 | 0.8 | 20 | 4 | 0 | 1 | 0 |
| 15 | 51 | 31 | 28 | 3 | 16 | 19.2 | 5.7 | 32.7 | 24 | 19 | 39 | 3 | 2 | 3 | 64.1 | 1 | 24 | 0.77 | 19 | 4 | 0 | 1 | 0 |
| 20 | 51 | 30 | 23 | 7 | 16 | 16.5 | 4.2 | 31.4 | 24 | 17 | 39 | 3 | 2 | 3 | 64.1 | 1 | 23 | 0.74 | 18 | 4 | 0 | 1 | 0 |
| 30 | 51 | 30 | 23 | 7 | 16 | 13.4 | 0.8 | 29.3 | 24 | 15 | 39 | 3 | 2 | 3 | 64.1 | 1 | 21 | 0.67 | 17 | 3 | 0 | 1 | 0 |
