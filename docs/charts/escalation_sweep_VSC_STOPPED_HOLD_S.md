# Escalation scorecard vs VSC_STOPPED_HOLD_S (training races)

Replay of historical FastF1 data, 20 training races, 31.2 race hours. Each row reruns python -m src.eval.escalation with src.racecontrol.engine.VSC_STOPPED_HOLD_S set to that value (in the worker processes only). In-sample. The A7 holdout run tested the engine frozen before the race control changes of 26 and 27 Sept; those changes were checked on replays of the holdout, so they have no out-of-sample test.

| VSC_STOPPED_HOLD_S | official | matched | earlier | later | missed | median_lead_s | lead_p25_s | lead_p75_s | same_first_flag | our_vsc | our_sc | our_red | red_both | red_race_control_only | red_ours_only | red_median_lead_s | our_reds_uncalled | extra | extra_per_hour | extra_yellows_only | extra_no_official_flag | extra_outside_window | extra_red_uncalled | track_clears_under_neutral |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 51 | 29 | 28 | 1 | 20 | 20.1 | 12 | 33.9 | 22 | 20 | 33 | 23 | 3 | 2 | 13 | 51.4 | 18 | 39 | 1.25 | 23 | 4 | 0 | 12 | 0 |
| 10 | 51 | 29 | 27 | 2 | 20 | 20 | 9.1 | 33.9 | 22 | 17 | 33 | 23 | 3 | 2 | 13 | 51.4 | 18 | 36 | 1.15 | 20 | 4 | 0 | 12 | 0 |
| 15 | 51 | 29 | 26 | 3 | 20 | 17.5 | 7.2 | 32.7 | 22 | 16 | 33 | 23 | 3 | 2 | 13 | 51.4 | 18 | 35 | 1.12 | 19 | 4 | 0 | 12 | 0 |
| 20 | 51 | 28 | 22 | 6 | 20 | 16.9 | 6.2 | 28.9 | 22 | 13 | 33 | 23 | 3 | 2 | 13 | 51.4 | 18 | 33 | 1.06 | 18 | 3 | 0 | 12 | 0 |
| 30 | 51 | 28 | 22 | 6 | 20 | 15.6 | 2.4 | 26.7 | 22 | 12 | 33 | 23 | 3 | 2 | 13 | 51.4 | 18 | 32 | 1.02 | 17 | 3 | 0 | 12 | 0 |
