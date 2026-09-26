# Escalation scorecard vs SC_STOPPED_HOLD_S (training races)

Replay of historical FastF1 data, 20 training races, 31.2 race hours. Each row reruns python -m src.eval.escalation with src.racecontrol.engine.SC_STOPPED_HOLD_S set to that value (in the worker processes only). In-sample: choose before the freeze, the holdout run tests the choice.

| SC_STOPPED_HOLD_S | official | matched | earlier | later | missed | median_lead_s | lead_p25_s | lead_p75_s | same_first_flag | our_vsc | our_sc | our_red | extra | extra_per_hour | extra_yellows_only | extra_no_official_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 51 | 29 | 27 | 2 | 22 | 20 | 9.1 | 33.9 | 22 | 17 | 34 | 8 | 24 | 0.77 | 20 | 4 |
| 5 | 51 | 29 | 27 | 2 | 22 | 18.1 | 7.6 | 31.9 | 22 | 17 | 32 | 8 | 22 | 0.7 | 18 | 4 |
| 8 | 51 | 29 | 26 | 3 | 22 | 15.1 | 7.1 | 28.9 | 22 | 17 | 30 | 8 | 20 | 0.64 | 16 | 4 |
| 10 | 51 | 29 | 26 | 3 | 22 | 13.1 | 7 | 26.9 | 22 | 17 | 28 | 8 | 19 | 0.61 | 15 | 4 |
