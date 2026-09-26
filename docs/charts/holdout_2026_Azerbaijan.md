# Holdout: 2026_Azerbaijan

Replay of historical FastF1 data. Run 1 of this race, at 2026-09-26T18:54:40, with code and models frozen at commit 380ddc5 (2026-09-26T18:54:35). Every threshold is the frozen one; nothing was fitted on this race.

## Detection

- rule alerts: recall 70.0% (7 of 10 incidents, 0 held), median lead 0.8 s, earlier than race control in 85.7%, 1 false alarms in 1.63 race hours = 0.6/h
- anomaly advisories: recall 0.0% (0 of 10 incidents, 0 held), median lead n/a s, earlier than race control in n/a, 1 false alarms in 1.63 race hours = 0.6/h
- naive speed threshold: recall 80.0% (8 of 10 incidents, 0 held), median lead 2.0 s, earlier than race control in 100.0%, 27 false alarms in 1.63 race hours = 16.6/h

## Latency from crash onset (10 events with an onset car, 6 without)

- YELLOW: 7 events, caught 7, race control median 4.0 s, us 3.2 s, earlier in 85.7%
- DOUBLE_YELLOW: 2 events, caught 2, race control median 5.7 s, us 1.4 s, earlier in 100.0%
- SC: 1 events, caught 1, race control median 28.3 s, us 1.0 s, earlier in 100.0%

## Risk (8 car incidents, 429140 racing ticks)

- 10 s: precursor PR-AUC 0.0094 (ANOMALY score 0.0015, speed threshold 0.0010, base rate 0.0007); headline 0.193
- 30 s: precursor PR-AUC 0.0027 (ANOMALY score 0.0020, speed threshold 0.0015, base rate 0.0022); headline 0.020
- Frozen high-risk line (0.0271): 37.5% of 8 car incidents flagged at least 3 s before detection, 58 false high-risk episodes per race hour

## Escalations (VSC, SC, red): 2 official

- We recommended a VSC, SC or red for 2: 2 earlier, 0 later, median lead 44.0 s. Missed 0, already out 0.
- Extra escalations race control never made: 2 = 1.23 per race hour. Same first flag as race control: 1 of 2.

## Charts

- latency_2026_Azerbaijan.png
- case_2026_Azerbaijan_car23.png
- escalation_2026_Azerbaijan.png

## Crashes (timelines and exposure)

- ALB (car 23): our first alert +1.0 s, official yellow +9.3 s, our escalation SC +5.0 s, official SC +28.3 s
  - W1 onset -> our first alert (unavoidable): 0 cars
  - W2 our first alert -> official yellow: 2 cars
  - W3 our escalation recommendation -> official escalation (headline): 4 cars
  - context: official yellow -> official escalation (not a counterfactual): 3 cars
