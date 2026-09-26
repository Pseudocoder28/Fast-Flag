# Requests from the lab to the core

The lab never edits core files. Anything it needs or finds in the core goes here.

## 1. Resolved: `latency_by_type.csv` labelled a VSC as SC (2026 Australian)

Fixed on `main` in PR #7 ("the corrected VSC split"). The lab output and the core CSV now agree.

## 2. Onset grouping differs from `src.eval.latency_by_type` (for information)

`src/lab/delay_cost.py` groups official incidents into crashes (`group_crashes`). It diverges from `latency_by_type.race_events` in two ways, which the lab review found necessary for honest per-crash cards:

- A second crash within 120 s of an earlier one gets its own onset, instead of being credited to the first crash's onset car. Example: 2024 Canadian, the Sainz/Albon crash at 5062.5 s, which `latency_by_type` attributes to Perez at 4984.0 s along with its SC.
- For official incidents with only track-wide flags, alerts must be in a sector matching the onset car's sector, instead of any sector.

The core may want the same rules. No change is needed for the lab.

## 3. `src.eval.case_study.rest_position` can put the crash site in a pit box (for information)

It takes the first tick below 5 km/h within 30 s and does not exclude the pit lane. 2025 Australian Norris, onset 8955.25 s, ends up in his pit box 700 m from the incident. The lab wraps it (`rest_position_on_track`). The core case studies would need the same guard.
