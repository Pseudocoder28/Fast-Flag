# Requests from the lab to the core

The lab never edits core files. Anything it needs or finds in the core goes here.

## 1. Resolved: `latency_by_type.csv` labelled a VSC as SC (2026 Australian)

Fixed on `main` in PR #7 ("the corrected VSC split"). The lab output and the core CSV now agree.

## 2. Partly adopted: onset grouping in `src.eval.latency_by_type`

- **Adopted (aa56cdd):** a second crash inside the 120 s window now gets its own onset. The onset car is the first car of the last chain of collapses (less than 10 s apart) before race control's first message. The 2024 Canadian example now anchors on Sainz at 5062.5 s, as in the lab.
- **Not adopted:** for official incidents with only track-wide flags, the lab only accepts alerts in a sector matching the onset car's sector. The core keeps accepting any sector, because the A7 holdout has already run and changing the rule would split the training and holdout definitions. The lab keeps its stricter rule for its own per-crash cards. The two definitions differ only for track-wide-only incidents.

## 3. Resolved: `rest_position` could put the crash site in a pit box

`src.eval.case_study.rest_position` now skips pit-lane ticks. No published crash site changed. The lab's `rest_position_on_track` wrapper still works and gives the same result.

## 4. Lab numbers in `NUMBERS.md`

Added in PR #10 (38fcad2, 34ae014): `src/eval/numbers.py` reads `docs/lab/delay_cost.json`, section "Lab: cost of delay". The slides quote those numbers.
