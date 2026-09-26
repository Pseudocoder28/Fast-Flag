# Fixtures

Built by `python -m src.ingest.fixtures` from the 2023 Australian GP (training race), SessionTime 4240 to 4570 s.

- Real FastF1 data: `ticks_sample.jsonl`, `official_sample.jsonl`, `track_sample.json`.
- Hand-built from the real timing (only so B can build before the real detectors exist): `detections_sample.jsonl`, `recs_sample.jsonl`.
- Real risk model: `risk_sample.jsonl` (rebuild alone with `python -m src.ingest.fixtures --risk-only`). The values come from the leave-one-race-out model, which never trained on this race, so nearly every car stays well below the dashboard's high-risk line. The production model, which the real server runs, trained on this race and shows car 23 high 30 s before its crash: memory, not prediction. On the real server, only races the model never saw (2021 Azerbaijan, the 2026 Azerbaijan holdout) show honest risk. `top_features` are the production model's explanation for the same row. A retired car and a car after its crash have no forecast.
- Never quote a number from these files: they only feed the mock server.
- Incident: car 23 (Albon) crashes at Turn 6, marshal sector 9. IMPACT t=4387.25, STOPPED t=4390.5. Official YELLOW sector 9 t=4390.18, SC t=4402.18, RED t=4560.18.
