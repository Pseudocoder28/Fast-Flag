# Fixtures

Built by `python -m src.ingest.fixtures` from the 2023 Australian GP (training race), SessionTime 4240 to 4570 s.

- Real FastF1 data: `ticks_sample.jsonl`, `official_sample.jsonl`, `track_sample.json`.
- Hand-built from the real timing (only so B can build before the real detectors exist): `detections_sample.jsonl`, `risk_sample.jsonl`, `recs_sample.jsonl`.
- Incident: car 23 (Albon) crashes at Turn 6, marshal sector 9. IMPACT t=4387.25, STOPPED t=4390.5. Official YELLOW sector 9 t=4390.18, SC t=4402.18, RED t=4560.18.
