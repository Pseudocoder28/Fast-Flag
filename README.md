# Fast Flag

AI race control assistant (FormulaTech Hacks, team We Are So Back). Watches every car's telemetry during a **replay of historical FastF1 data**, detects incidents, predicts risk and recommends flags, benchmarked against the official race control feed.

See `PROJECT_BRIEF.md` for the full spec and `CLAUDE.md` for rules and contracts.

## Setup

```
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
python -m src.replay.mock_server
```

Then open http://localhost:8000.
