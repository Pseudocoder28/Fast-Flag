# Fast Flag (FormulaTech Hacks, team We Are So Back)

Short rules and contracts. `PROJECT_BRIEF.md` is the single source of truth for everything else. Read it fully at the start of every session.

## Who is who

- **A = Naman.** Claude Max account, Mac, main builder. Owns `src/ingest/`, `src/replay/` (including the FastAPI server and mock server), `src/detect/`, `src/predict/`, `src/eval/`, `src/racecontrol/`, `dashboard/`, `fixtures/`, `docs/charts/` and `requirements-a.txt`. Starts with `docs/KICKOFF.md`, then follows `docs/session_a.md`.
- **B = Ishaan.** Claude Pro account, and a Codex account while his Claude limit resets. Owns `src/bridge/`, `firmware/`, `requirements-b.txt`, the pitch slides and the demo script. Tests the dashboard in his browser after every push and reports issues to Naman. Follows `docs/session_ishaan.md`.
- **Handover (26 Sept, agreed by both):** `src/racecontrol/`, `dashboard/` and `tests/test_racecontrol.py` moved from B to A when Ishaan's Claude limit ran out.
- **Tests:** `tests/test_<area>.py` belongs to the owner of that area. `tests/test_contracts.py` stays shared.
- **Shared, changed only by agreement:** `CLAUDE.md`, `PROJECT_BRIEF.md`, `docs/` (except `docs/charts/`), `requirements.txt`, `tests/test_contracts.py`, `pytest.ini`, `.gitignore`, `README.md`.

## Hard rules

1. Only edit files in your own paths. Never patch the other person's code to make yours work. Report the mismatch to them instead.
2. Never change a contract (PROJECT_BRIEF.md Section 7) silently. Propose it, both agree, one person updates PROJECT_BRIEF.md and this file on `main`, then both merge `main` into their branch.
3. No future leakage. Every component only sees data with SessionTime <= the current tick.
4. The holdout race (2026 Azerbaijan GP, fallback 2026 Madrid) is never loaded by training or tuning code. Keep it in an explicit HOLDOUT list that training code checks.
5. Free tools only. No paid APIs at runtime, including the Claude API. Narration uses templates or a local Ollama model, and the LLM never decides anything.
6. The ANOMALY detector (IsolationForest) is never cut. It keeps a learned model at the core of the product, which Ampere requires.
7. Honesty: the demo always says it is a replay of historical FastF1 data. Claim "earlier than the race control feed", never "earlier than the marshals".
8. These rules apply to every AI tool we use (Claude Code, Codex, Claude chat).

## Contracts (summary; full spec in PROJECT_BRIEF.md Section 7)

- WebSocket `ws://localhost:8000/stream`, envelope `{"kind": "tick|detection|risk|rec|official", "data": {...}}`.
- Detection types: STOPPED, IMPACT, SPIN, DROPOUT, MULTI, ANOMALY. Recommendation flags: CLEAR, YELLOW, DOUBLE_YELLOW, VSC, SC, RED.
- racecontrol publishes by sending `rec` envelopes on the same socket, and the server rebroadcasts them.
- HTTP: `POST /replay`, `GET /track`, `GET /official`, `GET /status`, `GET /races`, dashboard served at `/`. The mock server implements all of these, plus `--no-recs`.
- `RaceControl.on_tick(tick)` returns `(recs, did_reset)`. `on_detection` and `on_risk` return a list of recs.
- Recs with flag VSC, SC or RED are track-wide (`msector` is the sector that caused them). A CLEAR rec with message `TRACK CLEAR` ends them, other CLEAR recs clear one sector.
- Parked until after M2: the proposed `human` envelope. It is not in the contract, so nothing sends or handles it yet.
- Units: seconds (SessionTime), metres, km/h. `drv` is the car number as a string.
- `official` data: `{t, category, message, flag, scope, msector, drivers}`, `msector` null when track-wide. Tick: cars with no fresh data are omitted, `gap_ahead_m` is -1.0 when unknown.
- Serial protocol: PROJECT_BRIEF.md Section 7.7.
- `tests/test_contracts.py` validates the fixtures against these shapes. Run `pytest` before every merge into `main`.

## Git workflow

- Branches: A works on `a-work`, B works on `b-work`. `main` only receives working code.
- Commit small and often. Commit messages start with `A:` or `B:`.
- Merge `main` into your branch often. Merge your branch into `main` at milestones (M1, M2, M3, M4) or when the other person needs your change, and only when `pytest` passes.
- Before merging into `main`, run `git diff --stat main...HEAD` and check that every changed file is in your own paths. If anything else shows up, stop and ask.
- Merge one person at a time: the first person merges and pushes, the second pulls `main`, merges it into their branch, runs `pytest` and then merges.
- New Python packages go in your own `requirements-a.txt` or `requirements-b.txt`, pinned, and you tell the other person. Nobody edits `requirements.txt` after kickoff.

## Environment

- Python 3.11 with `.venv` at the repo root (gitignored). Install with `pip install -r requirements.txt`.
- Run everything from the repo root with `python -m`, for example `python -m src.replay.mock_server`.
- FastF1 cache in `data/fastf1_cache`, engineered features in `data/features`, models in `data/models` (all gitignored).
- Code style: Python 3.11, type hints, small functions, every module runnable with `python -m`, minimal dependencies. No em dashes in code, comments, docs or messages.

## How to talk to each person

- **Naman (A):** short plain-language explanations, then do the work.
- **Ishaan (B):** minimal explanations, direct bullets, step-by-step commands.
- **Both:** if stuck for more than 20 minutes, message the other person with the error and what was tried.
