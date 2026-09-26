# Kickoff (A's first Claude Code session)

Run in Claude Code on Naman's Max laptop, inside the `fast-flag` folder. Target: everything below done by M0 (1:00pm). Move fast, keep explanations short and only stop to ask when truly blocked.

Read `CLAUDE.md` and `PROJECT_BRIEF.md` fully before starting.

## Step 1: Start the race scan now, in the background

- Create `.venv` with Python 3.11 and install `fastf1` and `pandas` into it (Step 2 pins everything properly).
- Write `src/ingest/scan.py`: for every race session from 2023 to 2026, excluding the holdout races, load messages only as in PROJECT_BRIEF.md Section 4.3, count yellow, double yellow, SC, VSC and red events, and save the ranking to `data/race_ranking.csv`. Enable the cache at `data/fastf1_cache` first.
- Give me the exact command to run it in a separate terminal, then keep going while it runs.

## Step 2: Scaffold

- Create the layout from PROJECT_BRIEF.md Section 8, with `__init__.py` files and placeholder modules.
- `requirements.txt`: pinned versions of what we will certainly use (fastf1, pandas, numpy, scipy, scikit-learn, lightgbm, fastapi, uvicorn with websocket support, pyarrow, pyserial, pytest), followed by `-r requirements-a.txt` and `-r requirements-b.txt`, which both start empty. Check that LightGBM installs on this Mac (it may need `libomp` from Homebrew) and tell me if it fails.
- `.gitignore` as in Section 8, including `.venv`.
- `tests/test_contracts.py`: validates every line of every fixture file against the Section 7 shapes (required keys, types and allowed enum values), using plain Python checks with no extra dependency.

## Step 3: Git and GitHub

- `git init`, commit with the message `A: M0 skeleton`, create a private GitHub repo (use the `gh` CLI if it is installed, otherwise walk me through the website) and push `main`.
- Create and push the branches `a-work` and `b-work`.
- Tell me how to add Ishaan as a collaborator.

## Step 4: Fixtures and mock server (Section 7.6)

- Pick one training race with a clear crash (from the scan ranking if it is ready, otherwise a race you know had a crash). Never a holdout race.
- Cut `ticks`, `detections`, `risk`, `recs` and `official` fixtures around one real incident (about 2 to 3 minutes before it to 1 minute after), plus `fixtures/track_sample.json` in the `GET /track` shape. Detections, risk and recs can be hand-built from what actually happened, as long as they match the contracts and the real timing. Keep all fixtures under 20 MB in total.
- `src/replay/mock_server.py` (FastAPI and uvicorn): plays the fixtures over `ws://localhost:8000/stream` with the exact envelope, supports speed and seek through `POST /replay`, rebroadcasts `rec` envelopes that clients send, supports `--no-recs`, serves `GET /track` and `GET /official` and serves the `dashboard/` folder at `/`.
- Run `pytest` until it passes. Commit, merge into `main` and push.

## Step 5: Verify the holdout

- Check that the 2026 Azerbaijan GP race loads in FastF1. Load it only into the cache and a throwaway check, never into training code. If it isn't available yet, tell me and retry every 30 minutes.
- List its official events from race control messages with SessionTime and fill in the "Verified official events" line in PROJECT_BRIEF.md Section 4.4 (allowed during kickoff). Compare against the news notes there. If the race turns out incident-light, switch to Madrid as Section 4.4 says.
- Add both holdout races to an explicit `HOLDOUT` list in `src/ingest/`, and make the scan and training code refuse to touch them.

## Step 6: Hand Ishaan his doc

- Check `docs/session_ishaan.md` against what you actually built (commands, paths, port, fixture names, the `--no-recs` flag). Fix any mismatch, replace `<REPO_URL>` with the real repo URL, commit to `main` and push.
- Then tell me: "Send Ishaan the repo link. He opens Claude Code in his clone and follows docs/session_ishaan.md."

## Step 7: Cache copy

- When the scan and the first full downloads are ready, give me the exact commands to copy `data/fastf1_cache` to a USB drive (or AirDrop it) so Ishaan doesn't download it again over hackathon WiFi.

## Step 8: Continue

Open `docs/session_a.md` and continue from task A1.
