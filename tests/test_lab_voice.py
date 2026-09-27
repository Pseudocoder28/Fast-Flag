"""Race control voice (src/lab/voice.py): templates, narrator state and the priority
speaker, driven by the fixture envelopes. Run: pytest tests/test_lab_voice.py"""

from __future__ import annotations

import json
from pathlib import Path

from src.lab.voice import (
    MAX_WAIT_S,
    MIN_GAP_S,
    RESET_JUMP_S,
    Narrator,
    PrintBackend,
    Speaker,
    Utterance,
    clear_phrase,
    confirm_phrase,
    flag_phrase,
)

FIX = Path(__file__).resolve().parent.parent / "fixtures"
KIND_ORDER = {"tick": 0, "detection": 1, "risk": 2, "rec": 3, "official": 4}   # mock server order at equal t


def read_jsonl(name: str) -> list[dict]:
    return [json.loads(ln) for ln in (FIX / name).read_text(encoding="utf-8").splitlines() if ln.strip()]


def fixture_stream() -> list[dict]:
    envs = []
    for kind, name in (("tick", "ticks_sample.jsonl"), ("detection", "detections_sample.jsonl"),
                       ("rec", "recs_sample.jsonl"), ("official", "official_sample.jsonl")):
        envs += [{"kind": kind, "data": row} for row in read_jsonl(name)]
    return sorted(envs, key=lambda e: (e["data"]["t"], KIND_ORDER[e["kind"]]))


def narrate(envs: list[dict]) -> list[Utterance]:
    n = Narrator()
    return [u for env in envs for u in n.on_envelope(env)]


def rec(t: float, flag: str, msector: int, sources: list[str], message: str | None = None) -> dict:
    msg = message or (f"{flag.replace('_', ' ')} IN TRACK SECTOR {msector}" if flag not in ("SC", "VSC", "RED")
                      else {"SC": "SAFETY CAR DEPLOYED", "VSC": "VIRTUAL SAFETY CAR DEPLOYED", "RED": "RED FLAG"}[flag])
    return {"kind": "rec", "data": {"id": f"rec-{t}", "t": t, "msector": msector, "flag": flag, "confidence": 0.8,
                                    "reason": "test", "message": msg, "source_detections": sources}}


def det(det_id: str, t: float, drivers: list[str], msector: int, dtype: str) -> dict:
    return {"kind": "detection", "data": {"id": det_id, "t": t, "drivers": drivers, "msector": msector,
                                          "type": dtype, "severity": 0.9, "evidence": "test"}}


def official(t: float, flag: str, msector: int | None) -> dict:
    return {"kind": "official", "data": {"t": t, "category": "Flag", "message": "x", "flag": flag,
                                         "scope": "Track" if msector is None else "Sector",
                                         "msector": msector, "drivers": []}}


def tick(t: float, msector: int = 9) -> dict:
    return {"kind": "tick", "data": {"t": t, "lap": 5, "track_status": "1",
                                     "cars": [{"drv": "23", "x": 0, "y": 0, "dist": 0, "lat_off": 0, "speed": 200,
                                               "throttle": 100, "brake": False, "gear": 7, "rpm": 10000,
                                               "msector": msector, "gap_ahead_m": -1.0, "in_pit": False}]}}


# --- templates ------------------------------------------------------------------

def test_templates() -> None:
    assert flag_phrase("SC", ["23"], "IMPACT", 7) == "Safety Car. Car 23, impact, sector 7."
    assert flag_phrase("RED", ["1", "10", "43"], "MULTI", 2) == "Red flag. Cars 1, 10 and 43, multi-car, sector 2."
    assert flag_phrase("YELLOW", [], None, 3) == "Yellow flag. Sector 3."
    assert flag_phrase("DOUBLE_YELLOW", ["4"], "SOMETHING_NEW", 5) == "Double yellow. Car 4, sector 5."
    assert clear_phrase(7) == "Sector 7 clear."
    assert clear_phrase(None) == "Track clear."
    assert confirm_phrase("SC", 8.68) == "Race control confirms Safety Car, 8.7 seconds after Fast Flag."


# --- narrator on the fixtures -----------------------------------------------------

def test_fixture_phrases_in_order() -> None:
    texts = [u.text for u in narrate(fixture_stream())]
    assert texts == [
        "Yellow flag. Car 23, impact, sector 9.",                          # rec YELLOW t=4387.5
        "Race control confirms Yellow flag, 2.7 seconds after Fast Flag.",  # official YELLOW 4390.18
        "Double yellow. Car 23, stopped, sector 9.",                       # rec DY 4390.75, latest source is STOPPED
        "Safety Car. Car 23, stopped, sector 9.",                          # rec SC 4393.5
        "Race control confirms Safety Car, 8.7 seconds after Fast Flag.",   # official SC 4402.18
        # official RED 4560.18: we never raised RED, so nothing; official CLEAR: nothing
    ]


def test_fixture_level_ups_interrupt_and_confirmations_do_not() -> None:
    utts = narrate(fixture_stream())
    assert [u.interrupt for u in utts] == [True, False, True, True, False]
    assert [u.rank for u in utts] == [1, 1, 2, 4, 4]


def test_race_control_first_says_nothing_about_lead_time() -> None:
    envs = [tick(100.0), det("d1", 100.0, ["23"], 9, "IMPACT"), official(101.0, "YELLOW", 9),
            rec(102.0, "YELLOW", 9, ["d1"])]
    texts = [u.text for u in narrate(envs)]
    assert texts == ["Yellow flag. Car 23, impact, sector 9."]


def test_official_in_adjacent_sector_confirms_and_only_once() -> None:
    envs = [tick(100.0, msector=21), det("d1", 100.0, ["18"], 20, "IMPACT"), rec(101.0, "YELLOW", 20, ["d1"]),
            official(104.0, "DOUBLE_YELLOW", 21),      # race control flags the next sector, harder than us
            official(105.0, "YELLOW", 20), official(130.0, "YELLOW", 20)]
    texts = [u.text for u in narrate(envs)]
    assert texts == ["Yellow flag. Car 18, impact, sector 20.",
                     "Race control confirms Double yellow, 3.0 seconds after Fast Flag."]


def test_official_three_sectors_upstream_does_not_confirm() -> None:
    envs = [tick(100.0, msector=21), det("d1", 100.0, ["18"], 20, "IMPACT"), rec(101.0, "YELLOW", 20, ["d1"]),
            official(104.0, "YELLOW", 17)]
    assert [u.text for u in narrate(envs)] == ["Yellow flag. Car 18, impact, sector 20."]


def test_lead_is_measured_from_our_first_raise() -> None:
    envs = [tick(100.0), det("d1", 100.0, ["23"], 9, "IMPACT"), rec(101.0, "YELLOW", 9, ["d1"]),
            rec(103.0, "DOUBLE_YELLOW", 9, ["d1"]), official(105.0, "DOUBLE_YELLOW", 9)]
    assert [u.text for u in narrate(envs)][-1] == "Race control confirms Double yellow, 4.0 seconds after Fast Flag."


def test_track_wide_official_needs_our_track_wide_flag() -> None:
    envs = [tick(100.0), det("d1", 100.0, ["23"], 9, "IMPACT"), rec(101.0, "DOUBLE_YELLOW", 9, ["d1"]),
            rec(105.0, "SC", 9, []), official(114.0, "SC", None)]
    assert [u.text for u in narrate(envs)][-2:] == ["Safety Car. Car 23, impact, sector 9.",
                                                   "Race control confirms Safety Car, 9.0 seconds after Fast Flag."]
    envs = [tick(100.0), det("d1", 100.0, ["23"], 9, "IMPACT"), rec(101.0, "DOUBLE_YELLOW", 9, ["d1"]),
            official(110.0, "SC", None)]
    assert [u.text for u in narrate(envs)] == ["Double yellow. Car 23, impact, sector 9."], "we never called SC"
    envs = [tick(100.0), det("d1", 100.0, ["23"], 9, "IMPACT"), rec(101.0, "SC", 9, ["d1"]),
            official(110.0, "RED", None)]
    assert [u.text for u in narrate(envs)] == ["Safety Car. Car 23, impact, sector 9."], "an SC does not confirm a red"


def test_race_control_first_stays_first_when_it_reissues_the_flag() -> None:
    envs = [tick(100.0), det("d1", 100.0, ["23"], 9, "IMPACT"), official(101.0, "YELLOW", 9),
            rec(102.0, "YELLOW", 9, ["d1"]), official(105.0, "YELLOW", 9)]
    assert [u.text for u in narrate(envs)] == ["Yellow flag. Car 23, impact, sector 9."]
    envs = [tick(100.0), det("d1", 100.0, ["23"], 9, "IMPACT"), rec(101.0, "DOUBLE_YELLOW", 9, ["d1"]),
            official(110.0, "SC", None), rec(111.0, "SC", 9, []), official(120.0, "SC", None)]
    assert [u.text for u in narrate(envs)][-1] == "Safety Car. Car 23, impact, sector 9."


def test_tie_is_not_a_lead() -> None:
    envs = [tick(100.0), det("d1", 100.0, ["23"], 9, "IMPACT"), rec(101.0, "YELLOW", 9, ["d1"]),
            official(101.0, "YELLOW", 9)]
    assert [u.text for u in narrate(envs)] == ["Yellow flag. Car 23, impact, sector 9."]


def test_sector_level_up_is_spoken_under_a_safety_car() -> None:
    envs = [tick(100.0), det("d1", 100.0, ["23"], 9, "STOPPED"), rec(100.0, "YELLOW", 9, ["d1"]),
            rec(110.0, "VSC", 9, []), det("d2", 115.0, ["4"], 9, "STOPPED"),
            rec(115.0, "DOUBLE_YELLOW", 9, ["d1", "d2"])]
    assert [u.text for u in narrate(envs)][-1] == "Double yellow. Cars 4 and 23, stopped, sector 9."


def test_silent_anomaly_corroboration_keeps_the_physical_cause() -> None:
    envs = [tick(100.0), det("d1", 100.0, ["23"], 9, "STOPPED"), det("d2", 101.0, ["23"], 9, "ANOMALY"),
            rec(100.0, "DOUBLE_YELLOW", 9, ["d1"]), rec(101.0, "DOUBLE_YELLOW", 9, ["d1", "d2"]),
            rec(104.0, "SC", 9, [])]
    assert [u.text for u in narrate(envs)][-1] == "Safety Car. Car 23, stopped, sector 9."


def test_car_numbers_are_sanitised_for_say() -> None:
    envs = [tick(100.0), det("d1", 100.0, ["23", "[[slnc 5000]]"], 9, "MULTI"), rec(101.0, "YELLOW", 9, ["d1"])]
    assert [u.text for u in narrate(envs)] == ["Yellow flag. Cars 23 and slnc5000, multi-car, sector 9."]


def test_global_escalation_without_sources_uses_sector_cause() -> None:
    envs = [tick(100.0), det("d1", 100.0, ["23"], 9, "STOPPED"), rec(101.0, "DOUBLE_YELLOW", 9, ["d1"]),
            rec(104.0, "SC", 9, [])]
    assert [u.text for u in narrate(envs)][-1] == "Safety Car. Car 23, stopped, sector 9."


def test_same_or_lower_level_is_silent_and_higher_speaks() -> None:
    envs = [tick(100.0), det("d1", 100.0, ["23"], 9, "IMPACT"), rec(101.0, "YELLOW", 9, ["d1"]),
            rec(102.0, "YELLOW", 9, ["d1"]), rec(103.0, "DOUBLE_YELLOW", 9, ["d1"]),
            rec(104.0, "YELLOW", 9, ["d1"]), rec(105.0, "SC", 9, ["d1"]), rec(106.0, "VSC", 9, ["d1"])]
    texts = [u.text for u in narrate(envs)]
    assert texts == ["Yellow flag. Car 23, impact, sector 9.", "Double yellow. Car 23, impact, sector 9.",
                     "Safety Car. Car 23, impact, sector 9."]


def test_multi_car_lists_cars_in_number_order() -> None:
    envs = [tick(100.0), det("d1", 100.0, ["43", "10", "1"], 2, "MULTI"), rec(101.0, "RED", 2, ["d1"])]
    assert [u.text for u in narrate(envs)] == ["Red flag. Cars 1, 10 and 43, multi-car, sector 2."]


def test_clear_phrases_and_new_episode_after_clear() -> None:
    envs = [tick(100.0), det("d1", 100.0, ["23"], 9, "IMPACT"), rec(101.0, "YELLOW", 9, ["d1"]),
            rec(103.0, "SC", 9, ["d1"]),
            rec(120.0, "CLEAR", 9, [], "CLEAR IN TRACK SECTOR 9"), rec(125.0, "CLEAR", 9, [], "TRACK CLEAR"),
            rec(126.0, "CLEAR", 9, [], "TRACK CLEAR"),            # already clear: silent
            det("d2", 200.0, ["4"], 9, "SPIN"), rec(201.0, "YELLOW", 9, ["d2"]), official(203.0, "YELLOW", 9)]
    texts = [u.text for u in narrate(envs)]
    assert texts == ["Yellow flag. Car 23, impact, sector 9.", "Safety Car. Car 23, impact, sector 9.",
                     "Sector 9 clear.", "Track clear.",
                     "Yellow flag. Car 4, spin, sector 9.",
                     "Race control confirms Yellow flag, 2.0 seconds after Fast Flag."]


def test_official_long_after_our_flag_is_not_a_confirmation() -> None:
    envs = [tick(100.0), det("d1", 100.0, ["23"], 9, "IMPACT"), rec(101.0, "YELLOW", 9, ["d1"]),
            rec(110.0, "CLEAR", 9, [], "CLEAR IN TRACK SECTOR 9"), official(400.0, "YELLOW", 9)]
    assert [u.text for u in narrate(envs)] == ["Yellow flag. Car 23, impact, sector 9.", "Sector 9 clear."]


def test_reset_on_backward_time_jump() -> None:
    n = Narrator()
    first = [u for env in fixture_stream() for u in n.on_envelope(env)]
    assert first
    again = [u for env in fixture_stream() for u in n.on_envelope(env)]     # mock server loops back
    assert [u.text for u in again] == [u.text for u in first]
    n.on_envelope(tick(4570.0))
    assert n.on_envelope(tick(4570.0 - RESET_JUMP_S)) == [] and n.level, "a wobble within the jump keeps state"


def test_malformed_input_never_raises() -> None:
    n = Narrator()
    for raw in ("", "not json", "[1, 2]", "null", '{"kind": "rec"}', '{"kind": "rec", "data": {}}',
                '{"kind": "rec", "data": {"flag": "YELLOW"}}', '{"kind": "tick", "data": {"cars": 3}}',
                '{"kind": "official", "data": {"flag": "PURPLE", "t": 1}}', '{"kind": "risk", "data": {"t": 1}}',
                '{"kind": "rec", "data": {"id": "x", "t": 1, "msector": 9, "flag": "YELLOW", "confidence": 1, '
                '"reason": "", "message": "", "source_detections": ["missing-det"]}}'):
        n.on_raw(raw)      # must not raise
    n.on_envelope(tick(1.5))    # live from here: before the first tick is catch-up, never spoken
    assert n.on_raw('{"kind": "rec", "data": {"t": 2, "msector": 9, "flag": "DOUBLE_YELLOW", "message": "",'
                    ' "source_detections": []}}')[0].text == "Double yellow. Sector 9."


# --- speaker --------------------------------------------------------------------

class FakeBackend:
    """Each phrase 'runs' for duration seconds of the fake clock."""

    instant = False

    def __init__(self, duration: float = 1.5) -> None:
        self.duration, self.now = duration, 0.0
        self.started: list[str] = []
        self.stopped: list[str] = []

    def start(self, text: str) -> object:
        self.started.append(text)
        return {"text": text, "end": self.now + self.duration, "stopped": False}

    def running(self, handle: object) -> bool:
        return isinstance(handle, dict) and not handle["stopped"] and self.now < handle["end"]

    def stop(self, handle: object) -> None:
        handle["stopped"] = True
        self.stopped.append(handle["text"])

    def error(self, handle: object) -> str | None:
        return None


def utt(text: str, rank: int, interrupt: bool = False, key: str | None = None, seq: int = 0) -> Utterance:
    return Utterance(text=text, rank=rank, interrupt=interrupt, keys=frozenset({key} if key else ()), seq=seq)


def run_speaker(speaker: Speaker, backend: FakeBackend, events: list[tuple[float, Utterance | None]],
                until: float, dt: float = 0.1) -> list[tuple[float, str]]:
    """Push each utterance at its wall time, step the clock, record (time, text) starts."""
    out, t, queue = [], 0.0, sorted(events, key=lambda e: e[0])
    while t <= until + 1e-9:
        backend.now = t
        while queue and queue[0][0] <= t + 1e-9:
            _, u = queue.pop(0)
            if u is not None:
                speaker.push(u, t)
        for text in speaker.step(t):
            out.append((round(t, 1), text))
        t += dt
    return out


def test_higher_level_interrupts_current_speech() -> None:
    b = FakeBackend(duration=3.0)
    s = Speaker(b)
    starts = run_speaker(s, b, [(0.0, utt("Yellow flag. Car 23, impact, sector 9.", 1, True, "9", 1)),
                                (0.5, utt("Safety Car. Car 23, stopped, sector 9.", 4, True, "track", 2))], 4.0)
    assert starts == [(0.0, "Yellow flag. Car 23, impact, sector 9."), (0.5, "Safety Car. Car 23, stopped, sector 9.")]
    assert b.stopped == ["Yellow flag. Car 23, impact, sector 9."]


def test_lower_levels_wait_for_speech_and_gap() -> None:
    b = FakeBackend(duration=1.0)
    s = Speaker(b)
    starts = run_speaker(s, b, [(0.0, utt("Yellow flag. Sector 1.", 1, True, "1", 1)),
                                (0.2, utt("Yellow flag. Sector 2.", 1, True, "2", 2)),
                                (0.3, utt("Race control confirms Yellow flag, 1.0 seconds after Fast Flag.", 1, False, None, 3))],
                        6.0)
    assert starts == [(0.0, "Yellow flag. Sector 1."), (MIN_GAP_S, "Yellow flag. Sector 2."),
                      (2 * MIN_GAP_S, "Race control confirms Yellow flag, 1.0 seconds after Fast Flag.")]
    assert b.stopped == []


def test_long_phrase_delays_lower_phrase_past_the_gap() -> None:
    b = FakeBackend(duration=3.5)
    s = Speaker(b)
    starts = run_speaker(s, b, [(0.0, utt("Safety Car. Car 23, stopped, sector 9.", 4, True, "track", 1)),
                                (0.1, utt("Yellow flag. Sector 2.", 1, True, "2", 2))], 6.0)
    assert starts == [(0.0, "Safety Car. Car 23, stopped, sector 9."), (3.5, "Yellow flag. Sector 2.")]


def test_most_urgent_queued_phrase_goes_first() -> None:
    b = FakeBackend(duration=1.0)
    s = Speaker(b)
    starts = run_speaker(s, b, [(0.0, utt("Yellow flag. Sector 1.", 1, True, "1", 1)),
                                (0.1, utt("Race control confirms Yellow flag, 1.0 seconds after Fast Flag.", 1, False, None, 2)),
                                (0.2, utt("Yellow flag. Sector 5.", 1, True, "5", 3)),
                                (0.3, utt("Double yellow. Sector 7.", 2, True, "7", 4))], 8.0)
    # level-ups go before confirmations, whatever their rank
    assert [t for _, t in starts] == ["Yellow flag. Sector 1.", "Double yellow. Sector 7.", "Yellow flag. Sector 5.",
                                      "Race control confirms Yellow flag, 1.0 seconds after Fast Flag."]


def test_queued_lower_level_for_same_scope_is_dropped() -> None:
    b = FakeBackend(duration=1.0)
    s = Speaker(b)
    starts = run_speaker(s, b, [(0.0, utt("Yellow flag. Sector 1.", 1, True, "1", 1)),
                                (0.1, utt("Yellow flag. Sector 9.", 1, True, "9", 2)),
                                (0.2, utt("Double yellow. Sector 9.", 2, True, "9", 3))], 6.0)
    assert [t for _, t in starts] == ["Yellow flag. Sector 1.", "Double yellow. Sector 9."]


def test_stale_confirmation_is_dropped_but_level_ups_never_are() -> None:
    b = FakeBackend(duration=MAX_WAIT_S + 5)
    s = Speaker(b)
    starts = run_speaker(s, b, [(0.0, utt("Safety Car. Sector 9.", 4, True, "track", 1)),
                                (1.0, utt("Race control confirms Safety Car, 8.7 seconds after Fast Flag.", 4, False, None, 2)),
                                (1.0, utt("Yellow flag. Sector 2.", 1, True, "2", 3))], MAX_WAIT_S + 10)
    assert [t for _, t in starts] == ["Safety Car. Sector 9.", "Yellow flag. Sector 2."]


def test_higher_level_while_idle_skips_the_gap_and_lower_waits() -> None:
    b = FakeBackend(duration=0.5)
    s = Speaker(b)
    starts = run_speaker(s, b, [(0.0, utt("Yellow flag. Sector 1.", 1, True, "1", 1)),
                                (1.0, utt("Safety Car. Sector 1.", 4, True, "track", 2)),
                                (1.5, utt("Yellow flag. Sector 5.", 1, True, "5", 3))], 5.0)
    assert starts == [(0.0, "Yellow flag. Sector 1."), (1.0, "Safety Car. Sector 1."), (3.0, "Yellow flag. Sector 5.")]
    assert b.stopped == []


def test_gap_is_two_seconds() -> None:
    assert MIN_GAP_S == 2.0


def test_print_backend_prints_everything_without_pacing(capsys) -> None:
    s = Speaker(PrintBackend())
    for k in range(8):
        s.push(utt(f"Yellow flag. Sector {k}.", 1, True, str(k), k), 0.0)
    out = [x for now in range(8) for x in s.step(now * 0.1 + 30.0)]
    assert out == [f"Yellow flag. Sector {k}." for k in range(8)], "nothing is paced or dropped when printing"
    assert s.current is None
    assert "Yellow flag. Sector 0." in capsys.readouterr().out


def test_fixture_end_to_end_through_speaker() -> None:
    b = FakeBackend(duration=1.8)
    s = Speaker(b)
    n = Narrator()
    envs = fixture_stream()
    t0 = envs[0]["data"]["t"]
    events = [(env["data"]["t"] - t0, u) for env in envs for u in n.on_envelope(env)]
    starts = run_speaker(s, b, events, envs[-1]["data"]["t"] - t0 + 5)
    assert [t for _, t in starts] == [
        "Yellow flag. Car 23, impact, sector 9.",
        "Race control confirms Yellow flag, 2.7 seconds after Fast Flag.",
        "Double yellow. Car 23, stopped, sector 9.",
        "Safety Car. Car 23, stopped, sector 9.",
        "Race control confirms Safety Car, 8.7 seconds after Fast Flag.",
    ]


def test_only_track_clear_ends_a_track_wide_flag() -> None:
    # contract: a CLEAR rec with message TRACK CLEAR ends VSC, SC and RED, any other CLEAR clears one sector
    envs = [tick(100.0), det("d1", 100.0, ["23"], 9, "STOPPED"), rec(100.0, "DOUBLE_YELLOW", 9, ["d1"]),
            rec(103.0, "SC", 9, []), rec(120.0, "CLEAR", 9, [], ""), rec(125.0, "CLEAR", 9, [], "TRACK CLEAR")]
    assert [u.text for u in narrate(envs)][-2:] == ["Sector 9 clear.", "Track clear."]


def test_forward_seek_resets_like_the_engine() -> None:
    # live test on 2021 Azerbaijan: a seek from Stroll's crash (SC out, never cleared) to Verstappen's
    # muted the new Safety Car and both confirmations until the voice reset on forward jumps too
    stroll = [tick(5262.0, 20), det("a", 5275.25, ["18"], 20, "IMPACT"), rec(5275.25, "YELLOW", 20, ["a"]),
              rec(5279.25, "SC", 20, ["a"]), official(5311.99, "SC", None), tick(5321.0, 20)]
    ver = [tick(7266.0, 21), det("c", 7279.0, ["33"], 21, "IMPACT"), rec(7279.0, "YELLOW", 21, ["c"]),
           rec(7283.25, "SC", 21, []), official(7297.99, "DOUBLE_YELLOW", 21), official(7366.99, "SC", None)]
    assert [u.text for u in narrate(stroll + ver)][-4:] == [
        "Yellow flag. Car 33, impact, sector 21.", "Safety Car. Car 33, impact, sector 21.",
        "Race control confirms Double yellow, 19.0 seconds after Fast Flag.",
        "Race control confirms Safety Car, 83.7 seconds after Fast Flag."]


def test_reconnect_waits_at_most_5_s() -> None:
    from src.lab.voice import BACKOFF_CAP_S
    assert BACKOFF_CAP_S == 5.0, "a server started after the voice is picked up within 5 s"


def advisory(t: float, msector: int) -> dict:
    env = rec(t, "DOUBLE_YELLOW", msector, [])
    env["data"]["reason"] = ("car 18 still at its crash site 120 s after it stopped: recovery taking long, "
                             "race control may need a red flag (advisory, flag unchanged)")
    return env


def test_seek_catchup_is_silent_and_only_new_calls_speak() -> None:
    """Live test on 2021 Azerbaijan: after a seek past Stroll's crash, the voice reset and then
    announced the long-recovery advisory as a fresh double yellow. The server now sends the tick
    of the seek, then (with ?catchup=1) everything from before it: that only sets the levels."""
    n = Narrator()
    live = [tick(5262.0, 20), det("a", 5275.25, ["18"], 20, "IMPACT"), rec(5275.25, "YELLOW", 20, ["a"])]
    assert [u.text for env in live for u in n.on_envelope(env)] == ["Yellow flag. Car 18, impact, sector 20."]
    seek = [tick(5396.0, 20),                                  # the tick of the seek comes first ...
            det("a", 5275.25, ["18"], 20, "IMPACT"), rec(5275.25, "YELLOW", 20, ["a"]),
            rec(5276.25, "DOUBLE_YELLOW", 20, ["a"]), rec(5279.25, "SC", 20, ["a"]),
            official(5277.99, "DOUBLE_YELLOW", 21), official(5311.99, "SC", None)]   # ... then the catch-up
    assert [u for env in seek for u in n.on_envelope(env)] == []
    assert n.level == {20: 2, "track": 4}, "the catch-up restores the flags already out"
    later = [tick(5396.25, 20), advisory(5396.25, 20), tick(5396.5, 20),
             rec(5396.5, "RED", 20, [])]                       # a new call after the catch-up still speaks
    assert [u.text for env in later for u in n.on_envelope(env)] == ["Red flag. Car 18, impact, sector 20."]


def test_before_the_first_tick_is_catchup() -> None:
    """The first hub catch-up comes before any tick: flags already out, never spoken."""
    envs = [det("a", 5275.25, ["18"], 20, "IMPACT"), rec(5279.25, "SC", 20, ["a"]), tick(5300.0, 20),
            official(5311.99, "SC", None)]
    assert [u.text for u in narrate(envs)] == ["Race control confirms Safety Car, 32.7 seconds after Fast Flag."]


def test_recovery_advisory_is_never_a_new_call() -> None:
    """Even with no catch-up (an old server), the advisory only sets the level."""
    n = Narrator()
    envs = [tick(5390.0, 20), tick(5390.25, 20), advisory(5396.25, 20)]
    assert [u for env in envs for u in n.on_envelope(env)] == []
    assert n.level == {20: 2}


def test_voice_asks_for_the_catchup() -> None:
    from src.lab.voice import URL, with_catchup
    assert URL.endswith("/stream?catchup=1")
    assert with_catchup("ws://10.0.0.5:8000/stream") == "ws://10.0.0.5:8000/stream?catchup=1"
    assert with_catchup("ws://h/stream?x=1") == "ws://h/stream?x=1&catchup=1"
    assert with_catchup("ws://h/stream?catchup=0") == "ws://h/stream?catchup=0"
