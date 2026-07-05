import numpy as np
import mne
import pytest
from types import SimpleNamespace
from mne_bids_pipeline._import_data import _find_breaks_func

def make_raw(annots, duration=300.0, sfreq=100.0):
    info = mne.create_info(1, sfreq, "eeg")
    raw = mne.io.RawArray(np.zeros((1, int(duration * sfreq))), info)
    raw.set_annotations(annots)
    return raw


def get_results(raw):
    breaks = [(a["description"], a["onset"], a["onset"] + a["duration"]) for a in raw.annotations if a["description"].startswith("BAD")]
    return breaks


def test_break_removal(capsys):
    mne.set_log_level("ERROR")
    base = dict(
        find_breaks=True,
        min_break_duration=15.0,
        t_break_annot_start_after_previous_event=5.0,
        t_break_annot_stop_before_next_event=5.0,
        task_event_regex=None,
        break_start_regex=None,
        break_end_regex=None,
    )
    run = dict(subject="01", session=None, run=None)
    duration = 300.0
    sfreq = 100.0

    
    """ Test break removal by marker detection. """
    cfg = SimpleNamespace(**{**base, "break_start_regex": r"block\d*_end",
                        "break_end_regex": r"block\d*_start"})

    # SCENARIO 1.a: break removal by marker with perfectly matching start and end markers"
    annots = mne.Annotations(
        [10, 80, 100, 170, 190, 260], [0] * 6,
        ["block_start", "block_end", "block_start", "block_end", "block_start", "block_end"])
    raw = make_raw(annots, duration=duration, sfreq=sfreq)
    _find_breaks_func(cfg=cfg, raw=raw, **run)
    breaks = get_results(raw)
    assert breaks == [
        ("BAD_beginning", 0.0, 10.0),
        ("BAD_break", 80.0, 100.0),
        ("BAD_break", 170.0, 190.0),
        ("BAD_ending", 260.0, duration - 1/sfreq),
    ]

    # SCENARIO 1.b: break removal by marker with non-matching start and end markers (two block_end (breal_start) in a row)
    annots = mne.Annotations(
    [10, 100, 170, 190, 260], [0] * 5,
    ["block_start", "block_end", "block_end", "block_start", "block_end"])
    raw = make_raw(annots, duration=duration, sfreq=sfreq)
    _find_breaks_func(cfg=cfg, raw=raw, **run)
    breaks = get_results(raw)
    assert breaks == [
        ("BAD_beginning", 0.0, 10.0),
        ("BAD_break", 170.0, 190.0),
        ("BAD_ending", 260.0, duration - 1/sfreq)
    ]
    captured = capsys.readouterr()
    assert "Found two break starts in a row" in captured.out

    # SCENARIO 1.c: break removal by marker with non-matching start and end markers (two block_start (breal_end) in a row)
    annots = mne.Annotations(
    [10, 100, 170, 190, 260], [0] * 5,
    ["block_start", "block_start", "block_end", "block_start", "block_end"])
    raw = make_raw(annots, duration=duration, sfreq=sfreq)
    _find_breaks_func(cfg=cfg, raw=raw, **run)
    breaks = get_results(raw)
    assert breaks == [
        ("BAD_beginning", 0.0, 10.0),
        ("BAD_break", 170.0, 190.0),
        ("BAD_ending", 260.0, duration - 1/sfreq)
    ]
    captured = capsys.readouterr()
    assert "Found two break ends in a row" in captured.out

    # SCENARIO 1.d: break removal by marker with overlapping start and end markers
    raw = make_raw(mne.Annotations(
        onset=[10, 80, 80, 170, 190, 260], duration=[0] * 6,
        description=["block_start", "block_end", "block_start", "block_end", "block_start", "block_end"]))
    _find_breaks_func(cfg=cfg, raw=raw, **run)
    breaks = get_results(raw)
    assert breaks == [
        ("BAD_beginning", 0.0, 10.0),
        ("BAD_break", 170.0, 190.0),
        ("BAD_ending", 260.0, duration - 1/sfreq)
    ]
    captured = capsys.readouterr()
    assert "Found two break starts in a row" in captured.out
    assert "Found two break ends in a row" in captured.out

    # SCENARIO 1.e: break removal by marker with regex matching
    raw = make_raw(mne.Annotations(
        onset=[10, 80, 100, 170, 190, 260], duration=[0] * 6,
        description=["block1_start", "block1_end", "block2_start", "block2_end", "block3_start", "block4_end"]))
    _find_breaks_func(cfg=cfg, raw=raw, **run)
    breaks = get_results(raw)
    assert breaks == [
        ("BAD_beginning", 0.0, 10.0),
        ("BAD_break", 80.0, 100.0),
        ("BAD_break", 170.0, 190.0),
        ("BAD_ending", 260.0, duration - 1/sfreq)
    ]

    # SCENARIO 1.f: marker path, clean blocks, regex, no matches (no breaks should be found)
    raw= make_raw(mne.Annotations(
    onset=[10, 80, 100, 170, 190, 260], duration=[0] * 6,
    description=["event1", "event2", "event3", "event4", "event5", "event6"]))
    _find_breaks_func(cfg=cfg, raw=raw, **run)
    breaks = get_results(raw)
    assert breaks == []


    """ Test break removal by gap detection. """
    ann = mne.Annotations(list(np.arange(0, 300, 1.0)) + [10, 250],
                        [0] * 300 + [0, 0],
                        ["sync"] * 300 + ["event_1", "event_2"])

    # SCENARIO 2.a: 1 Hz pulses, no event regex defined (pulses hide the break)
    cfg = SimpleNamespace(**base)
    raw = make_raw(ann)
    _find_breaks_func(cfg=cfg, raw=raw, **run)
    breaks = get_results(raw)
    assert breaks == []

    # SCENARIO 2.b: 1 Hz pulses, event regex defined (break reappears)
    cfg = SimpleNamespace(**{**base, "task_event_regex": "event"})
    raw = make_raw(ann)
    _find_breaks_func(cfg=cfg, raw=raw, **run)
    breaks = get_results(raw)
    assert breaks == [
       ("BAD_break", 15.0, 245.0), 
       ("BAD_break", 255.0, 299.99)
    ]
