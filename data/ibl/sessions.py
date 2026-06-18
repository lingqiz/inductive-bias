"""Unified, source-agnostic session interface across IBL-task datasets.

Two backends share one API behind a single entry point:
  - 'ibl' : processed ALF dataset           -> data.ibl.raw   (root: IBL_Full/data)
  - 'jrc' : in-house raw iblrig dataset      -> data.ibl.jrc   (root: IBL_JRC/data)

Both expose a Session with the SAME public surface (``SessionLike``), so
downstream analysis is dataset-agnostic. The CSV daily-aggregate layer lives in
``utils.py`` (load_behavior_data / load_behavior_complete) and is separate, since
the CSV is a different granularity (per-day, no wheel).

Usage:
    from data.ibl import sessions
    idx  = sessions.open_dataset('jrc')                 # session index DataFrame
    sess = sessions.open_session('jrc', subject='m01')  # a SessionLike
    t, p = sess.wheel(); seg = sess.trial_wheel(0)

Generated with Claude Code.
"""
from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np

# Normalized per-trial schema every backend guarantees in Session.trials.
# stimOn_times / feedback_times are in SECONDS on the SAME clock as wheel().
COMMON_COLS = [
    'trial_idx', 'choice', 'signed_contrast', 'abs_contrast', 'is_easy',
    'feedbackType', 'correct', 'stimOn_times', 'feedback_times',
]

# Canonical sign conventions both backends conform to:
#   choice          : +1 = rightward wheel turn, -1 = leftward, 0 = no-go
#   signed_contrast : + = stimulus on the RIGHT
#                     IBL: contrastRight - contrastLeft
#                     JRC: -sign(position) * |contrast|  (position < 0 => right)

DATASETS = {
    'ibl': {'module': 'raw', 'root': '/groups/zhang/home/zhangl5/IBL_Full/data'},
    'jrc': {'module': 'jrc', 'root': '/groups/zhang/home/zhangl5/IBL_JRC/data'},
}


@runtime_checkable
class SessionLike(Protocol):
    """Structural contract both backend Session types satisfy (no inheritance)."""
    subject: str
    trials: object  # pandas DataFrame containing COMMON_COLS (+ source extras)

    @property
    def session_id(self) -> str: ...
    def load_trials(self): ...
    def wheel(self) -> Tuple[np.ndarray, np.ndarray]: ...        # (t_s, pos_rad)
    def has_wheel(self) -> bool: ...
    def trial_wheel(self, i: int, align: str = 'stimOn_times',
                    t_pre: float = 0.5, t_post: float = 2.0,
                    zero_position: bool = True
                    ) -> Optional[Tuple[np.ndarray, np.ndarray]]: ...


def slice_wheel(trials, wheel, i, align='stimOn_times', t_pre=0.5, t_post=2.0,
                zero_position=True):
    """Slice a continuous wheel trace around trial ``i``'s alignment event.

    Args:
        trials: DataFrame with the ``align`` column (event times in seconds).
        wheel: (t_s, pos) arrays on the SAME clock as the event times.
        i: trial index.
        align: trial column to align to (default stimulus onset).
        t_pre, t_post: window (s) before/after the event.
        zero_position: baseline position to its value at the event onset.

    Returns (t_rel, pos) or None if the event time is non-finite. Shared by both
    backends so ``trial_wheel`` behaves identically regardless of dataset.
    """
    t0 = trials[align].iloc[i]
    if not np.isfinite(t0):
        return None
    wt, wp = wheel
    m = (wt >= t0 - t_pre) & (wt <= t0 + t_post)
    t_rel = wt[m] - t0
    pos = wp[m].astype(float)
    if zero_position and len(pos):
        base = wp[wt <= t0]
        pos = pos - (base[-1] if len(base) else pos[0])
    return t_rel, pos


def _backend(dataset):
    """Import the backend module for a dataset key. Works whether this module was
    imported as ``data.ibl.sessions`` (relative) or ``sessions`` (data/ibl/ on
    sys.path)."""
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; choices: {list(DATASETS)}")
    name = DATASETS[dataset]['module']
    if name == 'raw':
        try:
            from . import raw
        except (ImportError, ValueError):
            import raw
        return raw
    if name == 'jrc':
        try:
            from . import jrc
        except (ImportError, ValueError):
            import jrc
        return jrc
    raise KeyError(name)


def open_dataset(dataset, **kwargs):
    """Return the backend's per-session index (DataFrame) for ``dataset``."""
    return _backend(dataset).build_index(**kwargs)


def open_session(dataset, *, subject=None, session_id=None, **kwargs) -> SessionLike:
    """Return a Session (trials loaded) for one session of ``dataset``.

    Extra backend-specific kwargs pass through (IBL: date/number/csv_subset;
    JRC: session_name/root).
    """
    return _backend(dataset).get_session(subject=subject, session_id=session_id, **kwargs)


def iter_sessions(dataset, subject=None, load=False, **kwargs):
    """Yield Session objects for ``dataset`` (optionally one subject)."""
    return _backend(dataset).iter_sessions(subject=subject, load=load, **kwargs)
