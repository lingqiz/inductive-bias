"""Pipeline for reading the full raw IBL training dataset (ALF/ONE format).

The raw dataset lives at ``RAW_ROOT`` and follows the standard IBL layout::

    <lab>/Subjects/<subject>/<date>/<number>/alf/
        _ibl_trials.table.<uuid>.pqt        per-trial behaviour
        _ibl_wheel.timestamps.<uuid>.npy    continuous wheel clock (s)
        _ibl_wheel.position.<uuid>.npy      continuous wheel position (rad)
        _ibl_wheel.velocity.<uuid>.npy      (optional) wheel velocity
        _ibl_wheelMoves.intervals.<uuid>.npy
        _ibl_wheelMoves.peakAmplitude.<uuid>.npy

This is a superset of ``ibl_behavior.csv`` (the paper's Fig. 2 daily-aggregate
source data). Three access tiers:

  build_index()       -> one row per session (metadata + alf path), cached.
  load_trials()       -> flat per-trial DataFrame across sessions, cached.
  get_session(...)    -> a Session object exposing per-trial behaviour AND the
                         continuous wheel trace, for kinematic analysis.

Plus CSV-compatible reductions used to validate against the existing notebook:

  daily_aggregate()   -> per (subject, training_day) summary (CSV schema).
  load_behavior_raw() -> nested dict, drop-in for utils.load_behavior_data.

Wheel event times and trial event times (stimOn_times, feedback_times, ...)
share the same session clock, so per-trial wheel segments come from slicing the
continuous trace by trial interval (see Session.trial_wheel).

Generated with Claude Code.
"""
import os
import glob
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RAW_ROOT = '/groups/zhang/home/zhangl5/IBL_Full/data'
CSV_PATH = os.path.join(os.path.dirname(__file__), 'behavior.csv')
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')

# Physical lab folder -> anonymized institution code used in the eLife paper.
# Two institutions pool two physical labs each (Lab 3, Lab 6).
LAB_TO_INSTITUTION = {
    'cortexlab':      'Lab 1',
    'mainenlab':      'Lab 2',
    'churchlandlab':  'Lab 3',
    'zadorlab':       'Lab 3',
    'angelakilab':    'Lab 4',
    'wittenlab':      'Lab 5',
    'mrsicflogellab': 'Lab 6',
    'hoferlab':       'Lab 6',
    'danlab':         'Lab 7',
}

# IBL "easy" trials: |contrast| in {50%, 100%}.
EASY_CONTRASTS = (0.5, 1.0)

# CHOICE SIGN CONVENTION.
# The raw ALF ``choice`` is signed OPPOSITE to both the stimulus side and the wheel movement:
# raw choice = +1 goes with a stimulus on the LEFT and a NEGATIVE wheel displacement (verified:
# median displacement at response = -0.282 rad for raw choice +1). We flip it on load so that
# a single sign means the same thing everywhere:
#
#     choice = +1  <->  stimulus on the RIGHT      (signed_contrast > 0)
#                  <->  wheel driven POSITIVE      (displacement -> +threshold_rad)
#     correct      <->  choice == sign(signed_contrast)      (was: choice == -sign(...))
#     boundary this trial drove to = +choice * threshold_rad (was: -choice * ...)
#
# Verified on the full 55-mouse cohort (684,693 easy responded trials, 3,018 sessions): the
# raw relation `correct <=> raw_choice == -stim_side` holds with ZERO violations, so the flip
# is safe and uniform. ``correct`` comes from feedbackType and is NOT affected by the flip.
CHOICE_SIGN = -1

# Bump when a change to _enrich_trials alters cached columns, so load_trials() cannot
# silently serve a parquet built under the old convention.
TRIALS_SCHEMA = 'v2'

# Wheel/stimulus geometry (iblrig standard). The stimulus is driven to the
# response boundary when the wheel turns enough to move it STIM_ANGLE_DEG of
# visual angle at the session's gain (deg/mm); wheel angle = displacement / radius.
#   threshold_rad = STIM_ANGLE_DEG / stim_gain / WHEEL_RADIUS_MM
STIM_ANGLE_DEG = 35.0        # visual angle from center to response boundary
WHEEL_RADIUS_MM = 31.0       # IBL wheel radius
DEFAULT_STIM_GAIN = 4.0      # AG_MIN_VALUE; the value used once mice are trained

# trials.table columns carried through (others dropped to keep frames lean).
_TRIAL_COLS = [
    'choice', 'feedbackType', 'rewardVolume', 'probabilityLeft',
    'contrastLeft', 'contrastRight', 'stimOn_times', 'goCue_times',
    'firstMovement_times', 'response_times', 'feedback_times',
    'intervals_0', 'intervals_1',
]

# ---------------------------------------------------------------------------
# Helpers: locate files, derived per-trial quantities
# ---------------------------------------------------------------------------

def _find(alf_dir, pattern):
    """Return the first file in ``alf_dir`` matching ``pattern`` (ALF UUID glob)."""
    hits = glob.glob(os.path.join(alf_dir, pattern))
    return hits[0] if hits else None


# Per-field ALF .npy files -> trials.table column name. Older sessions (the
# recovered early ones) store trials this way instead of a consolidated .pqt.
_NPY_TRIAL_FIELDS = [
    'choice', 'feedbackType', 'rewardVolume', 'probabilityLeft',
    'contrastLeft', 'contrastRight', 'stimOn_times', 'goCue_times',
    'firstMovement_times', 'response_times', 'feedback_times',
]


def _read_trials_alf(alf_dir):
    """Read one session's trials as a DataFrame (subset of _TRIAL_COLS).

    Prefers the consolidated ``_ibl_trials.table.*.pqt``; falls back to the
    older per-field ``_ibl_trials.<field>.*.npy`` layout (assembling the same
    columns) so recovered early-format sessions parse identically. Returns None
    if neither is present.
    """
    pqt = _find(alf_dir, '_ibl_trials.table*.pqt')
    if pqt is not None:
        df = pd.read_parquet(pqt).reset_index(drop=True)
        return df[[c for c in _TRIAL_COLS if c in df.columns]]

    cols = {}
    for field in _NPY_TRIAL_FIELDS:
        f = _find(alf_dir, f'_ibl_trials.{field}.*.npy')
        if f is not None:
            cols[field] = np.load(f, allow_pickle=True)
    if not cols:
        return None
    iv = _find(alf_dir, '_ibl_trials.intervals.*.npy')
    if iv is not None:
        arr = np.load(iv, allow_pickle=True)
        if arr.ndim == 2 and arr.shape[1] == 2:
            cols['intervals_0'] = arr[:, 0]
            cols['intervals_1'] = arr[:, 1]
    n = max(len(v) for v in cols.values())
    cols = {k: v for k, v in cols.items() if len(v) == n}   # guard ragged fields
    df = pd.DataFrame(cols)
    return df[[c for c in _TRIAL_COLS if c in df.columns]]


def _wheel_paths(alf_dir):
    """Return (position_path, timestamps_path) with matching lengths.

    Older sessions ship two ``_ibl_wheel.timestamps`` files (different UUIDs and
    lengths); the dense one paired with ``position`` has the same byte size
    (both float64). We select by size so the (t, pos) arrays line up; falls back
    to the first timestamps file otherwise.
    """
    ps = _find(alf_dir, '_ibl_wheel.position*.npy')
    if ps is None:
        return None, None
    cands = sorted(glob.glob(os.path.join(alf_dir, '_ibl_wheel.timestamps*.npy')))
    if not cands:
        return ps, None
    psize = os.path.getsize(ps)
    for t in cands:
        if os.path.getsize(t) == psize:
            return ps, t
    return ps, cands[0]


def _read_settings(alf_dir):
    """Load a session's iblrig task settings JSON, or None if absent.

    Settings live in the sibling ``raw_behavior_data/`` of the ``alf/`` dir.
    """
    rbd = os.path.join(os.path.dirname(alf_dir), 'raw_behavior_data')
    f = _find(rbd, '_iblrig_taskSettings.raw*.json')
    if f is None:
        return None
    try:
        with open(f) as fh:
            return json.load(fh)
    except (ValueError, OSError):
        return None


def _csv_subject_map():
    """nickname -> institution_code, taken from the CSV (authoritative)."""
    df = pd.read_csv(CSV_PATH, usecols=['subject_nickname', 'institution_code'])
    return dict(zip(df['subject_nickname'], df['institution_code']))


def _enrich_trials(df):
    """Add signed/abs contrast, easy flag, correctness, reaction times in place.

    NOTE ``choice`` is SIGN-FLIPPED relative to the raw ALF value (see CHOICE_SIGN).
    """
    cl = df['contrastLeft'].fillna(0.0)
    cr = df['contrastRight'].fillna(0.0)
    df['signed_contrast'] = cr - cl            # + = stimulus on the right
    df['abs_contrast'] = cl + cr               # exactly one side is non-zero
    df['is_easy'] = df['abs_contrast'].round(4).isin(EASY_CONTRASTS)
    df['correct'] = (df['feedbackType'] == 1).astype(int)
    df['choice'] = CHOICE_SIGN * df['choice']  # -> same sign as stimulus / wheel (see above)
    if {'response_times', 'stimOn_times'} <= set(df.columns):
        df['rt_response'] = df['response_times'] - df['stimOn_times']
    if {'firstMovement_times', 'stimOn_times'} <= set(df.columns):
        df['rt_movement'] = df['firstMovement_times'] - df['stimOn_times']
    return df


# ---------------------------------------------------------------------------
# Tier 1: session index
# ---------------------------------------------------------------------------

def build_index(csv_subset=True, root=RAW_ROOT, cache=True, verbose=True):
    """One row per session: metadata + path to its alf/ directory.

    Columns: [subject, lab, institution_code, date, number, training_day,
              session_id, alf_dir]. ``training_day`` is the 1-based dense rank
            of session dates per subject; ``session_id`` is
            ``subject/date/number``.

    With ``csv_subset=True`` only the trained subjects in ibl_behavior.csv are
    indexed. Result is cached under ``data/cache/``.
    """
    tag = 'csvsubset' if csv_subset else 'all'
    cache_path = os.path.join(CACHE_DIR, f'session_index_{tag}.parquet')
    if cache and os.path.exists(cache_path):
        if verbose:
            print(f'index: loaded cache {os.path.relpath(cache_path)}')
        return pd.read_parquet(cache_path)

    csv_map = _csv_subject_map()
    want = set(csv_map) if csv_subset else None

    rows = []
    for lab in sorted(os.listdir(root)):
        subj_root = os.path.join(root, lab, 'Subjects')
        if not os.path.isdir(subj_root):
            continue
        inst = LAB_TO_INSTITUTION.get(lab)
        for nick in sorted(os.listdir(subj_root)):
            if want is not None and nick not in want:
                continue
            nick_dir = os.path.join(subj_root, nick)
            dates = sorted(d for d in os.listdir(nick_dir)
                           if os.path.isdir(os.path.join(nick_dir, d)))
            for day_idx, date in enumerate(sorted(dates), start=1):
                date_dir = os.path.join(nick_dir, date)
                for number in sorted(os.listdir(date_dir)):
                    alf_dir = os.path.join(date_dir, number, 'alf')
                    if not os.path.isdir(alf_dir):
                        continue
                    rows.append({
                        'subject': nick,
                        'lab': lab,
                        'institution_code': csv_map.get(nick, inst),
                        'date': date,
                        'number': number,
                        'training_day': day_idx,
                        'session_id': f'{nick}/{date}/{number}',
                        'alf_dir': alf_dir,
                    })

    idx = pd.DataFrame(rows)
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        idx.to_parquet(cache_path)
    if verbose:
        print(f'index: {len(idx)} sessions, {idx["subject"].nunique()} subjects')
    return idx


# ---------------------------------------------------------------------------
# Tier 2: flat per-trial table
# ---------------------------------------------------------------------------

def load_trials(csv_subset=True, root=RAW_ROOT, cache=True, verbose=True):
    """Flat per-trial DataFrame across all indexed sessions ("individual trials").

    One row per trial; carries session metadata (subject, lab,
    institution_code, session_date, session_number, training_day, session_id,
    trial_idx) plus the behaviour columns and derived quantities
    (signed_contrast, abs_contrast, is_easy, correct, rt_response, rt_movement).

    Result is cached under ``data/cache/`` as a single parquet.
    """
    tag = 'csvsubset' if csv_subset else 'all'
    cache_path = os.path.join(CACHE_DIR, f'trials_{tag}_{TRIALS_SCHEMA}.parquet')
    if cache and os.path.exists(cache_path):
        if verbose:
            print(f'trials: loaded cache {os.path.relpath(cache_path)}')
        return pd.read_parquet(cache_path)

    idx = build_index(csv_subset=csv_subset, root=root, cache=cache, verbose=verbose)
    frames = []
    for k, r in enumerate(idx.itertuples(index=False)):
        t = _read_trials_alf(r.alf_dir)
        if t is None or len(t) == 0:
            continue
        t = t.reset_index(drop=True)
        t.insert(0, 'trial_idx', np.arange(len(t)))
        t.insert(0, 'training_day', r.training_day)
        t.insert(0, 'session_number', r.number)
        t.insert(0, 'session_date', r.date)
        t.insert(0, 'session_id', r.session_id)
        t.insert(0, 'institution_code', r.institution_code)
        t.insert(0, 'lab', r.lab)
        t.insert(0, 'subject', r.subject)
        frames.append(t)
        if verbose and (k + 1) % 1000 == 0:
            print(f'  ...{k + 1}/{len(idx)} sessions')

    if not frames:
        raise RuntimeError('No trials loaded; check RAW_ROOT and subject filter.')

    df = _enrich_trials(pd.concat(frames, ignore_index=True))
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_parquet(cache_path)
    if verbose:
        print(f'trials: {len(df):,} trials x {df.shape[1]} cols, '
              f'{df["subject"].nunique()} subjects')
    return df


# backwards-compatible alias
load_trials_raw = load_trials


# ---------------------------------------------------------------------------
# Tier 3: session object with wheel access
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """One recording session: per-trial behaviour + continuous wheel trace.

    Trial behaviour is loaded eagerly; wheel arrays are loaded lazily on first
    access and cached on the instance. Trial event times and wheel timestamps
    share the same clock, so ``trial_wheel`` slices the continuous trace by time.
    """
    subject: str
    lab: str
    institution_code: str
    date: str
    number: str
    training_day: int
    alf_dir: str
    trials: pd.DataFrame = None
    _wheel: tuple = field(default=None, repr=False)
    _wheel_moves: tuple = field(default=None, repr=False)
    _settings: dict = field(default=None, repr=False)

    @property
    def session_id(self):
        return f'{self.subject}/{self.date}/{self.number}'

    # --- rig settings ---
    @property
    def settings(self):
        """iblrig task settings dict for this session (lazy, cached; {} if absent)."""
        if self._settings is None:
            self._settings = _read_settings(self.alf_dir) or {}
        return self._settings

    @property
    def stim_gain(self):
        """Stimulus/wheel gain (deg visual angle per mm of wheel), constant per session.

        Read from the session's task settings ``STIM_GAIN``. Adaptive-gain
        training sessions record 8 deg/mm on the mouse's first session and
        4 deg/mm thereafter; biased/ephys sessions are always 4. Falls back to
        ``DEFAULT_STIM_GAIN`` if settings are unavailable.
        """
        g = self.settings.get('STIM_GAIN')
        return float(g) if g is not None else DEFAULT_STIM_GAIN

    @property
    def gain_scale(self):
        """Factor mapping raw wheel angle -> gain-normalized angle.

        ``stim_gain / DEFAULT_STIM_GAIN``: 1.0 on a standard gain-4 session, 2.0 on a
        gain-8 session (where only half the wheel angle reaches the boundary, so the
        trace is doubled to be comparable).
        """
        return self.stim_gain / DEFAULT_STIM_GAIN

    def threshold_rad(self, gain_normalize=True):
        """Wheel angle (rad) at the stimulus response boundary.

        Returned in the SAME units as ``wheel(gain_normalize=...)``, so always pair
        the two with the same flag:

        - ``gain_normalize=True``  (default): the reference gain-4 boundary, a constant
          ``STIM_ANGLE_DEG / DEFAULT_STIM_GAIN / WHEEL_RADIUS_MM`` = 0.282 rad. This is
          the boundary for the *normalized* trace, so it is the same for every session.
        - ``gain_normalize=False``: this session's physical boundary,
          ``STIM_ANGLE_DEG / stim_gain / WHEEL_RADIUS_MM`` (0.282 at gain 4, 0.141 at gain 8).
        """
        gain = DEFAULT_STIM_GAIN if gain_normalize else self.stim_gain
        return STIM_ANGLE_DEG / gain / WHEEL_RADIUS_MM

    # --- behaviour ---
    def load_trials(self):
        t = _read_trials_alf(self.alf_dir)
        if t is None:
            raise FileNotFoundError(f'no trials data in {self.alf_dir}')
        t = t.reset_index(drop=True)
        t.insert(0, 'trial_idx', np.arange(len(t)))   # for COMMON_COLS conformance
        self.trials = _enrich_trials(t)
        return self.trials

    # --- wheel ---
    def wheel(self, gain_normalize=True):
        """Return (timestamps[s], position[rad]) for the continuous wheel.

        With ``gain_normalize=True`` (default) the position is rescaled by
        ``gain_scale`` to the reference gain-4 convention, so traces are directly
        comparable across sessions (gain-8 sessions are doubled). Pair with
        ``threshold_rad(gain_normalize=True)``. Pass False for raw physical angle.
        """
        if self._wheel is None:
            ps, ts = _wheel_paths(self.alf_dir)
            if ts is None or ps is None:
                raise FileNotFoundError(f'no wheel data in {self.alf_dir}')
            self._wheel = (np.load(ts), np.load(ps))
        wt, wp = self._wheel
        return (wt, wp * self.gain_scale) if gain_normalize else (wt, wp)

    def has_wheel(self):
        return _find(self.alf_dir, '_ibl_wheel.position*.npy') is not None

    def wheel_moves(self):
        """Return (intervals[n,2], peakAmplitude[n]) from the IBL move detector."""
        if self._wheel_moves is None:
            iv = _find(self.alf_dir, '_ibl_wheelMoves.intervals*.npy')
            amp = _find(self.alf_dir, '_ibl_wheelMoves.peakAmplitude*.npy')
            self._wheel_moves = (
                np.load(iv) if iv else None,
                np.load(amp) if amp else None,
            )
        return self._wheel_moves

    def trial_wheel(self, i, align='stimOn_times', t_pre=0.5, t_post=2.0,
                    zero_position=True, gain_normalize=True):
        """Slice the continuous wheel trace around one trial's event.

        Returns (t_rel, pos) where ``t_rel`` is time relative to the alignment
        event and ``pos`` is wheel position (optionally baselined to 0 at event
        onset). ``gain_normalize`` is passed through to :meth:`wheel`, so the
        result is comparable to ``threshold_rad(gain_normalize=...)``.
        """
        if self.trials is None:
            self.load_trials()
        t0 = self.trials[align].iloc[i]
        if not np.isfinite(t0):
            return None
        wt, wp = self.wheel(gain_normalize=gain_normalize)
        m = (wt >= t0 - t_pre) & (wt <= t0 + t_post)
        t_rel = wt[m] - t0
        pos = wp[m].astype(float)
        if zero_position and len(pos):
            base = wp[wt <= t0]
            pos = pos - (base[-1] if len(base) else pos[0])
        return t_rel, pos


def get_session(subject=None, date=None, number=None, session_id=None,
                csv_subset=True, root=RAW_ROOT):
    """Build a :class:`Session` (trials loaded) for one session.

    Identify by (subject, date, number) or by ``session_id`` ('subject/date/number').
    """
    if session_id is not None:
        subject, date, number = session_id.split('/')
    idx = build_index(csv_subset=csv_subset, root=root, verbose=False)
    sel = idx[idx['subject'] == subject]
    if date is not None:
        sel = sel[sel['date'] == date]
    if number is not None:
        sel = sel[sel['number'] == number]
    if sel.empty:
        raise KeyError(f'session not found: {subject}/{date}/{number}')
    r = sel.iloc[0]
    s = Session(subject=r['subject'], lab=r['lab'],
                institution_code=r['institution_code'], date=r['date'],
                number=r['number'], training_day=int(r['training_day']),
                alf_dir=r['alf_dir'])
    s.load_trials()
    return s


def iter_sessions(subject=None, csv_subset=True, root=RAW_ROOT, load=False):
    """Yield :class:`Session` objects from the index (optionally one subject).

    With ``load=False`` (default) trials are NOT read until you call
    ``.load_trials()`` / ``.wheel()`` — cheap for walking the index.
    """
    idx = build_index(csv_subset=csv_subset, root=root, verbose=False)
    if subject is not None:
        idx = idx[idx['subject'] == subject]
    for r in idx.itertuples(index=False):
        s = Session(subject=r.subject, lab=r.lab,
                    institution_code=r.institution_code, date=r.date,
                    number=r.number, training_day=int(r.training_day),
                    alf_dir=r.alf_dir)
        if load:
            s.load_trials()
        yield s


def subject_sessions(subject, csv_subset=True, root=RAW_ROOT):
    """Return the list of :class:`Session` for one subject, ordered by training day.

    Ascending, matching the training-day structure used in the learning-curve
    analysis, so ``subject_sessions(subj)[-1]`` is the last training day. Trials are
    lazy (call ``.load_trials()`` / ``.wheel()`` to read). Sessions sharing a training
    day are ordered by date then acquisition number.
    """
    sessions = iter_sessions(subject=subject, csv_subset=csv_subset, root=root)
    return sorted(sessions, key=lambda s: (s.training_day, s.date, s.number))


# ---------------------------------------------------------------------------
# CSV-compatible reductions (for validation against the existing notebook)
# ---------------------------------------------------------------------------

def daily_aggregate(trials):
    """Collapse a per-trial frame to per (subject, training_day) summary.

    Columns: [subject, training_day, institution_code, lab, session_date,
              n_trials_date, performance, performance_easy]. Sessions sharing a
            date are pooled (as in the CSV). ``performance_easy`` reproduces the
            CSV column; ``performance`` is the all-trial proportion correct.
    """
    def _agg(g):
        easy = g['is_easy']
        return pd.Series({
            'institution_code': g['institution_code'].iloc[0],
            'lab': g['lab'].iloc[0],
            'session_date': g['session_date'].iloc[0],
            'n_trials_date': len(g),
            'performance': g['correct'].mean(),
            'performance_easy': g.loc[easy, 'correct'].mean() if easy.any() else np.nan,
        })

    return (trials.groupby(['subject', 'training_day'], sort=True)
                  .apply(_agg, include_groups=False)
                  .reset_index())


def coverage_report(csv_subset=True, root=RAW_ROOT, verbose=False):
    """Per-subject coverage of the raw mirror vs the CSV training history.

    The public openalyx release (tag 2021_Q1_IBL_et_al_Behaviour) omits the
    earliest training sessions for some animals, so for those the raw data
    starts mid-learning. This compares, per subject, the CSV session dates to
    the dates present in the raw index.

    Returns a DataFrame with columns:
      [subject, institution_code, csv_days, raw_covered_days, missed_early_days,
       perf_easy_at_raw_start, full_coverage]
    where ``full_coverage`` (missed_early_days == 0) flags animals whose raw
    data begins at training day 1 — the clean set for learning-curve analysis.
    """
    csv = pd.read_csv(CSV_PATH)
    idx = build_index(csv_subset=csv_subset, root=root, verbose=verbose)
    raw_first = idx.groupby('subject')['date'].min().to_dict()

    rows = []
    for subj, c in csv.groupby('subject_nickname'):
        rf = raw_first.get(subj)
        if rf is None:
            continue
        c = c.sort_values('session_date')
        covered = c[c['session_date'] >= rf]
        missed = int((c['session_date'] < rf).sum())
        rows.append({
            'subject': subj,
            'institution_code': c['institution_code'].iloc[0],
            'csv_days': len(c),
            'raw_covered_days': len(covered),
            'missed_early_days': missed,
            'perf_easy_at_raw_start': (covered['performance_easy'].iloc[0]
                                       if len(covered) else np.nan),
            'full_coverage': missed == 0,
        })
    return pd.DataFrame(rows).sort_values(['institution_code', 'subject']).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Wheel QC + end-to-end coverage (choice + wheel)
# ---------------------------------------------------------------------------

MIN_WHEEL_SAMPLES = 100        # a real session has far more; guards empty/stub files
_WHEEL_COVER_TOL = 1.0         # s: wheel may start/end slightly inside trial window


def wheel_qc_session(alf_dir, behaviour_window=None, min_samples=MIN_WHEEL_SAMPLES):
    """Lightweight QC of one session's processed wheel data.

    Uses memory-mapped reads (only length + endpoints), so it is cheap to run
    over every session. ``behaviour_window`` is the (stimOn_min, feedback_max)
    time the decisions span; if given, we check the wheel trace covers it.

    Returns a dict:
      has_wheel, n_samples, lengths_match, time_increasing, covers_behaviour,
      ok    -- usable (present, sane, covers behaviour; tolerates a small
               position/timestamp length mismatch, common in early sessions)
      clean -- ok AND position/timestamps lengths match exactly
    """
    ps, ts = _wheel_paths(alf_dir)
    out = {'has_wheel': bool(ts and ps), 'n_samples': 0, 'lengths_match': False,
           'time_increasing': False, 'covers_behaviour': None,
           'ok': False, 'clean': False}
    if not (ts and ps):
        return out
    try:
        wt = np.load(ts, mmap_mode='r')
        wp = np.load(ps, mmap_mode='r')
    except Exception:
        return out
    n_pos = len(wp)
    out['n_samples'] = int(n_pos)
    if wt.ndim == 2 and wt.shape[1] == 2:        # sparse [idx, time]: defines dense pos
        t0, t1 = float(wt[0, 1]), float(wt[-1, 1])
        out['lengths_match'] = True
    else:
        out['lengths_match'] = (len(wt) == n_pos)
        if len(wt) < 2:
            return out
        t0, t1 = float(wt[0]), float(wt[-1])
    out['time_increasing'] = (t1 > t0) and np.isfinite(t1) and t1 < 1e5   # < ~27h
    if behaviour_window is not None and np.isfinite(behaviour_window).all():
        out['covers_behaviour'] = bool(t0 <= behaviour_window[0] + _WHEEL_COVER_TOL and
                                       t1 >= behaviour_window[1] - _WHEEL_COVER_TOL)
    out['ok'] = bool(out['has_wheel'] and n_pos >= min_samples
                     and out['time_increasing']
                     and out['covers_behaviour'] is not False)
    out['clean'] = bool(out['ok'] and out['lengths_match'])
    return out


def coverage_qc_report(csv_subset=True, root=RAW_ROOT, verbose=True):
    """End-to-end coverage of the 140 CSV animals: learning choice AND wheel.

    For each subject compares the CSV training session-dates to the raw data:
      - a date has CHOICE data if a raw session that day yields a trials table;
      - a date has WHEEL data if such a session also passes ``wheel_qc_session``.
    "End-to-end" means every CSV training date is covered.

    Returns a per-subject DataFrame:
      [subject, institution_code, csv_days,
       choice_days, wheel_days,                 # CSV dates covered
       choice_frac, wheel_frac,                 # /csv_days
       end_to_end_choice, end_to_end_wheel]     # == csv_days (100%)
    Also prints a summary count of mice with end-to-end choice and choice+wheel.
    """
    csv = pd.read_csv(CSV_PATH)
    idx = build_index(csv_subset=csv_subset, root=root, cache=True, verbose=False)
    trials = load_trials(csv_subset=csv_subset, root=root, cache=True, verbose=verbose)

    # per-session decision window (stimOn -> feedback) + sessions that have trials
    win = (trials.groupby('session_id')
                 .agg(subject=('subject', 'first'), date=('session_date', 'first'),
                      t0=('stimOn_times', 'min'), t1=('feedback_times', 'max')))
    win = win.join(idx.set_index('session_id')['alf_dir'])

    # wheel QC per session that has trials
    if verbose:
        print(f'wheel QC over {len(win)} sessions...')
    qc_ok, qc_clean = {}, {}
    for sid, r in win.iterrows():
        qc = wheel_qc_session(r['alf_dir'], np.array([r['t0'], r['t1']]))
        qc_ok[sid] = qc['ok']
        qc_clean[sid] = qc['clean']
    win['wheel_ok'] = win.index.map(qc_ok)
    win['wheel_clean'] = win.index.map(qc_clean)

    # collapse to per (subject, date): choice present, wheel usable, wheel clean
    by_date = (win.groupby(['subject', 'date'])
                  .agg(wheel_ok=('wheel_ok', 'any'),
                       wheel_clean=('wheel_clean', 'any')).reset_index())
    choice_dates = by_date.groupby('subject')['date'].agg(set).to_dict()
    wheel_dates = (by_date[by_date['wheel_ok']].groupby('subject')['date']
                   .agg(set).to_dict())
    clean_dates = (by_date[by_date['wheel_clean']].groupby('subject')['date']
                   .agg(set).to_dict())

    rows = []
    for subj, c in csv.groupby('subject_nickname'):
        csv_dates = set(c['session_date'])
        nd = len(csv_dates)
        ch = len(csv_dates & choice_dates.get(subj, set()))
        wh = len(csv_dates & wheel_dates.get(subj, set()))
        cl = len(csv_dates & clean_dates.get(subj, set()))
        rows.append({
            'subject': subj,
            'institution_code': c['institution_code'].iloc[0],
            'csv_days': nd,
            'choice_days': ch,
            'wheel_days': wh,
            'wheel_clean_days': cl,
            'choice_frac': ch / nd,
            'wheel_frac': wh / nd,
            'end_to_end_choice': ch == nd,
            'end_to_end_wheel': wh == nd,
            'end_to_end_clean': cl == nd,
        })
    rep = pd.DataFrame(rows).sort_values(['institution_code', 'subject']).reset_index(drop=True)

    if verbose:
        n = len(rep)
        print(f'\n=== End-to-end coverage of {n} CSV animals ===')
        print(f'  end-to-end CHOICE (100% of CSV days):        '
              f'{rep.end_to_end_choice.sum():3d}/{n}')
        print(f'  end-to-end CHOICE+WHEEL (100% of CSV days):  '
              f'{rep.end_to_end_wheel.sum():3d}/{n}')
        print(f'  end-to-end CHOICE+WHEEL, lengths clean:      '
              f'{rep.end_to_end_clean.sum():3d}/{n}')
        for thr in (0.95, 0.90, 0.80):
            print(f'  >= {thr:.0%} choice+wheel coverage:              '
                  f'{(rep.wheel_frac >= thr).sum():3d}/{n}')
        print('\nper-lab end-to-end:')
        g = rep.groupby('institution_code').agg(
            n=('subject', 'size'),
            e2e_choice=('end_to_end_choice', 'sum'),
            e2e_both=('end_to_end_wheel', 'sum')).astype(int)
        print(g.to_string())
    return rep


def load_behavior_raw(csv_subset=True, root=RAW_ROOT, cache=True, verbose=True):
    """Drop-in replacement for utils.load_behavior_data using raw data.

    Returns (data, id_to_subject, column_names) where ``data`` is
    {institution_code: {subject_id: 2D array
        [training_day, n_trials_date, performance, performance_easy]}}.
    Subject ids are ``subject_NNN``; ``id_to_subject`` maps them back to
    nicknames (the raw analogue of subject_uuid).
    """
    trials = load_trials(csv_subset=csv_subset, root=root, cache=cache, verbose=verbose)
    agg = daily_aggregate(trials)

    nicks = sorted(agg['subject'].unique())
    nick_to_id = {n: f'subject_{i:03d}' for i, n in enumerate(nicks)}
    id_to_nick = {v: k for k, v in nick_to_id.items()}

    cols = ['training_day', 'n_trials_date', 'performance', 'performance_easy']
    data = {}
    for inst, inst_g in agg.groupby('institution_code'):
        data[inst] = {}
        for nick, sub_g in inst_g.groupby('subject'):
            sub_g = sub_g.sort_values('training_day')
            data[inst][nick_to_id[nick]] = sub_g[cols].values
    return data, id_to_nick, cols
