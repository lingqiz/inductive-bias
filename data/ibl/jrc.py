"""Backend for the in-house JRC raw iblrig dataset (parallel to raw.py for ALF).

The JRC dataset lives at ``JRC_ROOT`` in a non-standard, RAW iblrig layout::

    <mouse>/<session>/<run>/raw_task_data_00/
        _iblrig_taskData.raw.jsonable        one JSON object per trial
        _iblrig_encoderPositions.raw.ssv     [time_us, position_clicks]
        _iblrig_encoderEvents.raw.ssv        [time_us, code]  (8=stim_on,1=stim_off,3=close_loop)
        _iblrig_encoderTrialInfo.raw.ssv     per-trial sync info (+datetime col)
        _iblrig_taskSettings.raw.json        STIM_GAIN, STIM_POSITIONS, ...
    <mouse>/<session>/<run>/raw_video_data/
        _iblrig_leftCamera.raw.avi           video (path exposed; not decoded)
        _iblrig_leftCamera.frameData.bin

Sessions named ``0hab*`` (habituation) are ignored; only ``1t*`` (training) are
used. <run> is normally ``001`` but a few sessions use ``002`` (we pick the run
whose taskData is present/non-empty).

TWO CLOCKS: the bpod clock (seconds, in each trial's ``behavior_data``) and the
rotary-encoder clock (microseconds, in the .ssv files) have different origins.
Wheel positions AND stim events are both on the ENCODER clock, so we keep them
together: ``stimOn_times``/``feedback_times`` come from the encoder events (s) on
the same clock as ``wheel()``; trial OUTCOMES (choice/contrast/correct) come from
taskData by trial order. This mirrors analysis_jrcIBL.ipynb and the MATLAB loaders.

Exposes the same Session API as raw.py (see data.ibl.sessions.SessionLike), with
a normalized common trial schema plus JRC-specific extras.

Generated with Claude Code.
"""
import os
import glob
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

JRC_ROOT = '/groups/zhang/home/zhangl5/IBL_JRC/data'
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
MICE = ['m01', 'm03', 'm04', 'm05', 'm06', 'm07', 'm08', 'm09', 'm10', 'm12']

# Reward/feedback-type group per mouse. NOT present in the raw iblrig files —
# harvested from JRC_IBLData.mat (`mouse.type`), the authoritative annotation.
MOUSE_TYPE = {
    'm01': 'jackpot', 'm03': 'jackpot', 'm04': 'jackpot', 'm05': 'jackpot',
    'm08': 'jackpot', 'm09': 'jackpot',
    'm06': 'standard', 'm07': 'standard', 'm10': 'standard', 'm12': 'standard',
}

MUS_TO_SEC = 1e-6
CLICK_TO_RAD = 2 * np.pi / 1024
WHEEL_RADIUS_MM = 31
EVENT = {'stim_on': 8, 'stim_off': 1, 'close_loop': 3}
EASY_CONTRASTS = (0.5, 1.0)
MIN_WHEEL_SAMPLES = 100

# taskData scalar fields in JSON field order = the notebook's INDEX (cols 0..18).
# Excludes the trailing `signed_contrast` (often null, recomputed) and the nested
# `behavior_data` (bpod-clock timestamps; loaded only for verification).
_TASK_SCALARS = [
    'contrast', 'stim_probability_left', 'position', 'quiescent_period',
    'reward_amount', 'reward_valve_time', 'stim_angle', 'stim_freq', 'stim_gain',
    'stim_phase', 'stim_reverse', 'stim_sigma', 'trial_num', 'pause_duration',
    'response_side', 'response_time', 'trial_correct', 'training_phase', 'debias_trial',
]

# Extra (JRC-specific) columns carried alongside the normalized common schema.
_EXTRA_COLS = [
    'closeloop_times', 'stim_side', 'position_deg', 'stim_gain', 'training_phase',
    'debias_trial', 'response_time', 'reward_amount', 'stim_probability_left',
]

_JSONABLE = '_iblrig_taskData.raw.jsonable'
_ENC_POS = '_iblrig_encoderPositions.raw.ssv'
_ENC_EVT = '_iblrig_encoderEvents.raw.ssv'


# ---------------------------------------------------------------------------
# Low-level readers (pure)
# ---------------------------------------------------------------------------

def _read_ssv(path, usecols):
    """Read whitespace-delimited numeric columns from an .ssv file (2D).

    Fast path via np.loadtxt; falls back to a tolerant line parser that drops
    malformed short rows (a few sessions have a truncated final line where
    acquisition was cut off mid-write, e.g. m01/1t22).
    """
    try:
        return np.loadtxt(path, usecols=usecols, ndmin=2)
    except ValueError:
        ncol = max(usecols) + 1
        rows = [[float(p[c]) for c in usecols]
                for p in (ln.split() for ln in open(path)) if len(p) >= ncol]
        return np.asarray(rows, dtype=float).reshape(-1, len(usecols))


def _read_taskdata(task_dir):
    """Parse the per-trial jsonable. Returns (scalar DataFrame, list[behavior_data])."""
    path = os.path.join(task_dir, _JSONABLE)
    rows, bdata = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append({k: d.get(k) for k in _TASK_SCALARS})
            bdata.append(d.get('behavior_data'))
    return pd.DataFrame(rows, columns=_TASK_SCALARS), bdata


def _read_encoder_events(task_dir):
    """Return (onset_s, offset_s, closeloop_s) from encoderEvents on the encoder clock."""
    arr = _read_ssv(os.path.join(task_dir, _ENC_EVT), usecols=(0, 1))
    t = arr[:, 0] * MUS_TO_SEC
    code = arr[:, 1].astype(int)
    return (t[code == EVENT['stim_on']],
            t[code == EVENT['stim_off']],
            t[code == EVENT['close_loop']])


def _read_wheel(task_dir):
    """Return (t_s, pos_rad) for the continuous wheel, sorted by time."""
    arr = _read_ssv(os.path.join(task_dir, _ENC_POS), usecols=(0, 1))
    t = arr[:, 0] * MUS_TO_SEC
    pos = arr[:, 1] * CLICK_TO_RAD
    order = np.argsort(t, kind='stable')
    return t[order], pos[order]


def _align(arr, n):
    """Length-``n`` view of ``arr`` (truncate/NaN-pad) for trial<->event alignment."""
    out = np.full(n, np.nan)
    k = min(n, len(arr))
    out[:k] = arr[:k]
    return out


def _normalize_trials(task, onset_s, offset_s, closeloop_s):
    """Build the normalized common schema (+ JRC extras) from raw taskData + events.

    See data.ibl.sessions.COMMON_COLS for the guaranteed columns and the sign
    convention. Event times are NaN-filled where trial/event counts diverge.
    """
    n = len(task)
    contrast = task['contrast'].to_numpy(dtype=float)
    position = task['position'].to_numpy(dtype=float)
    abs_c = np.abs(contrast)
    stim_side = -np.sign(position)                       # +1 = stimulus on the right
    correct = task['trial_correct'].astype(bool).to_numpy()

    df = pd.DataFrame({
        'trial_idx': np.arange(n),
        'choice': task['response_side'].to_numpy(dtype=float),
        'signed_contrast': stim_side * abs_c,            # + = stim on right (IBL convention)
        'abs_contrast': abs_c,
        'feedbackType': np.where(correct, 1, -1).astype(int),
        'correct': correct.astype(int),
        'stimOn_times': _align(onset_s, n),
        'feedback_times': _align(offset_s, n),
        # --- JRC extras ---
        'closeloop_times': _align(closeloop_s, n),
        'stim_side': stim_side,
        'position_deg': position,
        'stim_gain': task['stim_gain'].to_numpy(dtype=float),
        'training_phase': task['training_phase'].to_numpy(),
        'debias_trial': task['debias_trial'].to_numpy(),
        'response_time': task['response_time'].to_numpy(dtype=float),
        'reward_amount': task['reward_amount'].to_numpy(dtype=float),
        'stim_probability_left': task['stim_probability_left'].to_numpy(dtype=float),
    })
    df['is_easy'] = df['abs_contrast'].round(4).isin(EASY_CONTRASTS)
    return df


# ---------------------------------------------------------------------------
# Tier 1: session index
# ---------------------------------------------------------------------------

def build_index(root=JRC_ROOT, cache=True, verbose=True):
    """One row per training (1t*) session: metadata + path to raw_task_data_00.

    Columns: [subject, mnum, session_name, training_day, session_id, task_dir,
              session_root, institution_code]. ``training_day`` is the 1tNN
            ordinal parsed from the session name, so a session with no usable
            data (e.g. m01/1t03) leaves a gap rather than shifting later days.
            Habituation (0hab*) sessions are excluded. The <run> dir is the first
            (001, then 002, ...) whose taskData jsonable is present and non-empty.
    """
    cache_path = os.path.join(CACHE_DIR, 'jrc_session_index.parquet')
    if cache and os.path.exists(cache_path):
        if verbose:
            print(f'jrc index: loaded cache {os.path.relpath(cache_path)}')
        return pd.read_parquet(cache_path)

    rows = []
    for mouse in MICE:
        mdir = os.path.join(root, mouse)
        if not os.path.isdir(mdir):
            continue
        sessions = sorted(d for d in os.listdir(mdir) if d.startswith('1t'))
        for sname in sessions:
            # training_day = the 1tNN ordinal, so a missing session (e.g. m01/1t03)
            # leaves a gap rather than shifting later days.
            day_idx = int(sname[2:])
            sdir = os.path.join(mdir, sname)
            # Pick the run (001/002/...) with the largest taskData jsonable among
            # runs that also have non-empty encoder data. A restarted session
            # leaves a small stub in 001 with empty encoder files (e.g. m01/1t22,
            # m04/1t11) and the complete data in 002 -- this selects 002.
            cands = []
            for run in sorted(os.listdir(sdir)):
                cand = os.path.join(sdir, run, 'raw_task_data_00')
                jf = os.path.join(cand, _JSONABLE)
                ef = os.path.join(cand, _ENC_POS)
                if (os.path.isfile(jf) and os.path.getsize(jf) > 0
                        and os.path.isfile(ef) and os.path.getsize(ef) > 0):
                    cands.append((os.path.getsize(jf), run, cand))
            if not cands:
                continue
            _, run, task_dir = max(cands)
            session_root = os.path.join(sdir, run)
            rows.append({
                'subject': mouse, 'mnum': int(mouse[1:]), 'session_name': sname,
                'training_day': day_idx, 'session_id': f'{mouse}/{sname}',
                'task_dir': task_dir, 'session_root': session_root,
                'institution_code': 'JRC', 'feedback_type': MOUSE_TYPE.get(mouse),
            })

    idx = pd.DataFrame(rows)
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        idx.to_parquet(cache_path)
    if verbose:
        print(f'jrc index: {len(idx)} sessions, {idx["subject"].nunique()} mice')
    return idx


# ---------------------------------------------------------------------------
# Tier 2: flat per-trial table
# ---------------------------------------------------------------------------

def load_trials(root=JRC_ROOT, cache=True, verbose=True):
    """Flat per-trial DataFrame across all training sessions (mirrors raw.load_trials).

    Prepends session metadata (subject, institution_code, session_id, training_day)
    to the normalized schema. Cached to data/ibl/cache/jrc_trials.parquet.
    """
    cache_path = os.path.join(CACHE_DIR, 'jrc_trials.parquet')
    if cache and os.path.exists(cache_path):
        if verbose:
            print(f'jrc trials: loaded cache {os.path.relpath(cache_path)}')
        return pd.read_parquet(cache_path)

    idx = build_index(root=root, cache=cache, verbose=verbose)
    frames = []
    for k, r in enumerate(idx.itertuples(index=False)):
        task, _ = _read_taskdata(r.task_dir)
        onset, offset, closeloop = _read_encoder_events(r.task_dir)
        df = _normalize_trials(task, onset, offset, closeloop)
        df.insert(0, 'feedback_type', r.feedback_type)
        df.insert(0, 'training_day', r.training_day)
        df.insert(0, 'session_id', r.session_id)
        df.insert(0, 'institution_code', 'JRC')
        df.insert(0, 'subject', r.subject)
        frames.append(df)
        if verbose and (k + 1) % 50 == 0:
            print(f'  ...{k + 1}/{len(idx)} sessions')

    if not frames:
        raise RuntimeError('No JRC trials loaded; check JRC_ROOT.')
    out = pd.concat(frames, ignore_index=True)
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        out.to_parquet(cache_path)
    if verbose:
        print(f'jrc trials: {len(out):,} trials x {out.shape[1]} cols, '
              f'{out["subject"].nunique()} mice')
    return out


# ---------------------------------------------------------------------------
# Tier 3: session object (same API as raw.Session / sessions.SessionLike)
# ---------------------------------------------------------------------------

@dataclass
class JRCSession:
    subject: str
    mnum: int
    session_name: str
    training_day: int
    task_dir: str
    session_root: str
    feedback_type: str = None
    trials: pd.DataFrame = None
    _wheel: tuple = field(default=None, repr=False)
    _events: tuple = field(default=None, repr=False)

    @property
    def session_id(self):
        return f'{self.subject}/{self.session_name}'

    def _load_events(self):
        if self._events is None:
            self._events = _read_encoder_events(self.task_dir)
        return self._events

    def load_trials(self):
        task, _ = _read_taskdata(self.task_dir)
        onset, offset, closeloop = self._load_events()
        t = _normalize_trials(task, onset, offset, closeloop)
        t.insert(0, 'feedback_type', self.feedback_type)
        t.insert(0, 'training_day', self.training_day)
        t.insert(0, 'session_id', self.session_id)
        t.insert(0, 'institution_code', 'JRC')
        t.insert(0, 'subject', self.subject)
        self.trials = t
        return self.trials

    def wheel(self):
        """Return (timestamps[s], position[rad]) on the encoder clock."""
        if self._wheel is None:
            self._wheel = _read_wheel(self.task_dir)
        return self._wheel

    def has_wheel(self):
        return os.path.isfile(os.path.join(self.task_dir, _ENC_POS))

    def wheel_moves(self):
        """JRC has no IBL move detector; returns None (graceful degrade)."""
        return None

    def trial_wheel(self, i, align='stimOn_times', t_pre=0.5, t_post=2.0,
                    zero_position=True):
        if self.trials is None:
            self.load_trials()
        try:
            from . import sessions
        except (ImportError, ValueError):
            import sessions
        return sessions.slice_wheel(self.trials, self.wheel(), i, align,
                                    t_pre, t_post, zero_position)

    # --- video (path-only; no decoding/frame-time parsing this stage) ---
    @property
    def video_path(self):
        p = os.path.join(self.session_root, 'raw_video_data', '_iblrig_leftCamera.raw.avi')
        return p if os.path.isfile(p) else None

    @property
    def frame_data_path(self):
        p = os.path.join(self.session_root, 'raw_video_data', '_iblrig_leftCamera.frameData.bin')
        return p if os.path.isfile(p) else None


def _session_from_row(r):
    return JRCSession(subject=r['subject'], mnum=int(r['mnum']),
                      session_name=r['session_name'], training_day=int(r['training_day']),
                      task_dir=r['task_dir'], session_root=r['session_root'],
                      feedback_type=r.get('feedback_type'))


def get_session(subject=None, session_name=None, session_id=None, root=JRC_ROOT):
    """Build a :class:`JRCSession` (trials loaded). Identify by (subject,
    session_name) or ``session_id='subject/session_name'``."""
    if session_id is not None:
        subject, session_name = session_id.split('/')
    idx = build_index(root=root, verbose=False)
    sel = idx[idx['subject'] == subject]
    if session_name is not None:
        sel = sel[sel['session_name'] == session_name]
    if sel.empty:
        raise KeyError(f'JRC session not found: {subject}/{session_name}')
    s = _session_from_row(sel.iloc[0])
    s.load_trials()
    return s


def iter_sessions(subject=None, root=JRC_ROOT, load=False):
    """Yield :class:`JRCSession` objects (optionally one subject)."""
    idx = build_index(root=root, verbose=False)
    if subject is not None:
        idx = idx[idx['subject'] == subject]
    for r in idx.itertuples(index=False):
        s = _session_from_row(r._asdict())
        if load:
            s.load_trials()
        yield s


# ---------------------------------------------------------------------------
# Wheel QC (same return contract as raw.wheel_qc_session, + JRC-specific flags)
# ---------------------------------------------------------------------------

def wheel_qc_session(task_dir, behaviour_window=None, min_samples=MIN_WHEEL_SAMPLES):
    """Lightweight QC of one JRC session's encoder wheel data.

    Returns: has_wheel, n_samples, monotonic_raw, time_increasing,
    covers_behaviour, events_ok, ok, clean.
    """
    pos_file = os.path.join(task_dir, _ENC_POS)
    out = {'has_wheel': os.path.isfile(pos_file), 'n_samples': 0,
           'monotonic_raw': False, 'time_increasing': False,
           'covers_behaviour': None, 'events_ok': None, 'ok': False, 'clean': False}
    if not out['has_wheel']:
        return out
    arr = _read_ssv(pos_file, usecols=(0, 1))
    n = len(arr)
    out['n_samples'] = int(n)
    if n < 2:
        return out
    traw = arr[:, 0]
    out['monotonic_raw'] = bool(np.all(np.diff(traw) >= 0))
    t = np.sort(traw) * MUS_TO_SEC
    out['time_increasing'] = bool(t[-1] > t[0])
    if behaviour_window is not None and np.isfinite(behaviour_window).all():
        out['covers_behaviour'] = bool(t[0] <= behaviour_window[0] + 1.0 and
                                       t[-1] >= behaviour_window[1] - 1.0)
    evt = os.path.join(task_dir, _ENC_EVT)
    if os.path.isfile(evt):
        code = _read_ssv(evt, usecols=(0, 1))[:, 1].astype(int)
        n8 = int((code == EVENT['stim_on']).sum())
        n1 = int((code == EVENT['stim_off']).sum())
        n3 = int((code == EVENT['close_loop']).sum())
        out['events_ok'] = (n8 == n1 == n3)
    out['ok'] = bool(out['has_wheel'] and n >= min_samples and out['time_increasing']
                     and out['covers_behaviour'] is not False)
    out['clean'] = bool(out['ok'] and out['events_ok'] is not False)
    return out
