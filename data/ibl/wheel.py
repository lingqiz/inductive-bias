"""Per-trial wheel kinematics for the IBL dataset.

Reusable building blocks for the wheel analysis, operating on the ``Session``
objects from :mod:`data.ibl.raw` (per-trial behaviour + the continuous,
gain-normalized wheel trace). Three layers:

  filter_trials / select_trials     -- pick the usable trials of a session
  trial_velocity / movement_onset   -- kinematics from the continuous trace
  measure_trial                     -- one trial -> a TrialMeasurement

WHEEL SAMPLING. The wheel is EVENT-sampled: samples are emitted only while it
moves. During movement the native rate is ~260-900 Hz (median inter-sample
1-4 ms, varying by session), with multi-second gaps while the mouse is still, so
there is no single native frequency -- we interpolate onto fixed grids. ``DT``
(velocity grid) sits well above the bandwidth left after ``SIGMA`` smoothing, so
nothing is aliased.

CHOICE-SIGN CONVENTION (set in raw.py, CHOICE_SIGN = -1) -- one sign means the
same thing everywhere:

    choice = +1  <->  stimulus on the RIGHT  <->  wheel driven to +threshold_rad
    correct      <->  choice == stim_side          (stim_side = sign(signed_contrast))

So a trial's turn drives the wheel toward ``choice * threshold_rad``; folding a
trace by ``response_side`` (= choice) therefore points every turn positive.
"""
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d

# --- kinematics constants ---
DT, SIGMA = 0.005, 0.02      # velocity grid step (s), gaussian smoothing sigma (s)
V_THR = 0.1                  # movement-onset speed threshold (rad/s)

# MIN_RT: ~550 trials cohort-wide have a response BEFORE stimOn (RT < 0), which break the
# velocity estimate. A 0.1 s floor costs 0.45% of trials, drops evenly across training, and
# removes ~59%-correct (near-chance) anticipatory responses. Do NOT go above ~0.15: trained
# median RT is ~0.3 s, so a 0.2-0.3 s cut would delete genuine fast decisions (75-88% correct)
# from TRAINED sessions.
MIN_RT = 0.1                 # minimum reaction time kept (s)

# --- trace (measurement) constants ---
T_TAIL = 0.5                 # window kept past the response (s)
N_U = 100                    # points on the normalized-time trace


# ---------------------------------------------------------------------------
# Trial selection
# ---------------------------------------------------------------------------

def filter_trials(session, easy_only=True, min_rt=MIN_RT):
    """The usable trials of a session -- no stimulus/response conditioning.

    Usable = responded (choice != 0), valid stimOn/response times, RT > min_rt.
    """
    trials = session.load_trials()
    rt = trials['response_times'] - trials['stimOn_times']

    keep = ((trials['choice'] != 0)                     # responded (not a timeout)
            & np.isfinite(trials['stimOn_times'])
            & np.isfinite(trials['response_times'])
            & (rt > min_rt))
    if easy_only:
        keep &= trials['is_easy']
    return trials[keep]


def select_trials(session, stim_side=None, resp_side=None, easy_only=True, min_rt=MIN_RT):
    """Usable trials restricted to a stimulus side and/or a response side.

    stim_side / resp_side are +1 / -1; None means do not condition on that.
    """
    trials = filter_trials(session, easy_only=easy_only, min_rt=min_rt)
    if stim_side is not None:
        trials = trials[np.sign(trials['signed_contrast']) == stim_side]
    if resp_side is not None:
        trials = trials[trials['choice'] == resp_side]
    return trials


# ---------------------------------------------------------------------------
# Kinematics from the continuous trace
# ---------------------------------------------------------------------------

def trial_velocity(session, row, wheel=None, dt=DT, sigma=SIGMA):
    """Position and smoothed speed on a regular grid, stim onset -> response.

    Pass ``wheel`` (the session.wheel() tuple) to avoid re-fetching it per trial.
    """
    wheel_t, wheel_pos = session.wheel() if wheel is None else wheel
    on = row['stimOn_times']
    rt = row['response_times'] - on
    t_grid = np.arange(0.0, rt, dt)
    pos = np.interp(on + t_grid, wheel_t, wheel_pos) - np.interp(on, wheel_t, wheel_pos)
    vel = np.gradient(gaussian_filter1d(pos, sigma / dt), dt)
    return t_grid, pos, vel


def movement_onset(session, row, wheel=None, v_thr=V_THR, dt=DT, sigma=SIGMA):
    """Movement onset: FIRST crossing of |v| >= v_thr after stim onset.

    Returned relative to stim onset (s); NaN if the speed never reaches v_thr.
    """
    t_grid, _, vel = trial_velocity(session, row, wheel=wheel, dt=dt, sigma=sigma)
    above = np.flatnonzero(np.abs(vel) >= v_thr)
    return t_grid[above[0]] if len(above) else np.nan


# ---------------------------------------------------------------------------
# Per-trial measurement
# ---------------------------------------------------------------------------

@dataclass
class TrialMeasurement:
    """The measurements extracted from one trial's wheel trace.

    onset_time     movement onset - stim onset            (s)
    duration       response - movement onset              (s)
    response_side  choice, +1 (right) / -1 (left)         -- the turn's direction
    stim_side      sign(signed_contrast), +1 / -1         -- the conditioning variable
    correct        0 / 1                                  -- response_side == stim_side
    trace          normalized-time wheel trace (N_U pts), FOLDED to positive

    The trace spans [movement onset, response + T_TAIL] rescaled to u in [0, 1],
    then multiplied by ``response_side`` so every turn points the same way -- it
    captures turn SHAPE in a canonical orientation. Recover the signed trace as
    ``trace * response_side``.
    """
    onset_time: float
    duration: float
    response_side: int
    stim_side: int
    correct: int
    trace: np.ndarray


def measure_trial(session, row, wheel=None, t_tail=T_TAIL, v_thr=V_THR):
    """Convert one trial's wheel trace into a :class:`TrialMeasurement`.

    ``row`` is a row of ``session.load_trials()`` (or of :func:`filter_trials`).
    Pass ``wheel`` (the session.wheel() tuple) to avoid re-fetching it per trial.

    The two analysis knobs are exposed: ``t_tail`` (trace window kept past the
    response) and ``v_thr`` (movement-onset speed threshold). The velocity grid
    (``DT`` / ``SIGMA``) and trace resolution (``N_U``) are fixed module constants.

    Returns None when the trial has no detectable movement onset (speed never
    reaches v_thr) or a non-positive movement duration -- i.e. nothing to measure.
    """
    wheel_t, wheel_pos = session.wheel() if wheel is None else wheel
    on, resp = row['stimOn_times'], row['response_times']

    onset_rel = movement_onset(session, row, wheel=(wheel_t, wheel_pos), v_thr=v_thr)
    if not np.isfinite(onset_rel):
        return None
    onset = on + onset_rel
    duration = resp - onset
    if duration <= 0:
        return None

    resp_side = int(row['choice'])
    span = duration + t_tail                              # onset -> response + tail
    base = np.interp(on, wheel_t, wheel_pos)              # zero the trace at stim onset
    u = np.linspace(0.0, 1.0, N_U)
    curve = np.interp(onset + u * span, wheel_t, wheel_pos) - base
    
    return TrialMeasurement(
        onset_time=float(onset_rel),
        duration=float(duration),
        response_side=resp_side,
        stim_side=int(np.sign(row['signed_contrast'])),
        correct=int(row['correct']),
        trace=curve * resp_side,                         # fold to positive
    )


# ---------------------------------------------------------------------------
# Collecting a stretch of an animal's trials
# ---------------------------------------------------------------------------

@dataclass
class TrialSet:
    """A stretch of an animal's trials, measured and stacked into aligned arrays.

    All arrays have length n (the number of measured trials); ``trace`` is
    (n, N_U), folded positive (see :class:`TrialMeasurement`). ``training_day``
    records each trial's session day. Trials are in chronological order.
    """
    onset: np.ndarray
    duration: np.ndarray
    response_side: np.ndarray
    stim_side: np.ndarray
    correct: np.ndarray
    trace: np.ndarray
    training_day: np.ndarray

    def __len__(self):
        return len(self.onset)

    @property
    def accuracy(self):
        """Proportion correct over the collected trials."""
        return float(self.correct.mean())

    @classmethod
    def concat(cls, trialsets):
        """Pool several TrialSets into one (e.g. trials pooled across animals).

        Concatenates every field; ``trace`` is row-stacked. Empty sets are skipped.
        Use to build a combined "super animal" from per-animal collections.
        """
        ts = [t for t in trialsets if len(t)]
        return cls(
            onset=np.concatenate([t.onset for t in ts]),
            duration=np.concatenate([t.duration for t in ts]),
            response_side=np.concatenate([t.response_side for t in ts]),
            stim_side=np.concatenate([t.stim_side for t in ts]),
            correct=np.concatenate([t.correct for t in ts]),
            trace=np.vstack([t.trace for t in ts]),
            training_day=np.concatenate([t.training_day for t in ts]),
        )


def collect_trials(animal, n_trials=2000, from_end=False, easy_only=True,
                   t_tail=T_TAIL, v_thr=V_THR):
    """Measure the first (or last) ``n_trials`` usable EASY trials of an animal.

    Walks the animal's sessions in training order, measuring each usable trial
    with :func:`measure_trial` and crossing session boundaries until ``n_trials``
    are collected. ``from_end=True`` walks backward from the last session (the
    trained end). Returns a :class:`TrialSet` in chronological order either way.

    ``t_tail`` and ``v_thr`` are forwarded to :func:`measure_trial` (the two knobs).
    """
    sessions = animal.sessions()
    order = list(reversed(sessions)) if from_end else list(sessions)
    ms, days = [], []
    for s in order:
        if not s.has_wheel():
            continue
        sel = filter_trials(s, easy_only=easy_only)
        if from_end:
            sel = sel.iloc[::-1]                          # walk back within the session too
        wheel = s.wheel()
        wt, wp = wheel
        if wt.ndim != 1 or len(wt) != len(wp):   # malformed wheel (timestamps/position length
            continue                             # mismatch, e.g. KS016 day 49); skip the session
        for _, row in sel.iterrows():
            m = measure_trial(s, row, wheel=wheel, t_tail=t_tail, v_thr=v_thr)
            if m is None:
                continue
            ms.append(m); days.append(s.training_day)
            if len(ms) >= n_trials:
                break
        if len(ms) >= n_trials:
            break

    if from_end:                                          # restore chronological order
        ms = ms[::-1]; days = days[::-1]

    return TrialSet(
        onset=np.array([m.onset_time for m in ms]),
        duration=np.array([m.duration for m in ms]),
        response_side=np.array([m.response_side for m in ms]),
        stim_side=np.array([m.stim_side for m in ms]),
        correct=np.array([m.correct for m in ms]),
        trace=np.stack([m.trace for m in ms]),
        training_day=np.array(days),
    )
