import numpy as np
import pandas as pd
from typing import Dict, Tuple
from dataclasses import dataclass, field
import os

# Generated with Claude Code
def load_behavior_data(csv_path=None) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, str], list]:
    """
    Load behavioral data from CSV into a simple nested dictionary structure.

    Args:
        csv_path: Path to the CSV file (defaults to 'behavior.csv' in same directory as this file)

    Returns:
        Tuple of:
        1. Dictionary structured as:
           {
               institution_code: {
                   subject_id: 2D array with columns [training_day, n_trials_date, performance, performance_easy]
               }
           }
        2. Mapping from subject_id back to original subject_uuid
        3. List of column names for the 2D array
    """
    if csv_path is None:
        # Default path to the CSV file
        csv_path = os.path.join(os.path.dirname(__file__), 'behavior.csv')

    df = pd.read_csv(csv_path)

    # Create mapping from UUID to simple ID
    unique_uuids = df['subject_uuid'].unique()
    uuid_to_id = {uuid: f"subject_{i:03d}" for i, uuid in enumerate(unique_uuids)}
    id_to_uuid = {v: k for k, v in uuid_to_id.items()}

    data = {}

    for institution_code, institution_group in df.groupby('institution_code'):
        data[institution_code] = {}

        for subject_uuid, subject_group in institution_group.groupby('subject_uuid'):
            # Sort by training_day
            subject_sorted = subject_group.sort_values('training_day')

            # Create 2D array with [training_day, n_trials_date, performance, performance_easy]
            subject_array = subject_sorted[['training_day', 'n_trials_date', 'performance', 'performance_easy']].values

            # Use simplified subject ID
            subject_id = uuid_to_id[subject_uuid]
            data[institution_code][subject_id] = subject_array

    column_names = ['training_day', 'n_trials_date', 'performance', 'performance_easy']

    return data, id_to_uuid, column_names


def block_average(sub_data):
    '''
    sub_data: a list of data matrices for each subject
    columns are [(0) training_day, (1) n_trials_date,
                (2) performance, (3) performance_easy]
    '''
    n_max = 0
    for sub in sub_data:
        n_max = max(n_max, sub.shape[0])

    n_trials = np.zeros(n_max)
    n_correct = np.zeros(n_max)
    n_avg = np.zeros(n_max)

    # combine data from all subjects, blocked by training day
    for sub in sub_data:
        n_trials[:sub.shape[0]] += sub[:, 1]
        n_correct[:sub.shape[0]] += sub[:, 1] * sub[:, 3]
        n_avg[:sub.shape[0]] += 1

    # calculate average performance and number of trials per training day
    p_correct = n_correct / n_trials
    n_trials = n_trials / n_avg
    return n_trials, p_correct


def interpolate_average(sub_data, m_max=35000, quantile=0.841):
    m_trials = []
    p_correct = []

    # individual p_correct as a function of m_trials
    for sub in sub_data:
        n_trials = sub[:, 1]
        p_correct_sub = sub[:, 3]

        m_trials.append(np.cumsum(n_trials))
        p_correct.append(p_correct_sub)

    # setup sample points
    n_days = []
    n_trials = 0
    for m in m_trials:
        n_days.append(np.sum(m <= m_max))
        n_trials += m[0]
    n_days = np.ceil(np.quantile(n_days, quantile))
    n_trials = np.ceil(n_trials / len(m_trials))

    # compute average function
    m_average = np.linspace(int(n_trials), m_max, int(n_days))

    # interpolate each subject's performance onto common grid
    p_interpolated = []
    for i, (m_sub, p_sub) in enumerate(zip(m_trials, p_correct)):
        # only interpolate within the range of this subject's data
        m_sub_max = m_sub[-1]
        mask = (m_average <= m_sub_max)

        if np.any(mask):
            p_interp = np.full(len(m_average), np.nan)
            p_interp[mask] = np.interp(m_average[mask], m_sub, p_sub)
            p_interpolated.append(p_interp)

    # average across subjects (ignoring NaN values)
    p_interpolated = np.array(p_interpolated)
    p_average = np.nanmean(p_interpolated, axis=0)

    return m_average, p_average


def find_threshold(performance, threshold=0.8, min_days=3):
    """
    Find the earliest training day where performance consistently stays above threshold.

    Args:
        performance: 1D array of performance values (one per training day)
        threshold: Performance threshold (default 0.8 for 80%)
        min_days: Number of consecutive days above threshold required (default 3)

    Returns:
        int or float: Training day index where learning criterion is met, or np.inf if never met
    """
    if len(performance) < min_days:
        return np.inf

    for i in range(len(performance) - min_days + 1):
        if np.all(performance[i:i + min_days] >= threshold):
            return i

    return np.inf


# ---------------------------------------------------------------------------
# Raw end-to-end-complete cohort
#
# Same nested-dict interface as load_behavior_data(), but sourced from the raw
# ALF dataset (via the raw backend) and restricted to animals whose raw data covers
# every CSV training day. block_average / interpolate_average / find_threshold
# all work unchanged on the result (column 3 is performance_easy, as before).
# ---------------------------------------------------------------------------

_COMPLETE_FLAG = {'choice': 'end_to_end_choice', 'wheel': 'end_to_end_wheel'}
_COVERAGE_CSV = os.path.join(os.path.dirname(__file__), 'cache', 'coverage_qc_report.csv')


def _import_raw():
    """Import the raw backend module whether utils was imported as
    ``data.ibl.utils`` (relative) or ``utils`` (data/ibl/ on sys.path)."""
    try:
        from . import raw
    except (ImportError, ValueError):
        import raw
    return raw




def complete_subjects(level='wheel', report_csv=None):
    """Subject nicknames whose raw data is end-to-end complete vs the CSV.

    Args:
        level: 'wheel'  -> full choice AND wheel coverage every CSV day (55 mice)
               'choice' -> full choice coverage every CSV day (66 mice)
        report_csv: path to a cached coverage_qc_report.csv; if missing it is
                    recomputed via raw.coverage_qc_report().

    Note: the 11 'choice'-only mice (66 vs 55) DO have wheel data at every training
    day; they just have a small chunk of wheel samples that fails strict wheel QC on a
    few days. Their naive->trained ends are wheel-complete, so 'choice' is usable as the
    wheel cohort for end-to-end (naive vs trained) analyses.

    Returns:
        set of subject_nickname strings.
    """
    report_csv = report_csv or _COVERAGE_CSV
    if os.path.exists(report_csv):
        rep = pd.read_csv(report_csv)
    else:
        rep = _import_raw().coverage_qc_report(verbose=False)
    return set(rep.loc[rep[_COMPLETE_FLAG[level]], 'subject'])


def load_behavior_complete(level='wheel', report_csv=None):
    """Load the end-to-end-complete raw cohort in load_behavior_data() format.

    Args:
        level: 'wheel' (55 mice, choice+wheel) or 'choice' (66 mice, choice).

    Returns:
        (data, id_to_subject, column_names) where ``data`` is
        {institution_code: {subject_id: 2D array
            [training_day, n_trials_date, performance, performance_easy]}},
        restricted to the complete subjects. ``id_to_subject`` maps the
        synthetic subject ids back to nicknames.
    """
    keep = complete_subjects(level, report_csv=report_csv)
    data, id2nick, cols = _import_raw().load_behavior_raw(verbose=False)

    out, id_out = {}, {}
    for inst, subs in data.items():
        kept = {sid: arr for sid, arr in subs.items() if id2nick.get(sid) in keep}
        if kept:
            out[inst] = kept
            id_out.update({sid: id2nick[sid] for sid in kept})
    return out, id_out, cols


# ---------------------------------------------------------------------------
# Shared per-animal interface
#
# A single handle for the complete cohort, keyed by real subject nickname, that
# both analyses build on: the learning-curve fit reads Animal.curve, the wheel /
# kinematic analysis reads Animal.sessions(). Replaces the old two-call pattern
# (complete_subjects + load_behavior_complete), which returned the SAME animals
# under two different keys (nickname vs synthetic subject_NNN).
# ---------------------------------------------------------------------------

@dataclass
class Animal:
    """One animal in the complete cohort, keyed by its real nickname.

    ``curve`` is the per-day aggregate (columns :attr:`COLUMNS`), used for
    learning-curve fitting; :meth:`sessions` lazily exposes the raw ALF sessions
    (per-trial behaviour + continuous wheel) for kinematic analysis. Both share
    the single, consistent ``subject`` key, so per-animal results from the two
    analyses can be joined later.
    """

    COLUMNS = ('training_day', 'n_trials_date', 'performance', 'performance_easy')

    subject: str                 # real nickname; the shared key everywhere
    institution: str             # institution_code (e.g. 'Lab 5')
    curve: np.ndarray            # per-day aggregate, columns = COLUMNS
    uuid: str = None             # subject_uuid: the anonymized paper id
    fit: dict = None             # learning-curve params, set by fit_curve()
    _sessions: list = field(default=None, repr=False)

    @property
    def cum_trials(self):
        """Cumulative trial count at the end of each training day."""
        return np.cumsum(self.curve[:, 1])

    @property
    def error_easy(self):
        """Per-day easy-trial error rate (1 - performance_easy)."""
        return 1.0 - self.curve[:, 3]

    def trials_to_criterion(self, threshold=0.84):
        """Cumulative trials to first reach ``threshold`` easy-trial performance.

        First-order (per-day) learning-speed estimate from the aggregate curve: the
        cumulative trial count at the first training day whose performance_easy
        reaches ``threshold``; ``np.inf`` if the animal never reaches it.
        """
        hit = np.flatnonzero(self.curve[:, 3] >= threshold)
        return float(self.cum_trials[hit[0]]) if len(hit) else np.inf

    @property
    def alpha(self):
        """Fitted learning-curve exponent (None until fit_curve() is called)."""
        return self.fit.get('alpha') if self.fit else None

    @property
    def phi(self):
        """Fitted complexity scale phi (None until fit_curve(); PACBayes only)."""
        return self.fit.get('phi') if self.fit else None

    @property
    def r2(self):
        """Goodness of fit of the recorded learning-curve fit."""
        return self.fit.get('r2') if self.fit else None

    def fit_curve(self, model, cap=0.5):
        """Fit a learning-curve model and record its parameters on this Animal."""
        self.fit = model(cap=cap).fit(self.cum_trials, self.error_easy).params
        return self.fit

    def sessions(self):
        """Raw Session objects for this animal, ordered by training day (lazy)."""
        if self._sessions is None:
            self._sessions = _import_raw().subject_sessions(self.subject)
        return self._sessions


def nickname_to_uuid(csv_path=None):
    """Map each subject_nickname -> subject_uuid (the anonymized paper id).

    1:1 over the 140 CSV animals; use to report cohort results under the
    anonymized identifier instead of the real nickname.
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), 'behavior.csv')
    df = pd.read_csv(csv_path, usecols=['subject_nickname', 'subject_uuid'])
    return dict(zip(df['subject_nickname'], df['subject_uuid']))


def load_cohort(level='choice', report_csv=None):
    """Load the end-to-end-complete cohort as ``{nickname: Animal}``.

    Single shared entry point for both analyses (learning curve via
    ``Animal.curve``, wheel via ``Animal.sessions()``), replacing the old
    complete_subjects + load_behavior_complete pair. Each Animal also carries its
    anonymized ``uuid`` (see :func:`nickname_to_uuid`).

    Args:
        level: 'choice' (66 mice, choice-complete) or 'wheel' (55 mice,
               choice+wheel complete); see :func:`complete_subjects`.

    Returns:
        dict {subject_nickname: Animal}, ordered by institution then nickname.
    """
    data, id2nick, _ = load_behavior_complete(level, report_csv=report_csv)
    nick2uuid = nickname_to_uuid()
    cohort = {}
    for inst, subs in data.items():
        for sid, arr in subs.items():
            nick = id2nick[sid]
            cohort[nick] = Animal(subject=nick, institution=inst, curve=arr,
                                  uuid=nick2uuid.get(nick))
    return cohort