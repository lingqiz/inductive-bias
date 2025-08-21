import numpy as np
import pandas as pd
from typing import Dict, Tuple
import os

# Generated with Claude Code
def load_behavior_data(csv_path=None) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, str], list]:
    """
    Load behavioral data from CSV into a simple nested dictionary structure.

    Args:
        csv_path: Path to the CSV file (defaults to 'ibl_behavior.csv' in same directory as this file)

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
        csv_path = os.path.join(os.path.dirname(__file__), 'ibl_behavior.csv')

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