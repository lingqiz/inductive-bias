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