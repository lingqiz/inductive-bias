import os
from typing import List, Any, Tuple
import scipy.io
import numpy as np
from scipy.stats import norm


def load_tactile_data(data_path: str = None) -> Tuple[List[str], List[Any], List[Any]]:
    """
    Load tactile behavior data from MATLAB file.

    Args:
        data_path: Path to the .mat file. If None, uses default path.

    Returns:
        Tuple containing (animal_names, trial_data, trial_type)

    Raises:
        FileNotFoundError: If the data file doesn't exist
        KeyError: If expected keys are missing from the data
    """
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), 'tactile_behavior.mat')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    try:
        data = scipy.io.loadmat(data_path)
        subjects = data['allData'][0]
        n_subject = len(subjects)

        animal_names = [subjects[i]['animalName'][0] for i in range(n_subject)]
        trial_data = [subjects[i]['trialData'][0] for i in range(n_subject)]
        trial_type = [subjects[i]['trialType'][0] for i in range(n_subject)]

        return animal_names, trial_data, trial_type

    except KeyError as e:
        raise KeyError(f"Missing expected key in data file: {e}")


# Load data on import for easy access
animal_names, trial_data, trial_type = load_tactile_data()


# return data filterd by trianing stage
def select_stage(stages):
    all_data = []

    for i in range(len(animal_names)):
        select_index = np.zeros(trial_type[i].shape, dtype=bool)
        for stage in stages:
            select_index = np.logical_or(select_index, trial_type[i] == stage)

        selected = trial_data[i][select_index]
        all_data.append(selected)

    return all_data


def sliding_window(decision: np.ndarray, window_size: int = 128, step_size: int = 16):
    """
    Perform sliding window analysis on decision sequence to track performance over time.

    Args:
        decision: Array of decision codes (11, 10, 21, 20)
        window_size: Size of sliding window
        step_size: Step size for sliding window

    Returns:
        Arrays of trial_center, acc_rate, hit_rate, cr_rate, fa_rate, miss_rate, dprime
    """
    n_trials = len(decision)
    trial_centers = []
    acc_rates = []
    hit_rates = []
    cr_rates = []
    fa_rates = []
    miss_rates = []
    dprimes = []

    for start in range(0, n_trials - window_size + 1, step_size):
        end = start + window_size
        window_data = decision[start:end]

        # Count each decision type
        hit = np.sum(window_data == 11)
        miss = np.sum(window_data == 10)
        cr = np.sum(window_data == 21)
        fa = np.sum(window_data == 20)

        go_trials = hit + miss
        nogo_trials = cr + fa
        total_trials = go_trials + nogo_trials

        # Compute statistics
        trial_centers.append(start + window_size // 2)
        acc_rates.append((hit + cr) / total_trials if total_trials > 0 else 0)

        hit_rate = hit / go_trials if go_trials > 0 else 0
        cr_rate = cr / nogo_trials if nogo_trials > 0 else 0
        fa_rate = fa / nogo_trials if nogo_trials > 0 else 0
        miss_rate = miss / go_trials if go_trials > 0 else 0

        hit_rates.append(hit_rate)
        cr_rates.append(cr_rate)
        fa_rates.append(fa_rate)
        miss_rates.append(miss_rate)

        # Calculate d-prime: Z(hit_rate) - Z(false_alarm_rate)
        # Clip rates to avoid infinite values from norm.ppf
        hit_rate_clipped = np.clip(hit_rate, 0.001, 0.999)
        fa_rate_clipped = np.clip(fa_rate, 0.001, 0.999)
        dprime = norm.ppf(hit_rate_clipped) - norm.ppf(fa_rate_clipped)
        dprimes.append(dprime)

    return (np.array(trial_centers), np.array(acc_rates), np.array(hit_rates),
            np.array(cr_rates), np.array(fa_rates), np.array(miss_rates), np.array(dprimes))


def sliding_window_group(decision_sequences: list, window_size: int = 128, step_size: int = 16):
    """
    Perform sliding window analysis on multiple decision sequences.

    Args:
        decision_sequences: List of arrays, each containing decision codes (11, 10, 21, 20)
        window_size: Size of sliding window
        step_size: Step size for sliding window

    Returns:
        Arrays of trial_center, acc_rate, hit_rate, cr_rate, fa_rate, miss_rate, dprime
    """
    # Find the maximum length to determine window positions
    max_length = max(len(seq) for seq in decision_sequences)

    trial_centers = []
    acc_rates = []
    hit_rates = []
    cr_rates = []
    fa_rates = []
    miss_rates = []
    dprimes = []

    for start in range(0, max_length - window_size + 1, step_size):
        end = start + window_size

        # Pool data from all sequences for this window
        all_hits = 0
        all_misses = 0
        all_cr = 0
        all_fa = 0

        for seq in decision_sequences:
            if len(seq) > start:  # Only include sequences long enough for this window
                window_end = min(end, len(seq))
                window_data = seq[start:window_end]

                all_hits += np.sum(window_data == 11)
                all_misses += np.sum(window_data == 10)
                all_cr += np.sum(window_data == 21)
                all_fa += np.sum(window_data == 20)

        go_trials = all_hits + all_misses
        nogo_trials = all_cr + all_fa
        total_trials = go_trials + nogo_trials

        if total_trials > 0:  # Only add if we have data
            # Compute statistics
            trial_centers.append(start + window_size // 2)
            acc_rates.append((all_hits + all_cr) / total_trials)

            hit_rate = all_hits / go_trials if go_trials > 0 else 0
            cr_rate = all_cr / nogo_trials if nogo_trials > 0 else 0
            fa_rate = all_fa / nogo_trials if nogo_trials > 0 else 0
            miss_rate = all_misses / go_trials if go_trials > 0 else 0

            hit_rates.append(hit_rate)
            cr_rates.append(cr_rate)
            fa_rates.append(fa_rate)
            miss_rates.append(miss_rate)

            # Calculate d-prime: Z(hit_rate) - Z(false_alarm_rate)
            # Clip rates to avoid infinite values from norm.ppf
            hit_rate_clipped = np.clip(hit_rate, 0.001, 0.999)
            fa_rate_clipped = np.clip(fa_rate, 0.001, 0.999)
            dprime = norm.ppf(hit_rate_clipped) - norm.ppf(fa_rate_clipped)
            dprimes.append(dprime)

    return (np.array(trial_centers), np.array(acc_rates), np.array(hit_rates),
            np.array(cr_rates), np.array(fa_rates), np.array(miss_rates), np.array(dprimes))

