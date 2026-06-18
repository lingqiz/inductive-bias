# IBL decision-making dataset

Behavioural learning data from the IBL standardized decision-making task
(visual 2AFC on a steering wheel). Paper: *Standardized and reproducible
measurement of decision-making in mice*, IBL et al., eLife 2021 —
<https://elifesciences.org/articles/63711>.

All IBL-task files live under `data/ibl/`. Three sources are supported through
the loaders here:
- **`behavior.csv`** — the published daily-aggregate CSV (Fig. 2 source data).
- **IBL raw ALF dataset** (`/groups/zhang/home/zhangl5/IBL_Full/data`) — full
  per-trial + wheel; loaded by `raw.py`.
- **JRC raw iblrig dataset** (`/groups/zhang/home/zhangl5/IBL_JRC/data`) — the
  in-house cohort (10 mice); loaded by `jrc.py`.

Modules (`data/ibl/`): `utils.py` (CSV + complete-cohort daily aggregates +
analysis helpers), `raw.py` (IBL ALF backend), `jrc.py` (JRC raw-iblrig backend),
`sessions.py` (source-agnostic `SessionLike` interface + `open_session`
dispatcher). The legacy JRC processed `.mat` + MATLAB loaders were moved out of
the repo to `IBL_JRC/old_format/` (the `jrc.py` backend reads the raw iblrig data
directly and does not need them).

---

## 1. `behavior.csv` — published daily aggregates (Fig. 2 source data)

One row per (mouse, training day). 8,083 rows, **140 trained mice**, 7
laboratories anonymized as `Lab 1`–`Lab 7` (`institution_code`).

Key columns:

| column | meaning |
|---|---|
| `subject_nickname` | animal id (e.g. `CSHL047`, `KS014`) |
| `subject_uuid` | IBL database UUID |
| `institution_code` | anonymized lab, `Lab 1`–`Lab 7` |
| `session_date` | calendar date of the training day |
| `training_day` | 1-based day index within the animal's training |
| `n_trials_date` | trials that day (sessions pooled) |
| `performance` | fraction correct over **all** trials that day |
| `performance_easy` | fraction correct over **easy** trials (\|contrast\| ∈ {0.5, 1.0}) |

`performance_easy` is the metric used for the learning-curve / power-law
analysis. Loaded by `utils.load_behavior_data()` →
`{institution_code: {subject_id: [training_day, n_trials_date, performance,
performance_easy]}}`.

---

## 2. Raw ALF dataset — `/groups/zhang/home/zhangl5/IBL_Full/data`

Full per-trial + wheel data, a **superset** of the CSV. Standard IBL ALF/ONE
layout:

```
<lab>/Subjects/<subject>/<date>/<number>/alf/
    _ibl_trials.table.<uuid>.pqt          per-trial behaviour (newer sessions)
    _ibl_trials.<field>.<uuid>.npy        per-field layout (older/recovered sessions)
    _ibl_wheel.position.<uuid>.npy        continuous wheel position (rad)
    _ibl_wheel.timestamps.<uuid>.npy      wheel clock (s); see QC note below
    _ibl_wheel.velocity.<uuid>.npy
    _ibl_wheelMoves.intervals / .peakAmplitude
```

Per-trial fields: `choice`, `feedbackType`, `rewardVolume`, `contrastLeft/Right`,
`probabilityLeft`, and event times (`stimOn_times`, `goCue_times`,
`firstMovement_times`, `response_times`, `feedback_times`, `intervals_0/1`).
Trial event times and wheel timestamps share one clock, so per-trial wheel
segments come from slicing the continuous trace by time
(`raw.Session.trial_wheel`).

### How it was downloaded (scripts in `IBL_Full/`)

- Session list comes from the public **openalyx** Alyx server.
  `build_manifest.py` originally queried the release tag
  `2021_Q1_IBL_et_al_Behaviour`, which **omits each animal's early naive
  sessions**. `build_manifest_full.py` supplements this by querying **per
  subject** (the 140 CSV nicknames), restricted to each subject's CSV training
  date range, and unions the result (old manifest backed up to
  `manifest_tagged.json`).
- `download.py` mirrors `alf/` + `raw_behavior_data/` from the Flatiron file
  server. `SKIP_SUBSTRINGS` drops data we don't use: **camera/video, pose, and
  microphone audio** (`Camera`, `lightningPose`, `micData`, `audioSpectrogram`).
- Result: 6,848 sessions, **138 of the 140 CSV subjects** present
  (`ibl_witten_02`, `ibl_witten_03` have no public data).

---

## 2b. JRC dataset (in-house) — `/groups/zhang/home/zhangl5/IBL_JRC/data`

An in-house run of the same IBL task, in **raw iblrig format only** (no processed
ALF). 10 mice (`m01, m03, m04, m05, m06, m07, m08, m09, m10, m12`), 212 training
sessions. Non-standard layout:

```
<mouse>/<session>/<run>/raw_task_data_00/
    _iblrig_taskData.raw.jsonable        one JSON object per trial (21 fields incl. behavior_data)
    _iblrig_encoderPositions.raw.ssv     [time_us, position_clicks]   (wheel)
    _iblrig_encoderEvents.raw.ssv        [time_us, code]  8=stim_on, 1=stim_off, 3=close_loop
    _iblrig_encoderTrialInfo.raw.ssv     per-trial sync info (+datetime col)
    _iblrig_taskSettings.raw.json        STIM_GAIN, STIM_POSITIONS, ...
<mouse>/<session>/<run>/raw_video_data/
    _iblrig_leftCamera.raw.avi           video (path exposed; not decoded)
    _iblrig_leftCamera.frameData.bin
```

Conventions handled by the loader (`jrc.py`):
- **Sessions**: only `1t*` (training) are used; `0hab*` (habituation) ignored.
  `training_day` = the `1tNN` ordinal, so a missing session leaves a gap.
- **Run**: normally `001`; a restarted session leaves a stub in `001` and the
  complete data in `002` (e.g. `m01/1t22`, `m04/1t11`, `m08/1t18`). The loader
  picks the run with the largest taskData jsonable that also has encoder data.
- **Two clocks**: the bpod clock (seconds, in `behavior_data`) and the
  rotary-encoder clock (microseconds, in the `.ssv` files) differ. Wheel **and**
  stim events are on the encoder clock; trial outcomes come from taskData by
  order — so `stimOn_times`/`feedback_times` are on the encoder clock (same as
  `wheel()`), and `trial_wheel` works identically to the ALF backend. Units:
  µs→s, clicks→rad via `2π/1024`.
- **`feedback_type`** (`jackpot` vs `standard` reward): NOT in the raw files —
  harvested once from `JRC_IBLData.mat` into `jrc.MOUSE_TYPE`, exposed as a
  column. jackpot = {m01,m03,m04,m05,m08,m09} (6); standard = {m06,m07,m10,m12} (4).
- **Video**: `sess.video_path` / `sess.frame_data_path` (paths only; not decoded).

Access: `sessions.open_session('jrc', subject='m06')` or `jrc.load_trials()`.
Validated against the legacy `JRC_IBLData.mat` (moved to `IBL_JRC/old_format/`):
212/212 sessions internally consistent; sampled per-session arrays match exactly;
the jackpot/standard group-average learning curves match the original notebook
bit-for-bit (slopes −0.76 jackpot, −1.30 standard).

---

## 3. Reconciling raw ↔ CSV, and the "complete" cohort

**Lab mapping** (physical folder → anonymized `institution_code`); two
institutions pool two physical labs each:

| Lab | physical lab(s) |
|---|---|
| Lab 1 | cortexlab |
| Lab 2 | mainenlab |
| Lab 3 | churchlandlab + zadorlab |
| Lab 4 | angelakilab |
| Lab 5 | wittenlab |
| Lab 6 | mrsicflogellab + hoferlab |
| Lab 7 | danlab |

**Matching.** Raw and CSV are joined on `(subject_nickname, session_date)`.
Reconstructing the CSV columns from raw trials: `n_trials_date` and
`performance_easy` match to ~99% exact on shared days. The all-trial
`performance` only matches ~56% — the IBL pipeline uses a different convention
for that field on full-task (biased-block) days — but the learning-curve
analysis uses `performance_easy`, so this does not matter.

**Coverage ceiling.** The public release contains only ~70% of the CSV's
training-days; the rest (mostly the earliest naive sessions) were never
publicly released, so an exact CSV reproduction is impossible.

**End-to-end-complete cohort.** An animal is "complete" if the raw data covers
**every** CSV training day:

- **66 / 140** mice have end-to-end **choice** data;
- **55 / 140** also have end-to-end **wheel** data (choice + wheel) — the clean
  set for combined learning + kinematics analysis.

Filtering / QC lives in `raw.coverage_qc_report()`. Wheel QC
(`raw.wheel_qc_session`) marks a session usable if the wheel trace exists,
has enough samples, increases in time, and spans the stimOn→feedback window.

> **Wheel QC gotcha:** older sessions ship **two** `_ibl_wheel.timestamps`
> files (different UUIDs and lengths). The correct one matches
> `_ibl_wheel.position` in length — selected by byte-size in
> `raw._wheel_paths`. Picking the wrong one looks like a length mismatch.

---

## 4. Loader interface

```python
from data.ibl.utils import load_behavior_data, load_behavior_complete, complete_subjects
from data.ibl import sessions

# Daily aggregates (nested dict {institution_code:{subject_id: array}})
data, id2uuid, cols = load_behavior_data()                 # published CSV, 140 mice
data, id2nick, cols = load_behavior_complete(level='wheel') # raw end-to-end cohort
#   level='wheel' -> 55 mice (choice+wheel); level='choice' -> 66 mice (choice)

# Source-agnostic per-trial + wheel access (same API for 'ibl' and 'jrc')
idx  = sessions.open_dataset('jrc')                    # per-session index DataFrame
sess = sessions.open_session('ibl', subject='CSHL047') # or open_session('jrc', subject='m01')
t, pos = sess.wheel()                                  # (timestamps_s, position_rad)
seg    = sess.trial_wheel(0)                           # per-trial wheel segment
#   sess.trials -> normalized COMMON_COLS (choice, signed_contrast, feedbackType,
#   correct, stimOn_times, feedback_times, ...) + source-specific extras.
```

`block_average`, `interpolate_average`, and `find_threshold` (in `utils.py`)
operate identically on the output of either daily-aggregate loader. `raw.py` /
`jrc.py` cache the session index and per-trial frame under `data/ibl/cache/`
(regenerated on demand). JRC notes: only `1t*` training sessions (habituation
`0hab*` ignored); trial event times are on the rotary-encoder clock (shared with
the wheel); video is exposed as `sess.video_path` (not decoded). Each mouse has a
`feedback_type` (`jackpot` vs `standard` reward) — a column on the index/trials,
harvested once from the legacy `JRC_IBLData.mat` (now `IBL_JRC/old_format/`,
outside the repo) since it is not in the raw files.