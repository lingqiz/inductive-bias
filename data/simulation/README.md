# Simulated learning curves

## `mnist_pretrain/`: pretraining and the data scaling law (CNN, MNIST)

Test error and test cross-entropy of a small CNN on a 3-way MNIST task (digits 3, 5, 8) as a
function of the number of target training images `m`, trained from scratch or fine-tuned
from a CNN pretrained on other digits.

### How it was generated

`neural_network.ipynb`, **Part 1: Simulation** (all simulation code is in `theory.py`:
`load_mnist`, `split_digits`, `hold_out_per_class`, `merge_sources`, `augment`, `CNN`, `train`,
`evaluate`, `pretrain`, `pretrained_init`, `NetworkCurve`). To regenerate, delete the `.npz`
files here (and the pretrained networks in `cache/nn_pretrain_358_mix300/`) and run the setup
cells + Part 1. **Part 2** of the notebook reproduces the analysis and figures from these files.

**Data.** MNIST (torchvision), pixels scaled to [0, 1] and normalized with the training-set
mean / sd.
- Target: digits 3, 5, 8, labels 0, 1, 2. Test set: all 2,876 MNIST test images of these digits.
- A fixed subset of 300 training images per target digit (900 total, `hold_out_per_class`,
  seed 0) is held out for the *all digits* pretraining set and removed from the target
  training pool of **every** condition: the pool has 16,503 images.

**Conditions** (file `target_<condition>.npz`):

| file | pretraining set | classes | images |
|---|---|---|---|
| `target_scratch.npz` | none (random init) | | |
| `target_pretrain_2_digits.npz` | digits 2, 7 | 2 | 12,223 |
| `target_pretrain_4_digits.npz` | digits 0, 2, 4, 7 | 4 | 23,988 |
| `target_pretrain_all_digits.npz` | digits 0, 1, 2, 4, 6, 7, 9 + the 900 held-out target images | 10 | 43,497 |
| `target_pretrain_all_digits_waug.npz` | same as *all digits*, with random affine augmentation (`augment`: rotation within ±15°, scale 0.9 to 1.1, shift up to ~2 px) | 10 | 43,497 |

**Model.** `CNN(width=32)`: conv(1→32, 3×3)-ReLU-maxpool, conv(32→64, 3×3)-ReLU-maxpool,
fc(3136→128)-ReLU (`features`), then a linear head (`head`). 3-way head for the target.

**Training** (`train`, same learner at every `m`): AdamW (lr 1e-3, no weight decay), cosine
learning-rate decay, batch size min(128, m), `max(3000, ceil(20 m / 128))` SGD steps
(trained to ~zero training error at every `m`). No augmentation during target training.

**Pretraining** (`pretrain`): one network per repeat `r` (seed 1000 + r), trained on the full
pretraining set with the same protocol. Fine-tuning loads the pretrained `features` and
starts from a **new random head** (for *all digits*, the pretrained 3/5/8 outputs are
discarded too); all weights are trained.

**Repeats** (`NetworkCurve`): for each `m` in 2^4 .. 2^14 and repeat r = 0..7, seed r draws
the target training subset of size `m` (the same subset in every condition), then the
network (random init, or pretrained network r + new head) is trained and evaluated.

### File format

Each `.npz` (`np.load(path)`) holds:

| key | shape | content |
|---|---|---|
| `train_size` | (11,) | target training set sizes m = 2^4 .. 2^14 |
| `errors` | (11, 8) | test 0-1 error, per (m, repeat) |
| `losses` | (11, 8) | test cross-entropy (nats), per (m, repeat) |
| `train_errors` | (11, 8) | training 0-1 error on the m training images, per (m, repeat) |

### Environment

PyTorch 2.14.1 (CUDA 13.0), torchvision 0.29.1, NumPy 1.26.4, on an NVIDIA RTX 4090
(code at git commit c3df5af + the refactor that moved the simulation into `theory.py`).
cuDNN autotuning (`torch.backends.cudnn.benchmark = True`) is on, so re-running reproduces
the curves statistically but not bit for bit.
