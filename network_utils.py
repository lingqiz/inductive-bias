'''
Neural-network data scaling + pretraining simulation (MNIST, CNN), used by pretrain_nn.ipynb;
the simulated networks are saved in data/simulation/.
'''
import os
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def load_mnist(root='./data/vision'):
    '''
    Load full MNIST train / test split as normalized tensors on device.
    X: (N, 1, 28, 28), y: (N,)
    '''
    split = {}
    for train in (True, False):
        ds = datasets.MNIST(root, train=train, download=True)
        split[train] = (ds.data.float().div(255.0).unsqueeze(1), ds.targets.long())

    # normalize with training set statistics
    mu, sd = split[True][0].mean(), split[True][0].std()
    X_train, y_train = split[True]
    X_test, y_test = split[False]
    return ((X_train - mu) / sd).to(device), y_train.to(device), \
           ((X_test - mu) / sd).to(device), y_test.to(device)


def split_digits(data, digits):
    '''
    Keep only the given digits (train + test), relabeled to 0..len(digits)-1.
    '''
    digits = torch.as_tensor(sorted(digits), device=device)
    out = []
    for X, y in (data[:2], data[2:]):
        keep = torch.isin(y, digits)
        out += [X[keep], torch.searchsorted(digits, y[keep])]
    return tuple(out)


def hold_out_per_class(data, n_per_class, seed=0):
    '''
    Split a fixed random subset of n_per_class training images per class off the training set.
    Returns (data without the held-out images, held-out images with an empty test split).
    '''
    X, y, X_test, y_test = data
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed)).to(device)
    held = torch.cat([perm[y[perm] == c][:n_per_class] for c in range(int(y.max()) + 1)])
    keep = torch.ones(len(y), dtype=torch.bool, device=device)
    keep[held] = False
    return (X[keep], y[keep], X_test, y_test), (X[held], y[held], X_test[:0], y_test[:0])


def merge_sources(a, b, n_a):
    # concatenate two labeled datasets; b's labels are offset by a's n_a classes
    return tuple(torch.cat([u, v + n_a if i % 2 else v]) for i, (u, v) in enumerate(zip(a, b)))


def augment(X, max_rot=15, scale=(0.9, 1.1), max_shift=0.15):
    '''
    Random affine transform per image (rotation in degrees, scale, shift as a fraction
    of the half-width: 0.15 ~ 2 px), on device; the background stays at its normalized value.
    '''
    n = len(X)
    ang = torch.empty(n, device=X.device).uniform_(-max_rot, max_rot) * np.pi / 180
    s = torch.empty(n, device=X.device).uniform_(*scale)
    t = torch.empty(n, 2, device=X.device).uniform_(-max_shift, max_shift)

    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([torch.stack([cos, -sin, t[:, 0]], 1),
                         torch.stack([sin, cos, t[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, X.shape, align_corners=False)
    bg = X.amin()
    return F.grid_sample(X - bg, grid, align_corners=False) + bg


def subset(X_pool, y_pool, m, r):
    # random training subset of size m for repeat r
    # (same subset for every condition at a given (m, r))
    torch.manual_seed(r)
    idx = torch.randperm(len(X_pool), device=device)[:m]
    return X_pool[idx], y_pool[idx]


# -----------------------------------------------------------------------------
# Model, training, pretraining
# -----------------------------------------------------------------------------

# split into features + linear head, so the head can be replaced for fine-tuning
class CNN(nn.Module):
    def __init__(self, width=32, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(2 * width * 7 * 7, 128), nn.ReLU())
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.head(self.features(x))


def cnn_from_state(state):
    # CNN with the width / number of classes of a saved state dict, with those weights loaded
    net = CNN(width=state['features.0.weight'].shape[0], n_classes=state['head.weight'].shape[0])
    net.load_state_dict(state)
    return net.to(device)


def train(model, X, y, min_steps=3000, min_epochs=20, batch_size=128,
          lr=1e-3, weight_decay=0.0, augment=None, callback=None):
    '''
    Train to (approximate) convergence with AdamW + cosine decay for
    n_steps = max(min_steps, min_epochs passes over the data):
    small m -> many epochs (train loss ~0), large m -> at least min_epochs.
    augment: optional callable applied to each minibatch of images.
    callback: optional callable callback(step, model) after every step (e.g. to snapshot
              the model); it does not change the training.
    '''
    m = len(X)
    bs = min(batch_size, m)
    n_steps = max(min_steps, int(np.ceil(min_epochs * m / bs)))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)

    model.train()
    for t in range(n_steps):
        idx = torch.randint(0, m, (bs,), device=X.device)
        xb = X[idx] if augment is None else augment(X[idx])
        loss = F.cross_entropy(model(xb), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if callback is not None:
            callback(t + 1, model)
            model.train()
    return model


@torch.no_grad()
def evaluate(model, X, y, batch_size=4096):
    # returns (0-1 error, cross-entropy loss) on (X, y)
    model.eval()
    n_err, loss = 0.0, 0.0
    for i in range(0, len(X), batch_size):
        out = model(X[i:i + batch_size])
        n_err += (out.argmax(1) != y[i:i + batch_size]).sum().item()
        loss += F.cross_entropy(out, y[i:i + batch_size], reduction='sum').item()
    return n_err / len(X), loss / len(X)


def pretrain(data, n_classes, n_repeat=8, cache=None, **train_kw):
    '''
    Train one CNN per repeat on the full source training set (same protocol as train());
    returns the feature-extractor state dicts (the source head is discarded).
    train_kw: passed to train() (e.g. augment).
    '''
    if cache is not None and os.path.exists(cache):
        print(f'loaded {cache}')
        return torch.load(cache)

    X, y, X_test, y_test = data
    states = []
    for r in range(n_repeat):
        torch.manual_seed(1000 + r)
        net = train(CNN(n_classes=n_classes).to(device), X, y, **train_kw)
        print(f'pretrain {r}: source test error = {evaluate(net, X_test, y_test)[0]:.4f}')
        states.append({k: v.detach().cpu().clone() for k, v in net.features.state_dict().items()})

    if cache is not None:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save(states, cache)
    return states


def pretrained_init(states):
    '''
    init hook for NetworkCurve: load pretrained features for repeat r;
    the head stays at its random init (new head for the target labels).
    '''
    def init(net, r):
        net.features.load_state_dict(states[r])
        return net
    return init


# -----------------------------------------------------------------------------
# Learning-curve simulation: trained networks for every (m, repeat)
# -----------------------------------------------------------------------------

def _state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


# arrays in a NetworkCurve.run file (stored as tensors, for a safe weights_only load)
NETWORK_ARRAYS = ('train_size',
                  'final_errors', 'final_losses', 'final_train_errors',
                  'early_stop_step', 'early_stop_errors', 'early_stop_losses', 'early_stop_train_errors')


def load_networks(path):
    '''
    Load a NetworkCurve.run file: the initial weights of each repeat ('init', [r]), the
    early-stopped and final weights of every network ('early_stop', 'final', [m][r]) as state
    dicts, and the results recorded during the simulation (NETWORK_ARRAYS) as numpy arrays.
    '''
    res = torch.load(path)
    res.update({k: res[k].numpy() for k in NETWORK_ARRAYS})
    return res


class NetworkCurve:
    def __init__(self, data, model, init=None, train_size=2 ** np.arange(4, 15), n_repeat=8,
                 check_every=5, **train_kw):
        '''
        Args:
            data: (X_train, y_train, X_test, y_test); training subsets drawn from X_train.
            model: callable that returns an (untrained) nn.Module.
            init: optional callable init(model, r) -> model for repeat r, applied
                  before training (e.g. load pretrained features); None = random init.
            check_every: steps between the training-error checks for the early stop.
            train_kw: passed to train().
        '''
        self.X_pool, self.y_pool, self.X_D, self.y_D = data
        self.model = model
        self.init = init
        self.train_size = np.asarray(train_size)
        self.n_repeat = n_repeat
        self.check_every = check_every
        self.train_kw = train_kw

    def run(self, path):
        '''
        Train a network for every (m, repeat) and save to path (.pt), or load path if it exists:
        - 'init' [r]: initial weights of repeat r (the same for every m)
        - 'early_stop' [m][r]: weights at the first check (every check_every steps) with zero
          training error, i.e. the first point the network fits its training set
        - 'final' [m][r]: weights at the end of training
        - per (m, r) test error / test loss / train error of both ('early_stop_*', 'final_*'),
          and the early-stop step (0: never at zero training error, end of training used)
        (path is keyed by name only -- delete it after changing the protocol)
        '''
        if os.path.exists(path):
            res = load_networks(path)
            assert np.array_equal(res['train_size'], self.train_size)
            print(f'loaded {path}')
            return res

        res = self._simulate()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({**res, **{k: torch.as_tensor(res[k]) for k in NETWORK_ARRAYS}}, path)
        return res

    def subset(self, m, r):
        return subset(self.X_pool, self.y_pool, m, r)

    def _simulate(self):
        n_m = len(self.train_size)
        shape = (n_m, self.n_repeat)
        res = {'train_size': self.train_size, 'init': [None] * self.n_repeat,
               'early_stop': [[None] * self.n_repeat for _ in range(n_m)],
               'final': [[None] * self.n_repeat for _ in range(n_m)],
               'early_stop_step': np.zeros(shape, dtype=int)}
        for k in ('early_stop', 'final'):
            for q in ('errors', 'losses', 'train_errors'):
                res[f'{k}_{q}'] = np.zeros(shape)

        for i, m in enumerate(self.train_size):
            for r in range(self.n_repeat):
                X_S, y_S = self.subset(m, r)

                # (random or pretrained) init; the init of repeat r does not depend on m
                # (the subset is drawn on the CUDA generator)
                net = self.model().to(device)
                if self.init is not None:
                    net = self.init(net, r)
                w0 = _state(net)
                if i == 0:
                    res['init'][r] = w0
                assert all(torch.equal(w0[k], res['init'][r][k]) for k in w0)

                # early stop: snapshot at the first check with zero training error
                stop = {}
                def snapshot(step, net, X_S=X_S, y_S=y_S, stop=stop):
                    if not stop and step % self.check_every == 0 and evaluate(net, X_S, y_S)[0] == 0:
                        stop.update(step=step, state=_state(net))
                train(net, X_S, y_S, callback=snapshot, **self.train_kw)
                res['final'][i][r] = _state(net)
                if not stop:  # never at zero training error: use the end of training
                    stop.update(step=0, state=res['final'][i][r])
                res['early_stop'][i][r], res['early_stop_step'][i, r] = stop['state'], stop['step']

                # test error / loss and train error of both
                for k in ('early_stop', 'final'):
                    net_k = cnn_from_state(res[k][i][r])
                    res[f'{k}_errors'][i, r], res[f'{k}_losses'][i, r] = evaluate(net_k, self.X_D, self.y_D)
                    res[f'{k}_train_errors'][i, r] = evaluate(net_k, X_S, y_S)[0]

            print(f"m = {m:6d}  final error = {res['final_errors'][i].mean():.4f}  "
                  f"max train error = {res['final_train_errors'][i].max():.4f}  | "
                  f"early stop at step {res['early_stop_step'][i].mean():6.0f}: "
                  f"error = {res['early_stop_errors'][i].mean():.4f}")
        return res


def evaluate_networks(res, key, data):
    '''
    Evaluate the saved networks res[key][m][r] (key: 'early_stop' or 'final', see NetworkCurve.run):
    test error / test loss on the test split of data, and train error on each network's own
    training subset of the train split. Returns a dict of per (m, repeat) arrays
    (train_size, errors, losses, train_errors). Deterministic cuDNN settings, so the results do not
    depend on the session's cudnn.benchmark setting.
    '''
    X_pool, y_pool, X_test, y_test = data
    shape = (len(res['train_size']), len(res['init']))
    out = {'train_size': res['train_size'], 'errors': np.zeros(shape), 'losses': np.zeros(shape),
           'train_errors': np.zeros(shape)}
    with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True):
        for i, m in enumerate(res['train_size']):
            for r in range(shape[1]):
                net = cnn_from_state(res[key][i][r])
                out['errors'][i, r], out['losses'][i, r] = evaluate(net, X_test, y_test)
                out['train_errors'][i, r] = evaluate(net, *subset(X_pool, y_pool, m, r))[0]
    return out


# -----------------------------------------------------------------------------
# Network-side estimates of the complexity phi: distance from the init (the
# prior mean) and a flatness-calibrated PAC-Bayes KL (Jiang et al. 2019)
# -----------------------------------------------------------------------------

def param_distance(state, w0):
    '''
    Squared L2 distance between two state dicts: (total, {parameter name: distance}).
    '''
    layer = {k: (state[k].float() - w0[k].float()).pow(2).sum().item() for k in state}
    return sum(layer.values()), layer


@torch.no_grad()
def noisy_error(net, X, y, sigma, n_draw=10, seed=0):
    '''
    Mean 0-1 error on (X, y) of the net with isotropic Gaussian weight noise N(0, sigma^2 I),
    over n_draw noise draws (fixed seed: common random numbers across sigma).
    '''
    noisy = copy.deepcopy(net)
    gen = torch.Generator(device=device).manual_seed(seed)
    errs = []
    for _ in range(n_draw):
        for p, p0 in zip(noisy.parameters(), net.parameters()):
            p.copy_(p0 + sigma * torch.randn(p0.shape, generator=gen, device=device))
        errs.append(evaluate(noisy, X, y)[0])
    return np.mean(errs)


def flatness_sigma(net, X, y, eps=0.2, n_draw=10, lo=1e-4, hi=1.0, n_iter=15):
    '''
    Largest isotropic weight noise sigma that keeps the (noisy) training error within eps of
    the trained net's, by bisection on log sigma in [lo, hi].
    '''
    target = evaluate(net, X, y)[0] + eps
    log_lo, log_hi = np.log(lo), np.log(hi)
    for _ in range(n_iter):
        mid = (log_lo + log_hi) / 2
        if noisy_error(net, X, y, np.exp(mid), n_draw) <= target:
            log_lo = mid
        else:
            log_hi = mid
    return np.exp(log_lo)


def pac_bayes_kl(dist2, sigma, k):
    '''
    KL(N(w, sigma^2 I) || N(w0, sigma_P^2 I)) in nats, for ||w - w0||^2 = dist2 over k
    parameters, at the best prior width sigma_P^2 = sigma^2 + dist2 / k:
    (k/2) log(1 + dist2 / (k sigma^2))  (~ dist2 / (2 sigma^2) for small dist2).
    '''
    return 0.5 * k * np.log1p(dist2 / (k * sigma ** 2))


# -----------------------------------------------------------------------------
# Decoders on frozen representations: a linear ridge readout of the target labels
# from the features of a (pretrained or random) network, as one would decode
# neural recordings
# -----------------------------------------------------------------------------

# number of CNN.features modules up to each layer's output (after its ReLU and pooling)
FEATURE_LAYERS = {'conv1': 3, 'conv2': 7, 'fc': 9}


@torch.no_grad()
def frozen_features(state, X, layer='fc', batch_size=4096):
    # flattened features at `layer` (FEATURE_LAYERS) of the CNN with weights `state` (its head is not used)
    f = cnn_from_state(state).eval().features[:FEATURE_LAYERS[layer]]
    return torch.cat([f(X[i:i + batch_size]).flatten(1) for i in range(0, len(X), batch_size)])


def zscore(X_train, *others, eps=1e-6):
    # z-score each feature with the training-set mean / sd; applies the same transform to `others`
    mu, sd = X_train.mean(0), X_train.std(0) + eps
    return [(X - mu) / sd for X in (X_train, *others)]


def ridge_probe(X, y, n_classes, lambdas=np.logspace(-2, 6, 33)):
    '''
    Linear ridge-regression decoder of the class labels (one-hot targets coded +-1, with an
    unpenalized intercept) from features X, the ridge strength chosen by leave-one-out
    cross-validation of the squared error (closed form, from the SVD of the centered X).
    Returns (nn.Linear decoder on device, chosen lambda).
    '''
    X = X.double()
    Y = 2 * F.one_hot(y, n_classes).double() - 1
    x_mean, y_mean = X.mean(0), Y.mean(0)
    U, s, Vt = torch.linalg.svd(X - x_mean, full_matrices=False)
    UtY = U.T @ (Y - y_mean)

    def loo(lam):
        d = s ** 2 / (s ** 2 + lam)
        resid = (Y - y_mean) - U @ (d[:, None] * UtY)
        h = (U ** 2 * d).sum(1) + 1 / len(X)  # leverage, including the intercept
        return (resid / (1 - h)[:, None]).pow(2).mean().item()

    lam = min(lambdas, key=loo)
    W = Vt.T @ ((s / (s ** 2 + lam))[:, None] * UtY)  # (n_features, n_classes)
    probe = nn.Linear(X.shape[1], n_classes).to(device)
    probe.weight.data = W.T.float().contiguous()
    probe.bias.data = (y_mean - x_mean @ W).float()
    return probe, lam
