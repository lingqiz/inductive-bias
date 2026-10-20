import os
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from utils import PiecewiseLinear

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

class ConvergenceCurve():
    def __init__(self, distribution, model, asmp_error=None, n_test=4096,
                 train_size = 2 ** np.arange(1, 15, 1), n_repeat=128):
        """
        Args:
            distribution: callable that generates data.
            model: callable that returns model instance.
            asmp_error: expected asymptotic error rate.
            n_test: number of test samples.
            n_repeat: number of repeats for empirical error estimation.
        """
        self.distribution = distribution
        self.model = model
        self.asmp_error = asmp_error
        self.n_test = n_test
        self.train_size = train_size
        self.n_repeat = n_repeat

        self.X_D, self.y_D = distribution(n_test)

    def run(self):
        error = np.zeros(len(self.train_size))
        error_sd = np.zeros(len(self.train_size))
        for i in range(len(self.train_size)):
            m = self.train_size[i]
            test_error = []
            for _ in range(self.n_repeat):
                # sample data
                X_S, y_S = self.distribution(m)

                # fit a linear model
                lc = self.model()
                lc.fit(X_S, y_S)

                # predict on the test set
                pred = lc.predict(self.X_D)
                test_error.append(1 - np.mean(pred == self.y_D))                

            error[i] = np.mean(test_error)
            error_sd[i] = np.std(test_error)

        self.error = error
        self.error_sd = error_sd
        return self.train_size, self.error, self.error_sd

    def plot_error(self, ax, log_scale=True):
        # plot generalization error as a function of training set size
        ax.errorbar(self.train_size, self.error, yerr=self.error_sd,
                    marker='o', markersize=6, capsize=4)

        if log_scale:
            ax.set_xscale('log', base=2)
            ax.set_yscale('log', base=2)
            ax.minorticks_off()
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.ticklabel_format(style='plain', axis='y')

        ax.set_xlabel('training set size (m)')
        ax.set_ylabel(r'$ \epsilon = L_D(h)$')

    def plot_gap(self, ax, piecewise=False):
        # plot generalization gap as a function of log training set size
        log_m = np.log2(self.train_size)
        if self.asmp_error is None:
            print('estimate asymptotic error rate from data')
            asmp_error = self.error[-1]
            error_gap = self.error - asmp_error

            log_m = log_m[:-1]  # exclude last point
            log_gap = np.log2(error_gap[:-1])
        else:
            error_gap = self.error - self.asmp_error
            log_gap = np.log2(error_gap)

        # no error bars: this is the expected error, which can be estimated
        # to arbitrary precision with more simulation repeats
        l = ax.plot(log_m, log_gap, marker='o', markersize=6)
        ax.set_xlabel('$\log_2(m)$')
        ax.set_ylabel(r'$\log |\epsilon - \epsilon^*|$')

        if not piecewise:
            # fit a linear function to the error gap
            coeffs = np.polyfit(log_m, log_gap, 1)
            a, b = coeffs
            ax.plot(log_m, a * log_m + b, color=l[0].get_color(),
                    linestyle='--', label=f'$%.2f * \log_2(m) + %.2f$' % (a, b))
        else:
            # fit a piecewise linear function to the error gap
            pwl = PiecewiseLinear()
            pwl.fit(log_m, log_gap)
            x0, m2, b2 = pwl.params['x0'], pwl.params['m2'], pwl.params['b2']
            ax.plot(log_m, m2 * log_m + b2, linestyle='--', color=l[0].get_color(),
                    label=f'$%.2f * \log_2(m) + %.2f$' % (m2, b2))

        ax.legend()


# -----------------------------------------------------------------------------
# Neural-network data scaling + pretraining simulation (MNIST, CNN);
# used by neural_network.ipynb, results in data/simulation/
# -----------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def train(model, X, y, min_steps=3000, min_epochs=20, batch_size=128,
          lr=1e-3, weight_decay=0.0, augment=None):
    '''
    Train to (approximate) convergence with AdamW + cosine decay for
    n_steps = max(min_steps, min_epochs passes over the data):
    small m -> many epochs (train loss ~0), large m -> at least min_epochs.
    augment: optional callable applied to each minibatch of images.
    '''
    m = len(X)
    bs = min(batch_size, m)
    n_steps = max(min_steps, int(np.ceil(min_epochs * m / bs)))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)

    model.train()
    for _ in range(n_steps):
        idx = torch.randint(0, m, (bs,), device=X.device)
        xb = X[idx] if augment is None else augment(X[idx])
        loss = F.cross_entropy(model(xb), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
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
        states.append({k: v.detach().clone() for k, v in net.features.state_dict().items()})

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


class NetworkCurve(ConvergenceCurve):
    def __init__(self, data, model, init=None, asmp_error=None,
                 train_size=2 ** np.arange(4, 15), n_repeat=8, **train_kw):
        '''
        Args:
            data: (X_train, y_train, X_test, y_test); training subsets drawn from X_train.
            model: callable that returns an (untrained) nn.Module.
            init: optional callable init(model, r) -> model for repeat r, applied
                  before training (e.g. load pretrained features); None = random init.
            train_kw: passed to train().
        '''
        self.X_pool, self.y_pool, self.X_D, self.y_D = data
        self.model = model
        self.init = init
        self.asmp_error = asmp_error
        self.train_size = np.asarray(train_size)
        self.n_repeat = n_repeat
        self.train_kw = train_kw

    def run(self, cache=None):
        '''
        cache: optional .npz path; load results from it if it exists, else run and save
        (train_size, and per (m, repeat) test errors / test losses / train errors).
        (cache is keyed by path only -- delete it after changing the protocol)
        '''
        if cache is not None and os.path.exists(cache):
            res = np.load(cache)
            assert np.array_equal(res['train_size'], self.train_size)
            self.errors, self.losses = res['errors'], res['losses']
            self.train_errors = res['train_errors']
            print(f'loaded {cache}')
        else:
            self._simulate()
            if cache is not None:
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                np.savez(cache, train_size=self.train_size, errors=self.errors,
                         losses=self.losses, train_errors=self.train_errors)

        self.error = self.errors.mean(axis=1)
        self.error_sd = self.errors.std(axis=1)
        self.loss = self.losses.mean(axis=1)
        self.loss_sd = self.losses.std(axis=1)
        return self.train_size, self.error, self.error_sd

    def _simulate(self):
        n_pool = len(self.X_pool)
        shape = (len(self.train_size), self.n_repeat)
        self.errors, self.losses, self.train_errors = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        for i, m in enumerate(self.train_size):
            for r in range(self.n_repeat):
                torch.manual_seed(r)

                # sample a random training subset of size m
                # (same subset for every condition at a given (m, r))
                idx = torch.randperm(n_pool, device=device)[:m]
                X_S, y_S = self.X_pool[idx], self.y_pool[idx]

                # fit a network from (random or pretrained) init
                net = self.model().to(device)
                if self.init is not None:
                    net = self.init(net, r)
                train(net, X_S, y_S, **self.train_kw)

                # evaluate on the test set; train error to check convergence
                self.errors[i, r], self.losses[i, r] = evaluate(net, self.X_D, self.y_D)
                self.train_errors[i, r] = evaluate(net, X_S, y_S)[0]
            print(f'm = {m:6d}  error = {self.errors[i].mean():.4f}  '
                  f'max train error = {self.train_errors[i].max():.4f}')
