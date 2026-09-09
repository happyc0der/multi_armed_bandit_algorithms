"""
algorithms/fast_adswitch.py -- Fast, iterative, numpy-vectorised AdSwitch
(Auer, Gajane, Ortner), refactored onto the shared BanditAlgorithm
interface (see algorithms/base.py) so it can be benchmarked head-to-head
against TS-GE and any future baseline on the exact same environment.

Only the plumbing changed from the original fast_adswitch.py: it now
returns 'reward'/'chosen_arm' lists (for the common harness) in addition
to 'net_reward'. The algorithm logic itself (O(1) mean/count queries via
cumulative sums, vectorised change-detection, iterative main loop instead
of recursion) is unchanged from the version already benchmarked at
20-50x+ speedup over the original recursive ADSWITCH.py.
"""
from __future__ import annotations
import math
import numpy as np

from bandit_base import BanditAlgorithm


class FastAdSwitch(BanditAlgorithm):
    def __init__(self, K: int, T: int, C1: float = 1.0, seed=None):
        super().__init__(K, T)
        self.C1 = C1
        self.rng = np.random.default_rng(seed)

    def run(self, reward_fn):
        K, T, C1 = self.K, self.T, self.C1
        logT = math.log(max(T, 2))

        cum_reward = np.zeros((K, T + 2))
        cum_count = np.zeros((K, T + 2))
        last_chosen = np.zeros(K, dtype=np.int64)

        def mean_iv(a, s, e):
            if e < s:
                return 0.0
            c = cum_count[a, e + 1] - cum_count[a, s]
            if c == 0:
                return 0.0
            return (cum_reward[a, e + 1] - cum_reward[a, s]) / c

        def count_iv(a, s, e):
            if e < s:
                return 0
            return cum_count[a, e + 1] - cum_count[a, s]

        good, bad = set(range(K)), set()
        sampling_oblig = {a: [] for a in range(K)}
        eviction_gap = {a: None for a in range(K)}
        eviction_mean = {a: None for a in range(K)}

        net_reward = 0.0
        reward_hist, chosen_hist = [], []
        episode_start = 1
        t = 0

        def start_new_episode():
            nonlocal good, bad, sampling_oblig, eviction_gap, eviction_mean, episode_start
            good, bad = set(range(K)), set()
            sampling_oblig = {a: [] for a in range(K)}
            eviction_gap = {a: None for a in range(K)}
            eviction_mean = {a: None for a in range(K)}
            episode_start = t + 1

        start_new_episode()

        while t < T:
            t += 1
            cum_reward[:, t + 1] = cum_reward[:, t]
            cum_count[:, t + 1] = cum_count[:, t]
            new_bad = set(bad)

            for a in list(bad):
                d = eviction_gap[a] / 16.0
                i = 1
                obligs = []
                change = 2.0 ** (-i)
                ep = max(episode_start, 1)
                while change >= d:
                    p = change * math.sqrt(ep / (K * T * logT))
                    if self.rng.random() < p:
                        n = math.ceil((2 ** (2 * i + 1)) * logT)
                        obligs.append([change, n, t])
                    i += 1
                    change = 2.0 ** (-i)
                sampling_oblig[a] = obligs

            candidates = list(good) + [a for a in bad if sampling_oblig[a]]
            chosen = min(candidates, key=lambda a: last_chosen[a])
            r = reward_fn(chosen, t)
            cum_reward[chosen, t + 1] += r
            cum_count[chosen, t + 1] += 1
            last_chosen[chosen] = t
            net_reward += r
            reward_hist.append(r)
            chosen_hist.append(chosen)

            restart = False
            sigmas = np.arange(episode_start, t + 1)

            if len(sigmas) > 0 and len(good) >= 2:
                gl = list(good)
                means = np.zeros((len(gl), len(sigmas)))
                counts = np.zeros((len(gl), len(sigmas)))
                for idx, a in enumerate(gl):
                    c = cum_count[a, t + 1] - cum_count[a, sigmas]
                    s = cum_reward[a, t + 1] - cum_reward[a, sigmas]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        means[idx] = np.where(c > 0, s / np.where(c == 0, 1, c), 0.0)
                    counts[idx] = c
                thresh = np.sqrt(2 * logT / np.where(counts == 0, 1, counts))
                for i1 in range(len(gl)):
                    for i2 in range(i1 + 1, len(gl)):
                        diff = np.abs(means[i1] - means[i2])
                        bound = thresh[i1] + thresh[i2]
                        valid = (counts[i1] > 0) & (counts[i2] > 0)
                        if np.any(valid & (diff > bound)):
                            restart = True
                            break
                    if restart:
                        break

            if not restart:
                for a in bad:
                    c = cum_count[a, t + 1] - cum_count[a, sigmas]
                    s = cum_reward[a, t + 1] - cum_reward[a, sigmas]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        m = np.where(c > 0, s / np.where(c == 0, 1, c), 0.0)
                    bound = eviction_gap[a] / 4.0 + np.sqrt(2 * logT / np.where(c == 0, 1, c))
                    diff = np.abs(m - eviction_mean[a])
                    if np.any((c > 0) & (diff > bound)):
                        restart = True
                        break

            if restart:
                start_new_episode()
                continue

            for a, obligs in list(sampling_oblig.items()):
                keep = []
                for (d, n, s) in obligs:
                    if count_iv(a, s, t) < n:
                        keep.append([d, n, s])
                        new_bad.add(a)
                sampling_oblig[a] = keep

            gl = list(good)
            if len(gl) >= 2:
                means = np.zeros((len(gl), len(sigmas)))
                counts = np.zeros((len(gl), len(sigmas)))
                for idx, a in enumerate(gl):
                    c = cum_count[a, t + 1] - cum_count[a, sigmas]
                    s = cum_reward[a, t + 1] - cum_reward[a, sigmas]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        means[idx] = np.where(c > 0, s / np.where(c == 0, 1, c), 0.0)
                    counts[idx] = c
                best_val, evict = -np.inf, None
                for i2 in range(len(gl)):
                    c2 = counts[i2]
                    valid2 = c2 >= 2
                    if not np.any(valid2):
                        continue
                    right = np.sqrt(C1 * logT / np.where(c2 <= 1, 1, c2 - 1))
                    for i1 in range(len(gl)):
                        if i1 == i2:
                            continue
                        left = means[i1] - means[i2]
                        margin = np.where(valid2, left - right, -np.inf)
                        j = np.argmax(margin)
                        if margin[j] > best_val and margin[j] > 0:
                            best_val, evict = margin[j], (gl[i2], left[j], sigmas[j])
                if evict is not None:
                    a, left_val, s_val = evict
                    good.discard(a)
                    new_bad.add(a)
                    eviction_mean[a] = mean_iv(a, s_val, t)
                    eviction_gap[a] = left_val
                    sampling_oblig[a] = []

            bad = new_bad
            good = set(range(K)) - bad

        return dict(net_reward=net_reward, reward=reward_hist, chosen_arm=chosen_hist)
