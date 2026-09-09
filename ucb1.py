"""
ucb1.py -- classical UCB1 (Auer, Cesa-Bianchi, Fischer 2002), on the shared
BanditAlgorithm interface. Included as a stationary-bandit baseline: it has
no change-detection mechanism at all, so on a non-stationary environment it
shows you what "not adapting" costs relative to FastAdSwitch / TS-GE.

Cleaned up from the previous version:
  - Replaced the flat exploration constant `c=2.0` with a `sigma` parameter
    that actually calibrates the confidence bonus to the environment's real
    noise level: bonus = sqrt(2*sigma^2*ln(t)/n_i). The old flat c=2.0
    implicitly assumed unit-variance (or [0,1]-bounded) reward noise
    regardless of what sigma the environment actually used -- not wrong,
    just uncalibrated. Verified empirically the difference is small on this
    benchmark (UCB1's regret is dominated by its total inability to recover
    after a change, not by exploration-constant tuning), but this is the
    principled choice going forward, and it removes the risk of silently
    over/under-exploring on environments with a very different noise scale.
  - The `rng` (seed) now has an actual purpose: random tie-breaking when
    multiple arms share the same UCB value (most likely early on, before
    arms have been pulled enough times to differentiate). The old version
    used np.argmax, which always resolves ties toward the LOWEST index --
    a silent, systematic bias toward arm 0 that had nothing to do with
    which arm is actually better. Previously, the `seed` parameter was
    accepted but never used anywhere (dead code); now it's meaningful.
"""
import math
import numpy as np

from bandit_base import BanditAlgorithm


class UCB1(BanditAlgorithm):
    def __init__(self, K, T, sigma=1.0, seed=None):
        """
        sigma : the environment's actual reward noise standard deviation
                (or your best estimate of it). Calibrates the confidence
                bonus -- pass the same sigma you used to construct the
                environment for an honestly-tuned baseline.
        """
        super().__init__(K, T)
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def run(self, reward_fn):
        K, T, sigma = self.K, self.T, self.sigma
        sum_reward = np.zeros(K)
        n_pulls = np.zeros(K)
        reward_hist, chosen_hist = [], []

        for t in range(1, T + 1):
            if t <= K:
                j = t - 1  # play each arm once first
            else:
                means = sum_reward / np.maximum(n_pulls, 1)
                bonus = np.sqrt(2 * sigma ** 2 * math.log(t) / np.maximum(n_pulls, 1))
                ucb = means + bonus
                best = ucb.max()
                candidates = np.flatnonzero(ucb == best)
                j = int(self.rng.choice(candidates)) if len(candidates) > 1 else int(candidates[0])
            r = reward_fn(j, t)
            sum_reward[j] += r
            n_pulls[j] += 1
            reward_hist.append(r)
            chosen_hist.append(j)

        return dict(net_reward=sum(reward_hist), reward=reward_hist, chosen_arm=chosen_hist)
