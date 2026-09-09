"""
ucb1.py -- classical UCB1 (Auer, Cesa-Bianchi, Fischer 2002), on the shared
BanditAlgorithm interface. Included as a stationary-bandit baseline: it has
no change-detection mechanism, so on a non-stationary environment it shows
you what "not adapting at all" costs relative to FastAdSwitch / TS-GE.
"""
import math
import numpy as np

from bandit_base import BanditAlgorithm


class UCB1(BanditAlgorithm):
    def __init__(self, K, T, c=2.0, seed=None):
        """
        c : exploration constant (2.0 matches the classical Hoeffding-based
            UCB1 bound; larger = more exploration)
        """
        super().__init__(K, T)
        self.c = c
        self.rng = np.random.default_rng(seed)

    def run(self, reward_fn):
        K, T, c = self.K, self.T, self.c
        sum_reward = np.zeros(K)
        n_pulls = np.zeros(K)
        reward_hist, chosen_hist = [], []

        for t in range(1, T + 1):
            if t <= K:
                j = t - 1  # play each arm once first
            else:
                ucb = sum_reward / np.maximum(n_pulls, 1) + \
                      np.sqrt(c * math.log(t) / np.maximum(n_pulls, 1))
                j = int(np.argmax(ucb))
            r = reward_fn(j, t)
            sum_reward[j] += r
            n_pulls[j] += 1
            reward_hist.append(r)
            chosen_hist.append(j)

        return dict(net_reward=sum(reward_hist), reward=reward_hist, chosen_arm=chosen_hist)
