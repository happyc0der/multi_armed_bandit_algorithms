"""
epsilon_greedy.py -- classical epsilon-greedy baseline, on the shared
BanditAlgorithm interface. With decay=False and a moderate fixed epsilon,
its constant random exploration can accidentally help it recover from a
distribution change faster than a purely stationary algorithm like UCB1 --
useful contrast to have in the comparison table alongside the
change-aware algorithms.
"""
import numpy as np

from bandit_base import BanditAlgorithm


class EpsilonGreedy(BanditAlgorithm):
    def __init__(self, K, T, epsilon=0.1, decay=True, seed=None):
        """
        epsilon : exploration probability (or decay numerator if decay=True)
        decay   : if True, uses epsilon/t (shrinking exploration over time,
                  standard for stationary bandits); if False, uses a fixed
                  epsilon every round (better suited to non-stationary
                  environments, at the cost of some permanent regret)
        """
        super().__init__(K, T)
        self.epsilon = epsilon
        self.decay = decay
        self.rng = np.random.default_rng(seed)

    def run(self, reward_fn):
        K, T = self.K, self.T
        sum_reward = np.zeros(K)
        n_pulls = np.zeros(K)
        reward_hist, chosen_hist = [], []

        for t in range(1, T + 1):
            eps = self.epsilon / t if self.decay else self.epsilon
            if t <= K:
                j = t - 1  # play each arm once first
            elif self.rng.random() < eps:
                j = int(self.rng.integers(K))
            else:
                means = sum_reward / np.maximum(n_pulls, 1)
                j = int(np.argmax(means))
            r = reward_fn(j, t)
            sum_reward[j] += r
            n_pulls[j] += 1
            reward_hist.append(r)
            chosen_hist.append(j)

        return dict(net_reward=sum(reward_hist), reward=reward_hist, chosen_arm=chosen_hist)
