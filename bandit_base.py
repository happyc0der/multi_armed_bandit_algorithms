"""
algorithms/base.py -- common interface every bandit algorithm in this repo
must implement, so they can all be dropped into the same benchmark harness
and compared fairly on the same environment.
"""
from abc import ABC, abstractmethod


class BanditAlgorithm(ABC):
    def __init__(self, K, T, **kwargs):
        self.K = K
        self.T = T

    @abstractmethod
    def run(self, reward_fn):
        """
        reward_fn(arm_or_arms, t) -> float
            arm_or_arms : int (single arm index) or list[int] (group probe,
                          returns the mean reward across the group)
            t           : 1-indexed round number

        Must return a dict with at least:
          'reward'      : list[float], reward obtained each round the
                           algorithm actually played (length may be < T if
                           the algorithm has fixed per-phase batching that
                           doesn't divide T evenly)
          'chosen_arm'  : list matching 'reward' 1:1; each entry is an int
                          arm index for a genuine single-arm decision, or a
                          string tag (e.g. 'BP', 'GE:2') for rounds that
                          probe a group of arms rather than choosing one
        """
        raise NotImplementedError
