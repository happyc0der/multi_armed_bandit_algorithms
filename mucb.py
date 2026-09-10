"""
mucb.py -- M-UCB (Monitored UCB), Cao, Wen, Kveton, Xie, "Nearly Optimal
Adaptive Procedure with Change Detection for Piecewise-Stationary Bandit",
AISTATS 2019. This is the explicit competitor benchmark named in the
TS-GE paper's abstract alongside ADSWITCH, so it belongs in this
comparison for a complete, standardized benchmark.

Algorithm summary (Algorithm 1 = CD, Algorithm 2 = M-UCB in the paper):
  - Time is divided into repeating cycles of length floor(K/gamma).
  - Within each cycle, the first K steps are FORCED uniform round-robin
    sampling (one pull per arm) -- this guarantees every arm gets fresh
    data regularly, specifically so the change-detection test has
    observations even for arms that plain UCB would otherwise neglect.
  - The remaining floor(K/gamma) - K steps in each cycle use standard
    UCB1 selection, computed only from data gathered since the last
    detected change (tau).
  - After every pull, once an arm has >= w observations since tau, a
    simple sliding-window mean-shift test (Algorithm 1: CD) is run on its
    most recent w rewards: compare the sum of the second half against the
    sum of the first half; if the absolute difference exceeds a threshold
    b, a change is declared and the WHOLE algorithm resets (tau <- t,
    all counts/buffers cleared) -- unlike TS-GE, there's no attempt to
    localize *which* arm changed; a change in any one arm just restarts
    everything.

Parameter defaults follow the paper's own Remark 1 tuning guidance, given
an assumed minimum detectable change magnitude `delta` and a rough guess
`M_estimate` for the number of piecewise-stationary segments (only used to
set the uniform-exploration fraction gamma; the algorithm doesn't need to
know the true number of changes to run, this is just for calibrating how
aggressively to explore uniformly):
    w     ~ (4/delta^2) * (sqrt(log(2*K*T^2)) + sqrt(log(2*T)))^2
    b     = sqrt(w * log(2*K*T^2) / 2)
    gamma ~ sqrt((M_estimate-1) * K * min(w/2, ceil(b/delta) + 3*sqrt(w)) / (2*T))

Note: the paper's analysis assumes rewards are bounded in [0,1] (Section
3.1). The UCB1 exploration term here (sqrt(2*log(t-tau)/n)) is the
classical unit-variance-proxy form and is not separately calibrated to
arbitrary reward scales -- same caveat documented in ucb1.py. Pass a
`delta` appropriate to your actual reward scale (not [0,1]-normalized) and
this still works correctly for change-magnitude/window-size tuning, since
that part of the formula doesn't assume [0,1] rewards.
"""
import math
from collections import deque

import numpy as np

from bandit_base import BanditAlgorithm


class MUCB(BanditAlgorithm):
    def __init__(self, K, T, delta=1.0, M_estimate=2, w=None, b=None, gamma=None, seed=None):
        """
        delta      : assumed minimum detectable change magnitude (drives
                     the default window size w -- smaller delta means a
                     larger window is needed to detect it reliably)
        M_estimate : rough guess at the number of piecewise-stationary
                     segments, used only to calibrate gamma (uniform
                     exploration fraction); doesn't need to be exact
        w, b, gamma : override any of the paper's auto-computed defaults
                     directly if you have your own calibration
        """
        super().__init__(K, T)
        self.rng = np.random.default_rng(seed)

        if w is None:
            w = int(4 / delta ** 2 * (math.sqrt(math.log(2 * K * T ** 2)) + math.sqrt(math.log(2 * T))) ** 2)
            w = max(w + (w % 2), 4)  # CD (Algorithm 1) requires an even window
        self.w = w

        self.b = b if b is not None else math.sqrt(self.w * math.log(2 * K * T ** 2) / 2)

        if gamma is None:
            gamma = math.sqrt(
                max(M_estimate - 1, 1) * K *
                min(self.w / 2, math.ceil(self.b / delta) + 3 * math.sqrt(self.w)) / (2 * T)
            )
        self.gamma = min(max(gamma, 1e-6), 1.0)
        self.cycle_len = max(int(math.floor(K / self.gamma)), K)

    def run(self, reward_fn):
        K, T, w, b = self.K, self.T, self.w, self.b
        tau = 0
        n = np.zeros(K, dtype=int)
        # Running sums give the UCB means in O(K) per round. The previous
        # version called np.mean over each arm's full history every round,
        # which is O(t - tau) per round and therefore O(T^2) overall: 17.7 s at
        # K=64/T=60,000, and hours at T=1e6.
        sums = np.zeros(K, dtype=float)
        # CD (Algorithm 1) only ever inspects the most recent w rewards, so a
        # bounded deque is sufficient and keeps memory at O(K*w) rather than
        # O(T).
        buffers = [deque(maxlen=w) for _ in range(K)]
        reward_hist, chosen_hist, detections = [], [], []

        for t in range(1, T + 1):
            # position within the current exploration/exploitation cycle
            A = (t - tau - 1) % self.cycle_len
            if A < K:
                a_t = A  # forced uniform round-robin sampling
            else:
                means = sums / np.maximum(n, 1)
                ucb = means + np.sqrt(2 * math.log(max(t - tau, 2)) / np.maximum(n, 1))
                a_t = int(np.argmax(ucb))

            r = reward_fn(a_t, t)
            n[a_t] += 1
            sums[a_t] += r
            buffers[a_t].append(r)
            reward_hist.append(r)
            chosen_hist.append(a_t)

            # Algorithm 1 (CD): sliding-window mean-shift test
            if n[a_t] >= w:
                recent = list(buffers[a_t])
                stat = abs(sum(recent[w // 2:]) - sum(recent[:w // 2]))
                if stat > b:
                    tau = t
                    n[:] = 0
                    sums[:] = 0.0
                    buffers = [deque(maxlen=w) for _ in range(K)]
                    detections.append(t)

        return dict(net_reward=sum(reward_hist), reward=reward_hist,
                    chosen_arm=chosen_hist, detections=detections)
