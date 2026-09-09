"""
algorithms/ts_ge.py -- TS-GE (Ghatak, "Actively Tracking the Optimal Arm
in Non-Stationary Environments with Mandatory Probing"), refactored onto
the shared BanditAlgorithm interface so it can be benchmarked head-to-head
against FastAdSwitch (and future baselines) on the exact same environment.

What changed vs. the standalone TS_GE.py:
  - No longer owns its own reward simulation (_draw / _apply_changes /
    true_means). All rewards now come from the environment's
    `reward_fn(arm_or_arms, t)` callback -- the algorithm never sees the
    environment's true means, which is more realistic (the paper's own
    Assumption 1 only grants the algorithm knowledge of R_min/R_max, an
    assumed reward range, not the arm means themselves).
  - R_min / R_max are now REQUIRED constructor arguments instead of being
    auto-derived from the (now-hidden) true means -- you must supply your
    own assumed reward bounds, exactly as the paper's Assumption 1 requires.
  - Padding ("dummy") arms added to round K up to a power of two are
    handled internally with a fixed constant reward (never queried from
    the environment, since the environment only knows about your K real
    arms) -- see _single()/_group() below.

Everything else (ETC init, alternating TS/Broadcast-Probing episodes,
Group Exploration bit-coded super-arm localization) is the same algorithm
already validated in the standalone version: a scheduled mean-jump was
detected within ~1 episode length and correctly localized/adapted to.
"""
import math
import numpy as np

from bandit_base import BanditAlgorithm


class TS_GE(BanditAlgorithm):
    def __init__(self, K, T, delta, R_min, R_max, n_ge=None, seed=None, dummy_mean=None):
        """
        K        : number of real arms
        T        : time horizon
        delta    : localization threshold (see paper); smaller = more
                   sensitive detection but needs more ETC/GE samples
        R_min    : assumed lower bound on any real reward the algorithm
                   will see (used only to normalize the Bernoulli success
                   signal for the TS phase's Beta posteriors)
        R_max    : assumed upper bound on any real reward
        n_ge     : override for the GE phase's per-super-arm sample count
                   (not given as an explicit formula in the paper; default
                   is the same order as the ETC phase's sample count)
        dummy_mean : reward value used for padding arms (added to round K
                   up to a power of two); defaults to just below R_min so
                   their Bernoulli success probability is ~0
        """
        super().__init__(K, T)
        self.num_real = K
        Kp = 1
        while Kp < K:
            Kp *= 2
        self.K = Kp
        self.d = int(math.ceil(math.log2(Kp + 1)))
        self.delta = delta
        self.R_min, self.R_max = R_min, R_max
        self.dummy_mean = dummy_mean if dummy_mean is not None else (R_min - (R_max - R_min) / 2)
        self.n_ge = n_ge or max(int(1 / (2 * delta ** 2) * math.log(max(T, 2))), 3)
        self.rng = np.random.default_rng(seed)

    def _single(self, reward_fn, idx, t):
        """Reward for one arm; padding arms never touch the environment."""
        if idx < self.num_real:
            return reward_fn(idx, t)
        return self.dummy_mean

    def _group(self, reward_fn, indices, t):
        """Mean reward across a mixed real/padding group: query the
        environment only for the real arms in the group, and combine with
        the fixed constant for any padding arms in the group."""
        real_idx = [i for i in indices if i < self.num_real]
        n_dummy = len(indices) - len(real_idx)
        total = 0.0
        if real_idx:
            total += reward_fn(real_idx, t) * len(real_idx)
        total += n_dummy * self.dummy_mean
        return total / len(indices)

    def _bernoulli_signal(self, r):
        p = np.clip((r - self.R_min) / (self.R_max - self.R_min), 0.0, 1.0)
        return self.rng.binomial(1, p)

    def _construct_super_arms(self):
        """Algorithm 2 (CSA): arm i (0-indexed) belongs to super-arm k iff
        bit k of (i+1) is set (1-indexed coding avoids the all-zero
        codeword arm index 0 would otherwise get)."""
        B = [[] for _ in range(self.d)]
        for i in range(self.K):
            code = i + 1
            for k in range(self.d):
                if (code >> k) & 1:
                    B[k].append(i)
        return B

    def run(self, reward_fn):
        K, T, delta = self.K, self.T, self.delta
        alpha = np.ones(K)
        beta = np.ones(K)
        sum_reward = np.zeros(K)
        n_pulls = np.zeros(K)
        reward_hist, chosen_hist, detections = [], [], []
        t = 0

        def mu_hat(i):
            return sum_reward[i] / n_pulls[i] if n_pulls[i] > 0 else 0.0

        # ---- ETC initialization ----
        p_L = 1.0 / max(T, 2)
        n_etc = max(int(1 / (2 * delta ** 2) * math.log(1 / p_L)), 1)
        for i in range(K):
            for _ in range(n_etc):
                if t >= T:
                    break
                r = self._single(reward_fn, i, t + 1)
                sum_reward[i] += r
                n_pulls[i] += 1
                t += 1
                reward_hist.append(r)
                chosen_hist.append(i if i < self.num_real else "dummy")

        Tl = max(int(math.sqrt(T)), 2)
        T_TS = max(int(math.sqrt(T) - T ** 0.4), 1)
        T_BP = max(Tl - T_TS, 1)

        while t < T:
            # ---- Thompson Sampling phase ----
            for _ in range(T_TS):
                if t >= T:
                    break
                theta = self.rng.beta(alpha, beta)
                theta[self.num_real:] = -1
                j = int(np.argmax(theta))
                r = self._single(reward_fn, j, t + 1)
                r_pi = self._bernoulli_signal(r)
                alpha[j] += 1 - r_pi
                beta[j] += r_pi
                sum_reward[j] += r
                n_pulls[j] += 1
                t += 1
                reward_hist.append(r)
                chosen_hist.append(j)

            if t >= T:
                break

            mu_baseline = np.mean([mu_hat(i) for i in range(K)])

            # ---- Broadcast Probing phase ----
            bp_rewards = []
            for _ in range(T_BP):
                if t >= T:
                    break
                r_bp = self._group(reward_fn, list(range(K)), t + 1)
                bp_rewards.append(r_bp)
                t += 1
                reward_hist.append(r_bp)
                chosen_hist.append("BP")

            if not bp_rewards:
                break
            mu_bp = np.mean(bp_rewards)
            change_detected = abs(mu_baseline - mu_bp) >= 4 * delta

            if change_detected and t < T:
                detections.append(t)
                # ---- Group Exploration phase ----
                B = self._construct_super_arms()
                mu_hat_B_pre = [np.mean([mu_hat(i) for i in Bk]) for Bk in B]
                flagged = [False] * self.d
                mu_hat_B_new = [None] * self.d
                for k in range(self.d):
                    if t >= T:
                        break
                    group_rewards = []
                    for _ in range(self.n_ge):
                        if t >= T:
                            break
                        r_g = self._group(reward_fn, B[k], t + 1)
                        group_rewards.append(r_g)
                        t += 1
                        reward_hist.append(r_g)
                        chosen_hist.append(f"GE:{k}")
                    if group_rewards:
                        mu_hat_B_new[k] = np.mean(group_rewards)
                        flagged[k] = abs(mu_hat_B_pre[k] - mu_hat_B_new[k]) >= 2 * delta

                candidates = []
                for i in range(K):
                    code = i + 1
                    bits = [(code >> k) & 1 for k in range(self.d)]
                    belongs_only_to_flagged = all(
                        flagged[k] for k in range(self.d) if bits[k] == 1
                    )
                    touches_a_flagged_group = any(
                        bits[k] == 1 and flagged[k] for k in range(self.d)
                    )
                    if belongs_only_to_flagged and touches_a_flagged_group:
                        candidates.append(i)

                if len(candidates) == 1:
                    j = candidates[0]
                    estimates = []
                    for k in range(self.d):
                        if flagged[k] and mu_hat_B_new[k] is not None:
                            others = sum(mu_hat(i) for i in B[k] if i != j)
                            est = len(B[k]) * mu_hat_B_new[k] - others
                            estimates.append(est)
                    if estimates:
                        new_mu_j = float(np.mean(estimates))
                        sum_reward[j] = new_mu_j
                        n_pulls[j] = 1
                        others_idx = [i for i in range(K) if i != j]
                        closest = min(others_idx, key=lambda i: abs(mu_hat(i) - new_mu_j))
                        alpha[j], beta[j] = alpha[closest], beta[closest]

        return dict(net_reward=sum(reward_hist), reward=reward_hist,
                    chosen_arm=chosen_hist, detections=detections)
