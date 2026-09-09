"""
ts_ge.py -- TS-GE (Ghatak). Cross-checked against the arXiv version history
of this paper (arXiv:2205.10366): v1 (May 2022, matches the PDF you
originally gave me) -> v2 (Nov 2022, WITHDRAWN) -> v3 (Nov 2022, current,
retitled "Fast Change Identification in Multi-Play Bandits...", changelog:
"Corrected the assumptions and removed the case-study due to an error").

=====================================================================
Fix (v3-confirmed): NO extra bit needed for the K=2^d edge case
=====================================================================
d = log2(K) exactly, no padding. v3 clarifies: for EVERY group (flagged
or not), the changed arm must be in that group if flagged, or in its
COMPLEMENT if not; intersecting across all groups naturally identifies
arm index 0 (all-zero bit-code, member of zero groups) as the unique arm
consistent with ALL groups being unflagged. My exact-bit-match check
already implements this correctly.

=====================================================================
NEW gap found while testing the above: Eq. 8 cannot estimate arm 0's mean
=====================================================================
Removing the extra bit exposed a real problem, distinct from the
identification question: v3 explains how to IDENTIFY arm 0 as the changed
arm (it's the one in zero groups, all consistent-with-unflagged), but its
mean-recovery formula (Eq. 8) sums over "flagged groups containing the
changed arm" -- and arm 0 belongs to NO groups, flagged or not, so that
sum is structurally empty. Verified directly: without a fallback, arm 0's
mean/prior never got updated after being correctly identified, so the BP
phase kept re-flagging the same uncorrected drift every episode
indefinitely (detections at t=3024, 3535, 4046, 4557, 5068, 5579, ...,
instead of a single genuine detection).
FIXED by falling back to the Broadcast Probing phase's own aggregate
deviation to recover arm 0's new mean directly: since exactly one arm
changed and BP phase reward is the K-arm average, new_mu_j = old mu_hat(j)
+ K * (mu_bp - mu_baseline). This is only used for the (rare, K=2^d-only)
case where the identified arm belongs to zero groups; all other arms still
use the paper's Eq. 8 group-subtraction recovery.

=====================================================================
Still-unresolved: the alpha/beta update direction (verified against v3)
=====================================================================
v3 Algorithm 1, lines 11-12, are BYTE-IDENTICAL to v1:
    alpha_j <- alpha_j + 1 - R*
    beta_j  <- beta_j + R*
Backwards from standard Thompson Sampling convention. NOT part of what
got fixed in the v1->v3 revision, so no official errata confirms it's a
typo -- but empirically, the literal version gives ~3.5x worse regret and
systematically favors low-reward arms. Default here is the standard-
convention fix (`paper_literal=False`); set `paper_literal=True` to
reproduce the paper's literal formula.

Two other bugs found and fixed earlier (implementation errors on my end,
not paper issues):
  FIX #1 -- Beta-prior reseed after Group Exploration: reinitialize
    (alpha, beta) from the changed arm's own new mean, borrowing only
    the pseudo-count (confidence) from the closest arm, not its success
    rate (copying verbatim was wrong whenever "closest" wasn't actually
    close in absolute terms).
  FIX #2 -- Changed-arm identification requires an EXACT bit-vector
    match against the flagged-groups pattern (implements v3's
    complement-intersection rule, Eq. 5), not a "subset" test.
"""
import math
import numpy as np

from bandit_base import BanditAlgorithm


class TS_GE(BanditAlgorithm):
    def __init__(self, K, T, delta, R_max, n_ge=None, seed=None,
                 dummy_mean=None, paper_literal=False):
        """
        K              : number of real arms
        T              : time horizon
        delta          : localization threshold (see paper)
        R_max          : known upper bound on reward (rewards assumed
                         non-negative; R/R_max is a direct ratio)
        n_ge           : override for the GE phase's per-super-arm
                         sample count
        dummy_mean     : reward for padding arms (added to round K up to
                         a power of two)
        paper_literal  : if True, use the paper's literal (apparently
                         inverted) Beta update. Default False uses the
                         standard TS convention that actually works.
        """
        super().__init__(K, T)
        self.num_real = K
        Kp = 1
        while Kp < K:
            Kp *= 2
        self.K = Kp
        self.d = max(int(round(math.log2(Kp))), 1)
        self.delta = delta
        self.R_max = R_max
        self.dummy_mean = dummy_mean if dummy_mean is not None else -10 * R_max
        self.n_ge = n_ge or max(int(1 / (2 * delta ** 2) * math.log(max(T, 2))), 3)
        self.paper_literal = paper_literal
        self.rng = np.random.default_rng(seed)

    def _single(self, reward_fn, idx, t):
        if idx < self.num_real:
            return reward_fn(idx, t)
        return self.dummy_mean

    def _group(self, reward_fn, indices, t):
        real_idx = [i for i in indices if i < self.num_real]
        n_dummy = len(indices) - len(real_idx)
        total = 0.0
        if real_idx:
            total += reward_fn(real_idx, t) * len(real_idx)
        total += n_dummy * self.dummy_mean
        return total / len(indices)

    def _bernoulli_signal(self, r):
        p = np.clip(r / self.R_max, 0.0, 1.0)
        return self.rng.binomial(1, p)

    def _construct_super_arms(self):
        """Algorithm 2 (CSA): arm i (0-indexed) belongs to super-arm k iff
        bit k of i is set. K = 2^d, exactly d groups."""
        B = [[] for _ in range(self.d)]
        for i in range(self.K):
            for k in range(self.d):
                if (i >> k) & 1:
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
                if self.paper_literal:
                    alpha[j] += 1 - r_pi
                    beta[j] += r_pi
                else:
                    alpha[j] += r_pi
                    beta[j] += 1 - r_pi
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
                    bits = [bool((i >> k) & 1) for k in range(self.d)]
                    if bits == flagged:
                        candidates.append(i)

                if len(candidates) == 1:
                    j = candidates[0]
                    estimates = []
                    for k in range(self.d):
                        if flagged[k] and mu_hat_B_new[k] is not None:
                            others = sum(mu_hat(i) for i in B[k] if i != j)
                            est = len(B[k]) * mu_hat_B_new[k] - others
                            estimates.append(est)

                    # FIX: fallback for arms belonging to zero groups (e.g.
                    # arm 0), where Eq.8's group-based recovery is
                    # structurally impossible -- use the BP phase's own
                    # aggregate deviation to recover the new mean instead.
                    if not estimates:
                        new_mu_j = mu_hat(j) + K * (mu_bp - mu_baseline)
                    else:
                        new_mu_j = float(np.mean(estimates))

                    sum_reward[j] = new_mu_j
                    n_pulls[j] = 1
                    others_idx = [i for i in range(K) if i != j]
                    closest = min(others_idx, key=lambda i: abs(mu_hat(i) - new_mu_j))
                    total_pseudo_count = alpha[closest] + beta[closest]
                    p_j = np.clip(new_mu_j / self.R_max, 1e-3, 1 - 1e-3)
                    if self.paper_literal:
                        alpha[j] = (1 - p_j) * total_pseudo_count
                        beta[j] = p_j * total_pseudo_count
                    else:
                        alpha[j] = p_j * total_pseudo_count
                        beta[j] = (1 - p_j) * total_pseudo_count

        return dict(net_reward=sum(reward_hist), reward=reward_hist,
                    chosen_arm=chosen_hist, detections=detections)
