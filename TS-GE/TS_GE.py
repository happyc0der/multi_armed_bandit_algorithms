"""
TS_GE.py -- TS-GE: "Actively Tracking the Optimal Arm in Non-Stationary
Environments with Mandatory Probing" (Gourab Ghatak, IEEE).

This is a from-scratch implementation of the actual TS-GE algorithm
described in the paper you were working from (Algorithm 1 "TS-GE" and
Algorithm 2 "Construct Super-Arms, CSA"), replacing the old stub file that
never implemented the core loop.

Algorithm summary (see the paper for full derivations):
  1. ETC (explore-then-commit) initialization: pull every arm
     n_ETC = ceil(1/(2*delta^2) * ln(1/p_L)) times to get an initial mean
     estimate for each arm, well-localized within +-delta w.h.p.
  2. The horizon is divided into N_l = sqrt(T) episodes of length
     T_l = sqrt(T), each split into:
       - a Thompson Sampling (TS) phase of length T_TS = sqrt(T) - T^0.4,
         using classical Beta-Bernoulli TS (on a normalized Bernoulli
         success signal derived from the real-valued reward), and
       - a mandatory Broadcast Probing (BP) phase of length
         T_BP = T^0.4, where ALL arms are played simultaneously and the
         agent observes only the group-averaged reward.
  3. At the end of each BP phase, the pre-BP average of all arms'
     estimated means is compared to the BP phase's observed average
     reward. If they differ by >= 4*delta, a change is flagged.
  4. If flagged, a Group Exploration (GE) phase runs: arms are grouped
     into d = log2(K) "super-arms" via a bit-coding scheme (arm i, coded
     as i+1 in binary, belongs to super-arm k iff bit k of (i+1) is set).
     Each super-arm is played n_ge times; a super-arm whose new observed
     average deviates from its pre-computed estimate by >= 2*delta is
     "flagged". The single arm whose bit-code exactly matches the set of
     flagged super-arms is identified as the changed arm, its mean is
     algebraically recovered from the flagged groups' new averages, and
     its Beta prior is re-seeded from the (still-unflagged) arm with the
     closest current mean estimate, so it doesn't have to re-explore from
     a neutral prior.

Implementation notes / deviations from the paper (documented, not hidden):
  - K is padded to the next power of two if needed; the paper suggests
    padding with arms of constant reward -infinity. Using literal -1e9
    values wrecks the BP-phase averages numerically for demonstration
    purposes, so padding arms instead get a mean far enough below the
    real R_min that their Bernoulli success probability is ~0 (so TS
    practically never selects them), without blowing up the reward scale.
  - The paper assumes rewards are non-negative and bounded above by
    R_max, appropriate for its wireless-power/data-rate case study. Here
    R_min/R_max default to a symmetric window around the initial arm
    means so this also works for arms with negative Gaussian means; you
    can override R_min/R_max explicitly if you know your reward's true
    range.
  - The exact formula for n_ge (GE phase per-super-arm sample count)
    isn't spelled out numerically in the paper (only appears symbolically
    in the algorithm and regret proof). It's set here to the same order
    as n_ETC (both are "O(log T)" concentration-based sample counts);
    override `n_ge` explicitly if you have a tighter formula from your
    own derivation.
  - Detectability of a change of size Delta_C requires
    Delta_C / K >= 4*delta (a single arm's change only shifts the
    K-arm group average by Delta_C/K), so for many arms K you need a
    proportionally larger Delta_C or smaller delta to reliably detect a
    change -- this is an inherent property of the algorithm's group-test
    design, not a bug in this implementation.

Usage:
    from TS_GE import TS_GE
    algo = TS_GE(K=4, T=6000, delta=0.1, means=[0.2, 0.5, 0.3, 0.8],
                 change_schedule=[(3000, 0, 1.5)], seed=1)
    history = algo.run()   # dict with 'reward', 'regret', 'detections'
"""
import numpy as np
import math


class TS_GE:
    def __init__(self, K, T, delta, means, sigma=1.0, change_schedule=None,
                 n_ge=None, seed=None, R_min=None, R_max=None):
        self.rng = np.random.default_rng(seed)
        self.num_real = K
        self.sigma = sigma

        Kp = 1
        while Kp < K:
            Kp *= 2
        self.K = Kp
        self.d = int(math.ceil(math.log2(Kp + 1)))

        real_means = list(means)
        self.R_min = R_min if R_min is not None else min(real_means) - 5 * sigma
        self.R_max = R_max if R_max is not None else max(real_means) + 5 * sigma
        dummy_mean = self.R_min - 5 * sigma
        self.true_means = np.array(real_means + [dummy_mean] * (Kp - K), dtype=float)

        self.T = T
        self.delta = delta
        self.n_ge = n_ge or max(int(1 / (2 * delta ** 2) * math.log(max(T, 2))), 3)
        self.change_schedule = sorted(change_schedule or [], key=lambda x: x[0])
        self.t = 0
        self.history = {"reward": [], "regret": [], "detections": []}

    def _apply_changes(self):
        while self.change_schedule and self.change_schedule[0][0] == self.t:
            _, idx, new_mean = self.change_schedule.pop(0)
            self.true_means[idx] = new_mean

    def _draw(self, idx):
        return self.rng.normal(self.true_means[idx], self.sigma)

    def _bernoulli_signal(self, r):
        p = np.clip((r - self.R_min) / (self.R_max - self.R_min), 0.0, 1.0)
        return self.rng.binomial(1, p)

    def _log_step(self, reward):
        self.t += 1
        self._apply_changes()
        best = self.true_means[:self.num_real].max()
        self.history["reward"].append(reward)
        self.history["regret"].append(best - reward)

    def _construct_super_arms(self):
        """Algorithm 2 (CSA): arm i (0-indexed) belongs to super-arm k iff
        bit k of (i+1) is set. Using i+1 (not i) avoids the all-zero
        codeword that arm index 0 would otherwise get, which would make
        it structurally unidentifiable by the GE phase."""
        B = [[] for _ in range(self.d)]
        for i in range(self.K):
            code = i + 1
            for k in range(self.d):
                if (code >> k) & 1:
                    B[k].append(i)
        return B

    def run(self):
        K, T, delta = self.K, self.T, self.delta
        alpha = np.ones(K)
        beta = np.ones(K)
        sum_reward = np.zeros(K)
        n_pulls = np.zeros(K)

        def mu_hat(i):
            return sum_reward[i] / n_pulls[i] if n_pulls[i] > 0 else self.true_means[i]

        # ---- ETC initialization (Section II-B) ----
        p_L = 1.0 / max(T, 2)
        n_etc = max(int(1 / (2 * delta ** 2) * math.log(1 / p_L)), 1)
        for i in range(K):
            for _ in range(n_etc):
                if self.t >= T:
                    break
                r = self._draw(i)
                sum_reward[i] += r
                n_pulls[i] += 1
                self._log_step(r)

        Tl = max(int(math.sqrt(T)), 2)
        T_TS = max(int(math.sqrt(T) - T ** 0.4), 1)
        T_BP = max(Tl - T_TS, 1)

        while self.t < T:
            # ---- Thompson Sampling phase ----
            for _ in range(T_TS):
                if self.t >= T:
                    break
                theta = self.rng.beta(alpha, beta)
                theta[self.num_real:] = -1  # padding arms never chosen
                j = int(np.argmax(theta))
                r = self._draw(j)
                r_pi = self._bernoulli_signal(r)
                alpha[j] += 1 - r_pi
                beta[j] += r_pi
                sum_reward[j] += r
                n_pulls[j] += 1
                self._log_step(r)

            if self.t >= T:
                break

            mu_baseline = np.mean([mu_hat(i) for i in range(K)])

            # ---- Broadcast Probing phase (Eq. 5) ----
            bp_rewards = []
            for _ in range(T_BP):
                if self.t >= T:
                    break
                r_each = np.array([self._draw(i) for i in range(K)])
                r_bp = r_each.mean()
                bp_rewards.append(r_bp)
                self._log_step(r_bp)

            if not bp_rewards:
                break
            mu_bp = np.mean(bp_rewards)
            change_detected = abs(mu_baseline - mu_bp) >= 4 * delta

            if change_detected and self.t < T:
                self.history["detections"].append(self.t)
                # ---- Group Exploration phase (Eq. 6-8, Algorithm 2) ----
                B = self._construct_super_arms()
                mu_hat_B_pre = [np.mean([mu_hat(i) for i in Bk]) for Bk in B]
                flagged = [False] * self.d
                mu_hat_B_new = [None] * self.d
                for k in range(self.d):
                    if self.t >= T:
                        break
                    group_rewards = []
                    for _ in range(self.n_ge):
                        if self.t >= T:
                            break
                        r_each = np.array([self._draw(i) for i in B[k]])
                        r_g = r_each.mean()
                        group_rewards.append(r_g)
                        self._log_step(r_g)
                    if group_rewards:
                        mu_hat_B_new[k] = np.mean(group_rewards)
                        flagged[k] = abs(mu_hat_B_pre[k] - mu_hat_B_new[k]) >= 2 * delta

                # Identify the (unique) arm whose bit-code exactly matches
                # the flagged-group pattern (Eq. 7): it must belong to
                # every flagged group and to no unflagged group.
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
                    # Recover the changed arm's new mean (Eq. 8): from each
                    # flagged group containing j, subtract out the (still
                    # trusted) means of the other members.
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
                        # Re-seed the changed arm's TS prior from the
                        # closest-matching (still trusted) arm instead of
                        # starting from a neutral Beta(1,1) prior.
                        others_idx = [i for i in range(K) if i != j]
                        closest = min(others_idx, key=lambda i: abs(mu_hat(i) - new_mu_j))
                        alpha[j], beta[j] = alpha[closest], beta[closest]

        return self.history


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    ARTIFACT_DIR = "artifacts"
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    T = 6000
    means = [0.2, 0.6]
    change_schedule = [(3000, 0, 1.6)]  # arm 0 jumps from 0.2 to 1.6 at t=3000
    algo = TS_GE(K=2, T=T, delta=0.1, means=means, sigma=1.0,
                 change_schedule=change_schedule, seed=5)
    h = algo.run()

    total_reward = sum(h["reward"])
    total_regret = sum(h["regret"])
    print(f"TS-GE ran for T={T}: total reward={total_reward:.2f}, "
          f"cumulative regret={total_regret:.2f}, "
          f"change detected at t={h['detections']} (true change at t=3000)")

    window = 100
    rewards = np.array(h["reward"])
    def rolling(x, w):
        c = np.cumsum(np.insert(x, 0, 0))
        return (c[w:] - c[:-w]) / w
    roll = rolling(rewards, window)

    plt.figure(figsize=(9, 4.5))
    plt.plot(roll, color="blue", label="reward (rolling mean, window=100)")
    plt.axvline(3000, color="red", linestyle="--", label="true change point")
    for d_t in h["detections"]:
        plt.axvline(d_t, color="green", linestyle=":", alpha=0.7)
    plt.xlabel("Round")
    plt.ylabel("Reward")
    plt.title("TS-GE tracking a mean-shift change (green = detected change)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "ts_ge_tracking.png"), dpi=150)
    plt.close()
    print("Saved artifacts/ts_ge_tracking.png")
