"""
TS_GE.py — Thompson Sampling with (partial) Group-Exploration restart logic
for non-stationary bandits.

Patched bugs (compared to the version currently in the repo):
  1. `import Arms` then calling `Arms(i*0.1, 1)` treated the *module* as a
     callable class. Fixed to `from Arms import ARMS` and construct with the
     ARMS class's real 5-argument signature.
  2. `Arm_list` was always freshly randomised inside the function, silently
     ignoring the `Arm` argument the caller passed in. Fixed so the function
     actually uses the arms it was given, and only pads with dummy
     "dead" arms up to the next power of two (and those dummy arms are now
     explicitly excluded from selection -- see fix #7 below).
  3. `while (~(K and ~(K & (K-1)))):` used bitwise NOT on a Python bool/int,
     which does not mean logical negation in Python (`~True == -2`, always
     truthy) -> dead/broken loop condition. Fixed to the standard
     power-of-two test: `while (K & (K - 1)) != 0:`.
  4. `np.bernoulli(p)` does not exist in numpy. Fixed to
     `np.random.binomial(1, p)`, with `p` clipped to [0, 1] since simulated
     rewards aren't bounded and would otherwise produce an invalid
     probability.
  5. `np.argmax(<generator expression>)` does not do what it looks like it
     does -- numpy can't reduce a bare Python generator the way it reduces a
     list/array, so this silently always behaved as if index 0 were passed.
     Fixed to build a numpy array first: `np.argmax(np.array([...]))`.
  6. `estimates = [0 for i in range(K)]` followed by `estimates.append(...)`
     inside a loop appended past the pre-filled zeros instead of filling
     the list in by index. Fixed to `estimates[i] = estimate`.
  7. Padding/dummy arms (added to round K to a power of two) had an
     extreme placeholder mean and, if ever actually selected by Thompson
     Sampling, produced reward ~ -1e9 and corrupted the whole run. Fixed by
     restricting arm selection and the best-expected-reward computation to
     the real arms only (`num_real_arms`), so dummy arms exist structurally
     but are never played.

Remaining stubs (left as documented TODOs, not invented from scratch since
they weren't specified in the original file): `super_arms(...)` and the
"Broadcast Probing Phase" change-detection block are placeholders here too
-- fill these in once you have the exact group-exploration test your
professor wants.
"""
import numpy as np
import math
import random
from Arms import ARMS


def TS_GE(Arm, Time_Horizon, Delta):  # K, T, delta
    """
    Arm : list of ARMS instances (the real arms of the problem)
    Time_Horizon : time horizon for which we want to run the algorithm
    Delta : maximum error allowed in the mean reward of the arms
    """

    def super_arms(arms, n):
        # TODO: build "super-arm" groupings for the group-exploration
        # phase. Left as a stub, matching the original file -- fill in
        # once the exact grouping rule is specified.
        s = {}
        K = len(arms)
        for k in range(1, n):
            for i in range(1, K + 1):
                pass
        return s

    num_real_arms = len(Arm)
    K = len(Arm)
    Arm_list = list(Arm)          # FIX: use the arms passed in, don't fabricate new ones
    priors = [[1, 1] for _ in range(K)]  # [alpha, beta] per arm

    # pad with "dead" dummy arms up to the next power of two
    # FIX: correct power-of-two test (bitwise NOT on a Python int/bool is
    # NOT logical negation, so the original condition never behaved as
    # intended)
    while (K & (K - 1)) != 0:
        Arm_list.append(ARMS(K, -1e9, 1, change=0, Time_Horizon=Time_Horizon))
        priors.append([1, 1])
        K += 1

    upper_bound = max(a.arm_mean for a in Arm_list[:num_real_arms]) + 1  # to normalize rewards

    probability_of_change = 1 - (1 / Time_Horizon) ** (1 / (Time_Horizon ** 0.5 - Time_Horizon ** 0.4))
    probability_of_change = random.uniform(probability_of_change, 1)

    Rewards_List = []
    Time_Step = []
    Regret_List = []
    t = 0

    # ---- Explore-Then-Commit style initial mean estimation ----
    n_etc = int(1 / (2 * Delta ** 2) * math.log(Time_Horizon))  # ~ O(log T) pulls/arm
    n_etc = max(n_etc, 1)
    estimates = [0.0 for _ in range(K)]
    count_arms = [0 for _ in range(K)]
    for i in range(K):
        mean = Arm_list[i].arm_mean
        for _ in range(n_etc):
            mean += np.random.normal(mean, 1)
        estimates[i] = mean / n_etc  # FIX: assign by index instead of appending

    Length_Episode = max(int(math.sqrt(Time_Horizon)), 2)

    # ---- Main loop: alternating Thompson-Sampling / checking episodes ----
    for e in range(0, Time_Horizon, Length_Episode):
        Length_TS = max(int(Length_Episode - Time_Horizon ** 0.4), 1)
        for t_ts in range(Length_TS):
            if t >= Time_Horizon:
                break
            # FIX: padding/dummy arms (added above to reach a power of two)
            # must never actually be played -- restrict the argmax to real arms
            samples = np.array([np.random.beta(priors[a][0], priors[a][1]) for a in range(num_real_arms)])
            max_index = int(np.argmax(samples))
            t += 1
            reward = float(np.random.normal(Arm_list[max_index].arm_mean, 1))
            Rewards_List.append(reward)
            Time_Step.append(t)
            best_expected_reward = max(a.arm_mean for a in Arm_list[:num_real_arms])
            Regret_List.append(best_expected_reward - reward)

            # FIX: np.bernoulli doesn't exist; use np.random.binomial, and
            # clip the probability into [0, 1] since `reward` isn't bounded
            success_event_probability = float(np.clip(reward / upper_bound, 0.0, 1.0))
            R_value = np.random.binomial(1, success_event_probability)
            priors[max_index][0] += 1 - R_value
            priors[max_index][1] += R_value
            count_arms[max_index] += 1

        if t >= Time_Horizon:
            break

        # ---- Broadcast Probing / change-detection phase ----
        # TODO: this is left as a stub in the original file too. Implement
        # the actual statistical test here (e.g. compare recent empirical
        # means against `estimates` within tolerance `Delta`) and set
        # `flag = True` to trigger a restart/group-exploration phase.
        flag = False
        if flag:
            # change detected -> restart estimation / group exploration
            pass

    return Rewards_List, Time_Step, Regret_List


if __name__ == "__main__":
    T = 2000
    means = [0.2, 0.5, 0.8]
    arms = [ARMS(i, m, 1.0, change=0, Time_Horizon=T) for i, m in enumerate(means)]
    rewards, steps, regret = TS_GE(arms, T, Delta=0.1)
    print(f"Ran TS_GE for T={T}: total reward={sum(rewards):.2f}, "
          f"cumulative regret={sum(regret):.2f}, rounds played={len(rewards)}")
