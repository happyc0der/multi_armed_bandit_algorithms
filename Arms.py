"""
Arms.py — arm implementation used by TS_GE.py.

No functional changes were needed here (this file itself is fine); the
bug was in TS_GE.py, which imported this module and then called it as if
it were a class:  `Arms(i*0.1, 1)`  instead of  `Arms.ARMS(...)`.
"""
import random
import numpy as np


class ARMS:
    def __init__(self, arm_index, arm_mean, arm_std, change, Time_Horizon):
        """
        Args:
            arm_index (int): index of the arm
            arm_mean (float): average expected reward of the arm
            arm_std (float): standard deviation of the arm
            change (int): 0 if the arm is stationary, else 1 (non-stationary;
                not fully implemented yet -- see note below)
            Time_Horizon (int): the time horizon for the problem
        """
        self.arm_index = arm_index
        self.arm_mean = arm_mean
        self.arm_std = arm_std
        self.Time_Horizon = Time_Horizon
        self.number_of_times_chosen = [0 for _ in range(Time_Horizon + 1)]
        self.time_when_first_chosen = None
        self.time_when_last_chosen = 0
        self.eviction_observed_mean_reward = None
        self.eviction_gap = None

        if change == 1:
            # NOTE: non-stationary reward generation isn't implemented yet
            # (the original code only rolled a random change-count and did
            # nothing with it, leaving reward_List empty -> IndexError on
            # first use). Fall back to a stationary reward stream so the
            # arm is at least usable; replace this with real changepoint
            # logic if/when you implement the non-stationary case.
            self.reward_List = np.random.normal(arm_mean, arm_std, Time_Horizon + 1)
        else:
            self.reward_List = np.random.normal(arm_mean, arm_std, Time_Horizon + 1)

    def tell_reward(self, time_step):
        return self.reward_List[time_step]

    def __str__(self):
        return f"Arm Index: {self.arm_index} Arm Mean: {self.arm_mean} Arm Std: {self.arm_std}"

    def __repr__(self):
        return f"Arm Index: {self.arm_index}"

    def update_chosen(self, time):
        if self.time_when_first_chosen is None:
            self.time_when_first_chosen = time
        self.number_of_times_chosen[time] += 1
        self.time_when_last_chosen = time

    def chosen_in_interval(self, start, end):
        return self.number_of_times_chosen[start:end + 1].count(1)

    def mean_rewards_observed(self, start, end):
        num = denom = 0
        flag = False
        for i in range(start, end + 1):
            if self.number_of_times_chosen[i] == 1:
                flag = True
                num += self.reward_List[i]
                denom += 1
        return num / denom if flag else 0
