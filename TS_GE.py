# The upper bound for the reward is going to be max of reward mean of all current arms + 1 and then we divide to get the success rate probability
import numpy as np
import matplotlib.pyplot as plt
import math

def TS_GE(Arms,Time_Horizon,Delta):
    """
    Arms : List of arms
    Time_Horizon : Time horizon for which we want to run the algorithm
    Delta : Maximum error allowed in the mean reward of the arms beyond which it gives an error
    """
    # Thompson-Sampling Phase
    K = Arms.size()
    alphas = [1 for i in range(0,K)]
    betas = [1 for i in range(0,K)]
    
    Rewards = []
    Time_Step = []
    Regret = []
    p = 0
    Max_Episodes = math.sqrt(Time_Horizon)
    for e in range(0,int(Max_Episodes)):
        # Thompson Sampling phase
        for t in range(0,Time_Horizon):
            pass 
        p +=1 
        # Broadcast Probing Phase
        # Build the estimate
        flag = False 
        # Checking if change is detected
        
        if (flag):
            # change is detected hence we go to group exploration phase
            pass
            
        

