# multi-armed-bandit_algorithms
Working on various multi-armed bandit algorithms for further uses in financial markets


The paper for TS-GE misidentifies the alpha beta update formula, if we account for it then the regret is much much lower.

Benchmark details 
1. Epsilon Greedy: Keep decay false since it is for stationary bandits. Otherwise it has no way of finding the changed arms after some time, unless the decay function is very complex and can be changed to estimate when the arms change so that it can go exploring again.
2. 