import numpy as np
from MILP_method import MILP_method_for_incentive_synthesis
from CCP_method import CCP_method_for_incentive_synthesis
from MDP_class import MDP
from MDP_model_for_discounting import reward_functions, discount_model


model=discount_model()
rewards=reward_functions()
init = 0
target = [15]
blocks = []
epsilon= 1e-2
#print(rewards)

incentive_amount, incentivized_rewards, optimal_policy, compute_time =MILP_method_for_incentive_synthesis(model,rewards,init,target,blocks, epsilon)
#incentivized_rewards, optimal_policy=optimal_incentive_CCP_variant2(model,rewards,init,target,epsilon,'D')
print(incentivized_rewards)
print(optimal_policy)
