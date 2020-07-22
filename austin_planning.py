import numpy as np
from austin_city_model import austin_city_MDP
from random import randint,seed,uniform
from MILP_method import MILP_method_for_incentive_synthesis
from CCP_method import CCP_method_for_incentive_synthesis
from MDP_class import MDP
import csv


# Problem parameters
num_of_types = 3 # number of agent types
epsilon = 1e-1 # parameter to make an action uniquely optimal
target = [29] # Target state in the grid world
blocks=[] # absorbing states as blocks
init = 11 # Initial state of the MDP
absorb_states=[29]

# Construct MDP model
model = austin_city_MDP(absorb_states)


dist_matrix = np.loadtxt(open("distance_map.csv", "rb"), delimiter=",", skiprows=0)
time_matrix = np.loadtxt(open("time_map.csv", "rb"), delimiter=",", skiprows=0)

#print(np.size(dist_matrix,1))
rewards=[{} for k in range(num_of_types)]
for pair in model[1]:
    if pair[0] not in absorb_states and pair[0] != pair[1]:
        rewards[0][pair]=-dist_matrix[pair[0],pair[1]]
        rewards[1][pair]=-2*time_matrix[pair[1]]
        rewards[2][pair]=-(0.8*dist_matrix[pair[0],pair[1]]+0.2*time_matrix[pair[1]])
    else:
        rewards[0][pair],rewards[1][pair],rewards[2][pair] = 0,0,0

# Compute the lower bound on the minimum total cost to the principal
lower_bound_vector=[]
for i in range(num_of_types):
    cost_of_control=MDP(model).cost_of_control(target,rewards[i],epsilon)
    total_cost,_=MDP(model).compute_min_cost_subject_to_max_reach(init,target,blocks,cost_of_control)
    lower_bound_vector.append(total_cost)
print(lower_bound_vector)

# Compute a global optimal solution to the N-BMP using the MILP
incentives, incentivized_rewards, optimal_policy, compute_time= MILP_method_for_incentive_synthesis(model,rewards,init,target,blocks,epsilon)

# Compute a local optimal solution to the N-BMP using the CCP
incentives, incentivized_rewards, optimal_policy=CCP_method_for_incentive_synthesis(model,rewards,init,target,blocks,epsilon,'D')
print(incentives)
