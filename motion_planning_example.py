import numpy as np
from MDP_model_for_grid_world import grid_world
from random import randint,seed,uniform
from MILP_method import MILP_method_for_incentive_synthesis
from CCP_method import CCP_method_for_incentive_synthesis
from MDP_class import MDP
import csv


# Problem parameters
num_of_types = 3 # number of agent types
epsilon = 1e-1 # parameter to make an action uniquely optimal
row , column , slip =7 ,7 , 0  # Grid world environment parameters
target = [row*column-1] # Target state in the grid world
blocks=[] # absorbing states as blocks
init = 0 # Initial state of the MDP
absorb_states=[row*column-1]


# Construct MDP model
model = grid_world(row,column,absorb_states,slip)

# Generate reward functions for all agent types
seed(1000)
rewards=[{} for k in range(num_of_types)]
for i in range(num_of_types):
    for pair in model[1]:
        if pair[0] not in absorb_states:
            if i == 0:
                if pair[1]=='L':
                    rewards[i][pair]=0
                elif pair[0]< 1*column and pair[1] == 'E':
                    rewards[i][pair]=-2
                else:
                    rewards[i][pair]=-randint(300,500)/100
            elif i == 1:
                if pair[1]=='L':
                    rewards[i][pair]=0
                elif pair[0]%column <=0 and pair[1] == 'N':
                    rewards[i][pair]=-2
                else:
                    rewards[i][pair]=-randint(300,500)/100
            elif i == 2:
                if pair[1]=='L':
                    rewards[i][pair]=0
                elif pair[0]%column >=4  and pair[0]%column <=4  and pair[1] == 'N':
                    rewards[i][pair]=-2
                elif pair[0]<column  and pair[1] == 'E':
                    rewards[i][pair]=-2
                else:
                    rewards[i][pair]=-round(uniform(800,1000)/100,3)
        else:
            rewards[i][pair]=0

# Compute the lower bound on the minimum total cost to the principal
lower_bound_vector=[]
for i in range(num_of_types):
    cost_of_control=MDP(model).cost_of_control(target,rewards[i],epsilon)
    total_cost,_=MDP(model).compute_min_cost_subject_to_max_reach(init,target,blocks,cost_of_control)
    lower_bound_vector.append(total_cost)
#print(lower_bound_vector)

# Compute a global optimal solution to the N-BMP using the MILP
incentives, incentivized_rewards, optimal_policy, compute_time= MILP_method_for_incentive_synthesis(model,rewards,init,target,blocks,epsilon)

# Compute a local optimal solution to the N-BMP using the CCP
incentives, incentivized_rewards, optimal_policy=CCP_method_for_incentive_synthesis(model,rewards,init,target,blocks,epsilon,'D')
print(incentives)
