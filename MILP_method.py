import numpy as np
import gurobipy as gp
from gurobipy import GRB
from random import randint
from MDP_class import MDP

def MILP_method_for_incentive_synthesis(model,rewards,init,target,blocks,epsilon):
    # This function synthesizes a sequence of incentives for a given N-BMP instance
    # by solving the corresponding MILP

    # Inputs:
    # model: a class variable corresponding to the given MDP instance
    # rewards: a dict variable corresponding to reward functions of all possible agent types
    # init: a scalar variable corresponding to the initial state of the MDP
    # target: a list variable corresponding to the set of target states
    # blocks: a list variable corresponding to the set of absorbing states
    # epsilon: a scalar variable indicating the suboptimality of the solution

    MDP_m=MDP(model)
    num_of_states = len(MDP_m.states())
    num_of_actions = len(MDP_m.actions())
    num_of_types = len(rewards)

    # Construct initial distribution
    alpha = np.zeros((num_of_states,1))
    alpha[init]=1

    # Compute big-M parameters
    R_max=max( [rewards[i][max(rewards[i], key=rewards[i].get)] for i in range(num_of_types)] )
    R_min=min( [rewards[i][min(rewards[i], key=rewards[i].get)] for i in range(num_of_types)] )
    M_e = 2 * (R_max-R_min+epsilon)

    C={}
    for pair in MDP_m.state_action_pairs():
        array=[]
        for types in range(num_of_types):
            max_reward = R_min
            for act in MDP_m.active_actions()[pair[0]]:
                if rewards[types][(pair[0],act)] > max_reward:
                    max_reward = rewards[types][(pair[0],act)]
            array.append(max_reward-rewards[types][pair])
        C[pair] = max(array)+epsilon

    min_cost , _ =  MDP_m.compute_min_cost_subject_to_max_reach(init,target,blocks,C)
    M_e_bar=min_cost

    M_e2_bar=num_of_states

    ## Compute max reachability probability and construct reach reward function
    max_reach , _ =  MDP_m.compute_max_reach_value_and_policy(init,target,blocks)
    reach_rew = MDP_m.reward_for_reach(target)

    ##### Construct the optimization problem
    ## We implement the algorithm without using the variables Q_theta(s,a) and Q_{p,theta}(s,a).
    ## Note that the constraints (20b) and (20k) are redundant equality constraints which are
    ## just introduced to define these variables.
    m = gp.Model()
#    m.setParam( 'OutputFlag', False )

    # Variables
    gamma=m.addVars(num_of_states,num_of_actions, lb=0.0, name='incentives')
    omega=m.addVar(lb=-GRB.INFINITY)
    X=m.addVars(num_of_types,num_of_states,num_of_actions, vtype=GRB.BINARY, name='actions') # binary variables
    V_theta=m.addVars(num_of_types,num_of_states, lb=-GRB.INFINITY, name='V_theta')
    V_p=m.addVars(num_of_types,num_of_states, lb=0.0, name='V_p')
    lambda_theta=m.addVars(num_of_types,num_of_states,num_of_actions, lb=0.0, name='lambda_theta')
    mu_theta=m.addVars(num_of_types,num_of_states,num_of_actions, name='mu_theta')

    sum_successor_p={}
    dummy_integer_counter={}
    sum_actions_counter={}
    sum_mu_counter={}
    total_reach_prob={}

    for theta in range(num_of_types):
        for state in range(num_of_states):
            sum_mu_counter[(theta,state)]=gp.LinExpr()
            dummy_integer_counter[(theta,state)]=gp.LinExpr()
            sum_actions_counter[(theta,state)]=gp.LinExpr()

            for i,act in enumerate(model[0][state]): # available actions in a state
                sum_actions_counter[(theta,state)].add(mu_theta[theta,state,i] , 1)
                dummy_integer_counter[(theta,state)].add(X[theta,state,i] , 1)

                ################ McCormick envelopes for mu_theta variables
                ################ Corresponding constraints (20i), (20j)
                m.addConstr(0 <= mu_theta[theta,state,i])
                m.addConstr(mu_theta[theta,state,i] <= M_e2_bar*X[theta,state,i])
                m.addConstr(lambda_theta[theta,state,i]-M_e2_bar*(1-X[theta,state,i])<= mu_theta[theta,state,i])
                m.addConstr(mu_theta[theta,state,i] <= lambda_theta[theta,state,i])

                ################ Value function of the agent type theta
                ################ Corresponding constraints -- (20c), (20d)

                m.addConstr( V_theta[theta,state] >= rewards[theta][(state,act)]+gamma[state,i])
                m.addConstr( V_theta[theta,state] <= rewards[theta][(state,act)]+gamma[state,i]+(1-X[theta,state,i])*M_e)


                ############## Uniqueness of the agent theta's optimal policy
                ################ Corresponding constraints -- (20e)
                for j,act2 in enumerate(model[0][state]):
                    if act2 != act:
                        m.addConstr( rewards[theta][(state,act2)]+gamma[state,j]+epsilon <= \
                                     rewards[theta][(state,act)]+gamma[state,i]+(1-X[theta,state,i])*M_e)


                ############### Principal's total cost for type theta
                ############### Corresponding constraints -- (20l), (20m)
                sum_successor_p[(theta,state,i)]=gp.LinExpr()
                for k,succ in enumerate(model[1][(state,act)][1]):
                    sum_successor_p[(theta,state,i)].add(V_p[theta,succ] , model[1][(state,act)][0][k])
                m.addConstr( V_p[theta,state] >= sum_successor_p[(theta,state,i)]+gamma[state,i]-(1-X[theta,state,i])*M_e_bar)
                m.addConstr( V_p[theta,state] <= sum_successor_p[(theta,state,i)]+gamma[state,i]+(1-X[theta,state,i])*M_e_bar)

            ################ Principal's flow constraint for type theta and state s
            ################ Corresponding constraint -- (20f)
            for pair in model[1]:
                if state in model[1][pair][1]:
                    trans_prob=model[1][pair][1].index(state)
                    act_index=model[0][pair[0]].index(pair[1])
                    sum_mu_counter[(theta,state)].add(mu_theta[theta,pair[0],act_index],model[1][pair][0][trans_prob])
            if state not in target and state not in blocks:
                m.addConstr( sum_actions_counter[(theta,state)]-sum_mu_counter[(theta,state)] == alpha[state] )

            ################ Corresponding constraint -- (20o)
            m.addConstr( dummy_integer_counter[(theta,state)] == 1)

        ################ Principal's reachability constraint for type theta
        ################ Corresponding constraint -- (20h)
        total_reach_prob[theta]=gp.LinExpr()
        for pair in model[1]:
            index_action=model[0][pair[0]].index(pair[1])
            total_reach_prob[theta].add(mu_theta[theta,pair[0],index_action],reach_rew[pair])
        m.addConstr( total_reach_prob[theta]>=max_reach)

        ################ Corresponding constraint -- (20n)
        m.addConstr( omega >= V_p[theta,init])

    m.setObjective(omega, GRB.MINIMIZE)
    m.setParam("IntFeasTol", 1e-4);

    # Optimize model
    m.optimize()

    incentivized_rewards=[{} for k in range(num_of_types)]
    incentive_amount={}
    for i in range(num_of_types):
        for pair in model[1]:
            index_action=model[0][pair[0]].index(pair[1])
            incentivized_rewards[i][pair]=rewards[i][pair]+gamma[pair[0],index_action].x
            incentive_amount[pair]=gamma[pair[0],index_action].x
    compute_time=m.Runtime

    optimal_policy=[{} for k in range(num_of_types)]
    for types in range(num_of_types):
        for state in range(num_of_states):
            optimal_policy[types][state]=model[0][state][0]
            opt_reward=incentivized_rewards[types][(state,optimal_policy[types][state])]
            for action in model[0][state]:
                if incentivized_rewards[types][(state,action)]-opt_reward >= epsilon/10:
                    optimal_policy[types][state]=action
                    opt_reward=incentivized_rewards[types][(state,action)]

    return incentive_amount, incentivized_rewards, optimal_policy, compute_time
