import numpy as np
import gurobipy as gp
from gurobipy import GRB
from random import uniform,seed,randint
from MDP_class import MDP
import time
import csv

def linearize_H1(x,x_bar):
    # Linearize the function H_1(x)=1/2*(x_1+x_2)^2 about the point x_bar
    # See the paper titled 'Synthesis in pMDPs: A tale of 1001 parameters' by Cubuktepe et al. for further details

    # Inputs:
    # x : a list variable of size n
    # x_bar : a list variable of size n

    expression = 1/2*(x_bar[0]+x_bar[1])**2+(x_bar[0]+x_bar[1])*(x[0]-x_bar[0])+(x_bar[0]+x_bar[1])*(x[1]-x_bar[1])
    return expression

def linearize_H2(x,x_bar):
    # Linearize the function H_2(x)=1/2*(x_1^2+x_2^2) about the point x_bar
    # See the paper titled 'Synthesis in pMDPs: A tale of 1001 parameters' by Cubuktepe et al. for further details

    # Inputs:
    # x : a list variable of size n
    # x_bar : a list variable of size n

    expression = 1/2*(x_bar[0]**2+x_bar[1]**2)+x_bar[0]*(x[0]-x_bar[0])+x_bar[1]*(x[1]-x_bar[1])
    return expression

def CCP_method_for_incentive_synthesis(model,rewards,init,target,blocks,epsilon,Model_type=None):
    # This function synthesizes a sequence of incentives for a given N-BMP instance
    # by solving the corresponding nonlinear optimization problem through convex-concave procedure

    # Inputs:
    # model: a class variable corresponding to the given MDP instance
    # rewards: a dict variable corresponding to reward functions of all possible agent types
    # init: a scalar variable corresponding to the initial state of the MDP
    # target: a list variable corresponding to the set of target states
    # blocks: a list variable corresponding to the set of absorbing states
    # epsilon: a scalar variable indicating the suboptimality of the solution
    # Model_type: a str variable indicating whether or not the model is deterministic ('D')



    MDP_m = MDP(model)
    num_of_states = len(MDP_m.states())
    num_of_actions = len(MDP_m.actions())
    num_of_types = len(rewards)


    # Construct initial distribution
    alpha = np.zeros((num_of_states,1))
    alpha[init] = 1

    # CCP parameters
    delta=1e-2 # termination condition
    tau=0.0001 # initial slack parameter
    eta=1.1 # slack multiplier
    tau_max=1e6 # max slack parameter

    ## Compute max reachability probability and construct reach reward function (r)
    max_reach , max_reach_policy= MDP_m.compute_max_reach_value_and_policy(init,target,blocks)
    reach_rew = MDP_m.reward_for_reach(target)


    ### Compute an initial feasible point
    gamma_bar = {}
    lambda_theta_bar = [{} for k in range(num_of_types)]
    Q_theta_bar = [{} for k in range(num_of_types)]
    V_p_bar = [{} for k in range(num_of_types)]
    Q_p_bar = [{} for k in range(num_of_types)]
    nu_theta_bar = [{} for k in range(num_of_types)]
    for typee in range(num_of_types):
        for statee in MDP_m.states():
            for actt in MDP_m.active_actions()[statee]:
                gamma_bar[(statee,actt)], nu_theta_bar[typee][(statee,actt)] =  randint(5,10),uniform(0,1)
                lambda_theta_bar[typee][(statee,actt)] =  uniform(0,1)
                Q_p_bar[typee][(statee,actt)] = randint(20,40)
                Q_theta_bar[typee][(statee,actt)] = rewards[typee][(statee,actt)]+gamma_bar[(statee,actt)]

    new_obj =0
    old_obj=new_obj+100

    ##### Construct the optimization problem
    start = time.time()
    total_time=0
    m = gp.Model()
    m.setParam('BarHomogeneous',1)
    m.setParam( 'OutputFlag', False )
    # Variables
    gamma = m.addVars(num_of_states,num_of_actions, lb=0, name='incentives')
    slacks1 = m.addVars(num_of_types,num_of_states,num_of_actions, lb=0, name='slacks')
    slacks2 = m.addVars(num_of_types,num_of_states, lb=0, name='slacks2')
    slacks3 = m.addVars(num_of_types,num_of_states, lb=0, name='slacks3')
    slacks4 = m.addVars(num_of_types,num_of_states,num_of_actions,  lb=0, name='slacks4')
    omega = m.addVar(lb=-GRB.INFINITY)
    nu = m.addVars(num_of_types,num_of_states,num_of_actions,lb=0, name='actions')
    V_theta = m.addVars(num_of_types,num_of_states,lb=-GRB.INFINITY, name='V_theta')
    Q_theta = m.addVars(num_of_types,num_of_states,num_of_actions, lb=-GRB.INFINITY,name='Q_theta')
    V_p = m.addVars(num_of_types,num_of_states, lb=0, name='V_p')
    Q_p = m.addVars(num_of_types,num_of_states,num_of_actions,lb=0, name='Q_p')
    lambda_theta = m.addVars(num_of_types,num_of_states,num_of_actions, lb=0,name='lambda_theta')
    mu_theta = m.addVars(num_of_types,num_of_states,num_of_actions, lb=0, name='mu_theta')

    ## NOTE:
    ## We implement the algorithm without using the variables Q_theta(s,a) and Q_{p,theta}(s,a).
    ## Note that the constraints (20b) and (20k) are redundant equality constraints which are
    ## just introduced to define these variables.


    outflow_counter,inflow_counter = {},{}
    total_reach_prob,sum_successor_p, total_action_counter = {},{},{}
    slack_sum=gp.LinExpr()
    for theta_O in range(num_of_types):
        total_reach_prob[theta_O] = gp.LinExpr()
        for state_O in range(num_of_states):
            slack_sum.add( slacks2[theta_O,state_O],1)
            slack_sum.add( slacks3[theta_O,state_O],1)
            if state_O not in target and state_O not in blocks:
                outflow_counter[(theta_O,state_O)] = gp.LinExpr()
                total_action_counter[(theta_O,state_O)] = gp.LinExpr()
                for ind_a_O,act_O in enumerate(model[0][state_O]): # available actions in a state
                    slack_sum.add( slacks1[theta_O,state_O,ind_a_O],1)
                    slack_sum.add( slacks4[theta_O,state_O,ind_a_O],1)
                    total_reach_prob[theta_O].add(mu_theta[theta_O,state_O,ind_a_O],reach_rew[(state_O,act_O)])
                    outflow_counter[(theta_O,state_O)].add(mu_theta[theta_O,state_O,ind_a_O] , 1)
                    total_action_counter[(theta_O,state_O)].add(nu[theta_O,state_O,ind_a_O] , 1)
                    m.addConstr(Q_theta[theta_O,state_O,ind_a_O] == rewards[theta_O][(state_O,act_O)]+gamma[state_O,ind_a_O])
                    m.addConstr(V_theta[theta_O,state_O] >= Q_theta[theta_O,state_O,ind_a_O])

                    # Principal's total cost for type theta
                    sum_successor_p[(theta_O,state_O,ind_a_O)] = gp.LinExpr()
                    for k2,succ2 in enumerate(model[1][(state_O,act_O)][1]):
                        sum_successor_p[(theta_O,state_O,ind_a_O)].add(V_p[theta_O,succ2] , model[1][(state_O,act_O)][0][k2])

                    m.addConstr( Q_p[theta_O,state_O,ind_a_O] == gamma[state_O,ind_a_O]+sum_successor_p[(theta_O,state_O,ind_a_O)])

                # Principal's flow constraint for type theta and state s
                inflow_counter[(theta_O,state_O)] = gp.LinExpr()
                for pair_O in model[1]:
                    if state_O in model[1][pair_O][1]:
                        trans_prob = model[1][pair_O][1].index(state_O)
                        act_index_O = model[0][pair_O[0]].index(pair_O[1])
                        inflow_counter[(theta_O,state_O)].add(mu_theta[theta_O,pair_O[0],act_index_O],model[1][pair_O][0][trans_prob])

                m.addConstr( outflow_counter[(theta_O,state_O)]-inflow_counter[(theta_O,state_O)] == alpha[state_O] )
                m.addConstr( total_action_counter[(theta_O,state_O)] == 1)

        #Principal's reachability constraint for type theta
        m.addConstr( total_reach_prob[theta_O] >= max_reach)

    # At each iteration verify whether the solution is feasible
    verify_reach_for_type=[False for i in range(num_of_types)]


    num_of_iter_CCP=0
    while abs(new_obj-old_obj) > delta or slack_var_sum > 1e-2:

        cons_list=[]
        V_theta_sum_bilinears = [{} for i in range(num_of_types)]
        V_p_sum_bilinears = [{} for i in range(num_of_types)]

        for theta_O in range(num_of_types):
            for state_O in range(num_of_states):
                if state_O not in target and state_O not in blocks:
                    V_theta_sum_bilinears[theta_O][state_O],V_p_sum_bilinears[theta_O][state_O] = [],[]
                    for ind_a_O,act_O in enumerate(model[0][state_O]): # available actions in a state

                        # The following summation will be used as the upper bound on V_theta[theta, state] at the end of the for loop
                        V_theta_sum_bilinears[theta_O][state_O].append(linearize_H1([nu[theta_O,state_O,ind_a_O],Q_theta[theta_O,state_O,ind_a_O]],\
                                                                     [nu_theta_bar[theta_O][(state_O,act_O)],Q_theta_bar[theta_O][(state_O,act_O)]])\
                                                        -1/2*(nu[theta_O,state_O,ind_a_O]*nu[theta_O,state_O,ind_a_O]+Q_theta[theta_O,state_O,ind_a_O]*Q_theta[theta_O,state_O,ind_a_O]))


                        # Uniqueness of the agent theta's optimal policy
                        for inner_ind_a_O, inner_act_O in enumerate(model[0][state_O]):
                            if inner_ind_a_O != ind_a_O:
                                a=m.addQConstr(slacks1[theta_O,state_O,ind_a_O]+linearize_H1([nu[theta_O,state_O,ind_a_O],Q_theta[theta_O,state_O,ind_a_O]],\
                                                         [nu_theta_bar[theta_O][(state_O,act_O)],Q_theta_bar[theta_O][(state_O,act_O)]])\
                                            -1/2*(nu[theta_O,state_O,ind_a_O]*nu[theta_O,state_O,ind_a_O]+Q_theta[theta_O,state_O,ind_a_O]*Q_theta[theta_O,state_O,ind_a_O])\
                                            >= -linearize_H2([nu[theta_O,state_O,ind_a_O],Q_theta[theta_O,state_O,inner_ind_a_O]],\
                                                             [nu_theta_bar[theta_O][(state_O,act_O)],Q_theta_bar[theta_O][(state_O,inner_act_O)]])\
                                            +1/2*(nu[theta_O,state_O,ind_a_O]+Q_theta[theta_O,state_O,inner_ind_a_O])*(nu[theta_O,state_O,ind_a_O]+Q_theta[theta_O,state_O,inner_ind_a_O])\
                                            +nu[theta_O,state_O,ind_a_O]*epsilon)
                                cons_list.append(a)

                        # The following summation will be used as the lower bound on V_p[theta, state] at the end of the for loop
                        V_p_sum_bilinears[theta_O][state_O].append( -linearize_H2([nu[theta_O,state_O,ind_a_O],Q_p[theta_O,state_O,ind_a_O]],\
                                                                     [nu_theta_bar[theta_O][(state_O,act_O)],Q_p_bar[theta_O][(state_O,act_O)]])\
                                                        +1/2*(nu[theta_O,state_O,ind_a_O]+Q_p[theta_O,state_O,ind_a_O])*(nu[theta_O,state_O,ind_a_O]+Q_p[theta_O,state_O,ind_a_O]))

                        # Introduce the bilinear equality constraint mu(s,a)=nu(s,a)*lambda(s,a)
                        if Model_type == 'D':
                            a=m.addConstr(mu_theta[theta_O,state_O,ind_a_O] >=0)
                            cons_list.append(a)
                            a=m.addConstr(mu_theta[theta_O,state_O,ind_a_O] <=nu[theta_O,state_O,ind_a_O])
                            cons_list.append(a)
                            a=m.addConstr(mu_theta[theta_O,state_O,ind_a_O] <=lambda_theta[theta_O,state_O,ind_a_O]+slacks4[theta_O,state_O,ind_a_O])
                            cons_list.append(a)
                            a=m.addConstr(mu_theta[theta_O,state_O,ind_a_O] >=nu[theta_O,state_O,ind_a_O]+lambda_theta[theta_O,state_O,ind_a_O]-1)
                            cons_list.append(a)
                        else:
                            a=m.addQConstr(slacks4[theta_O,state_O,ind_a_O]+mu_theta[theta_O,state_O,ind_a_O] >= -linearize_H2([nu[theta_O,state_O,ind_a_O],lambda_theta[theta_O,state_O,ind_a_O]],\
                                                                                [nu_theta_bar[theta_O][(state_O,act_O)],lambda_theta_bar[theta_O][(state_O,act_O)]])\
                                                                  +1/2*(nu[theta_O,state_O,ind_a_O]+lambda_theta[theta_O,state_O,ind_a_O])*(nu[theta_O,state_O,ind_a_O]+lambda_theta[theta_O,state_O,ind_a_O]))
                            cons_list.append(a)
                            a=m.addQConstr(-slacks5[theta_O,state_O,ind_a_O]+mu_theta[theta_O,state_O,ind_a_O] <= linearize_H1([nu[theta_O,state_O,ind_a_O],lambda_theta[theta_O,state_O,ind_a_O]],\
                                                                                [nu_theta_bar[theta_O][(state_O,act_O)],lambda_theta_bar[theta_O][(state_O,act_O)]])\
                                                                  -1/2*(nu[theta_O,state_O,ind_a_O]*nu[theta_O,state_O,ind_a_O]+lambda_theta[theta_O,state_O,ind_a_O]*lambda_theta[theta_O,state_O,ind_a_O]))
                            cons_list.append(a)


                    a=m.addQConstr(V_theta[theta_O,state_O] <= gp.quicksum(V_theta_sum_bilinears[theta_O][state_O])+slacks2[theta_O,state_O])
                    cons_list.append(a)
                    a=m.addQConstr(slacks3[theta_O,state_O] + V_p[theta_O,state_O] >= gp.quicksum(V_p_sum_bilinears[theta_O][state_O]))
                    cons_list.append(a)

            m.addConstr( omega >= V_p[theta_O,init])
        m.setObjective(omega+tau*slack_sum, GRB.MINIMIZE)
        m.setParam("FeasibilityTol", 1e-3);

        # Optimize model
        m.optimize()
        m.remove(cons_list)

        # Extract incentives from the solution
        incentivized_rewards=[{} for k in range(num_of_types)]
        incentives_as_rewards={}
        for i in range(num_of_types):
            for pair in model[1]:
                index_action=model[0][pair[0]].index(pair[1])
                incentivized_rewards[i][pair]=rewards[i][pair]+gamma[pair[0],index_action].x
                incentives_as_rewards[pair]=gamma[pair[0],index_action].x

        # Compute optimal policies of all agent types under provided incentives
        optimal_policy=[{} for k in range(num_of_types)]
        for types in range(num_of_types):
            for state in range(num_of_states):
                reward_vec= [round(incentivized_rewards[types][(state,a)],5) for a in MDP_m.active_actions()[state] ]
                optimal_policy[types][state]=model[0][state][reward_vec.index(max(reward_vec))]
        #print(optimal_policy)

        # Verify whether the solution is feasible
        verify_reach_for_type=[]
        for type in range(num_of_types):
            verify_reach_for_type.append(MDP_m.verify_reach_deterministic(init,target,optimal_policy[type]))
        print(verify_reach_for_type)
        verify_cost_for_type=[]
        if False not in verify_reach_for_type:
            for type in range(num_of_types):
                verify_cost_for_type.append(MDP_m.verify_cost_deterministic(init,target,optimal_policy[type],incentives_as_rewards))
            print('Actual total cost: '+str(max(verify_cost_for_type)))


        # Update the point about which the linearization is performed
        for type2 in range(num_of_types):
            for state2 in MDP_m.states():
                for k222,act2 in enumerate(model[0][state2]):
                    if gamma[state2,k222].x >= 1e-3:
                        gamma_bar[(state2,act2)] = round(gamma[state2,k222].x ,3)
                    else:
                        gamma_bar[(state2,act2)] = 0
                    if  nu[type2,state2,k222].x >= 1e-3:
                        nu_theta_bar[type2][(state2,act2)] = round(nu[type2,state2,k222].x,3)
                    else:
                        nu_theta_bar[type2][(state2,act2)] = 0
                    lambda_theta_bar[type2][(state2,act2)] = round(lambda_theta[type2,state2,k222].x,3)
                    Q_theta_bar[type2][(state2,act2)] = round(Q_theta[type2,state2,k222].x,3)
                    Q_p_bar[type2][(state2,act2)] = round(Q_p[type2,state2,k222].x,3)

        # Print the results of the iteration
        old_obj=new_obj
        new_obj = omega.x
        slack_var_sum = (m.objVal-omega.x)/tau
        total_time += m.Runtime
        num_of_iter_CCP += 1
        print('Number of iterations: '+str(num_of_iter_CCP))
        print('Elapsed time: '+str(total_time))
        print('Solver Status: '+str(GRB.OPTIMAL))
        print('Total cost to the principal: '+str(omega.x))
        print('Real objective: '+str(m.objVal))
        print('Total slack: '+str(slack_var_sum))
        print('------------------------------------------------')

        # Increase the multiplier for the slack variables
        tau = tau*eta
        if tau >=tau_max:
            tau = tau_max
        if False not in verify_reach_for_type:
            break

    return incentives_as_rewards, incentivized_rewards, optimal_policy
