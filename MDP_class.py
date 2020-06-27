import numpy as np
import gurobipy as gp
from gurobipy import GRB

# MDP class object
class MDP(object):

    def __init__(self, MDP=None):
        self.MDP = MDP


    def states(self):
    # MDP.states[i] gives the 'set' of all states
        states=set(i for i in self.MDP[0])
        return states

    def actions(self):
    # MDP.actions[i] gives the 'set' of all actions
        actions=set()
        for state in self.MDP[0]:
             actions.update(self.MDP[0][state])
        return actions

    def active_actions(self):
    # MDP.active_actions[i] gives the 'list' of successor states for the state i
        return self.MDP[0]

    def state_action_pairs(self):
    # MDP.state_action_pairs[i] gives the 'list' of state action pairs
        return self.MDP[1].keys()

    def successors(self):
     # MDP.successors[i] gives the 'set' of successor states for the state i
        succ={i : set() for i in self.states() }
        for pair in self.MDP[1]:
            succ[pair[0]].update(self.MDP[1][pair][1])
        return succ

    def predecessors(self):
     # MDP.predecessors[i] gives the 'set' of predecessor states for the state i
        pre={i : set() for i in self.states() }
        for pair in self.MDP[1]:
            for succ in self.MDP[1][pair][1]:
                pre[succ].add(pair[0])
        return pre

    def pre_state_action_pair(self):
     # MDP.predecessors[i] gives the 'set' of predecessor states for the state i
        pre={i : set() for i in self.states() }
        for pair in self.MDP[1]:
            for succ in self.MDP[1][pair][1]:
                pre[succ].add(pair)
        return pre

    def reward_for_reach(self,target):
        # Construct a reward function expressing the reachability to a set of target states
        # Inputs:
        # target: a 'list' of target states
        reach_rew = {}
        for pair in self.MDP[1]:
            if pair[0] not in target and set(self.MDP[1][pair][1]).intersection(set(target)) != set():
                reach_rew[pair] = 1
            else:
                reach_rew[pair] = 0

        return reach_rew

    def value_evaluate(self,reward ,policy,discount=1):
        # Value evaluation function. Computes the value of each state for a given
        # policy and a discount factor
        Q_val, V_val_new, V_val, diff = {}, {}, {} , {}
        eps=1e-8
        if  discount >= 1:
            discount = 1
        else:
            discount = discount
        for pair in self.MDP[1]:
            Q_val[pair] = 0
            V_val[pair[0]], V_val_new[pair[0]] = 0,0
            diff[pair[0]] = 1
        while diff[max(diff, key=diff.get)] > eps:
            for pair in self.MDP[1]:
                succ_sum = 0
                for k,succ in enumerate(self.MDP[1][pair][1]):
                    succ_sum += self.MDP[1][pair][0][k]*V_val[succ]
                Q_val[pair] = reward[pair] + discount*succ_sum
            for state in self.MDP[0]:
                V_val_new[state] = Q_val[(state,policy[state])]
                diff[state] = abs(V_val_new[state] - V_val[state])
                V_val[state] = V_val_new[state]
        return V_val, Q_val

    def verify_reach_deterministic(self,init,target,policy):
    # For deterministic MDPs, verify whether the given policy reaches the target state
        state=init
        for i in range(len(self.states())):
            state=self.MDP[1][(state,policy[state])][1][0]
        if state in target:
            return True
        else:
            return False

    def verify_cost_deterministic(self,init,target,policy,incentives):
    # For deterministic MDPs, verify the total cost incurred by a given policy
        state=init
        total_cost=0
        for i in range(len(self.states())):
            if state not in target:
                total_cost += incentives[(state,policy[state])]
                state=self.MDP[1][(state,policy[state])][1][0]
        return total_cost

    def cost_of_control(self,target,reward,epsilon):
    # Cost of control function computes the minimum incentive amount required to
    # make an action uniquely optimal
        V,cost={},{}
        for state in self.MDP[0]:
            if state not in target:
                nominal_rewards=[reward[(state,a)] for a in self.MDP[0][state]]
                V[state] = max(nominal_rewards)
                for act in self.MDP[0][state]:
                    cost[(state,act)] = V[state]-reward[(state,act)]+epsilon
            else:
                for act in self.MDP[0][state]:
                    cost[(state,act)] = 0
        return cost



    def compute_max_reach_value_and_policy(self,init, target,blocks):
     # MDP.compute_max_reach_value_and_policy(init,target) returns
     # (i) the maximum reachability probability to a 'list' of target states from an initial state,
     # (ii) a PROPER policy that maximizes the reachability probability

     # Inputs:
     # init : unique initial state
     # target: the 'list' of ABSORBING target states

        # Define reachability reward
        # Define UNIFORM!!! initial distribution
        alpha = np.ones((len(self.states()),1))
        reach_rew=self.reward_for_reach(target)
        alpha2 = np.zeros((len(self.states()),1))
        alpha2[init]=1

        # The following optimization problem is known as the 'dual program'.
        # The variables X(s,a) represent the expected residence time in pair (s,a)
        m = gp.Model("max_reach_value_computation")
        m.setParam( 'OutputFlag', False )
        m2 = gp.Model("max_reach_policy_computation")
        m2.setParam( 'OutputFlag', False )
        X = m.addVars(len(self.states()), len(self.actions()), lb=0.0, name='lambda')
        X2 = m2.addVars(len(self.states()), len(self.actions()), lb=0.0, name='lambda')
        pre_lambda_sum, post_lambda_sum, total_reach_prob = {} , {} , gp.LinExpr()
        pre_lambda_sum2, post_lambda_sum2, total_reach_prob2, total_expected  = {}, {} , gp.LinExpr(), gp.LinExpr()
        for state in self.states():
            pre_lambda_sum[state], post_lambda_sum[state] = gp.LinExpr() , gp.LinExpr()
            pre_lambda_sum2[state] , post_lambda_sum2[state] = gp.LinExpr() , gp.LinExpr()

            for act in self.active_actions()[state]:
                act_ind = self.MDP[0][state].index(act)
                post_lambda_sum[state].add(X[state,act_ind] , 1)
                total_reach_prob.add(X[state,act_ind] , reach_rew[(state,act)])
                post_lambda_sum2[state].add(X2[state,act_ind] , 1)
                total_reach_prob2.add(X2[state,act_ind] , reach_rew[(state,act)])
                total_expected.add(X2[state,act_ind] , 1)

            for pre in self.pre_state_action_pair()[state]:
                trans_prob_index = self.MDP[1][pre][1].index(state)
                act_index = self.MDP[0][pre[0]].index(pre[1])
                pre_lambda_sum[state].add( X[pre[0],act_index] , self.MDP[1][pre][0][trans_prob_index])
                pre_lambda_sum2[state].add( X2[pre[0],act_index] , self.MDP[1][pre][0][trans_prob_index])

            # Flow equation for each state
            if state not in target and state not in blocks:
                m.addConstr( post_lambda_sum[state] - pre_lambda_sum[state] == alpha[state])
                m2.addConstr( post_lambda_sum2[state] - pre_lambda_sum2[state] == alpha2[state])

        m.setObjective(total_reach_prob, GRB.MAXIMIZE)
        m.optimize()

        policy={}
        for state in range(len(self.states())):
            policy[state]=self.active_actions()[state][0]
            for action in self.active_actions()[state]:
                act_ind = self.MDP[0][state].index(action)
                if X[state,act_ind].x >= 1e-4:
                    policy[state] = action
        V_val,_=self.value_evaluate(reach_rew,policy)
        max_reach_val=V_val[init]

        m2.addConstr( total_reach_prob2 >= max_reach_val)
        m2.setObjective( total_expected, GRB.MINIMIZE)
        m2.optimize()
        optimal_policy={}
        for state in range(len(self.states())):
            optimal_policy[state]=self.active_actions()[state][0]
            for action in self.active_actions()[state]:
                act_ind = self.MDP[0][state].index(action)
                if X2[state,act_ind].x >= 1e-4:
                    optimal_policy[state] = action
        print(optimal_policy)
        #residence_times=self.compute_residence(init,target,blocks,optimal_policy)
        return max_reach_val, optimal_policy

    def compute_min_cost_subject_to_max_reach(self,init,target,blocks,cost):
     # MDP.compute_max_reach_value_and_policy(init,target) returns
     # (i) the minimum cost to reach a 'list' of target states with maximum probability,
     # (ii) a policy that minimizes the expected total cost while maximizing the reachability probability

     # Inputs:
     # init: the unique initial state init \in S
     # cost: the 'dictionary' of NONNEGATIVE cost values for each state-action pair

        # Define initial distribution
        alpha = np.zeros((len(self.states()),1))
        alpha[init] = 1

        # Compute maximum reachability probability to the target set
        max_reach_val, _ =self.compute_max_reach_value_and_policy(init,target,blocks)
        reach_rew=self.reward_for_reach(target)

        # The following optimization problem is known as the 'dual program'.
        # The variables X(s,a) represent the expected residence time in pair (s,a)
        m = gp.Model("min_cost_value_computation")
        m.setParam( 'OutputFlag', False )
        m2 = gp.Model("min_cost_policy_computation")
        m2.setParam( 'OutputFlag', False )
        X = m.addVars(len(self.states()), len(self.actions()), lb=0.0, name='lambda')
        X2 = m2.addVars(len(self.states()), len(self.actions()), lb=0.0, name='lambda')
        pre_lambda_sum, post_lambda_sum, total_reach_prob, total_expected_cost= {} , {} , gp.LinExpr(), gp.LinExpr()
        pre_lambda_sum2, post_lambda_sum2, total_reach_prob2 = {}, {} , gp.LinExpr()
        total_expected_cost2, total_expected_time = gp.LinExpr(), gp.LinExpr()

        for state in self.states():
            pre_lambda_sum[state], post_lambda_sum[state] = gp.LinExpr() , gp.LinExpr()
            pre_lambda_sum2[state] , post_lambda_sum2[state] = gp.LinExpr() , gp.LinExpr()

            for act in self.active_actions()[state]:
                act_ind = self.MDP[0][state].index(act)

                post_lambda_sum[state].add(X[state,act_ind] , 1)
                total_reach_prob.add(X[state,act_ind] , reach_rew[(state,act)])
                total_expected_cost.add(X[state,act_ind] , cost[(state,act)])

                post_lambda_sum2[state].add(X2[state,act_ind] , 1)
                total_reach_prob2.add(X2[state,act_ind] , reach_rew[(state,act)])
                total_expected_cost2.add(X2[state,act_ind] , cost[(state,act)])
                total_expected_time.add(X2[state,act_ind] , 1)

            for pre in self.pre_state_action_pair()[state]:
                trans_prob_index = self.MDP[1][pre][1].index(state)
                act_index = self.MDP[0][pre[0]].index(pre[1])
                pre_lambda_sum[state].add( X[pre[0],act_index] , self.MDP[1][pre][0][trans_prob_index])
                pre_lambda_sum2[state].add( X2[pre[0],act_index] , self.MDP[1][pre][0][trans_prob_index])

            # Flow equation for each state
            if state not in target and  state not in blocks:
                m.addConstr( post_lambda_sum[state] - pre_lambda_sum[state] == alpha[state])
                m2.addConstr( post_lambda_sum2[state] - pre_lambda_sum2[state] == alpha[state])

        m.addConstr( total_reach_prob >= max_reach_val)
        m.setObjective(total_expected_cost, GRB.MINIMIZE)
        m.optimize()

        min_cost_val=m.objVal

        m2.addConstr( total_expected_cost2 <= min_cost_val)
        m2.addConstr( total_reach_prob2 >= max_reach_val)
        m2.setObjective( total_expected_time, GRB.MINIMIZE)
        m2.optimize()

        optimal_policy={}
        for state in range(len(self.states())):
            optimal_policy[state]=self.active_actions()[state][0]
            for action in self.active_actions()[state]:
                act_ind = self.MDP[0][state].index(action)
                if X2[state,act_ind].x >= 1e-4:
                    optimal_policy[state]=action
        return min_cost_val, optimal_policy
