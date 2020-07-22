# Sequential-Incentive-Design

This repository includes the implementations of two algorithms that compute globally or locally optimal solutions to a class of sequential incentive design problems. For details on the considered incentive design problem and the solution techniques, please refer to http://arxiv.org/abs/2007.08548 

To run the algorithms, you will need an active Gurobi optimization package license.

The file 'MILP_method.py' includes an implementation of a mixed-integer linear program which computes a globally optimal solution to small scale sequential incentive design problems. 

The file 'CCP_method.py' includes an implementation of a nonlinear optimization problem with bilinear constraints which computes a locally optimal solution to mid-scale sequential incentive design problems.

The main example files are 'austin_planning.py', 'discount_planning_example.py', and 'motion_planning_example.py'.


