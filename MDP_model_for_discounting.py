import numpy as np

def reward_functions():
    rew=[{},{},{}]
    rew[0][(0,'N')], rew[0][(0,'1')], rew[0][(0,'2')], rew[0][(0,'3')], rew[0][(0,'4')] = 0, -1, -1, -1, -1
    rew[0][(1,'N')], rew[0][(1,'2')], rew[0][(1,'3')], rew[0][(1,'4')] = 0, -1, -2, -2
    rew[0][(2,'N')], rew[0][(2,'1')], rew[0][(2,'3')], rew[0][(2,'4')] = 0, -1, -2, -2
    rew[0][(3,'N')], rew[0][(3,'1')], rew[0][(3,'2')], rew[0][(3,'4')] = 0, -2, -2, -1
    rew[0][(4,'N')], rew[0][(4,'1')], rew[0][(4,'2')], rew[0][(4,'3')] = 0, -2, -2, -1
    rew[0][(5,'N')], rew[0][(5,'3')], rew[0][(5,'4')] = 0, -2, -2
    rew[0][(6,'N')], rew[0][(6,'2')], rew[0][(6,'4')] = 0, -1, -1
    rew[0][(7,'N')], rew[0][(7,'2')], rew[0][(7,'3')] = 0, -1, -1
    rew[0][(8,'N')], rew[0][(8,'1')], rew[0][(8,'4')] = 0, -1, -1
    rew[0][(9,'N')], rew[0][(9,'1')], rew[0][(9,'3')] = 0, -1, -1
    rew[0][(10,'N')], rew[0][(10,'1')], rew[0][(10,'2')] = 0, -3, -3
    rew[0][(11,'N')], rew[0][(11,'4')] = 0, -1
    rew[0][(12,'N')], rew[0][(12,'3')] = 0, -1
    rew[0][(13,'N')], rew[0][(13,'2')] = 0, -1
    rew[0][(14,'N')], rew[0][(14,'1')] = 0, -1
    rew[0][(15,'N')] = 0

    rew[1][(0,'N')], rew[1][(0,'1')], rew[1][(0,'2')], rew[1][(0,'3')], rew[1][(0,'4')] = 0, -1, -1, -1, -1
    rew[1][(1,'N')], rew[1][(1,'2')], rew[1][(1,'3')], rew[1][(1,'4')] = 0, -2, -1, -2
    rew[1][(2,'N')], rew[1][(2,'1')], rew[1][(2,'3')], rew[1][(2,'4')] = 0, -2, -2, -1
    rew[1][(3,'N')], rew[1][(3,'1')], rew[1][(3,'2')], rew[1][(3,'4')] = 0, -1, -2, -2
    rew[1][(4,'N')], rew[1][(4,'1')], rew[1][(4,'2')], rew[1][(4,'3')] = 0, -2, -1, -2
    rew[1][(5,'N')], rew[1][(5,'3')], rew[1][(5,'4')] = 0, -1, -1
    rew[1][(6,'N')], rew[1][(6,'2')], rew[1][(6,'4')] = 0, -2, -2
    rew[1][(7,'N')], rew[1][(7,'2')], rew[1][(7,'3')] = 0, -1, -1
    rew[1][(8,'N')], rew[1][(8,'1')], rew[1][(8,'4')] = 0, -1, -1
    rew[1][(9,'N')], rew[1][(9,'1')], rew[1][(9,'3')] = 0, -3, -3
    rew[1][(10,'N')], rew[1][(10,'1')], rew[1][(10,'2')] = 0, -1, -1
    rew[1][(11,'N')], rew[1][(11,'4')] = 0, -1
    rew[1][(12,'N')], rew[1][(12,'3')] = 0, -1
    rew[1][(13,'N')], rew[1][(13,'2')] = 0, -1
    rew[1][(14,'N')], rew[1][(14,'1')] = 0, -1
    rew[1][(15,'N')] = 0

    rew[2][(0,'N')], rew[2][(0,'1')], rew[2][(0,'2')], rew[2][(0,'3')], rew[2][(0,'4')] = 0, -1, -1, -1, -1
    rew[2][(1,'N')], rew[2][(1,'2')], rew[2][(1,'3')], rew[2][(1,'4')] = 0, -2, -2, -1
    rew[2][(2,'N')], rew[2][(2,'1')], rew[2][(2,'3')], rew[2][(2,'4')] = 0, -2, -1, -2
    rew[2][(3,'N')], rew[2][(3,'1')], rew[2][(3,'2')], rew[2][(3,'4')] = 0, -2, -1, -2
    rew[2][(4,'N')], rew[2][(4,'1')], rew[2][(4,'2')], rew[2][(4,'3')] = 0, -1, -2, -2
    rew[2][(5,'N')], rew[2][(5,'3')], rew[2][(5,'4')] = 0, -1, -1
    rew[2][(6,'N')], rew[2][(6,'2')], rew[2][(6,'4')] = 0, -1, -1
    rew[2][(7,'N')], rew[2][(7,'2')], rew[2][(7,'3')] = 0, -2, -2
    rew[2][(8,'N')], rew[2][(8,'1')], rew[2][(8,'4')] = 0, -2, -2
    rew[2][(9,'N')], rew[2][(9,'1')], rew[2][(9,'3')] = 0, -1, -1
    rew[2][(10,'N')], rew[2][(10,'1')], rew[2][(10,'2')] = 0, -1, -1
    rew[2][(11,'N')], rew[2][(11,'4')] = 0, -1
    rew[2][(12,'N')], rew[2][(12,'3')] = 0, -1
    rew[2][(13,'N')], rew[2][(13,'2')] = 0, -1
    rew[2][(14,'N')], rew[2][(14,'1')] = 0, -1
    rew[2][(15,'N')] = 0

    return rew


def discount_model():
    model_s={}
    model_sa={}
    model=[model_s,model_sa]
    model_s[0]=['N','1','2','3','4']
    model_s[1]=['N','2','3','4']
    model_s[2]=['N','1','3','4']
    model_s[3]=['N','1','2','4']
    model_s[4]=['N','1','2','3']
    model_s[5]=['N','3','4']
    model_s[6]=['N','2','4']
    model_s[7]=['N','2','3']
    model_s[8]=['N','1','4']
    model_s[9]=['N','1','3']
    model_s[10]=['N','1','2']
    model_s[11]=['N','4']
    model_s[12]=['N','3']
    model_s[13]=['N','2']
    model_s[14]=['N','1']
    model_s[15]=['N']
    for state in range(len(model_s)):
        model_sa[(state,'N')] = ([1],[state])
    model_sa[(0,'1')] = ([1],[1])
    model_sa[(0,'2')] = ([1],[2])
    model_sa[(0,'3')] = ([1],[3])
    model_sa[(0,'4')] = ([1],[4])
    model_sa[(1,'2')] = ([1],[5])
    model_sa[(1,'3')] = ([1],[6])
    model_sa[(1,'4')] = ([1],[7])
    model_sa[(2,'1')] = ([1],[5])
    model_sa[(2,'3')] = ([1],[8])
    model_sa[(2,'4')] = ([1],[9])
    model_sa[(3,'1')] = ([1],[6])
    model_sa[(3,'2')] = ([1],[8])
    model_sa[(3,'4')] = ([1],[10])
    model_sa[(4,'1')] = ([1],[7])
    model_sa[(4,'2')] = ([1],[9])
    model_sa[(4,'3')] = ([1],[10])
    model_sa[(5,'3')] = ([1],[11])
    model_sa[(5,'4')] = ([1],[12])
    model_sa[(6,'2')] = ([1],[11])
    model_sa[(6,'4')] = ([1],[13])
    model_sa[(7,'2')] = ([1],[12])
    model_sa[(7,'3')] = ([1],[13])
    model_sa[(8,'1')] = ([1],[11])
    model_sa[(8,'4')] = ([1],[14])
    model_sa[(9,'1')] = ([1],[12])
    model_sa[(9,'3')] = ([1],[14])
    model_sa[(10,'1')] = ([1],[13])
    model_sa[(10,'2')] = ([1],[14])
    model_sa[(11,'4')] = ([1],[15])
    model_sa[(12,'3')] = ([1],[15])
    model_sa[(13,'2')] = ([1],[15])
    model_sa[(14,'1')] = ([1],[15])
    return model
