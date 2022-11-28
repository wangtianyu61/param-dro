import pandas as pd
import math
import numpy as np
from gurobipy import *
from sklearn.model_selection import KFold

#use the obj example in Section 5.2 [Simulation Experiment]
#https://jmlr.org/papers/volume20/17-750/17-750.pdf

class simulation_test:
    def __init__(self, sample_size, feature_dim):
        self.sample_size = sample_size
        self.feature_dim = feature_dim
        self.reparam = True
        self.bd = 10
        #self.ambiguity_size = self.bd*5/self.sample_size
        self.ambiguity_size = 0.5
    def ERM(self, dom_order = 2):
        sample_size = self.sample_size
        sample = self.sample
        if self.reparam != False:
            self.param_est()
        m = Model('ERM')
        decision = pd.Series(m.addVars(self.feature_dim, lb = -GRB.INFINITY))
        m.addConstr(quicksum(decision[i] * decision[i] for i in range(self.feature_dim)) <= self.bd * self.bd, 'budget')

        opt_val = self.bd/(2*math.sqrt(self.feature_dim))*np.ones(self.feature_dim)
        
        if dom_order == 2:
            obj = quicksum((decision[i] - opt_val[i])*(decision[i] - opt_val[i]) for i in range(self.feature_dim))/2
        elif dom_order == 4:
            aux_y = m.addVar(name = 'aux_y', lb = 0)
            m.addConstr(aux_y >= quicksum((decision[i] - opt_val[i])*(decision[i] - opt_val[i]) for i in range(self.feature_dim)))
            obj = aux_y*aux_y/2
        for i in range(self.sample_size):
            obj += np.dot(self.sample[i], decision - opt_val)/self.sample_size
        
        m.setObjective(obj, GRB.MINIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        #reset
        self.sample_size = sample_size
        self.sample = sample

        opt_decision = [v.x for v in decision]
        return np.linalg.norm(opt_decision - opt_val, ord = 2)**dom_order/2

    def DRO_W1(self, dom_order = 2):
        sample_size = self.sample_size
        sample = self.sample
        if self.reparam != False:
            self.param_est()
        m = Model('DRO-W1')
        decision = pd.Series(m.addVars(self.feature_dim, lb = -GRB.INFINITY))
        m.addConstr(quicksum(decision[i] * decision[i] for i in range(self.feature_dim)) <= self.bd * self.bd, 'budget')
        opt_val = self.bd/(2*math.sqrt(self.feature_dim))*np.ones(self.feature_dim)
        
        if dom_order == 2:
            obj = quicksum((decision[i] - opt_val[i])*(decision[i] - opt_val[i]) for i in range(self.feature_dim))/2
        elif dom_order == 4:
            aux_y = m.addVar(name = 'aux_y', lb = 0)
            m.addConstr(aux_y >= quicksum((decision[i] - opt_val[i])*(decision[i] - opt_val[i]) for i in range(self.feature_dim)))
            obj = aux_y*aux_y/2
        for i in range(self.sample_size):
            obj += np.dot(self.sample[i], decision - opt_val)/self.sample_size
        reg_t = m.addVar(lb = 0)
        decision_new = pd.Series(m.addVars(self.feature_dim, lb = -GRB.INFINITY))
        m.addConstrs((decision_new[i] == decision[i] - opt_val[i] for i in range(self.feature_dim)), 'c0')
        m.addConstr((quicksum(decision_new[i] * decision_new[i] for i in range(self.feature_dim)) <= reg_t * reg_t), 'reg_term')
        obj += reg_t * self.ambiguity_size
        m.setObjective(obj, GRB.MINIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        #reset
        self.sample_size = sample_size
        self.sample = sample

        opt_decision = [v.x for v in decision]
        return np.linalg.norm(opt_decision - opt_val, ord = 2)**dom_order/2

    def DRO_chi2(self, dom_order = 2):
        flag = 0
        while flag == 0:
            sample_size = self.sample_size
            sample = self.sample

            if self.reparam != False:
                self.param_est()
            m = Model('DRO-chi2')
            decision = pd.Series(m.addVars(self.feature_dim, lb = -GRB.INFINITY))
            m.addConstr(quicksum(decision[i] * decision[i] for i in range(self.feature_dim)) <= self.bd * self.bd, 'budget')
            opt_val = self.bd/(2*math.sqrt(self.feature_dim))*np.ones(self.feature_dim)
            eta = m.addVar(name = 'adj_eta', lb = -GRB.INFINITY)
            aux_lambda = m.addVar(name = 'adj_lambda', lb = 0)
            aux_t = pd.Series(m.addVars(self.sample_size, lb = 0))
            aux_u = pd.Series(m.addVars(self.sample_size, lb = -GRB.INFINITY))
            aux_beta = pd.Series(m.addVars(self.sample_size, lb = 0))
            aux_gamma = m.addVar(lb = 0)

            # if dom_order == 2:
            #     aux_alpha = pd.Series(m.addVars(self.sample_size, self.feature_dim, lb = -GRB.INFINITY))

            #     m.addConstrs((aux_alpha[i][j] == decision[j] - opt_val[j] + self.sample[i][j]
            #                 for i in range(self.sample_size) for j in range(self.feature_dim)), 'c0')
                
                
                
            #     m.addConstrs((aux_gamma[i] == np.linalg.norm(self.sample[i], ord = 2)**2/2 + aux_u[i] + eta
            #                 for i in range(self.sample_size)), 'c10')
            #     m.addConstrs((quicksum(aux_alpha[i][j]*aux_alpha[i][j] for j in range(self.feature_dim)) <= 2*aux_gamma[i]
            #                 for i in range(self.sample_size)),'c20')
            decision_y = pd.Series(m.addVars(self.feature_dim, lb = -GRB.INFINITY))

            m.addConstrs((decision_y[i] == decision[i] - opt_val[i]
                        for i in range(self.feature_dim)), 'c11')        
            #print(np.dot(self.sample[0], decision_y))
            m.addConstrs((aux_gamma + np.dot(self.sample[i], decision_y) == aux_u[i] + eta
                        for i in range(self.sample_size)), 'c12')
            decision_z = m.addVar(name = 'y^2', lb = 0)                
            m.addConstr((quicksum(decision_y[i]*decision_y[i] for i in range(self.feature_dim)) <= decision_z), 'c13')                    
            if dom_order == 2:
                m.addConstr((decision_z <= 2*aux_gamma),'c0')
            elif dom_order == 4:
                
                #m.update()

                #print(np.dot(self.sample[0], decision_y))
                m.addConstr((decision_z*decision_z <= 2*aux_gamma),'c22')


            m.addConstrs((aux_beta[i] == aux_lambda - aux_u[i]/2 for i in range(self.sample_size)), 'c2')
            m.addConstrs((aux_t[i]*aux_t[i] + aux_u[i]*aux_u[i]/4<=aux_beta[i]*aux_beta[i]
                        for i in range(self.sample_size)), 'c3')
            obj = eta + (2 + self.ambiguity_size)*aux_lambda - 2*quicksum(aux_t[i] for i in range(self.sample_size))/self.sample_size
            m.setObjective(obj, GRB.MINIMIZE)
            m.setParam('OutputFlag', 0)
            m.optimize()
            #reset
            self.sample_size = sample_size
            self.sample = sample
            try:
                opt_decision = [v.x for v in decision]
                print(aux_lambda.x)
                flag = 1
            except Exception as e:
                print(e)
        return np.linalg.norm(opt_decision - opt_val, ord = 2)**dom_order/2


    def DRO_OT(self):
        pass
    def param_est(self):
        sample_number = self.reparam*self.sample_size
        sample_mean = np.mean(self.sample, axis = 0)
        sample_cov = np.cov(self.sample.T)
        #follow normal distribution
        self.sample = np.random.multivariate_normal(mean = sample_mean, cov = sample_cov, size = sample_number)
        self.sample_size = sample_number
    
    #load data
    def simulate_test_base(self, train_sample):
        self.sample = train_sample

