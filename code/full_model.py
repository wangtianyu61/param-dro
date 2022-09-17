from pickle import TRUE
from xml.etree.ElementTree import TreeBuilder
import pandas as pd
import numpy as np
import math
from gurobipy import *
from scipy.special import beta
from add_param import *
from sklearn.model_selection import KFold
import wgan
import os
import cvxpy as cp

class data_opt_portfolio:
    def __init__(self, window_size):
        self.reparam = True
        self.window_size = window_size
        self.epsilon = 0.05
        self.tradeoff_param = 0
        self.tradeoff_MVR = 1
        self.ambiguity_size = 0.5
        self.target_return = 5
        #the lower bound that the weight can attain guarantees 
        self.lower_weight = -2
    def Downside_Risk_ERM(self, option = 'CVXPY'):
        window_size = self.window_size
        history_return = self.history_return
        if self.reparam != False:
            self.param_est()
        if option == 'GUROBI':
            m = Model('Downside-Risk-ERM')
            weight = pd.Series(m.addVars(self.port_num, lb = self.lower_weight))
            m.addConstr(weight.sum() == 1, 'budget')
            v = self.target_return
            aux_s = pd.Series(m.addVars(self.window_size, lb = -GRB.INFINITY, ub = 0))
            m.addConstrs((aux_s[i] <= np.dot(self.history_return[i], weight) - v
                        for i in range(self.window_size)), 'c0')
            var_upd = m.addVar(lb = 0)
            if self.dom_order == 1:
                aux_s2 = pd.Series(m.addVars(self.window_size, lb = 0))
                m.addConstrs((-aux_s[i] <= aux_s2[i]
                            for i in range(self.window_size)), 'c0')
                m.addConstr(quicksum(aux_s2[i] for i in range(self.window_size)) <= self.window_size * var_upd, 'c1')
            elif self.dom_order == 2:
                m.addConstr(quicksum(aux_s[i]*aux_s[i] for i in range(self.window_size)) <= self.window_size * var_upd, 'c1')

            elif self.dom_order == 4:
                var_upd_mid = pd.Series(m.addVars(self.window_size, lb = 0))
                m.addConstrs((aux_s[i] * aux_s[i] <= var_upd_mid[i] for i in range(self.window_size)), 'c01')
                m.addConstr(quicksum(var_upd_mid[i] * var_upd_mid[i] for i in range(self.window_size)) <= self.window_size * var_upd, 'c1')
            obj = var_upd - self.tradeoff_param * np.dot(np.mean(self.history_return, axis = 0), weight)
            m.setObjective(obj, GRB.MINIMIZE)
            m.setParam('OutputFlag', 0)
            m.optimize()
            self.weight = [v.x for v in weight]
        else:
            weight = cp.Variable(self.port_num)
            loss = cp.sum(cp.power(cp.pos(self.target_return - self.history_return@weight), self.dom_order))
            prob = cp.Problem(cp.Minimize(loss), [cp.sum(weight) == 1, weight >= self.lower_weight])
            prob.solve(solver = cp.GUROBI)
            self.weight = weight.value

        #print(self.weight)
        self.window_size = window_size
        self.history_return = history_return
    def CVaR_ERM(self):
        window_size = self.window_size
        history_return = self.history_return
        if self.reparam != False:
            self.param_est() 

        m = Model("CVaR_ERM")
        ## no shortsale constraint (decision constraint)
        weight = pd.Series(m.addVars(self.port_num, lb = self.lower_weight))
        m.addConstr(weight.sum() == 1, 'budget')
        #used in the cvar formulation 
        v = m.addVar(name = 'adj_v', lb = -GRB.INFINITY)
        ## auxiliary variables
        aux_s = pd.Series(m.addVars(self.window_size, lb = -GRB.INFINITY))
        # constraint part

        ## nominal constraint
        #m.addConstrs((-self.lost_cost*order + self.history_demand[i]*aux_gamma[i][0] + (self.demand_upper_bound - self.history_demand[i])*aux_gamma[i][1] >=0 for i in range(self.sample_number)), "c0")
        m.addConstrs((-(1/self.epsilon + self.tradeoff_param)*np.dot(self.history_return[i], weight) + v*(1 - 1/self.epsilon) <= aux_s[i]
                      for i in range(self.window_size)), 'c0')        
        m.addConstrs((-self.tradeoff_param*np.dot(self.history_return[i], weight) + v <= aux_s[i] 
                      for i in range(self.window_size)), 'c1')        
        #target for the worst-case CVaR
        
        obj = 0

        for i in range(self.window_size):
            obj += aux_s[i]/self.window_size
        m.setObjective(obj, GRB.MINIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        self.weight = [v.x for v in weight]   
        
        self.window_size = window_size
        self.history_return = history_return

    def CVaR_DRO(self):
        #parametrize and data resample
        window_size = self.window_size
        history_return = self.history_return

        if self.reparam != False:
            self.param_est() 
        else:
            self.compute_chi2_distance()
        m = Model("CVaR-DRO")
        lag_lambda = m.addVar(name = 'lambda', lb = 0)
        weight = pd.Series(m.addVars(self.port_num, lb = self.lower_weight))
        m.addConstr(weight.sum() == 1, 'budget')
        v = m.addVar(name = 'adj_v', lb = -GRB.INFINITY)
        aux_s = pd.Series(m.addVars(self.window_size, lb = -GRB.INFINITY))
        m.addConstrs((-(1/self.epsilon + self.tradeoff_param)*np.dot(self.history_return[i], weight) + v*(1 - 1/self.epsilon) <= aux_s[i]
                      for i in range(self.window_size)), 'c0')        
        m.addConstrs((-self.tradeoff_param*np.dot(self.history_return[i], weight) + v <= aux_s[i] 
                      for i in range(self.window_size)), 'c1')
        ## auxiliary constraint
        m.addConstrs(((1/self.epsilon + self.tradeoff_param)*weight[i]<= lag_lambda
                      for i in range(self.port_num)), 'c10')
        m.addConstrs(((1/self.epsilon + self.tradeoff_param)*weight[i]>= -lag_lambda
                      for i in range(self.port_num)), 'c11')        
        m.addConstrs((self.tradeoff_param*weight[i]<= lag_lambda
                      for i in range(self.port_num)), 'c20')
        m.addConstrs((self.tradeoff_param*weight[i]>= -lag_lambda
                      for i in range(self.port_num)), 'c21')
        
        #target for the worst-case CVaR
        
        obj = self.ambiguity_size * lag_lambda

        for i in range(self.window_size):
            obj += aux_s[i]/self.window_size
        m.setObjective(obj, GRB.MINIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()

        self.weight = [v.x for v in weight]
        #reset
        self.window_size = window_size
        self.history_return = history_return

    def CVaR_DRO_chi2_div(self):
        window_size = self.window_size
        history_return = self.history_return


        if self.reparam != False:
            self.param_est() 
        m = Model("CVaR-DRO-ch2-divergence")
        weight = pd.Series(m.addVars(self.port_num, lb = self.lower_weight))
        m.addConstr(weight.sum() == 1, 'budget')
        v = m.addVar(name = 'adj_v', lb = -GRB.INFINITY)
        eta = m.addVar(name = 'adj_eta', lb = -GRB.INFINITY)
        aux_lambda = m.addVar(name = 'adj_lambda', lb = 0)
        aux_s = pd.Series(m.addVars(self.window_size, lb = 0))
        m.addConstrs((aux_s[i] >= -np.dot(self.history_return[i], weight) - v
                    for i in range(self.window_size)), 'c0')
        aux_t = pd.Series(m.addVars(self.window_size, lb = 0))
        aux_u = pd.Series(m.addVars(self.window_size, lb = -GRB.INFINITY))
        m.addConstrs((aux_u[i] == v + aux_s[i]/self.epsilon - self.tradeoff_param*np.dot(self.history_return[i], weight) - eta                   
                    for i in range(self.window_size)), 'c1')
        m.addConstrs((aux_u[i] <= aux_lambda for i in range(self.window_size)), 'c11')
        aux_beta = pd.Series(m.addVars(self.window_size, lb = 0))
        m.addConstrs((aux_beta[i] == aux_lambda - aux_u[i]/2 for i in range(self.window_size)), 'c2')
        m.addConstrs((aux_t[i]*aux_t[i] + aux_u[i]*aux_u[i]/4<=aux_beta[i]*aux_beta[i]
                    for i in range(self.window_size)), 'c3')

        obj = eta + (2 + self.ambiguity_size)*aux_lambda - 2*quicksum(aux_t[i] for i in range(self.window_size))/self.window_size
        
        m.setObjective(obj, GRB.MINIMIZE)
        m.setParam('OutputFlag', 0)

        m.optimize()

        self.weight = [v.x for v in weight]
        #reset
        self.window_size = window_size
        self.history_return = history_return

    def Downside_Risk_DRO_chi2_div(self, option = 'CVXPY'):
        window_size = self.window_size
        history_return = self.history_return

        if self.reparam != False:
            self.param_est() 
        else:
            self.compute_chi2_distance()
        self.window_size = self.history_return.shape[0]
        if option == 'GUROBI':
            m = Model("Downside_Risk_DRO_chi2_div")
            ## no shortsale constraint (decision constraint)
            weight = pd.Series(m.addVars(self.port_num, lb = self.lower_weight))
            m.addConstr(weight.sum() == 1, 'budget')
            #used in the cvar formulation 
            v = self.target_return
            ## auxiliary variables
            aux_s = pd.Series(m.addVars(self.window_size, lb = -GRB.INFINITY, ub = 0))        
            aux_s2 = pd.Series(m.addVars(self.window_size, lb = 0))
            
            eta = m.addVar(name = 'adj_eta', lb = -GRB.INFINITY)
            aux_lambda = m.addVar(name = 'adj_lambda', lb = 0, ub = GRB.INFINITY)
            aux_t = pd.Series(m.addVars(self.window_size, lb = 0))
            aux_u = pd.Series(m.addVars(self.window_size, lb = -GRB.INFINITY))
            
            m.addConstrs((aux_s[i] <= np.dot(self.history_return[i], weight) - v
                        for i in range(self.window_size)), 'c0')
            if self.dom_order == 1:
                m.addConstrs((-aux_s[i] <= aux_s2[i] 
                            for i in range(self.window_size)), 'c01')
            else:
                if self.dom_order == 2:
                    m.addConstrs((aux_s[i] * aux_s[i] <= aux_s2[i]
                            for i in range(self.window_size)), 'c01')
                elif self.dom_order == 4:
                    aux_s3 = pd.Series(m.addVars(self.window_size, lb = 0))
                    m.addConstrs((aux_s[i] * aux_s[i] <= aux_s3[i]
                        for i in range(self.window_size)), 'c00')
                    m.addConstrs((aux_s3[i] * aux_s3[i] <= aux_s2[i]
                        for i in range(self.window_size)), 'c01')
            m.addConstrs((aux_u[i] == aux_s2[i] - self.tradeoff_param*np.dot(self.history_return[i], weight) - eta                   
                    for i in range(self.window_size)), 'c1')
            m.addConstrs((aux_u[i] <= aux_lambda for i in range(self.window_size)), 'c11')
            aux_beta = pd.Series(m.addVars(self.window_size, lb = 0))
            m.addConstrs((aux_beta[i] == aux_lambda - aux_u[i]/2 for i in range(self.window_size)), 'c2')
            m.addConstrs((aux_t[i]*aux_t[i] + aux_u[i]*aux_u[i]/4<=aux_beta[i]*aux_beta[i]
                        for i in range(self.window_size)), 'c3')

            obj = eta + (2 + self.ambiguity_size)*aux_lambda - 2*quicksum(aux_t[i] for i in range(self.window_size))/self.window_size



            m.setObjective(obj, GRB.MINIMIZE)
            m.setParam('OutputFlag', 0)

            m.optimize()

            self.weight = [v.x for v in weight]
        else:
            
            weight = cp.Variable(self.port_num)
            aux_s = cp.Variable(self.window_size)
            eta = cp.Variable()
            loss = math.sqrt(1 + self.ambiguity_size) / math.sqrt(self.window_size) * cp.norm(aux_s - eta, 2) + eta
            prob = cp.Problem(cp.Minimize(loss), [cp.sum(weight) == 1, weight >= self.lower_weight, aux_s >= cp.power(cp.pos(self.target_return - self.history_return@weight), self.dom_order)])
            prob.solve(solver = cp.GUROBI)
            self.weight = weight.value
        #print(aux_lambda.x)
        #print(self.weight)
        self.window_size = window_size
        self.history_return = history_return
    def MVR_ERM(self):
        pass
    def MVR_DRO(self):
        window_size = self.window_size
        if self.reparam == True:
            self.param_est()
        m = Model('MVR-DRO')
        history_return_cov = np.dot(self.history_return.T)
        weight = pd.Series(m.addVars(self.window_size, lb = -GRB.INFINITY))
        std_up_bd = m.addVar(name = 'std_up_bd', lb = 0)
        norm = m.addVar(name = 'norm_weight', lb = 0)
        m.addConstr((np.dot(history_return_cov, weight).dot(weight) <= std_up_bd*std_up_bd),'c0')
    
        obj = self.tradeoff_MVR*std_up_bd + math.sqrt(self.ambiguity_size)*norm
        for i in range(self.window_size):
            obj -= np.dot(weight,self.history_return[i])/self.window_size
        m.setObjective(obj, GRB.MINIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()

        self.weight = [v.x for v in weight]
        self.window_size = window_size

    def param_est(self):
        #return a new dataset of history returns
        sample_number = self.reparam * self.window_size
        if self.dist_type == 'normal':
            sample_mean = np.mean(self.history_return, axis = 0)
            sample_cov = np.cov(self.history_return.T)
            self.history_return = np.random.multivariate_normal(mean = sample_mean, cov = sample_cov, size = sample_number)
        elif self.dist_type == 'beta':
            # rescale
            self.history_return = self.history_return.T
            ## simulation
            bound = dist_bound
            #if mode == 'true-data':
            ## real data      
                
            for i in range(self.port_num):
                bound[i] = max(np.max(self.history_return[i]), np.max(-self.history_return[i]))
    
            
            
                
            scale_history_return = np.ones(self.history_return.shape)

            est_alpha = np.zeros(self.port_num)
            est_beta = np.zeros(self.port_num)
            for i in range(self.port_num):
                for j in range(self.window_size):
                    scale_history_return[i][j] = max(threshold, 0.5 + self.history_return[i][j]/(2*bound[i]))
                #parameter fit oracle
                log_mean = np.mean([math.log(scale_history_return[i][j]) for j in range(self.window_size)])
                est_alpha[i] = (-log_mean - 2 - math.sqrt(log_mean**2 + 4))/(2*log_mean)
                est_alpha[i] = max(min(est_alpha[i], alpha_ub), alpha_lb)
                est_beta[i] = 2


            #resample
            self.history_return = np.zeros((self.port_num, sample_number))
            for i in range(self.port_num):
                self.history_return[i] = 2*bound[i] * np.random.beta(est_alpha[i], est_beta[i], size = sample_number) - bound[i]
            self.history_return = self.history_return.T

            #estimate the ambiguity-size to cover the true distribution
            ##Method 1: calculate by the formal (Exact Value)
            if ambiguity_selection == -1:
                self.compute_chi2_distance()
                # div = 1
                # for i in range(self.port_num):
                #     div = div*beta(self.true_alpha[i], self.true_beta[i]) * beta(2 * est_alpha[i] - self.   true_alpha[i], 2 * est_beta[i] - self.true_beta[i])/(beta(est_alpha[i], est_beta[i])**2)
                # self.ambiguity_size = div - 1
            #print(div)
            #self.ambiguity_size = div - 1
            #print(self.ambiguity_size)
            ##Method 2: Prob upper bound

            ##Method 3: Simulation Counterpart of Method 1

        elif self.dist_type == 'WGAN':
            self.WGAN_train()

        self.window_size = sample_number
    
    #
    def WGAN_train(self, resample_time):
        sample_number = resample_time * self.window_size
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        #train a NN with unconditional assets return 
        continuous_vars_0 = ['r' + str(i) for i in range(self.port_num)]
        df_return = pd.DataFrame(self.history_return, columns = continuous_vars_0)
        continuous_lw_bd_0 = {}
        categorical_vars_0 = []
        context_vars_0 = []
        data_wrapper = wgan.DataWrapper(df_return, continuous_vars_0, categorical_vars_0,
                                        context_vars_0, continuous_lw_bd_0)
        spec = wgan.Specifications(data_wrapper, batch_size = 64, max_epochs = 2000, critic_d_hidden = [32, 32], generator_d_hidden = [32, 32], critic_lr = 2e-3, generator_lr = 2e-3)
        generator = wgan.Generator(spec)
        critic = wgan.Critic(spec)

        X, context = data_wrapper.preprocess(df_return)
        wgan.train(generator, critic, X, context, spec)
        #simulate data
        df_generated = data_wrapper.apply_generator(generator, df_return.sample(sample_number, replace = True))
        self.history_return = np.array(df_generated)

    #handle true data
    def roll_window_test_base(self, df_select, feature_name, label_name, dist_type = 'normal'):
        self.feature_name = feature_name
        self.label_name = label_name
        self.all_return = np.array(df_select[label_name])
        self.all_covariate = np.array(df_select[feature_name])
        self.port_num = len(label_name)
        self.sample_number = len(df_select)
        self.test_number = self.sample_number - self.window_size
        
        self.dist_type = dist_type
    def roll_window_test_method(self, optimize_method, CV, dom_order = 2, metric = 'W1'):
        self.dom_order = dom_order
        self.port_return = np.zeros(self.test_number)
        for i in range(self.test_number):
            if i%100 == 0:
                print(i)
            self.history_return = self.all_return[i:(i+self.window_size)]
            self.test_return = self.all_return[self.window_size + i]
            if CV != 0:
                self.CV_method(optimize_method, CV, metric)

            #optimize thhe obtain the portfolio weight
            if optimize_method == 'CVaR_ERM':
                self.CVaR_ERM()
            elif optimize_method == 'CVaR_DRO':
                #print(i)
                if metric == 'W1':
                    self.CVaR_DRO()
                elif metric == 'chi2':
                    self.CVaR_DRO_chi2_div()
            elif optimize_method == 'Down_Risk_ERM':
                self.Downside_Risk_ERM()
            elif optimize_method == 'Down_Risk_DRO':
                self.Downside_Risk_DRO_chi2_div()
            elif optimize_method == 'MVR_ERM':
                self.MVR_ERM()
            elif optimize_method == 'MVR_DRO':
                self.MVR_DRO()
            self.port_return[i] = np.dot(self.weight, self.test_return)
        if optimize_method == 'Down_Risk_ERM' or optimize_method == 'Down_Risk_DRO':
            return self.evaluate_downside_risk()
        else:
            return self.evaluate()
        

    #handle simulate data
    def simulate_test_base(self, history_return, new_return, dist_type = 'normal'):
        #define trivial variables
        self.history_return = history_return
        self.test_return = new_return

        self.port_num = np.shape(history_return)[1]
        self.window_size = np.shape(history_return)[0]

        self.dist_type = dist_type
        if dist_type == 'beta':
            self.true_alpha = np.zeros(self.port_num)
            self.true_beta = np.zeros(self.port_num)
    def freq_compute(self, sample, bin_num):
    #histogram for sample in [0, 1]
        region = np.array(range(bin_num))/bin_num
        freq = np.zeros(len(region))
        for element in sample:
            #return the index i such that a[i - 1] <= v < a[i]
            idx = np.searchsorted(region, element, side = 'right')
            freq[idx - 1] += 1
        return freq[1:]/sum(freq[1:])
    def chi2_dist_oracle(self, p, q):
        div = 0
        for a, b in zip(p, q):
            if a != 0:
                div += (a-b)**2/a
        return div 
    def compute_chi2_distance(self):
        if self.dist_type == 'beta' and ambiguity_selection == -1:
            div = 1
            true_sample_num = 100000
            bin_num = 10
            for i in range(self.port_num):    
                true_sample = np.random.beta(self.true_alpha[i], self.true_beta[i], size = true_sample_num)
                p = self.freq_compute(true_sample, bin_num)
                #print(self.history_return.T[i].shape)
                history_return_transform = self.history_return.T[i]/(2 * dist_bound[i]) + 1/2
                q = self.freq_compute(history_return_transform, bin_num)
                dist = self.chi2_dist_oracle(p, q)
                
                div = div * (dist + 1)
            self.ambiguity_size =  div - 1
            #print(self.ambiguity_size)

    def simulate_test_method(self, optimize_method, CV, dom_order = 2, metric = 'W1'):
        self.dom_order = dom_order
        if CV != 0:
            self.CV_method(optimize_method, CV, metric)
        #optimize thhe obtain the portfolio weight
        if optimize_method == 'CVaR_ERM':
            self.CVaR_ERM()
        elif optimize_method == 'CVaR_DRO':
            # if self.reparam == False:
            #     self.ambiguity_size = 0.5*math.pow(self.window_size, -1/self.port_num)
            # else:
            #     self.ambiguity_size = 0.02*self.port_num**2/math.sqrt(self.window_size)
            self.ambiguity_size = 0.02*self.port_num**2/math.sqrt(self.window_size)
            if metric == 'W1':
                self.CVaR_DRO()
            elif metric == 'chi2':
                self.CVaR_DRO_chi2_div()
        elif optimize_method == 'Down_Risk_ERM':
            self.Downside_Risk_ERM()
        elif optimize_method == 'Down_Risk_DRO':
            #choose the ambiguity size to cover the true distribution
            ##compute the f-divergence
            self.Downside_Risk_DRO_chi2_div()

        elif optimize_method == 'MVR_ERM':
            self.MVR_ERM()
        elif optimize_method == 'MVR_DRO':
            self.MVR_DRO()
        self.port_return = np.dot(self.test_return, self.weight)
        if optimize_method == 'Down_Risk_ERM' or optimize_method == 'Down_Risk_DRO':
            return self.evaluate_downside_risk()
        else:
            return self.evaluate()

    #
    def CV_method(self, optimize_method, choice_list, metric, fold_num = given_fold_num, eval = 'true-obj'):
        if optimize_method == 'CVaR_DRO':
            kf = KFold(n_splits = fold_num, shuffle = True)
            perform_metric = np.zeros(len(choice_list))
            true_history_return = self.history_return
            true_window_size = self.window_size
            for train_label, test_label in kf.split(self.history_return):
                self.history_return = true_history_return[train_label]
                #print(self.history_return)
                self.window_size = len(self.history_return)
                validation_return = true_history_return[test_label]
                res = [[] for i in range(len(choice_list))]
                for i, param in enumerate(choice_list):
                    self.ambiguity_size = param
                    if metric == 'W1':
                        self.CVaR_DRO()
                    elif metric == 'chi2':
                        self.CVaR_DRO_chi2_div()
                    port_return = list(np.dot(validation_return, self.weight))
                    res[i].extend(port_return)
                for i in range(len(choice_list)):
                    temp_res = np.array(res[i])
                    perform_metric[i] += self.cv_eval(temp_res, eval)

            self.history_return = true_history_return
            self.window_size = true_window_size
            self.ambiguity_size = choice_list[np.argmin(perform_metric)]

        elif optimize_method == 'Down_Risk_DRO':
            kf = KFold(n_splits = fold_num, shuffle = True)
            perform_metric = np.zeros(len(choice_list))
            true_history_return = self.history_return
            true_window_size = self.window_size
            for train_label, test_label in kf.split(self.history_return):
                self.history_return = true_history_return[train_label]
                self.window_size = len(self.history_return)
                validation_return = true_history_return
                res = [[] for i in range(len(choice_list))]
                for i, param in enumerate(choice_list):
                    self.ambiguity_size = param
                    self.Downside_Risk_DRO_chi2_div()
                    port_return = list(np.dot(validation_return, self.weight))
                    res[i].extend(port_return)
                for i in range(len(choice_list)):
                    temp_res = np.array(res[i])
                    perform_metric[i] += self.cv_eval_downrisk(temp_res)

            self.history_return = true_history_return
            self.window_size = true_window_size
            self.ambiguity_size = choice_list[np.argmin(perform_metric)]      
                    


    def cv_eval(self, res, eval):
        if eval == 'true-obj':
            return (-np.mean(np.sort(res)[0:int(len(res)*self.epsilon)]) - self.tradeoff_param*np.mean(res))/100
    def cv_eval_downrisk(self, res):
        if self.dom_order > 1:
            true_obj = np.mean([min(0, res[i] - self.target_return)**self.dom_order for i in range(len(res))]) - self.tradeoff_param * np.mean(res)
        elif self.dom_order == 1:
            true_obj = np.mean([abs(min(0, res[i] - self.target_return)) for i in range(len(res))]) - self.tradeoff_param * np.mean(res) 
        return true_obj 

    #evaluation
    def evaluate(self):
        true_obj = (-np.mean(np.sort(self.port_return)[0:int(len(self.port_return)*self.epsilon)]) - self.tradeoff_param*np.mean(self.port_return))/100
        SR = np.mean(self.port_return)/np.std(self.port_return)
        print('True obj is: ', true_obj)
        print('Sharpe Ratio is: ', SR)
        print('============================')
        return true_obj, SR

    #evaluation-v2
    def evaluate_downside_risk(self):
        if self.dom_order > 1:
            true_obj = np.sum([min(0, self.port_return[i] - self.target_return)**self.dom_order for i in range(len(self.port_return))])/len(self.port_return) - self.tradeoff_param*np.mean(self.port_return)
        elif self.dom_order == 1:
            true_obj = np.sum([abs(min(0, self.port_return[i] - self.target_return)) for i in range(len(self.port_return))])/len(self.port_return) - self.tradeoff_param*np.mean(self.port_return)
        # print('True obj is ', true_obj)
        # print('===============================')
        return true_obj