import numpy as np
feature_dim = 50
#rescale bound for beta
dist_bound = np.ones(feature_dim)*3

alpha_ub = 3
alpha_lb = 1.5

ambiguity_selection = 0
#here for the sign:
##-1 means to cheat and choose,
##0 means to use the given ambiguity level
##1 CV?
given_fold_num = 5

#control parameter in beta distribution
threshold = 0.0001


mode = 'simulation-v2'
dist_type = 'beta'