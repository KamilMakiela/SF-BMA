The package is for Bayesian Model Averaging in non-Gaussian error regression models, mainly for Stochastic Frontier Analysis. Main featueres of the package use paralelization toolbox, so it is better to have it installed in MATLAB.

Key functions for model estimation:
sfa_fit_rep - calculates a single model and returns its results structure

Functions for BMA:
fast_ES - the gold standard in model search, though time-consuming. It performs exhaustive (brute force) search over the entire model space. The algorithm uses paralelization and optional fast pre-screening algorithm; see Makieła (2026) Model Uncertainty under Non-Gaussian Errors: Bayesian Model Averaging and Selection in Stochastic Frontier Models (forthcoming).
fast_fs - fast heuristic method for model building from buttom up (more logical in case of sfa, imho)
fast_bs - fast heuristic method for model building from top (the most general case) to buttom.
full_fs, full_bs - similar algorithms to the above ones, with the exception that they are more focused on 'search' (not select), which means they do not have any built in stop, instead they go over all possible options in their respective model builing strategies (ie., p(p+1)/2, where p is the number of candidate regressors)
fast_fbs - a stepwise model-selection algorithm based on a forward-backward search; I would still classify this as a selection algorith in the sense needed from BMA/S


See comments in BMA_script for more details on how to use them. 
