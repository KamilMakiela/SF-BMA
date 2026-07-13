The package is for Bayesian Model Averaging in non-Gaussian error regression models like Stochastic Frontier Analysis. Main featueres of the package use paralelization toolbox, so it is better to have it  in MATLAB.

Key functions for model estimation:
sfa_fit_rep - calculates a single model and returns its results structure

Functions for BMA:

- fast_ES - the gold standard in model search, though time-consuming. It performs exhaustive (brute force) search over the entire model space. The algorithm uses paralelization and optional fast pre-screening algorithm; see Makieła (2026) Model Uncertainty under Non-Gaussian Errors: Bayesian Model Averaging and Selection in Stochastic Frontier Models (forthcoming).

- fast_fs - fast heuristic method for model building from buttom up (more logical in case of sfa, imho)

- fast_bs - fast heuristic method for model building from top (the most general case) to buttom.

- full_fs, full_bs - similar algorithms to the above ones, with the exception that they are more focused on 'search' (not select), which means they do not have any built in stop, instead they go over all possible options in their respective model builing strategies (ie., p(p+1)/2, where p is the number of candidate regressors); so around these procedures one can build a simple, fast BMA algorithm. 

- fast_fb - a stepwise model-selection algorithm based on a forward-backward search (with a stop criterion); I would still classify this as a selection algorith in the sense needed from BMA/S

- full_fbs_unique - a bottom-up model-space search algorithm based on forward-backward-swap model search strategy, with duplicate models removed. The algorithm explores three types of moves: 1 Forward: add one excluded variable; 2 Backward: remove one included variable; 3 Swap: remove one included variable and simultaneously add one excluded variable. The unique part means that models reached through different paths are stored only once. THis is a relatively good deterministic neighbourhood-search heuristic algorithm. 

- fast_ES - full exhaustive search with fast prescreening methods. Details for this algorithm can be found in Makieła (2026); references below. 

See comments in BMA_script for more details on how to use them. 

Reference: 
Makieła, Kamil (2026). Model Uncertainty under Non-Gaussian Errors: Bayesian Model Averaging and Selection in Stochastic Frontier Models. 
 
