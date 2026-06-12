The package is for Bayesian Model Averaging in non-Gaussian error regression models, mainly for Stochastic Frontier Analysis. Main featueres of the package use paralelization toolbox, so it is better to have it installed in MATLAB.

Main functions:
sfa_fit_rep - calculates a single model and returns its results structure

Functions for BMA:
fast_ES - the gold standard, though time-consuming. It performs exhaustive (brute force) search over the entire model space. The algorithm uses paralelization and optional fast pre-screening algorithm; see Makieła (2026) Model Uncertainty under Non-Gaussian Errors: Bayesian Model Averaging and Selection in Stochastic Frontier Models (forthcoming).
