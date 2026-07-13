# Bayesian Model Averaging for Non-Gaussian Regression Models: SFA

This MATLAB package implements Bayesian Model Averaging (BMA) for regression models with non-Gaussian error structures: stochastic frontier analysis (SFA) models.

Several of the main model-search procedures use MATLAB’s Parallel Computing Toolbox. Although the package can run without parallel execution in some cases, access to the toolbox is recommended for computationally intensive searches.

## Model Estimation

### `model = sfa_fit_rep(X,y, n, T, sfa_opt, dec_crit, if_mdd)`

Estimates a single regression or stochastic frontier model and returns structure `model` containing the estimation results. In particular, it returns information criteria such as: BIC, AIC, and more importantly the integrated likelihood value (aka marginal data density), which is a Bayesian measure of quality of model fit. Any of these can later be used by search algorithms. I recommend using integrated likelihood for an exact, fully Bayesian approach (dec_crit=1 and if_mdd=1), or BIC for speed (dec_crit=0 and if_mdd=0). For sfa_opt you can choose: 0 (cnlrm), 1 (SF normal-exponential), 2 (SF normal-half-normal), 3 (panel SF normal-exponential), 4 (panel SF normal-half-normal), 5 (panel RE). Note: X should already contain the constant (for the intercept), which I reccomend to place it as the first column from the left in matrix X (this is important mainly if you want to use X as an input for one of the search algorithms). 

## Bayesian Model Averaging and Model-Space Search

### `fast_ES`

Performs exhaustive search over the full model space and may therefore be treated as the benchmark model-search procedure when the number of candidate regressors is sufficiently small.

The function supports:

* parallel model evaluation;
* full enumeration of the model space;
* optional fast pre-screening for larger model spaces.

Because exhaustive search evaluates all possible subsets of candidate regressors (i.e., potential expalanatory variables), its computational cost increases exponentially with the number of regressors. Also, in SFA some variable choices are driven by theory rather than by pure statistical fit. That is why the algorithm allows us to provide an initial set of variables, which are not subjected to BMA/S (i.e., PIP=1 by default). Just place their columns as first ones from the left right after the constant (which should, by default, be the first column from the left in X). Parameter `init_var` determines how many columns in X, counting from left, are to be left out from the search (treated as fixed, as the baseline model). The minimum and in many cases probably the default value is `init_var'=1, which means that only the intercept is left out from BMA (so the baseline model is an empty one).  

Further details are provided in Makieła (2026); reference below.

### `fast_fs`

A fast forward-selection heuristic method.

The procedure starts from the mandatory-variable model and builds models from the bottom up by sequentially adding candidate regressors. An improvement-based stopping criterion is used.

This direction of search may be particularly natural in stochastic frontier applications, where the preferred specification is built gradually from a restricted baseline model (the baseline being usually motivated by theory).

### `fast_bs`

A fast backward-elimination.

The procedure starts from the most general model and works from the top down by sequentially removing candidate regressors. An improvement-based stopping criterion is used.

### `full_fs`

A full forward-search procedure without an early stopping criterion.

At each stage, the algorithm evaluates all possible additions to the currently selected model. For (p) candidate regressors, the procedure evaluates up to

[
\frac{p(p+1)}{2}
]

candidate models, excluding the initial mandatory-variables-only model.

Unlike `fast_fs`, the purpose of `full_fs` is broader model-space exploration rather than the selection of a single locally preferred model so a fast BMA may be used with it.

### `full_bs`

A full backward-search procedure without stopping criterion.

The algorithm starts from the most general specification and sequentially explores all possible variable removals within the backward-search path.

As with `full_fs`, the procedure is intended primarily for model-space exploration and can be used as the basis for a computationally efficient BMA procedure.

### `fast_fb`

A stepwise model-selection algorithm based on forward-backward search.

At each iteration, the procedure evaluates one-variable additions and removals. The search stops when no neighbouring model improves the selected information criterion.

Because it uses an improvement-based stopping rule, `fast_fb` should be regarded primarily as a model-selection algorithm rather than a general model-space search procedure for BMA.

### `full_fbs_unique`

A bottom-up model-space search procedure based on a forward-backward-swap strategy.

The algorithm considers three types of moves:

1. **Forward move:** add one excluded variable.
2. **Backward move:** remove one included variable.
3. **Swap move:** remove one included variable and simultaneously add one excluded variable.

The `unique` suffix indicates that duplicate model specifications obtained through different search paths are removed.

This procedure provides a relatively broad deterministic neighbourhood search while remaining substantially less computationally demanding than exhaustive enumeration.

## Usage

See the comments and examples in `BMA_script` for details on model specification, function arguments, and interpretation of the returned results.

## Reference

Makieła, K. (2026). *Model uncertainty under non-Gaussian errors: Bayesian model averaging and selection in stochastic frontier models*. Forthcoming.
