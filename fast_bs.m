function output = fast_bs(tabela, y, n, T, sfa, init_var, dec_cr)
%FAST_BS Fast backward search path for SFA models.
%
% The algorithm starts from the full model containing X0 and all candidate
% regressors. At each step it proposes removing the candidate regressor with
% the lowest absolute t-ratio in the currently fitted model. The reduced
% model is accepted only if it improves mdl.inf_cr. Otherwise the algorithm
% stops.
%
% Inputs are intentionally kept the same as in full_fs:
%   tabela   - table or numeric matrix of regressors
%   y        - dependent variable
%   n, T     - cross-section/panel dimensions used by sfa_fit_rep
%   sfa      - SFA option used by sfa_fit_rep
%   init_var - number of mandatory columns in tabela, e.g. intercept X0
%   dec_cr   - decision criterion option used by sfa_fit_rep
%
% Output structure mirrors full_fs as closely as possible.

if dec_cr == 1
    if_mdd = 1;
else
    if_mdd = 0;
end

% Data and labels
if istable(tabela)
    X = table2array(tabela);
    nazwy = tabela.Properties.VariableNames;
else
    X = tabela;
    nazwy = strcat("x", string(1:size(X,2)));
end
etykiety = string(nazwy);

if istable(y)
    y = table2array(y);
end

k_total = size(X,2);
X0      = X(:,1:init_var);
X_cand  = X(:,init_var+1:end);
p       = size(X_cand,2);

if sfa == 0
    sig_vars = 1;
else
    sig_vars = 2;
end
sig_idx = k_total+1:k_total+sig_vars;

% Storage: full model, then p removals, ending with X0-only
max_models = p + 1;   % upper bound: full model plus at most p accepted removals
cykli = 0;

all_lgmdd = zeros(max_models,1);
all_ML    = zeros(max_models,1);
all_pmax  = zeros(max_models,1);
all_steps = cell(max_models,2);
all_par   = zeros(max_models,k_total+sig_vars);
all_se    = zeros(max_models,k_total+sig_vars);

path_vars       = cell(max_models,1);
path_removed    = zeros(p,1);
path_removedLab = strings(p,1);
path_min_t      = zeros(p,1);
path_inf_crit   = zeros(max_models,1);
path_proposed_crit = NaN(p,1);

% Start from the full candidate set. These indices refer to columns of
% X_cand, not to columns of X.
current_vars = 1:p;

for step = 1:max_models

    Xtemp = [X0 X_cand(:,current_vars)];
    mdl_curr = sfa_fit_rep(Xtemp,y,n,T,sfa,dec_cr,if_mdd);

    cykli = cykli + 1;

    all_steps{cykli,1} = current_vars;
    all_steps{cykli,2} = etykiety([1:init_var, init_var + current_vars]);

    all_lgmdd(cykli) = mdl_curr.mdd;
    all_ML(cykli)    = mdl_curr.l_max;
    all_pmax(cykli)  = mdl_curr.p_max;
    path_inf_crit(cykli) = mdl_curr.inf_cr;
    path_vars{cykli} = current_vars;

    % Store parameters in the full-column layout used by full_fs:
    % mandatory vars in 1:init_var, optional vars in init_var+current_vars,
    % scale parameters at the end.
    if if_mdd == 1
        theta = mdl_curr.bayes.theta_post(:);
        se    = mdl_curr.bayes.theta_post_se(:);
    else
        theta = mdl_curr.theta_ml(:);
        se    = mdl_curr.theta_ml_se(:);
    end

    all_par(cykli,1:init_var) = theta(1:init_var)';
    all_se(cykli,1:init_var)  = se(1:init_var)';

    if ~isempty(current_vars)
        beta_pos_model = init_var + (1:numel(current_vars));
        beta_pos_full  = init_var + current_vars;

        all_par(cykli,beta_pos_full) = theta(beta_pos_model)';
        all_se(cykli,beta_pos_full)  = se(beta_pos_model)';
    end

    all_par(cykli,(end-sig_vars+1):end) = theta((end-sig_vars+1):end)';
    all_se(cykli,(end-sig_vars+1):end)  = se((end-sig_vars+1):end)';

    % Stop if the current model is already X0-only.
    if isempty(current_vars)
        break
    end

    % Backward proposal: remove variable with the lowest absolute t-ratio.
    % Then check whether the reduced model improves the information criterion.
    % Lower mdl.inf_cr is assumed to be better, as in fast_fs/full_fs.
    
    beta_pos_model = init_var + (1:numel(current_vars));
    t_abs = abs(theta(beta_pos_model) ./ se(beta_pos_model));

    % If numerical issues occur, propose removing that variable first.
    bad_t = ~isfinite(t_abs) | ~isreal(t_abs);
    t_abs(bad_t) = -Inf;

    [min_t, loc_remove] = min(t_abs);
    var_remove = current_vars(loc_remove);

    if isinf(min_t) && min_t < 0
        path_min_t(cykli) = NaN;
    else
        path_min_t(cykli) = min_t;
    end
    path_removed(cykli)    = var_remove;
    path_removedLab(cykli) = etykiety(init_var + var_remove);

    proposed_vars = current_vars;
    proposed_vars(loc_remove) = [];
    Xprop = [X0 X_cand(:,proposed_vars)];
    mdl_prop = sfa_fit_rep(Xprop,y,n,T,sfa,dec_cr,if_mdd);
    path_proposed_crit(cykli) = mdl_prop.inf_cr;

    % Stop criterion: accept the removal only if the reduced model improves
    % the current model. Otherwise keep the current model and stop.
    if mdl_prop.inf_cr < mdl_curr.inf_cr
        current_vars = proposed_vars;
    else
        break
    end
end

% Trim storage
all_lgmdd = all_lgmdd(1:cykli);
all_ML    = all_ML(1:cykli);
all_pmax  = all_pmax(1:cykli);
all_steps = all_steps(1:cykli,:);
all_par   = all_par(1:cykli,:);
all_se    = all_se(1:cykli,:);
path_vars = path_vars(1:cykli);
path_inf_crit = path_inf_crit(1:cykli);
path_removed = path_removed(1:max(cykli-1,0));
path_removedLab = path_removedLab(1:max(cykli-1,0));
path_min_t = path_min_t(1:max(cykli-1,0));
path_proposed_crit = path_proposed_crit(1:max(cykli-1,0));

% Best model and refit, same idea as in full_fs
[~, best_step] = max(all_lgmdd);
best_vars = all_steps{best_step,1};
X_best = [X0 X_cand(:,best_vars)];

best_mdl = sfa_fit_rep(X_best,y,n,T,sfa,dec_cr,1);
best_mdl.X = X_best;

if istable(tabela)
    labels_best = nazwy([1:init_var, init_var + best_vars]);
else
    labels_best = [ones(1,init_var), best_vars];
end
best_mdl.labels = labels_best;

% BMA table, following full_fs
w = exp(all_lgmdd - max(all_lgmdd));
p_prob_mdl = w / sum(w);

BMA_PIP = logical(all_par)' * p_prob_mdl;
BMA_par = all_par' * p_prob_mdl;

row_sum = sum(all_se,2);
bad_ind = find(~isfinite(row_sum) | ~isreal(row_sum));

all_se_for_var = all_se;
all_par_for_var = all_par;
p_prob_for_var = p_prob_mdl;

all_se_for_var(bad_ind,:) = [];
all_par_for_var(bad_ind,:) = [];
p_prob_for_var(bad_ind) = [];

% Re-normalize if any problematic rows were removed.
if ~isempty(p_prob_for_var)
    p_prob_for_var = p_prob_for_var / sum(p_prob_for_var);
end

var_within = (all_se_for_var.^2)' * p_prob_for_var;
var_between = ((all_par_for_var - BMA_par').^2)' * p_prob_for_var;
BMA_se = sqrt(var_within + var_between);

% Convert scale parameters from log scale to level scale.
BMA_par(sig_idx) = exp(BMA_par(sig_idx));
BMA_se(sig_idx) = BMA_par(sig_idx) .* BMA_se(sig_idx);

% Positive sign probability
all_par_psp = all_par;
all_se_psp = all_se;
all_par_psp(bad_ind,:) = [];
all_se_psp(bad_ind,:) = [];

Z = all_par_psp ./ all_se_psp;
Z(~isfinite(Z)) = 0;
Ppos_m = normcdf(Z);
BMA_PSP = round(Ppos_m' * p_prob_for_var, 6);
BMA_PSP(sig_idx) = 1;

if sfa ~= 0
    BMA_etykiety = [etykiety, 'sig_v','sig_u'];
else
    BMA_etykiety = [etykiety, 'sig_v'];
end

kol_lab = {'PIP','Est','SE','|t-ratio|','PSP'};
Tab = table(BMA_PIP, BMA_par, BMA_se, abs(BMA_par./BMA_se), BMA_PSP, ...
    'VariableNames', kol_lab, 'RowNames', BMA_etykiety);

% Output
output.summary = Tab;
output.best_model = best_mdl;
output.best_step = best_step;
output.all_steps = all_steps;
output.all_model_prob = p_prob_mdl;
output.all_mdd = all_lgmdd;
output.all_mlik = all_ML;
output.all_pmax = all_pmax;
output.bestCrit_path = path_inf_crit;
output.path_vars = path_vars;
output.path_removed = path_removed;
output.path_removed_labels = path_removedLab;
output.path_removed_abs_t = path_min_t;
output.path_proposed_crit = path_proposed_crit;
output.model_type = sfa;

end
