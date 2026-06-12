function output = exhaustive_search_screaning(exp_v, dep_v, n, T, sfa_opt, init_var, dec_cr, topN, fast)
%EXHAUSTIVE_SEARCH_FAST Large-model exhaustive-search branch with BIC screening.
%
% output = exhaustive_search_fast(exp_v, dep_v, n, T, sfa_opt, init_var, dec_cr)
%
% Strategy:
%   The wrapper fast_ES should decide whether to call full exhaustive 
%   search or this fast screened version.
%
%   1. Run quick exhaustive BIC screening over all 2^p models in parfor,
%      with dec_cr = 0 and if_mdd = 0.
%   2. Keep the topN models according to BIC/inf_cr.
%   3. Re-run the requested BMA strategy only on those topN models,
%      using the user-supplied dec_cr.
%
% Required external function:
%   mdl = sfa_fit_rep(Xtemp, dep_v, n, T, sfa_opt, dec_cr, if_mdd)
%
% Model ID convention:
%   model_id = m, where m = 1,...,2^p.
%   The included optional variables are reconstructed by:
%       mask = logical(bitget(uint64(model_id-1), 1:p));
%       vars = find(mask);
%
% Notes:
%   - The first init_var columns of exp_v are always included.
%   - The optional candidates are exp_v(:, init_var+1:end).
%   - In the large-model case, posterior probabilities are conditional on
%     the BIC-screened top 1000 models, not the full 2^p model space.

% Fixed settings (turned off, the wrapper decides on topN)
%topN = 1000;

% Bayesian part for final stage
if dec_cr == 1
    if_mdd_final = 1;
else
    if_mdd_final = 0;
end

% Convert inputs
if istable(exp_v)
    X = table2array(exp_v);
    var_names = string(exp_v.Properties.VariableNames);
else
    X = exp_v;
    var_names = "x" + string(1:size(X,2));
end

if istable(dep_v)
    y = table2array(dep_v);
else
    y = dep_v;
end

labels = string(var_names);
k_full = size(X,2);

if sfa_opt == 0
    sig_vars = 1;
    sig_idx = k_full + 1;
else
    sig_vars = 2;
    sig_idx = k_full + 1:k_full + 2;
end

X0     = X(:,1:init_var);
X_cand = X(:,init_var+1:end);
p      = size(X_cand,2);

if p > 63
    error('This implementation uses uint64 bit masks and supports at most 63 optional covariates.');
end

M = 2^p;
nT = n*T;

% Stage 1: quick BIC screening over all candidate models
strategy = "bic_screen_then_bma";

bic_all = NaN(M,1);
%logLik_all = NaN(M,1);
model_size_all = NaN(M,1);
ok_screen = false(M,1);

% Quick screening only. Each worker writes only to the sliced arrays.
parfor m = 1:M
    mask = logical(bitget(uint64(m-1), 1:p));
    Xtemp = [X0, X_cand(:,mask)];

    try
        if fast == 1
            mdl0 = pre_screen(Xtemp, y, nT);
        else
            mdl0 = sfa_fit_rep(Xtemp, y, n, T, sfa_opt, 0, 0);
        end
        bic_all(m) = mdl0.inf_cr;
        %logLik_all(m) = mdl0.l_max;
        model_size_all(m) = init_var + sum(mask);
        ok_screen(m) = isfinite(mdl0.inf_cr);
    catch
        % Leave this model as failed.
    end
end

good = ok_screen & isfinite(bic_all);
good_ids = find(good);

if isempty(good_ids)
    error('No model passed the quick BIC screening stage.');
end

[~, ord] = sort(bic_all(good_ids), 'ascend');
% in case somehow we get fewer models than topN
n_keep = min(topN, numel(good_ids));
selected_model_ids = good_ids(ord(1:n_keep));

% Build readable screening table only for retained models.
top_vars = cell(n_keep,1);
top_labels = cell(n_keep,1);

for j = 1:n_keep
    m = selected_model_ids(j);
    mask = logical(bitget(uint64(m-1), 1:p));
    vars = find(mask);
    top_vars{j} = vars;
    top_labels{j} = labels(init_var + vars);
end

screen_top = table(selected_model_ids, bic_all(selected_model_ids), ...
    model_size_all(selected_model_ids), top_vars, top_labels, ...
    'VariableNames', {'model_id','BIC','model_size','vars','labels'});

screen_top = sortrows(screen_top, 'BIC', 'ascend');

% Stage 2: final estimation on all selected models
S = numel(selected_model_ids);

all_lgmdd = NaN(S,1);
all_ML    = NaN(S,1);
all_pmax  = NaN(S,1);
all_infcr = NaN(S,1);
all_bic   = NaN(S,1);
all_id    = NaN(S,1);
all_par   = zeros(S,k_full+sig_vars);
all_se    = zeros(S,k_full+sig_vars);
all_vars  = cell(S,1);
all_labs  = cell(S,1);
exit_ok   = false(S,1);

parfor s = 1:S
    m = selected_model_ids(s);
    mask = logical(bitget(uint64(m-1), 1:p));
    vars = find(mask);
    Xtemp = [X0, X_cand(:,mask)];

    try
        mdl_curr = sfa_fit_rep(Xtemp, y, n, T, sfa_opt, dec_cr, if_mdd_final);

        par_row = zeros(1,k_full+sig_vars);
        se_row  = zeros(1,k_full+sig_vars);

        if if_mdd_final == 1
            theta = mdl_curr.bayes.theta_post(:)';
            se    = mdl_curr.bayes.theta_post_se(:)';
        else
            theta = mdl_curr.theta_ml(:)';
            se    = mdl_curr.theta_ml_se(:)';
        end

        % Always-included variables.
        par_row(1:init_var) = theta(1:init_var);
        se_row(1:init_var)  = se(1:init_var);

        % Optional variables: compact positions in Xtemp -> full positions in X.
        if ~isempty(vars)
            full_optional_pos = init_var + vars;
            compact_optional_pos = init_var + (1:numel(vars));

            par_row(full_optional_pos) = theta(compact_optional_pos);
            se_row(full_optional_pos)  = se(compact_optional_pos);
        end

        % Scale parameters at the end.
        par_row(end-sig_vars+1:end) = theta(end-sig_vars+1:end);
        se_row(end-sig_vars+1:end)  = se(end-sig_vars+1:end);

        all_lgmdd(s) = mdl_curr.mdd;
        all_bic(s) = mdl_curr.bic;
        all_ML(s)    = mdl_curr.l_max;
        all_pmax(s)  = mdl_curr.p_max;
        all_infcr(s) = mdl_curr.inf_cr;
        all_par(s,:) = par_row;
        all_se(s,:)  = se_row;
        all_vars{s}  = vars;
        all_id(s) = sum(2.^(vars - 1));
        all_labs{s}  = labels(init_var + vars);
        exit_ok(s)   = true;
    catch
        all_vars{s} = vars;
        all_labs{s} = labels(init_var + vars);
    end
end

all_steps = [all_vars, all_labs];

% Remove failed final models if there are any
good_final = exit_ok & isfinite(all_lgmdd);

if ~any(good_final)
    error('No model passed the final estimation stage.');
end

selected_model_ids = selected_model_ids(good_final);
all_lgmdd = all_lgmdd(good_final);
all_bic = all_bic(good_final); 
all_ML    = all_ML(good_final);
all_id = all_id(good_final);
all_pmax  = all_pmax(good_final);
all_infcr = all_infcr(good_final);
all_par   = all_par(good_final,:);
all_se    = all_se(good_final,:);
all_steps = all_steps(good_final,:);

% Best model
[~, best_step] = max(all_lgmdd);
best_model_id = selected_model_ids(best_step);

best_included_full = logical(all_par(best_step,1:end-sig_vars));
X_best = X(:,best_included_full);

best_mdl = sfa_fit_rep(X_best, y, n, T, sfa_opt, dec_cr, 1);
best_mdl.X = X_best;

if istable(exp_v)
    best_mdl.labels = var_names(best_included_full);
else
    best_mdl.labels = find(best_included_full);
end
best_mdl.included = best_included_full;

% Posterior model probabilities over evaluated final models
w = exp(all_lgmdd - max(all_lgmdd));
p_prob_mdl = w / sum(w);

% BMA posterior means and PIP
BMA_PIP = double(logical(all_par))' * p_prob_mdl;
BMA_par = all_par' * p_prob_mdl;

% BMA SE and PSP
row_sum = sum(all_se,2);
good_se = isfinite(row_sum) & isreal(row_sum);

all_se_good  = all_se(good_se,:);
all_par_good = all_par(good_se,:);
p_prob_good  = p_prob_mdl(good_se);
p_prob_good  = p_prob_good / sum(p_prob_good);

BMA_par_good = all_par_good' * p_prob_good;

var_within  = (all_se_good.^2)' * p_prob_good;
var_between = ((all_par_good - BMA_par_good').^2)' * p_prob_good;
BMA_se = sqrt(var_within + var_between);

% Back-transform scale parameters from logs.
BMA_par(sig_idx) = exp(BMA_par(sig_idx));
BMA_se(sig_idx)  = BMA_par(sig_idx) .* BMA_se(sig_idx);

% Posterior positive sign probability.
Z = all_par_good ./ all_se_good;
Z(~isfinite(Z)) = 0;
Ppos_m = normcdf(Z);
Ppos = Ppos_m' * p_prob_good;

BMA_PSP = round(Ppos,6);
BMA_PSP(sig_idx) = 1;

% Summary table
if sfa_opt ~= 0
    BMA_labels = [labels, "sig_v", "sig_u"];
else
    BMA_labels = [labels, "sig_v"];
end

Tab = table(BMA_PIP, BMA_par, BMA_se, abs(BMA_par./BMA_se), BMA_PSP, ...
    'VariableNames', {'PIP','Est','SE','abs_t_ratio','PSP'}, ...
    'RowNames', cellstr(BMA_labels));

% Output
output.summary = Tab;
output.best_model = best_mdl;
output.best_step = best_step;
output.best_model_id = best_model_id;
output.selected_model_ids = selected_model_ids;
output.all_steps = all_steps;
output.all_model_prob = p_prob_mdl;
output.all_mdd = all_lgmdd;
output.all_bic = all_bic;
output.all_mlik = all_ML;
output.all_pmax = all_pmax;
output.all_id = all_id;
output.all_infcr = all_infcr;
output.all_par = all_par;
output.all_se = all_se;
output.model_type = sfa_opt;
output.strategy = strategy;
output.screen_top = screen_top;
output.topN = topN;
output.total_possible_models = M;
output.total_models_evaluated_final = numel(all_lgmdd);

output.warning = "Posterior probabilities are conditional on the BIC-screened top models, not on the full 2^p model space.";

end
