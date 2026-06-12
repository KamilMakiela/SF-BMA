function output = exhaustive_search_parallel(tabela, y, n, T, sfa, init_var, dec_cr)
%EXHAUSTIVE_SEARCH_PARALLEL Exhaustive model search with parallel evaluation.
%
% This is the exhaustive-search version based on full_fs.m.
% It evaluates all 2^p models, where p is the number of optional
% candidate regressors. The first init_var variables are always included.
%
% Required external function:
%   mdl = sfa_fit_rep(Xtemp, y, n, T, sfa, dec_cr, if_mdd)

% Settings
if dec_cr == 1
    if_mdd = 1;
else
    if_mdd = 0;
end

% Convert inputs
if istable(tabela)
    X = table2array(tabela);
    nazwy = string(tabela.Properties.VariableNames);
else
    X = tabela;
    nazwy = "x" + string(1:size(X,2));
end
etykiety = string(nazwy);

if istable(y)
    y = table2array(y);
end

k = size(X,2);

if sfa == 0
    sig_vars = 1;
    sig_idx = k + 1;
else
    sig_vars = 2;
    sig_idx = k + 1:k + 2;
end

% Split mandatory and candidate regressors
X0     = X(:,1:init_var);
X_cand = X(:,init_var+1:end);
p      = size(X_cand,2);

% Total number of candidate subsets, including empty subset
M = 2^p;

% Preallocate storage for paralelization
all_lgmdd = NaN(M,1);
all_bic = NaN(M,1);
all_ML    = NaN(M,1);
all_pmax  = NaN(M,1);
all_id  = NaN(M,1);
all_vars   = cell(M,1);
all_labels = cell(M,1);
all_par   = zeros(M,k+sig_vars);
all_se    = zeros(M,k+sig_vars);
all_infcr = NaN(M,1);
exit_ok   = true(M,1);

% Exhaustive parallel search over all 2^p subsets
% Model m = 1 corresponds to the baseline model with X0 only.
parfor m = 1:M

    % mask(j) = true means candidate variable j is included.
    % bitget uses least-significant bit first, so candidate 1 corresponds
    % to X_cand(:,1), candidate 2 to X_cand(:,2), etc.
    mask = logical(bitget(uint64(m-1), 1:p));
    vars = find(mask);

    Xtemp = [X0, X_cand(:,mask)];

    try
        mdl_curr = sfa_fit_rep(Xtemp, y, n, T, sfa, dec_cr, if_mdd);

        par_row = zeros(1,k+sig_vars);
        se_row  = zeros(1,k+sig_vars);

        if if_mdd == 1
            theta = mdl_curr.bayes.theta_post(:)';
            se    = mdl_curr.bayes.theta_post_se(:)';
        else
            theta = mdl_curr.theta_ml(:)';
            se    = mdl_curr.theta_ml_se(:)';
        end

        % Mandatory regressors
        par_row(1:init_var) = theta(1:init_var);
        se_row(1:init_var)  = se(1:init_var);

        % Optional regressors: theta positions are compact inside Xtemp,
        % but par_row positions are expanded to the original full-X layout.
        full_optional_pos = init_var + vars;
        compact_optional_pos = init_var + (1:numel(vars));

        if ~isempty(vars)
            par_row(full_optional_pos) = theta(compact_optional_pos);
            se_row(full_optional_pos)  = se(compact_optional_pos);
        end

        % Scale parameters at the end
        par_row(end-sig_vars+1:end) = theta(end-sig_vars+1:end);
        se_row(end-sig_vars+1:end)  = se(end-sig_vars+1:end);

        all_vars{m}   = vars;
        all_id(m) = sum(2.^(vars - 1));
        all_labels{m} = etykiety(init_var + vars);
        all_lgmdd(m)   = mdl_curr.mdd;
        all_bic(m) = mdl_curr.bic;
        all_ML(m)      = mdl_curr.l_max;
        all_pmax(m)    = mdl_curr.p_max;
        all_infcr(m)   = mdl_curr.inf_cr;
        all_par(m,:)   = par_row;
        all_se(m,:)    = se_row;

    catch
        all_vars{m}   = vars;
        all_labels{m} = etykiety(init_var + vars);
        exit_ok(m) = false;
    end
end

% Remove models that failed completely
good_model = exit_ok & isfinite(all_lgmdd);

all_lgmdd = all_lgmdd(good_model);
all_bic = all_bic(good_model); 
all_ML    = all_ML(good_model);
all_pmax  = all_pmax(good_model);
all_id = all_id(good_model);
all_infcr = all_infcr(good_model);
all_vars   = all_vars(good_model);
all_labels = all_labels(good_model);
all_steps  = [all_vars, all_labels];
all_par   = all_par(good_model,:);
all_se    = all_se(good_model,:);

% Best model by maximum log marginal data density
[~, best_step] = max(all_lgmdd);

best_included_full = logical(all_par(best_step,1:end-sig_vars));
X_best = X(:,best_included_full);

best_mdl = sfa_fit_rep(X_best, y, n, T, sfa, dec_cr, 1);
best_mdl.X = X_best;

if istable(tabela)
    labels_best = nazwy(best_included_full);
else
    labels_best = find(best_included_full);
end
best_mdl.labels = labels_best;
best_mdl.included = best_included_full;

% Posterior model probabilities
w = exp(all_lgmdd - max(all_lgmdd));
p_prob_mdl = w / sum(w);

% BMA posterior means and PIP
BMA_PIP = double(logical(all_par))' * p_prob_mdl;
BMA_par = all_par' * p_prob_mdl;

% Remove numerically problematic SE rows only for SE/PSP calculations
row_sum = sum(all_se,2);
good_se = isfinite(row_sum) & isreal(row_sum);

all_se_good  = all_se(good_se,:);
all_par_good = all_par(good_se,:);
p_prob_good  = p_prob_mdl(good_se);
p_prob_good  = p_prob_good / sum(p_prob_good);

BMA_par_good = all_par_good' * p_prob_good;

var_within = (all_se_good.^2)' * p_prob_good;
var_between = ((all_par_good - BMA_par_good').^2)' * p_prob_good;
BMA_se = sqrt(var_within + var_between);

% Back-transform scale parameters from logs
BMA_par(sig_idx) = exp(BMA_par(sig_idx));
BMA_se(sig_idx)  = BMA_par(sig_idx) .* BMA_se(sig_idx);

% Posterior positive sign probability
Z = all_par_good ./ all_se_good;
Z(~isfinite(Z)) = 0;
Ppos_m = normcdf(Z);
Ppos = Ppos_m' * p_prob_good;

BMA_PSP = round(Ppos,6);       % positive sign probability
BMA_PSP(sig_idx) = 1;          % scale parameters are positive by construction

% Summary table
if sfa ~= 0
    BMA_etykiety = [etykiety, "sig_v", "sig_u"];
else
    BMA_etykiety = [etykiety, "sig_v"];
end

kol_lab = {'PIP','Est','SE','|t-ratio|','PSP'};
Tab = table(BMA_PIP, BMA_par, BMA_se, abs(BMA_par./BMA_se), BMA_PSP, ...
    'VariableNames', kol_lab, 'RowNames', cellstr(BMA_etykiety));

% Output
output.summary = Tab;
output.best_model = best_mdl;
output.best_step = best_step;
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
output.model_type = sfa;
output.total_models_evaluated = numel(all_lgmdd);
output.total_possible_models = M;

end
