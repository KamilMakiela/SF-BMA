function output = fast_fb(tabela, y, n, T, sfa, init_var, dec_cr)
%FAST_FBS Fast forward-backward search for SFA models.
%
% This replaces the pure forward search in full_fs.m.
% At each accepted model it evaluates the one-change neighbourhood:
%   1) add one candidate variable,
%   2) remove one currently included variable.
%
% Swap models are not evaluated here. Candidate models are evaluated in
% parallel with parfor. The outer search remains sequential because the next
% neighbourhood depends on the currently accepted model.

if dec_cr == 1
    if_mdd = 1;
else
    if_mdd = 0;
end

if istable(tabela)
    X = table2array(tabela);
    nazwy = tabela.Properties.VariableNames;
else
    X = tabela;
    nazwy = "x" + string(1:size(X,2));
end
etykiety = string(nazwy);

k = size(X,2);

if istable(y)
    y = table2array(y);
end

if sfa == 0
    sig_vars = 1;
    sig_idx = k+1;
else
    sig_vars = 2;
    sig_idx = k+1:k+2;
end

% Mandatory variables and candidate variables
X0 = X(:,1:init_var);
X_cand = X(:,init_var+1:end);
p = size(X_cand,2);

% Current accepted model, expressed relative to X_cand
current_vars = [];
included = false(1,p);

% Storage. Use dynamic growth because the number of visited neighbourhoods
% depends on when the local search stops.
cykli = 0;
all_lgmdd = [];
all_ML = [];
all_pmax = [];
all_steps = cell(0,2);
all_par = zeros(0,k+sig_vars);
all_se  = zeros(0,k+sig_vars);
path_vars = cell(0,1);
path_inf_crit = [];

% Evaluate and store initial mandatory-variable-only model
mdl0 = sfa_fit_rep(X0, y, n, T, sfa, dec_cr, if_mdd);
current_inf_crit = mdl0.inf_cr;
current_mdl = mdl0;
cykli = store_model(cykli, current_vars, current_mdl);

step = 0;
improved = true;

% the loop breaks if there is no improvement
% that is why this does not go over every model possible in the algorithm
% which is p(1+p) but stops earlier
while improved

    step = step + 1;
    improved = false;

    % Build add/remove/swap neighbourhood
    candidates = build_neighbourhood(included);
    nCand = size(candidates,1);

    if nCand == 0
        break
    end

    inf_crit_vec = Inf(nCand,1);
    mdl_vec  = cell(nCand,1);
    vars_vec = cell(nCand,1);

    % Parallel evaluation of all add/remove/swap candidates
    parfor t = 1:nCand

        mask = candidates(t,:);
        vars = find(mask);
        Xtemp = [X0 X_cand(:,vars)];

        mdl_curr = sfa_fit_rep(Xtemp, y, n, T, sfa, dec_cr, if_mdd);

        inf_crit_vec(t) = mdl_curr.inf_cr;
        mdl_vec{t} = mdl_curr;
        vars_vec{t} = vars;

    end

    % Store all tested candidates sequentially
    for t = 1:nCand
        cykli = store_model(cykli, vars_vec{t}, mdl_vec{t});
    end

    % Select best neighbour: lower inf_crit is better
    [best_inf_crit, best_t] = min(inf_crit_vec);

    if best_inf_crit < current_inf_crit
        included = candidates(best_t,:);
        current_vars = vars_vec{best_t};
        current_inf_crit = best_inf_crit;
        current_mdl = mdl_vec{best_t};
        improved = true;
    end

    path_vars{step,1} = current_vars;
    path_inf_crit(step,1) = current_inf_crit;

end

% Select best model by marginal data density, as in full_fs.m
[~, best_step] = max(all_lgmdd);
X_best = X(:,logical(all_par(best_step,1:end-sig_vars)));

included_best = false(1,p);
included_best(logical(all_par(best_step,(init_var+1):(end-sig_vars)))) = true;

best_mdl = sfa_fit_rep(X_best, y, n, T, sfa, dec_cr, 1);
best_mdl.X = X_best;

if istable(tabela)
    labels_best = nazwy(logical([ones(1,init_var) included_best]));
else
    labels_best = [ones(1,init_var) included_best];
end
best_mdl.labels = labels_best;

% BMA-style summaries over all visited models
w = exp(all_lgmdd - max(all_lgmdd));
p_prob_mdl = w / sum(w);
BMA_PIP = logical(all_par)' * p_prob_mdl;
BMA_par = all_par' * p_prob_mdl;

% Remove problematic rows for SE calculation only
row_sum = sum(all_se,2);
bad_ind = find(~isfinite(row_sum) | ~isreal(row_sum));
all_se_good = all_se;
all_se_good(bad_ind,:) = [];
p_prob_se = p_prob_mdl;
p_prob_se(bad_ind) = [];

% Renormalize SE weights after dropping bad SE rows
if ~isempty(p_prob_se)
    p_prob_se = p_prob_se / sum(p_prob_se);
end

var_within = (all_se_good.^2)' * p_prob_se;
var_between = ((all_par - BMA_par').^2') * p_prob_mdl;
BMA_se = sqrt(var_within + var_between);

BMA_par(sig_idx) = exp(BMA_par(sig_idx));
BMA_se(sig_idx) = BMA_par(sig_idx) .* BMA_se(sig_idx);

% Positive sign probability
all_par_good = all_par;
all_par_good(bad_ind,:) = [];
Z = all_par_good ./ all_se_good;
Z(~isfinite(Z)) = 0;
Ppos_m = normcdf(Z);
Ppos = Ppos_m' * p_prob_se;

BMA_PSP = round(Ppos,6); % positive sign probability
BMA_PSP(sig_idx) = 1;    % scale parameters are positive by construction

if sfa ~= 0
    BMA_etykiety = [etykiety, 'sig_v', 'sig_u'];
else
    BMA_etykiety = [etykiety, 'sig_v'];
end

kol_lab = {'PIP','Est','SE','|t-ratio|','PSP'};
Tab = table(BMA_PIP, BMA_par, BMA_se, abs(BMA_par./BMA_se), BMA_PSP, ...
    'VariableNames', kol_lab, 'RowNames', BMA_etykiety);

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
output.model_type = sfa;

% Nested helper: build neighbourhood of current model
    function candidates = build_neighbourhood(included_now)

        % Fast one-change neighbourhood only: additions and removals.
        % No add/remove swap models are evaluated here.
        add_vars = find(~included_now);
        rem_vars = find(included_now);

        candidates = false(numel(add_vars) + numel(rem_vars), p);
        c = 0;

        for a = 1:numel(add_vars)
            mask = included_now;
            mask(add_vars(a)) = true;
            c = c + 1;
            candidates(c,:) = mask;
        end

        for r = 1:numel(rem_vars)
            mask = included_now;
            mask(rem_vars(r)) = false;
            c = c + 1;
            candidates(c,:) = mask;
        end

        candidates = candidates(1:c,:);

        % Exclude models with no optional/candidate regressors.
        % The initial mandatory-only model X0 is evaluated once before the loop.
        % During the search, do not re-evaluate the intercept-only / mandatory-only model.
        if ~isempty(candidates)
            has_candidate_regressor = sum(candidates,2) > 0;
            candidates = candidates(has_candidate_regressor,:);
        end

    end


% Nested helper: store one fitted model in full-size arrays
    function cykli_out = store_model(cykli_in, vars, mdl_curr)

        cykli_out = cykli_in + 1;

        all_steps{cykli_out,1} = vars;
        all_steps{cykli_out,2} = etykiety(vars + init_var);
        all_lgmdd(cykli_out,1) = mdl_curr.mdd;
        all_ML(cykli_out,1) = mdl_curr.l_max;
        all_pmax(cykli_out,1) = mdl_curr.p_max;

        all_par(cykli_out,1:k+sig_vars) = 0;
        all_se(cykli_out,1:k+sig_vars) = 0;

        if if_mdd == 1
            theta = mdl_curr.bayes.theta_post;
            theta_se = mdl_curr.bayes.theta_post_se;
        else
            theta = mdl_curr.theta_ml;
            theta_se = mdl_curr.theta_ml_se;
        end

        % Mandatory variables
        all_par(cykli_out,1:init_var) = theta(1:init_var)';
        all_se(cykli_out,1:init_var) = theta_se(1:init_var)';

        % Selected variables. vars are relative to X_cand, so add init_var.
        if ~isempty(vars)
            all_par(cykli_out,vars + init_var) = theta((init_var+1):(end-sig_vars))';
            all_se(cykli_out,vars + init_var) = theta_se((init_var+1):(end-sig_vars))';
        end

        % Scale parameters
        all_par(cykli_out,(end-sig_vars+1):end) = theta((end-sig_vars+1):end)';
        all_se(cykli_out,(end-sig_vars+1):end) = theta_se((end-sig_vars+1):end)';

    end

end
