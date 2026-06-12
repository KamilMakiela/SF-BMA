function output = full_fs(tabela, y, n, T, sfa, init_var, dec_cr)

if dec_cr == 1
    if_mdd = 1;
else
    if_mdd = 0;
end

if istable(tabela)
    %kowersja do tablicy
    X = table2array(tabela);
    nazwy = tabela.Properties.VariableNames;
    %typowanie zmiennych ważnych; tu zakładam x0 (zawsze) oraz x5
else
    X = tabela;
    nazwy = string(tabela.Properties.VariableNames);
end
etykiety = string(nazwy);

k = size(X,2);

if istable(y)
    %kowersja do tablicy
    y = table2array(y);
    %y_lab = dep.Properties.VariableNames;
    %typowanie zmiennych ważnych; tu zakładam x0 (zawsze) oraz x5
end

if sfa == 0
    sig_vars=1;
    sig_idx = k+1;
else
    sig_vars=2;
    sig_idx = k+1:k+2;
end

%na początku są tylko zmienne które muszą być w modelu
X0 = X(:,1:init_var);
X_cand = X(:,init_var+1:end);
p = size(X_cand,2); %number of variables for selection

r = corr(X_cand, y);
[~, idx] = sort(abs(r), 'descend');
included = zeros(1,p);
remaining = idx;       % ordered candidate list
current_vars = [];

%simple model
%initial model:
%mdl = sfa_fit_rep(X0,y,n,T,sfa,dec_cr,if_mdd);
%current_BIC = mdl.inf_cr;

%tic;
cykli = 0;

% I need: mdd, vars_order, beta, se
% I need: mdd, vars_order, beta, se
max_models = 1 + p*(p+1)/2;

all_lgmdd = zeros(max_models,1);
all_ML = zeros(max_models,1);
all_pmax = zeros(max_models,1);
all_steps = cell(max_models,2);
all_par   = zeros(max_models,k+sig_vars);
all_se    = zeros(max_models,k+sig_vars);

path_vars     = cell(p,1);
path_inf_crit = zeros(p,1);

% Initial baseline model: mandatory regressors only (X0)
% This is usually the intercept-only model when init_var = 1.
% It is stored as current_vars = [] because no optional variables
% have been selected yet.
mdl0 = sfa_fit_rep(X0,y,n,T,sfa,dec_cr,if_mdd);
cykli = cykli + 1;

all_steps{cykli,1} = [];
all_steps{cykli,2} = etykiety(1:init_var);
all_lgmdd(cykli) = mdl0.mdd;
all_ML(cykli) = mdl0.l_max;
all_pmax(cykli) = mdl0.p_max;

if if_mdd == 1
    all_par(cykli,1:init_var) = mdl0.bayes.theta_post(1:init_var)';
    all_par(cykli,(end-sig_vars+1):end) = mdl0.bayes.theta_post((end-sig_vars+1):end)';
    all_se(cykli,1:init_var) = mdl0.bayes.theta_post_se(1:init_var)';
    all_se(cykli,(end-sig_vars+1):end) = mdl0.bayes.theta_post_se((end-sig_vars+1):end)';
else
    all_par(cykli,1:init_var) = mdl0.theta_ml(1:init_var)';
    all_par(cykli,(end-sig_vars+1):end) = mdl0.theta_ml((end-sig_vars+1):end)';
    all_se(cykli,1:init_var) = mdl0.theta_ml_se(1:init_var)';
    all_se(cykli,(end-sig_vars+1):end) = mdl0.theta_ml_se((end-sig_vars+1):end)';
end


for step = 1:p

    nr = length(remaining);

    inf_crit_vec = Inf(nr,1);
    mdl_vec  = cell(nr,1);
    vars_vec = cell(nr,1);

    % Parallel evaluation of all remaining candidate models
    parfor t = 1:nr

        j = remaining(t);

        vars = [current_vars j];
        Xtemp = [X0 X_cand(:,vars)];

        mdl_curr = sfa_fit_rep(Xtemp,y,n,T,sfa,dec_cr,if_mdd);

        inf_crit_vec(t) = mdl_curr.inf_cr;
        mdl_vec{t} = mdl_curr;
        vars_vec{t} = vars;

    end

    % Store results sequentially
    for t = 1:nr

        cykli = cykli + 1;

        vars = vars_vec{t};
        mdl_curr = mdl_vec{t};

        all_steps{cykli,1} = vars;
        all_steps{cykli,2} = etykiety(vars+1);
        all_lgmdd(cykli) = mdl_curr.mdd;
        all_ML(cykli) = mdl_curr.l_max;
        all_pmax(cykli) = mdl_curr.p_max;
        if if_mdd == 1
            all_par(cykli,1:init_var) = mdl_curr.bayes.theta_post(1:init_var)';
            all_par(cykli,(vars+init_var)) = mdl_curr.bayes.theta_post((init_var+1):(end-sig_vars))';
            all_par(cykli,(end-sig_vars+1):end) = mdl_curr.bayes.theta_post((end-sig_vars+1):end)';
            all_se(cykli,1:init_var) = mdl_curr.bayes.theta_post_se(1:init_var)';
            all_se(cykli,(vars+init_var)) = mdl_curr.bayes.theta_post_se((init_var+1):(end-sig_vars))';
            all_se(cykli,(end-sig_vars+1):end) = mdl_curr.bayes.theta_post_se((end-sig_vars+1):end)';
        else
            all_par(cykli,1:init_var) = mdl_curr.theta_ml(1:init_var)';
            all_par(cykli,(vars+init_var)) = mdl_curr.theta_ml((init_var+1):(end-sig_vars))';
            all_par(cykli,(end-sig_vars+1):end) = mdl_curr.theta_ml((end-sig_vars+1):end)';
            all_se(cykli,1:init_var) = mdl_curr.theta_ml_se(1:init_var)';
            all_se(cykli,(vars+init_var)) = mdl_curr.theta_ml_se((init_var+1):(end-sig_vars))';
            all_se(cykli,(end-sig_vars+1):end) = mdl_curr.theta_ml_se((end-sig_vars+1):end)';
        end
    end

    % Select best candidate: lower inf_crit is better
    [best_inf_crit, best_t] = min(inf_crit_vec);
    best_var = remaining(best_t);

    current_vars = [current_vars best_var];
    included(best_var) = 1;

    remaining(remaining == best_var) = [];

    path_vars{step} = current_vars;
    path_nBIC(step) = best_inf_crit;

end

all_lgmdd = all_lgmdd(1:cykli);
all_steps = all_steps(1:cykli,:);
all_par   = all_par(1:cykli,:);
all_se    = all_se(1:cykli,:);
path_inf_crit = path_inf_crit(1:step);
% [~, best_step] = max(path_BIC);
% best_vars = path_vars{best_step};
% included = zeros(1,p);
% included(best_vars) = 1;

[~, best_step] = max(all_lgmdd);
X_best = X(:,logical(all_par(best_step,1:end-sig_vars)));
%best_vars = path_vars{best_step};
included = zeros(1,p);
included(logical(all_par(best_step,(init_var+1):(end-sig_vars)))) = 1;

%X_best = [X0, X_cand(:,logical(included))];
best_mdl = sfa_fit_rep(X_best,y,n,T,sfa,dec_cr,1);
best_mdl.X = X_best;

if istable(tabela)
    labels_best = nazwy(logical([ones(1,init_var) included]));
else
    labels_best = [ones(1,init_var) included];
end
best_mdl.labels = labels_best;
%disp(lista);
%disp(string(labels_best));
%disp(cykli);
%disp(best_step);
%%
w = exp(all_lgmdd - max(all_lgmdd));
p_prob_mdl = w / sum(w);
BMA_PIP = logical(all_par)'*p_prob_mdl;
BMA_par = all_par'*p_prob_mdl;

%usuwam bardzo problematyczne modele
row_sum = sum(all_se,2);
bad_ind = find(~isfinite(row_sum) | ~isreal(row_sum));
all_se(bad_ind,:) = [];
p_prob_se = p_prob_mdl;
p_prob_se(bad_ind) = [];
%%tu zrobić prognozę SE!!!!

var_within = (all_se.^2)'*p_prob_se;
var_btween = ((all_par - BMA_par').^2')*p_prob_mdl;
BMA_se = sqrt(var_within + var_btween);

BMA_par(sig_idx) = exp(BMA_par(sig_idx));
BMA_se(sig_idx) = BMA_par(sig_idx).*BMA_se(sig_idx);

%PSP
all_par(bad_ind,:) = [];
Z = all_par./ all_se;
Z(~isfinite(Z)) = 0;
Ppos_m = normcdf(Z);
Ppos = Ppos_m' * p_prob_mdl;

BMA_PSP = round(Ppos,6); %positive sign probability!!
BMA_PSP(sig_idx) = 1; %by construction!

%BMA_par(sig_idx) = exp(BMA_par(sig_idx));

if sfa ~= 0
    BMA_etykiety = [etykiety, 'sig_v','sig_u'];
else
    BMA_etykiety = [etykiety, 'sig_v'];
end 
kol_lab = {'PIP','Est','SE','|t-ratio|','PSP'};
Tab = table(BMA_PIP, BMA_par, BMA_se, abs(BMA_par./BMA_se),BMA_PSP, 'VariableNames', kol_lab, 'RowNames', BMA_etykiety);
%disp(Tab);


output.summary = Tab;
output.best_model = best_mdl;
output.best_step = best_step;
output.all_steps = all_steps;
output.all_model_prob = p_prob_mdl;
output.all_mdd = all_lgmdd;
output.all_mlik = all_ML;
output.all_pmax = all_pmax;

%output.all_pmax = 
output.bestCrit_path = path_nBIC;

output.model_type = sfa;


end