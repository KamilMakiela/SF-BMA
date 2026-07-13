%% Presettings
sfa_opt = 1;
init_var = 1; %including the constant
dec_cr = 1;
if_mdd = 1;


%% DATASETS
% OPTION 1: artificial data
rng(0);
n = 500;
T = 1;
generate_data;

% OPTION 2: dataset Produc form plm package in R
% Croissant, Y., & Millo, G. (2008). Panel Data Econometrics in R: 
% The plm Package. Journal of Statistical Software, 27(2), 1–43
%{
Tab = readtable('Produc.xlsx', 'sheet', 'm1');
Tab(:, [1 2 3]) = [];
vars = Tab.Properties.VariableNames;

%Tab{:,:} = log(Tab{:,:});
dep_v = Tab(:,1);
exp_v = Tab(:,2:end);
init_var = 1;
T=1;
%}

y = dep_v{:,:};
X = exp_v{:,:};
n = numel(y)/T;

%% Estimation

% estimate the full model
sf_full_model = sfa_fit_rep(X,y,n,T,sfa_opt,dec_cr,1);

% forward selection
% with a stop, variables ordered by their marginal correlation with 
% the dependent
selection_forward = fast_fs(exp_v,dep_v,n,T,sfa_opt,init_var,dec_cr);

% backward selection
% with a stop, reduction by one, based on the lowest abs(t-ratio)
selection_backward = fast_bs(exp_v,dep_v,n,T,sfa_opt,init_var,dec_cr);

% forward search
% no stop, full fs; variables still ordered but it is not necessary really
% more of a search/walk, not a selection
search_forward = full_fs(exp_v,dep_v,n,T,sfa_opt,init_var,dec_cr);

% backward search/walk
% no stop
search_backward = full_bs(exp_v,dep_v,n,T,sfa_opt,init_var,dec_cr);

% model-selection based on a forward-backward search
% bottom-up growing: add/remove one, no swaps.
% evaluate all add-one-variable moves,
% evaluate all remove-one-variable moves,
% stop criterion - yes
search_forward_backward = fast_fb(exp_v,dep_v,n,T,sfa_opt,init_var,dec_cr);

% forward-backward-swap algorithm
% bottom-up growing model-space search: best-improvement local search over 
% the add/remove/swap neighborhood. evaluate all pair-swap moves, restrict 
% to unique models. A best-improvement local search over the 
% add/delete/exchange neighborhood. stop criterion - yes
% this is the "gold standard" in heuristic search
% see, e.g.:
% Miller, A. (2002). Subset selection in regression (2nd ed.). Boca Raton,
% FL: Chapman & Hall/CRC. https://doi.org/10.1201/9781420035933
% tic;
search_fbs = full_fbs_unique(exp_v,dep_v,n,T,sfa_opt,init_var,dec_cr);
% toc;

% algorithm for full exhaustive search
tic;
search_es = fast_ES(exp_v,dep_v,n,T,sfa_opt,init_var,dec_cr);
toc;
% disp(t_score);
disp(search_es.summary);
try
    disp("True parameters (labels)");
    disp(lista);
catch
    
end