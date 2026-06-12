function output = fast_ES(exp_v,dep_v,n,T,sfa_opt,init_var,dec_cr)

if istable(exp_v)
    X = table2array(exp_v);
    p = size(X,2) - init_var;
else
    p = size(exp_v,2) - init_var;
end
topN = 4000;
kryt = 2^p;
fast=1;
if kryt <= topN
    disp('Running full ES');
    output = exhaustive_search_parallel(exp_v,dep_v,n,T,sfa_opt,init_var,dec_cr);
else
    disp('Running fast ES');
    output = exhaustive_search_screaning(exp_v,dep_v,n,T,sfa_opt,init_var,dec_cr, topN, fast);
end