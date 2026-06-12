function [nmap, ng] = nlgMAP_nexP_b(theta,X,y,n,T)

[nll, g_nll] = nlgl_nexP_b(theta, X, y,n,T);
[lp, g_lp] = lg_pr_nex_b(theta);

nmap  = nll - lp;
ng = g_nll - g_lp;

end