function [nmap, ng] = nlgMAP_nex_b(theta,X,y)

[nll, g_nll] = nlgl_nex_b(theta, X, y);
[lp, g_lp] = lg_pr_nex_b(theta);

nmap  = nll - lp;
ng = g_nll - g_lp;

end