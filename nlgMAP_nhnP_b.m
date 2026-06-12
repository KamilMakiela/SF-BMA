function [nmap, ng] = nlgMAP_nhnP_b(theta,X,y,n,T)

[nll, g_nll] = nlgl_nhnP_b(theta, X, y,n,T);
[lp, g_lp] = lg_pr_nhn_b(theta);

nmap  = nll - lp;
ng = g_nll - g_lp;

end