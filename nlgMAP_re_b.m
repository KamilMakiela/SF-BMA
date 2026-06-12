function [nmap, ng] = nlgMAP_re_b(theta, X, y, n, T)

[nL, ng] = nlgl_re_b(theta, X, y, n, T);
[lgp, glgp] = lg_pr_re_b(theta);

nmap  = nL - lgp;
ng = ng - glgp;

end