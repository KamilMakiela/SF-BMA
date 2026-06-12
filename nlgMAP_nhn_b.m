function [nmap, ng] = nlgMAP_nhn_b(theta, X, y)
[a, b] = nlgl_nhn_b(theta, X, y);
[i, j] = lg_pr_nhn_b(theta); 
nmap =  a-i;
ng = b-j;

end