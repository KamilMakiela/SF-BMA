function [nmap, ng] = nlgMAP_kmnrl(theta,X,y)

[l1, g1] = lgl_kmnrl(theta, X, y); 
[l2, g2] = lg_pr_kmnrl(theta);

nmap = -l1 - l2;
ng = -g1 - g2;

end
