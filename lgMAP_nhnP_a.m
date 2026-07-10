function map = lgMAP_nhnP_a(theta,X,y,n,T)
map = lgl_nhnP_a(theta,X,y,n,T) + lg_pr_nhn_a(theta);
end