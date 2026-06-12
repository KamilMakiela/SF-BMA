function map = lgMAP_nhnP_a(theta,X,y,n,T)
map = lgl_nexP_a(theta,X,y,n,T) + lg_pr_nex_a(theta);
end