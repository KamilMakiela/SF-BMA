function lgmap = lgMAP_nex_a(theta,X,y)

lgmap = lgl_nex_a(theta, X, y) + lg_pr_nex_a(theta);

end
