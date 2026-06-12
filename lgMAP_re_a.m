function lgmap = lgMAP_re_a(theta,X,y,n,T)

lgmap = lgl_re_a(theta, X, y, n, T) + lg_pr_re_a(theta);

end
