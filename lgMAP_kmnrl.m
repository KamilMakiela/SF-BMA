function lgmap = lgMAP_kmnrl(theta,X,y)

[a,~] = lgl_kmnrl(theta, X, y); 
[b,~] = lg_pr_kmnrl(theta);

lgmap = a + b;

end
