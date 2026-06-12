function lgp = lg_pr_re_a(theta)

k = length(theta)-2;

b  = theta(1:k);
lv = theta(end-1);   % log(s_v)
la = theta(end);     % log(s_alpha)

av = 0.00005;
bv = 0.00005;

aa = 0.5;
ba = 0.00005;

lgp_b = -0.5*k*log(2*pi*100) - 0.5*(b'*b)/100;

% s_v^2 ~ IG(av,bv)
lgp_sv = av*log(bv) - gammaln(av) ...
         - (av+1)*(2*lv) ...
         - bv*exp(-2*lv) ...
         + log(2) + 2*lv;

% s_alpha^2 ~ IG(aa,ba)
lgp_sa = aa*log(ba) - gammaln(aa) ...
         - (aa+1)*(2*la) ...
         - ba*exp(-2*la) ...
         + log(2) + 2*la;

lgp = lgp_b + lgp_sv + lgp_sa;

end