function lgp = lg_pr_nex_a(theta)

k = length(theta)-2;

b  = theta(1:end-2);
lv = theta(end-1);   % log(s_v)
lu = theta(end);     % log(s_u)

av = 0.00005;
bv = 0.00005;

au = 1;
bu = 0.2877; %~-log(0.75), 0.75 is priormedian efficiency

lgp_b = -0.5*k*log(2*pi*100) - 0.5*(b'*b)/100;

% s_v^2 prior + Jacobian
%lgp_sv = av*log(bv) - gammaln(av) ...
%         - (av+1)*log(exp(2*lv)) ...
%         - bv/exp(2*lv) ...
%         + log(2) + 2*lv;

% simplified:
lgp_sv = av*log(bv) - gammaln(av) - 2*av*lv - bv*exp(-2*lv) + log(2);

% s_u prior + Jacobian
%lgp_su = au*log(bu) - gammaln(au) ...
%         - (au+1)*lu ...
%         - bu*exp(-lu) ...
%         + lu;

% simplified:
lgp_su = au*log(bu) - gammaln(au) - au*lu - bu*exp(-lu);

lgp = lgp_b + lgp_sv + lgp_su;

end