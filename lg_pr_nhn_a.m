function lgp = lg_pr_nhn_a(theta)
% -------------------------------------------------------------
% Logged prior for Normal-Half-Normal SFA
%
% Parametrization A:
% theta = [beta ; log(sv) ; log(su)]
%
% Priors:
% beta  ~ N(0, 100 I)
% s2v   ~ IG(0.00005, 0.00005)
% s2u   ~ IG(5, 10*log(0.75)^2)
%
% where:
% sv = exp(theta(end-1))
% su = exp(theta(end))
% s2v = sv^2
% s2u = su^2
%
% Returns:
% lgp = log prior density evaluated at theta
% -------------------------------------------------------------

k = length(theta) - 2;

% Parameters
b  = theta(1:k);
lv = theta(end-1);   % log(sv)
lu = theta(end);     % log(su)

% Variances
s2v = exp(2*lv);
s2u = exp(2*lu);

% Hyperparameters
av = 0.00005;
bv = 0.00005;

au = 5;
bu = 10 * log(0.75)^2;   % approx 0.8276

%% ------------------------------------------------------------
% beta prior: N(0,100I)
%% ------------------------------------------------------------
lgp_b = -0.5*k*log(2*pi*100) -0.5*(b' * b)/100;

%% ------------------------------------------------------------
% Prior on s2v ~ IG(av,bv)
% transformed through s2v = exp(2*lv)
% includes Jacobian log(2*s2v)
%% ------------------------------------------------------------
%lgp_sv = av*log(bv) ...
%         - gammaln(av) ...
%         - (av + 1)*log(s2v) ...
%         - bv/s2v ...
%         + log(2) + 2*lv;
lgp_sv = av*log(bv) - gammaln(av) - 2*av*lv - bv*exp(-2*lv) + log(2);

%% ------------------------------------------------------------
% Prior on s2u ~ IG(au,bu)
% transformed through s2u = exp(2*lu)
% includes Jacobian log(2*s2u)
%% ------------------------------------------------------------
%lgp_su = au*log(bu) ...
%         - gammaln(au) ...
%         - (au + 1)*log(s2u) ...
%         - bu/s2u ...
%         + log(2) + 2*lu;

lgp_su = au*log(bu) - gammaln(au) - 2*au*lu - bu*exp(-2*lu) + log(2);
%% ------------------------------------------------------------
% Total log prior
%% ------------------------------------------------------------
lgp = lgp_b + lgp_sv + lgp_su;

end
%{
%UNTITLED Summary of this function goes here

% Log prior for normal-half-normal SFA
% Parametrization A:
% theta = (beta, log(sv), log(su))

k = length(theta) - 2;

beta  = theta(1:k);
th_sv = theta(end-1);   % log(sv)
th_su = theta(end);     % log(su)

% =========================
% PRIOR HYPERPARAMETERS
% =========================

% beta ~ N(0, 100 I)
Vb = 100;

% sv^2 ~ InvGamma(a_sv, b_sv)
a_sv = 0.00005;
b_sv = 0.00005;

% su^2 ~ InvGamma(a_su, b_su)
a_su = 5;
b_su = 10 * (log(0.75))^2;

% =========================
% LOG PRIOR FOR beta
% =========================
lgp_b = -0.5*k*log(2*pi) -0.5*k*log(Vb) - (beta' * beta) / (2*Vb);

% =========================
% LOG PRIOR FOR sv
% sv^2 = exp(2*th_sv)
% Jacobian = 2*exp(2*th_sv)
% =========================
lgp_sv = a_sv*log(b_sv) - gammaln(a_sv) - 2*a_sv*th_sv - b_sv*exp(-2*th_sv) + log(2);

% =========================
% LOG PRIOR FOR su
% su^2 = exp(2*th_su)
% Jacobian = 2*exp(2*th_su)
% =========================
lgp_su = a_su*log(b_su) - gammaln(a_su) - 2*a_su*th_su - b_su*exp(-2*th_su) + log(2);

% =========================
% TOTAL LOG PRIOR
% =========================
lgp = lgp_b + lgp_sv + lgp_su;



%lgp_b  = log(mvnpdf(theta(1:end-2)',zeros(1,k),eye(k)*100)); %p to musi być wektor leżący!!
%lgp_sv = log(invgampdf(exp(2*theta(end-1)), 0.00005, 0.00005)*2*exp(2*theta(end-1)));
%lgp_su = log(invgampdf(exp(2*theta(end)), 5, 10*log(0.75)^2)*2*exp(2*theta(end))); %0.8277~=10*ln(0.75)^2
%lgp = lgp_b + lgp_sv + lgp_su;
end

%}