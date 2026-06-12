function [lgp, glgp] = lg_pr_re_b(theta)

k = length(theta)-2;

beta  = theta(1:end-2);
logs2 = theta(end-1);
eta  = theta(end);

sigma2 = exp(logs2);

gamma  = 1 ./ (1 + exp(-eta));

s2v = (1-gamma)*sigma2;
s2u = gamma*sigma2;

% Prior hyperparameters
av = 0.00005;
bv = 0.00005;

au = 0.5;
bu = 0.00005;

% -------------------------------------------------
% Log prior for beta: beta ~ N(0,100I)
% -------------------------------------------------
lgp_b = -0.5*k*log(2*pi) ...
        -0.5*k*log(100) ...
        -0.5*(beta'*beta)/100;

g_beta = -beta/100;

% -------------------------------------------------
% Log inverse-gamma priors
% IG density: p(x) = b^a/Gamma(a) * x^(-a-1) * exp(-b/x)
% -------------------------------------------------
lgp_s2v = av*log(bv) - gammaln(av) ...
          - (av+1)*log(s2v) - bv/s2v;

lgp_s2u = au*log(bu) - gammaln(au) ...
          - (au+1)*log(s2u) - bu/s2u;

% -------------------------------------------------
% Jacobian for B parametrization
% |d(s2v,s2u)/d(logsigma2,logitgamma)| = s2v*s2u
% -------------------------------------------------
log_jac = log(s2v) + log(s2u);

lgp = lgp_b + lgp_s2v + lgp_s2u + log_jac;

% =================================================
% Gradient
% =================================================

% Derivatives of log IG density wrt log(s2)
d_lgp_s2v_dlogs2v = -(av+1) + bv/s2v;
d_lgp_s2u_dlogs2u = -(au+1) + bu/s2u;

% Jacobian contributes +1 wrt log(s2v) and +1 wrt log(s2u)
qv = d_lgp_s2v_dlogs2v + 1;
qu = d_lgp_s2u_dlogs2u + 1;

% Chain rule:
%
% log(s2v) = log(sigma2) + log(1-gamma)
% log(s2u) = log(sigma2) + log(gamma)
%
% d log(s2v) / d logsigma2 = 1
% d log(s2u) / d logsigma2 = 1
%
% d log(s2v) / d logitgamma = -gamma
% d log(s2u) / d logitgamma = 1-gamma

g_logs2 = qv + qu;

g_lgam = -gamma*qv + (1-gamma)*qu;

glgp = [g_beta; g_logs2; g_lgam];

end