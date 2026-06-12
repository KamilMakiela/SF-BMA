function [lgp, g] = lg_pr_nhn_b(theta)
% Log prior + analytic gradient for normal-half-normal SFA
% Parametrization B:
% theta = (beta, log(sigma^2), logit(gamma))

k = length(theta) - 2;

beta        = theta(1:k);
log_sigma2  = theta(end-1);
eta         = theta(end);

% =========================
% HYPERPARAMETERS
% =========================
Vb   = 100;

a_sv = 0.00005;
b_sv = 0.00005;

a_su = 5;
b_su = 10 * (log(0.75))^2;

% =========================
% TRANSFORMED PARAMETERS
% =========================
sigma2 = exp(log_sigma2);
%numerical stability
%gamma  = 1 / (1 + exp(-eta));
if eta >= 0
    gamma = 1 / (1 + exp(-eta));
else
    eeta = exp(eta);
    gamma = eeta / (1 + eeta);
end

% optional safety
%gamma = min(max(gamma,1e-12),1-1e-12);

sv2 = (1-gamma) * sigma2;
su2 = gamma * sigma2;

% =========================
% LOG PRIOR FOR beta
% beta ~ N(0, 100 I)
% =========================
lgp_b = -0.5*k*log(2*pi) ...
        -0.5*k*log(Vb) ...
        - (beta' * beta) / (2*Vb);

% =========================
% LOG PRIOR FOR sv^2 ~ IG(a_sv, b_sv)
% =========================
lgp_sv2 = a_sv*log(b_sv) ...
          - gammaln(a_sv) ...
          - (a_sv + 1)*log(sv2) ...
          - b_sv / sv2;

% =========================
% LOG PRIOR FOR su^2 ~ IG(a_su, b_su)
% =========================
lgp_su2 = a_su*log(b_su) ...
          - gammaln(a_su) ...
          - (a_su + 1)*log(su2) ...
          - b_su / su2;

% =========================
% LOG JACOBIAN
% (log sigma^2, logit gamma) -> (sv^2, su^2)
% |J| = sigma2^2 * gamma * (1-gamma)
% =========================
lgJ = 2*log_sigma2 + log(gamma) + log(1-gamma);

% =========================
% TOTAL LOG PRIOR
% =========================
lgp = lgp_b + lgp_sv2 + lgp_su2 + lgJ;

% =========================
% GRADIENT
% =========================
g = zeros(size(theta));

% ---- wrt beta ----
g(1:k) = -beta / Vb;

% ---- wrt log_sigma2 ----
% Since d sv2 / d log_sigma2 = sv2 and d su2 / d log_sigma2 = su2:
% d/dlog_sigma2 [ -(a+1)log(x) - b/x ] = -(a+1) + b/x
d_lgp_sv2_dlogs2 = -(a_sv + 1) + b_sv / sv2;
d_lgp_su2_dlogs2 = -(a_su + 1) + b_su / su2;
d_lgJ_dlogs2     = 2;

g(end-1) = d_lgp_sv2_dlogs2 + d_lgp_su2_dlogs2 + d_lgJ_dlogs2;

% ---- wrt eta ----
% gamma' = gamma*(1-gamma)
%
% sv2 = (1-gamma)sigma2  => d sv2 / d eta = -gamma*sv2
% su2 = gamma sigma2     => d su2 / d eta = (1-gamma)*su2
%
% Therefore:
% d/deta [ -(a_sv+1)log(sv2) - b_sv/sv2 ] = gamma*(a_sv+1) - gamma*b_sv/sv2
% d/deta [ -(a_su+1)log(su2) - b_su/su2 ] = -(1-gamma)*(a_su+1) + (1-gamma)*b_su/su2
%
% Jacobian derivative:
% d/deta [log(gamma)+log(1-gamma)] = 1 - 2*gamma

d_lgp_sv2_deta = gamma * (a_sv + 1) - gamma * b_sv / sv2;
d_lgp_su2_deta = -(1-gamma) * (a_su + 1) + (1-gamma) * b_su / su2;
d_lgJ_deta     = 1 - 2*gamma;

g(end) = d_lgp_sv2_deta + d_lgp_su2_deta + d_lgJ_deta;

g = g(:);

end