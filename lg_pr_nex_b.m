function [lgp, g] = lg_pr_nex_b(theta)
% Log prior and analytic gradient for parametrization B:
% theta = [beta ; log_sigma2 ; eta]
% eta = logit(gamma), gamma = expit(eta)
%
% Returns:
%   lgp : scalar log prior
%   g   : gradient of lgp w.r.t. theta

k = length(theta) - 2;

% =========================
% PARAMETERS
% =========================
beta        = theta(1:k);
log_sigma2  = theta(end-1);
eta         = theta(end);

% =========================
% HYPERPARAMETERS
% =========================
a_v = 0.00005;
b_v = 0.00005;

a_u = 1;
b_u = 0.2877;

% =========================
% TRANSFORMATIONS
% =========================
sigma2 = exp(log_sigma2);

% stable logistic
%numerical stability
%gamma  = 1 / (1 + exp(-eta));
if eta >= 0
    gamma = 1 / (1 + exp(-eta));
else
    eeta = exp(eta);
    gamma = eeta / (1 + eeta);
end

% Optional safety clamp:
% If you clamp gamma, the gradient is no longer the exact analytic 
% gradient at the clamp points. So only use this if truly needed.
% gamma = min(max(gamma,1e-12),1 -1e-12);

s_u = sqrt(gamma * sigma2);      % sigma_u
s2v = (1 - gamma) * sigma2;      % sigma_v^2

% =========================
% LOG PRIOR COMPONENTS
% =========================

% beta ~ N(0, 100 I)
% log pdf up to exact constant
lgp_b = -0.5 * k * log(2*pi) -0.5 * log(det(100 * eye(k))) -0.5 * (beta' * beta) / 100;

% sigma_v^2 ~ InvGamma(a_v, b_v)
lgp_sv = a_v*log(b_v) - gammaln(a_v) - (a_v + 1) * log(s2v) - b_v / s2v;

% sigma_u ~ InvGamma(a_u, b_u)
lgp_su = a_u*log(b_u) - gammaln(a_u) - (a_u + 1) * log(s_u) - b_u / s_u;

% Jacobian: (log sigma^2, gamma) -> (sigma_v^2, sigma_u)
%lgJ1 = log(sigma2) + log(gamma) + log(1 - gamma) - log(2);

% Jacobian: eta -> gamma
%lgJ2 = log(gamma) + log(1 - gamma);
logJ = log(0.5) + log(s2v) + log(s_u);

lgp = lgp_b + lgp_sv + lgp_su + logJ;
% total log prior
%lgp = lgp_b + lgp_sv + lgp_su + lgJ1 + lgJ2;

% =========================
% GRADIENT
% =========================
g = zeros(size(theta));

% --- d/d beta ---
% beta prior: N(0,100I)
g(1:k) = -beta / 100;

% --- d/d log_sigma2 ---
% From lgp_sv:
% d/d log_sigma2 = -(a_v+1) + b_v/s2v
d_lgp_sv_dlogs2 = -(a_v + 1) + b_v / s2v;

% From lgp_su:
% s_u = sqrt(gamma*sigma2), so d s_u / d log_sigma2 = 0.5*s_u
% therefore contribution = -(a_u+1)/2 + b_u/(2*s_u)
d_lgp_su_dlogs2 = -0.5*(a_u + 1) + 0.5*b_u / s_u;

% From Jacobian lgJ1 only: derivative wrt log_sigma2 is 1
d_lgJ_dlogs2 = 1.5;

g(end-1) = d_lgp_sv_dlogs2 + d_lgp_su_dlogs2 + d_lgJ_dlogs2;

% --- d/d eta ---
% gamma' = gamma*(1-gamma)

% From lgp_sv:
% s2v = (1-gamma)*sigma2
% d s2v / d eta = -sigma2*gamma*(1-gamma) = -gamma*s2v
% contribution:
d_lgp_sv_deta = gamma * (a_v + 1) - gamma * b_v / s2v;

% From lgp_su:
% s_u = sqrt(gamma*sigma2)
% d s_u / d eta = 0.5*s_u*(1-gamma)
d_lgp_su_deta = 0.5*(1 - gamma) * (-(a_u + 1) + b_u / s_u);

% From Jacobians:
% lgJ1 + lgJ2 = 2*log(gamma) + 2*log(1-gamma) + log(sigma2) - log(2)
% derivative wrt eta = 2*(1 - 2*gamma)
d_lgJ_deta = 0.5 - 1.5*gamma;

g(end) = d_lgp_sv_deta + d_lgp_su_deta + d_lgJ_deta;

% ensure column vector
g = g(:);

end