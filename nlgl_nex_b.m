function [nL, ngrad] = nlgl_nex_b(theta, X, y)

k = size(X,2);

% PARAMETERS
beta         = theta(1:k);
log_sigma2   = theta(k+1);
%log_sigma2 = max(min(log_sigma2, 40), -40);
eta  = theta(k+2);

% transforms
sigma2 = exp(log_sigma2);
%numerical stability #1
%gamma  = 1 / (1 + exp(-eta));
if eta >= 0
    gamma = 1 / (1 + exp(-eta));
else
    eeta = exp(eta);
    gamma = eeta / (1 + eeta);
end
%gamma  = 1 / (1 + exp(-logit_gamma));
%numerical stability #2
gamma = min(max(gamma,1e-12),1-1e-12);

s_u = sqrt(gamma * sigma2);
s_v = sqrt((1 - gamma) * sigma2);

%numerical stability #3
s_v = max(s_v, 1e-10);
s_u = max(s_u, 1e-10);

% RESIDUALS
e = y - X*beta;

% CORE TERMS
a = -e./s_v - s_v./s_u;

logPhi = zeros(size(a));
idxPhi = a < -10;
logPhi(idxPhi) = log(0.5) + log(erfcx(-a(idxPhi)./sqrt(2))) - 0.5*a(idxPhi).^2;
logPhi(~idxPhi) = log(0.5 * erfc(-a(~idxPhi)./sqrt(2)));
logphi = -0.5*a.^2 - 0.5*log(2*pi);

% normal case
logdelta = logphi - logPhi;
% numerical stability #4
delta = zeros(size(a));
idx = a < -10;
delta(~idx) = exp(min(logdelta(~idx), log(1e12)));
% asymptotic approximation for very negative a
aa = a(idx);
delta(idx) = -aa - 1./aa + 2./(aa.^3);
% safety
delta(~isfinite(delta)) = 1e14;
delta = min(delta, 1e14);

% LOG-LIKELIHOOD
nL = -sum(-log(s_u) + 0.5*(s_v^2)/(s_u^2) + e./s_u + logPhi);

% GRADIENT
if nargout > 1
    % ---- beta ----
    g_beta = X' * ( -1/s_u + delta./s_v );
    
    % Derivatives wrt s_v and s_u (needed for chain rule)
    
    % d a / d s_v and d s_u
    da_dsv = (e ./ s_v.^2) - (1 / s_u);
    da_dsu = (s_v) / (s_u^2);
    
    % d/ds_v
    d_sv = (s_v / s_u^2) + delta .* da_dsv;
    
    % d/ds_u
    d_su = -1/s_u - e./(s_u^2) - (s_v^2)/(s_u^3) + delta .* da_dsu;
    
    % Chain rule to (log_sigma2, logit_gamma)
    % derivatives of s_v, s_u wrt sigma2 and gamma
    dsu_dsigma2 = gamma / (2*s_u);
    dsv_dsigma2 = (1-gamma) / (2*s_v);
    
    dsu_dgamma = sigma2 / (2*s_u);
    dsv_dgamma = -sigma2 / (2*s_v);
    
    % ---- log_sigma2 ----
    g_sigma2 = sum( d_su .* dsu_dsigma2 + d_sv .* dsv_dsigma2 );
    g_log_sigma2 = g_sigma2 * sigma2;
    
    % ---- gamma ----
    g_gamma = sum( d_su .* dsu_dgamma + d_sv .* dsv_dgamma );
    
    % logit transform
    g_logit_gamma = g_gamma * gamma * (1 - gamma);
    
    % GRADIENT
    ngrad = -[g_beta; g_log_sigma2; g_logit_gamma];
end
end