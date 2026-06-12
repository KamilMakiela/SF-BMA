function [nL, ngrad] = nlgl_nhn_b(theta, X, y)

k = size(X,2);

% PARAMETERS
beta        = theta(1:k);
log_sigma2  = theta(k+1);
eta = theta(k+2);

sigma2 = exp(log_sigma2);
% stability #1
%gamma  = 1 / (1 + exp(-eta));
if eta >= 0
    gamma = 1 / (1 + exp(-eta));
else
    eeta = exp(eta);
    gamma = eeta / (1 + eeta);
end
gamma = min(max(gamma,1e-12),1-1e-12);

s_u = sqrt(gamma * sigma2);
s_v = sqrt((1 - gamma) * sigma2);

sigma = sqrt(sigma2);
lambda = s_u / s_v;

% RESIDUALS
e = y - X*beta;
a = e ./ sigma;
w = -lambda .* a;

% LOG COMPONENTS
logphi_a = -0.5*a.^2 - 0.5*log(2*pi);

% stability #2
%logPhi_w = log(0.5 * erfc(-w ./ sqrt(2)));
logPhi_w = zeros(size(w));
idxPhi = w < -10;
logPhi_w(idxPhi) = log(0.5) + log(erfcx(-w(idxPhi)./sqrt(2))) - 0.5*w(idxPhi).^2;
logPhi_w(~idxPhi) = log(0.5 * erfc(-w(~idxPhi)./sqrt(2)));

% LOG-LIKELIHOOD
nL = -sum(-log(sigma) + log(2) + logphi_a + logPhi_w);

% GRADIENT
if nargout > 1
    % stability %3
    %phi_w = exp(-0.5*w.^2 - 0.5*log(2*pi));
    %delta = phi_w ./ exp(logPhi_w);
    logphi_w = -0.5*w.^2 - 0.5*log(2*pi);
    logdelta = logphi_w - logPhi_w;
    delta = zeros(size(w));
    idx = w < -10;
    delta(~idx) = exp(min(logdelta(~idx), log(1e12)));
    ww = w(idx);
    delta(idx) = -ww - 1./ww + 2./(ww.^3);
    delta(~isfinite(delta)) = 1e12;
    delta = min(delta, 1e14);
    % ---- beta ----
    g_beta = X' * ( a./sigma + delta .* (lambda./sigma) );

    % DERIVATIVES wrt sigma and lambda
    % d/d_sigma
    d_sigma = -1/sigma + a.^2/sigma + delta .* (lambda .* a / sigma);
    
    % d/d_lambda
    d_lambda = -delta .* a;
    
    % CHAIN RULE to (sigma, lambda)
    
    % sigma = sqrt(sigma^2)
    d_sigma2 = d_sigma * (1/(2*sigma));
    
    % lambda = sqrt(gamma/(1-gamma))
    d_lambda_dgamma = 1 ./ (2*lambda.*(1-gamma).^2);
    
    % sigma^2
    g_sigma2 = d_sigma2;
    
    % gamma
    g_gamma = d_lambda .* d_lambda_dgamma;
    
    % log transforms
    g_log_sigma2 = sum(g_sigma2) * sigma2;
    g_logit_gamma = sum(g_gamma) * gamma * (1-gamma);
    
    % FINAL GRADIENT
    ngrad = -[g_beta; g_log_sigma2; g_logit_gamma];
end
end