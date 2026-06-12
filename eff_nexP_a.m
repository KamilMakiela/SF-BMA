function [u_hat, TE_hat] = eff_nexP_a(theta, X, y, n, T)
% -------------------------------------------------------------
% Efficiency estimates for panel Normal-Exponential SFA
%
% Parametrization A:
% theta = [beta ; log(sv) ; log(su)]
%
% Model:
% y_it = x_it'*beta + v_it - u_i
%
% v_it ~ N(0, sv^2)
% u_i ~ Exp(scale = su)
%
% Priors typically used in your setup:
% s2v = sv^2 ~ IG(0.00005, 0.00005)
% su       ~ IG(1, 0.2877)   where 0.2877 = -log(0.75)
%
% Observations stacked:
% unit 1 over all T periods,
% then unit 2 over all T periods, etc.
%
% Outputs:
% u_hat  = E(u_i | e_i)      [n x 1]
% TE_hat = E(exp(-u_i)|e_i)  [n x 1]
% -------------------------------------------------------------

k = length(theta) - 2;

% Parameters
b  = theta(1:k);
sv = exp(theta(end-1));
su = exp(theta(end));

s2v = sv^2;

% Residuals
e = y - X*b;

% Reshape to T x n
ee = reshape(e, T, n);

% Sufficient statistic
r = sum(ee,1)';          % n x 1

% -------------------------------------------------------------
% Posterior:
% u_i | e_i  ~ Truncated Normal(mu_i, sig2), lower bound 0
%
% Kernel:
% exp( -0.5*u^2/sig2 + mu*u/sig2 ), u>=0
% -------------------------------------------------------------

sig2 = s2v / T;
sig  = sv / sqrt(T);

mu = -r ./ T - s2v ./ (T*su);

z = mu ./ sig;

% Stable inverse Mills ratio phi(z)/Phi(z)
logPhi = zeros(n,1);
idx = z < -10;

logPhi(idx)  = log(0.5) + log(erfcx(-z(idx)/sqrt(2))) - 0.5*z(idx).^2;
logPhi(~idx) = log(0.5 * erfc(-z(~idx)/sqrt(2)));

lambda = exp(-0.5*z.^2 - 0.5*log(2*pi) - logPhi);

% -------------------------------------------------------------
% Conditional mean inefficiency
% -------------------------------------------------------------

u_hat = mu + sig .* lambda;

% -------------------------------------------------------------
% Technical efficiency:
% E(exp(-u)|e)
% exact truncated-normal moment
% -------------------------------------------------------------

TE_hat = exp(-mu + 0.5*sig2) .* ...
         normcdf((mu - sig2)./sig) ./ ...
         normcdf(z);

end