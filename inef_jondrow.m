function [u_hat, TE_hat] = inef_jondrow(model, X, y)
% function that calculates (in)efficiency scores based on jondrow et al.

switch model.name
    case 'nex'
        [u_hat, TE_hat] = eff_nex_a(model.params_ml, X, y);
    case 'nhn'
        [u_hat, TE_hat] = eff_nhn_a(model.params_ml, X, y);
    case 'nexp'
        [u_hat, TE_hat] = eff_nexP_a(model.params_ml, X, y, model.n, model.T);
	case 'nhnp'
        [u_hat, TE_hat] = eff_nhnP_a(model.params_ml, X, y, model.n, model.T);
    otherwise
        u_hat = 0;
        TE_hat =1;
end
end

function [u_hat, TE_hat] = eff_nex_a(theta, X, y)
k = size(X,2);
beta = theta(1:k);
s_v  = exp(theta(k+1));
s_u  = exp(theta(k+2));

e = y - X*beta;

a = -e./s_v - s_v./s_u;

logPhi = log(0.5 * erfc(-a./sqrt(2)));
logphi = -0.5*a.^2 - 0.5*log(2*pi);

delta = exp(logphi - logPhi);   % phi(a)/Phi(a)

u_hat = -e - (s_v^2 ./ s_u) + s_v .* delta;

TE_hat = exp(-u_hat);

end

function [u_hat, TE_hat] = eff_nhn_a(theta, X, y)

k = size(X,2);

beta = theta(1:k);
s_v  = exp(theta(k+1));
s_u  = exp(theta(k+2));

e = y - X*beta;

sigma2 = s_u^2 + s_v^2;

mu_star  = -e .* (s_u^2 / sigma2);
sig_star = (s_u * s_v) / sqrt(sigma2);

z = mu_star ./ sig_star;

logPhi = log(0.5 * erfc(-z./sqrt(2)));
logphi = -0.5*z.^2 - 0.5*log(2*pi);

delta = exp(logphi - logPhi);

u_hat = mu_star + sig_star .* delta;

% numerical guard
u_hat = max(u_hat,0);

TE_hat = exp(-u_hat);

end

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

function [u_hat, TE_hat] = eff_nhnP_a(theta, X, y, n, T)
% -------------------------------------------------------------
% Efficiency estimates for panel Normal-Half-Normal SFA
%
% Parametrization A:
% theta = [beta ; log(sv) ; log(su)]
%
% Model:
% y_it = x_it'*beta + v_it - u_i
%
% v_it ~ N(0, sv^2)
% u_i ~ |N(0, su^2)|
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
s2u = su^2;

% Residuals
e = y - X*b;

% Reshape to T x n
ee = reshape(e, T, n);

% Sum across time
r = sum(ee,1)';      % n x 1

% -------------------------------------------------------------
% Posterior moments:
% u_i | e_i ~ Truncated Normal(mu_i, sigma_i^2), u_i >= 0
% -------------------------------------------------------------

sig2 = (1/s2u + T/s2v)^(-1);
sig  = sqrt(sig2);

mu = - sig2 .* r ./ s2v;

z = mu ./ sig;

% Stable inverse Mills ratio: phi(z)/Phi(z)
logPhi = zeros(n,1);

idx = z < -10;

logPhi(idx)  = log(0.5) + log(erfcx(-z(idx)/sqrt(2))) - 0.5*z(idx).^2;
logPhi(~idx) = log(0.5 * erfc(-z(~idx)/sqrt(2)));

lambda = exp(-0.5*z.^2 - 0.5*log(2*pi) - logPhi);

% -------------------------------------------------------------
% Jondrow estimator
% -------------------------------------------------------------

u_hat = mu + sig .* lambda;

% -------------------------------------------------------------
% Technical efficiency:
% E(exp(-u)|e)
% exact truncated-normal formula
% -------------------------------------------------------------

TE_hat = exp(-mu + 0.5*sig2) .* ...
         normcdf((mu - sig2)./sig) ./ ...
         normcdf(z);

end
