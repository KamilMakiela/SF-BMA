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