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