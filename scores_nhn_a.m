function G = scores_nhn_a(theta, X, y)
% Score matrix for normal-half-normal SFA
% Parametrization A: theta = (beta, log(sv), log(su))
%
% Output:
%   G : n x (k+2) matrix of individual score contributions

k = size(X,2);

% Stability for log standard deviations
theta(end-1:end) = max(min(theta(end-1:end), 20), -20);

beta = theta(1:k);
s_v  = exp(theta(k+1));
s_u  = exp(theta(k+2));

e = y - X*beta;

s2 = s_v^2 + s_u^2;
s  = sqrt(s2);
lambda = s_u / s_v;

a = -lambda .* e ./ s;

% Stable log(phi(a))
logphi = -0.5*a.^2 - 0.5*log(2*pi);

% Stable log(Phi(a))
logPhi = zeros(size(a));

idx = a < -10;

logPhi(idx) = log(0.5) + log(erfcx(-a(idx)./sqrt(2))) - 0.5*a(idx).^2;

logPhi(~idx) = log(0.5 * erfc(-a(~idx)./sqrt(2)));

% Stable inverse Mills ratio: delta = phi(a) / Phi(a)
logdelta = logphi - logPhi;
delta = zeros(size(a));

% Normal calculation where safe
delta(~idx) = exp(min(logdelta(~idx), log(1e12)));

% Asymptotic approximation for very negative a
aa = a(idx);
delta(idx) = -aa - 1./aa + 2./(aa.^3);

% Safety cleanup
delta(~isfinite(delta)) = 1e12;
delta = min(delta, 1e14);

% beta score
G_beta = X .* (e ./ s2 + delta .* (lambda ./ s));

% log(sv) score
G_log_sv = -(s_v^2 ./ s2) + (e.^2) .* (s_v^2 ./ s2.^2)  + delta .* (lambda .* e ./ s + lambda .* e .* s_v^2 ./ s.^3);

% log(su) score
G_log_su = -(s_u^2 ./ s2) + (e.^2) .* (s_u^2 ./ s2.^2) + delta .* (-lambda .* e ./ s + lambda .* e .* s_u^2 ./ s.^3);

G = [G_beta, G_log_sv, G_log_su];

end