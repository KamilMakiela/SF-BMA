function G = scores_nhn_a(theta, X, y)
% Score matrix for normal-half-normal SFA
% Parametrization A: theta = (beta, log(sv), log(su))
%
% Output:
%   G : n x (k+2) matrix of individual score contributions

k = size(X,2);

beta = theta(1:k);
s_v  = exp(theta(k+1));
s_u  = exp(theta(k+2));

e = y - X*beta;

s2 = s_v^2 + s_u^2;
s  = sqrt(s2);
lambda = s_u / s_v;

a = -lambda .* e ./ s;

logphi = -0.5*a.^2 - 0.5*log(2*pi);
logPhi = log(0.5 * erfc(-a ./ sqrt(2)));
delta  = exp(logphi - logPhi);

% beta score
G_beta = X .* (e ./ s2 + delta .* (lambda ./ s));

% log(sv) score
G_log_sv = ...
    -(s_v^2 ./ s2) ...
    + (e.^2) .* (s_v^2 ./ s2.^2) ...
    + delta .* (lambda .* e ./ s + lambda .* e .* s_v^2 ./ s.^3);

% log(su) score
G_log_su = ...
    -(s_u^2 ./ s2) ...
    + (e.^2) .* (s_u^2 ./ s2.^2) ...
    + delta .* (-lambda .* e ./ s + lambda .* e .* s_u^2 ./ s.^3);

G = [G_beta, G_log_sv, G_log_su];

end