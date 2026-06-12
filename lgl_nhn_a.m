function L = lgl_nhn_a(theta, X, y)

n = numel(y);

theta(end-1:end) = max(min(theta(end-1:end), 20), -20);

s_v = exp(theta(end-1));
s_u = exp(theta(end));

s   = sqrt(s_v^2 + s_u^2);
lbd = s_u / s_v;

e = y - X*theta(1:(end-2));
a = -e .* lbd ./ s;
%stability %1
logPhi = zeros(size(a));
idx = a < -10;
logPhi(idx) = log(0.5) + log(erfcx(-a(idx)./sqrt(2))) - 0.5*a(idx).^2;
logPhi(~idx) = log(0.5 * erfc(-a(~idx)./sqrt(2)));

L = -0.5*n*log(pi/2) - n*log(s) + sum(logPhi) - 0.5*(e'*e)/s^2;

end