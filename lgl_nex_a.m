function L = lgl_nex_a(theta, X, y)
n = numel(y);

theta(end-1:end) = max(min(theta(end-1:end), 20), -20);

s_v = exp(theta(end-1));
s_u = exp(theta(end));

e = y - X*theta(1:end-2);
a = -e./s_v - s_v./s_u;
%stability #1
%logPhi = log(0.5 * erfc(-a ./ sqrt(2)));

logPhi = zeros(size(a));
idx = a < -10;
logPhi(idx) = log(0.5) + log(erfcx(-a(idx)./sqrt(2))) - 0.5*a(idx).^2;
logPhi(~idx) = log(0.5 * erfc(-a(~idx)./sqrt(2)));


L = - n*log(s_u) + 0.5*n*(s_v^2)/(s_u^2) + sum(e)/s_u + sum(logPhi);

end

