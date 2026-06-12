function G = scores_nex_a(theta, X, y)

k = size(X,2);
%stability #1
theta(end-1:end) = max(min(theta(end-1:end), 20), -20);

beta = theta(1:k);
s_v  = exp(theta(k+1));
s_u  = exp(theta(k+2));

e = y - X*beta;
a = -e./s_v - s_v./s_u;

%delta = stable_delta(a);
logPhi = zeros(size(a));
idxPhi = a < -10;
logPhi(idxPhi) = log(0.5) + log(erfcx(-a(idxPhi)./sqrt(2))) - 0.5*a(idxPhi).^2;
logPhi(~idxPhi) = log(0.5 * erfc(-a(~idxPhi)./sqrt(2)));

logphi = -0.5*a.^2 - 0.5*log(2*pi);
logdelta = logphi - logPhi;
delta = zeros(size(a));
idx = a < -10;
delta(~idx) = exp(min(logdelta(~idx), log(1e14)));
aa = a(idx);
delta(idx) = -aa - 1./aa + 2./(aa.^3);
delta(~isfinite(delta)) = 1e14;
delta = min(delta, 1e14);

% beta
G_beta = X .* (-1/s_u + delta./s_v);

% helpful derivatives
da_dsv = e ./ s_v.^2 - 1 ./ s_u;
da_dsu = s_v ./ s_u.^2;

% log(s_v)
G_log_sv = s_v .* (s_v ./ s_u.^2 + delta .* da_dsv);

% log(s_u)
G_log_su = s_u .* (-1 ./ s_u - s_v.^2 ./ s_u.^3 - e ./ s_u.^2 + delta .* da_dsu);

G = [G_beta, G_log_sv, G_log_su];
end

%{
function delta = stable_delta(a)

logPhi = zeros(size(a));
idxPhi = a < -10;
logPhi(idxPhi) = log(0.5) + log(erfcx(-a(idxPhi)./sqrt(2))) - 0.5*a(idxPhi).^2;
logPhi(~idxPhi) = log(0.5 * erfc(-a(~idxPhi)./sqrt(2)));

logphi = -0.5*a.^2 - 0.5*log(2*pi);
logdelta = logphi - logPhi;
delta = zeros(size(a));
idx = a < -10;
delta(~idx) = exp(min(logdelta(~idx), log(1e14)));
aa = a(idx);
delta(idx) = -aa - 1./aa + 2./(aa.^3);
delta(~isfinite(delta)) = 1e14;
delta = min(delta, 1e14);

end
%}
