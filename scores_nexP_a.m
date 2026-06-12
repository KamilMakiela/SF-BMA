function G = scores_nexP_a(theta, X, y, n, t)

% Score matrix for panel normal-exponential SFA
% theta = [beta; log(s_v); log(s_u)]
% G is n x length(theta)

k = size(X,2);

b   = theta(1:k);
s_v = exp(theta(end-1));
s_u = exp(theta(end));
s2v = s_v^2;

e = y - X*b;

ee   = reshape(e,t,n);
e_sr = mean(ee,1)';        % n x 1
sse  = sum(ee.^2,1)';      % n x 1

q = sse ./ t;

c = s2v ./ (t*s_u);
B = e_sr + c;

d = q - B.^2;

a = -sqrt(t)*e_sr/s_v - s_v/(sqrt(t)*s_u);

% stable log Phi
logPhi = zeros(n,1);
idx = a < -10;

logPhi(idx)  = log(0.5) + log(erfcx(-a(idx)./sqrt(2))) - 0.5*a(idx).^2;
logPhi(~idx) = log(0.5 * erfc(-a(~idx)./sqrt(2)));

% inverse Mills ratio phi(a)/Phi(a)
lambda = exp(-0.5*a.^2 - 0.5*log(2*pi) - logPhi);

% reshape X according to your stacking:
% unit 1 over all t, then unit 2 over all t, etc.
XX = reshape(X,t,n,k);

Xbar  = zeros(n,k);
EXbar = zeros(n,k);

for j = 1:k
    Xj = XX(:,:,j);
    Xbar(:,j)  = mean(Xj,1)';
    EXbar(:,j) = mean(ee .* Xj,1)';
end

G = zeros(n,k+2);

% beta scores
for j = 1:k
    G(:,j) = (t/s2v) .* (EXbar(:,j) - B .* Xbar(:,j)) + (sqrt(t)/s_v) .* lambda .* Xbar(:,j);
end

% score w.r.t. log(s_v)
G(:,end-1) = -(t-1) + (t/s2v).*d + 2*B./s_u + lambda .* (sqrt(t)*e_sr/s_v - s_v/(sqrt(t)*s_u));

% score w.r.t. log(s_u)
G(:,end) = -1 - B./s_u + lambda .* (s_v/(sqrt(t)*s_u));

end