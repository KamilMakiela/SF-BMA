function Hlgp = prior_hes_re_a(theta)

k = length(theta)-2;

% Hyperparameters
bv = 0.00005;
bu = 0.00005;

s2v = exp(2*theta(end-1));
s2u = exp(2*theta(end));

Hlgp = zeros(k+2,k+2);

% beta prior: beta ~ N(0,100I)
Hlgp(1:k,1:k) = -(1/100)*eye(k);

% log(sv) prior part
Hlgp(k+1,k+1) = -4*bv/s2v;

% log(su) prior part
Hlgp(k+2,k+2) = -4*bu/s2u;

end