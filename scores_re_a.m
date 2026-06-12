function G = scores_re_a(theta, X, y, n, t)
% Score matrix for random effects panel model
% Parametrization A:
% theta = [beta ; log(sv) ; log(su)]
%
% Output:
% G is n x (k+2)
% row i = score contribution of individual i

k = size(X,2);
beta = theta(1:k);
s2v = exp(2*theta(end-1));
s2u = exp(2*theta(end));
d = s2v + t*s2u;
e = y - X*beta;

% reshape by individual blocks
E = reshape(e,t,n);      % t x n

% reshape X into t x k x n
X3 = reshape(X,t,n,k);
X3 = permute(X3,[1 3 2]);   % t x k x n

% Useful terms
a = 1/s2v;
c = s2u/(s2v*d);

sum_e  = sum(E,1);       % 1 x n
sum_e2 = sum(E.^2,1);    % 1 x n

% Omega^{-1} e_i for all i
WE = a*E - c*ones(t,1)*sum_e;   % t x n

% Score wrt beta
% g_beta_i = X_i' Omega^{-1} e_i
G_beta = zeros(n,k);
for i = 1:n
    Xi = X3(:,:,i);       % t x k
    G_beta(i,:) = (Xi' * WE(:,i))';
end

% Score wrt log(sv) and log(su)
% First compute derivatives wrt log(s2v), log(s2u)
% because log(s2v)=2log(sv), log(s2u)=2log(su)

dlogdet_dlogA = (t-1) + s2v/d;
dlogdet_dlogB = t*s2u/d;

dq_dlogA_i = -a*sum_e2 + c*(1 + s2v/d).*(sum_e.^2);
dq_dlogB_i = -c*(1 - t*s2u/d).*(sum_e.^2);
score_logA_i = -0.5*dlogdet_dlogA -0.5*dq_dlogA_i;

score_logB_i = -0.5*dlogdet_dlogB -0.5*dq_dlogB_i;

% Chain rule:
% log(s2v) = 2 log(sv)
% log(s2u) = 2 log(su)

score_logsv_i = 2*score_logA_i;
score_logsu_i = 2*score_logB_i;

G = [G_beta, score_logsv_i', score_logsu_i'];

end