function H_prior = prior_hes_nhn_a(theta)
% Prior precision for NHN SFA
% Parametrization A: theta = (beta, log(sv), log(su))
%
% H_prior = - Hessian(log prior)

k = length(theta)-2;

% hyperparameters
b_sv = 0.00005;
b_su = 10 * (log(0.75))^2;

H_prior = zeros(k+2, k+2);

% beta ~ N(0, 100 I)
H_prior(1:k,1:k) = -eye(k) / 100;

% log(sv), induced by sv^2 ~ IG(0.00005, 0.00005)
H_prior(k+1,k+1) = -4 * b_sv * exp(-2 * theta(k+1));

% log(su), induced by su^2 ~ IG(5, 10*(log(0.75))^2)
H_prior(k+2,k+2) = -4 * b_su * exp(-2 * theta(k+2));

end