function [log_sv, log_su] = sv_su_from_sigma_gamma(log_sigma2, eta)

sigma2 = exp(log_sigma2);

gamma = 1 ./ (1 + exp(-eta));
gamma = min(max(gamma, 1e-12), 1 - 1e-12);

log_sv = 0.5 * (log(sigma2) + log(1 - gamma));
log_su = 0.5 * (log(sigma2) + log(gamma));


end