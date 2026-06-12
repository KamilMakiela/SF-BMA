function H_prior = prior_hes_nonsf(theta)

k = length(theta)-1;
b = 0.00005;

H_prior = zeros(k+1,k+1);

% beta prior: beta ~ N(0,100 I)
H_prior(1:k,1:k) = -eye(k) / 100;

% log(sigma) prior induced by sigma^2 ~ IG(0.00005, 0.00005)
H_prior(k+1,k+1) = -4 * b * exp(-2*theta(end));

end