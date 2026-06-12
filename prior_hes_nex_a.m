function C = prior_hes_nex_a(theta)

k = size(theta,1);
C = zeros(k,k);

% beta ~ N(0,100I)
C(1:k-2,1:k-2) = -0.01 .* eye(k-2);

% s_v^2 ~ IG(0.00005, 0.00005), theta_v = log(s_v)
C(k-1,k-1) = -4 * 0.00005 * exp(-2 * theta(end-1));

% s_u ~ IG(1, 0.2877=-log(0.75)), theta_u = log(s_u)
C(k,k) = -0.2877 * exp(-theta(end));

end