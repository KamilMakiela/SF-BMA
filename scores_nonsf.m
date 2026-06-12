function G = scores_nonsf(theta, X, y)

theta(end) = max(min(theta(end), 40), -40);


sigma2 = exp(2*theta(end));
sigma2 = max(sigma2, 1e-12);
beta   = theta(1:end-1);

e = y - X*beta;

% score wrt beta: n x k
G_beta = X .* (e ./ sigma2);

% score wrt log(sigma): n x 1
G_lsig = -1 + (e.^2) ./ sigma2;

G = [G_beta, G_lsig];

end