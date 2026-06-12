function [L, grad] = lgl_kmnrl(theta, X, y)

% type A parametrization
% theta = [beta ; log(sv)]

s2v = exp(2*theta(end));

n = numel(y);

beta = theta(1:end-1);

e = y - X*beta;

% log-likelihood
L = -0.5*n*log(2*pi) -0.5*n*log(s2v) - (e'*e)/(2*s2v);

% ===== Gradient =====

% wrt beta
g_beta = X'*e ./ s2v;

% wrt log(sv)
g_lsv = -n + (e'*e)./s2v;

grad = [g_beta; g_lsv];


%type A parametrization
%s2v = exp(2*theta(end));
%s2v = theta(end)^2;
%n = numel(y);
%e = y - X*theta(1:end-1);
%L = - 0.5*n*log(2*pi) - 0.5*n*log(s2v) - e'*e./(2*s2v);
end