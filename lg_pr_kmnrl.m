function [lgp, grad] = lg_pr_kmnrl(theta)

k = length(theta)-1;

beta = theta(1:end-1);
lsv  = theta(end);

a = 0.00005;
b = 0.00005;

% log prior for beta
lgp_b = log(mvnpdf(beta', zeros(1,k), eye(k)*100));

% log prior for s_v^2, with transformation s_v^2 = exp(2*lsv)
s2v = exp(2*lsv);
lgp_sv = log(invgampdf(s2v, a, b) * 2 * s2v);

lgp = lgp_b + lgp_sv;

% ===== Gradient =====

% beta prior gradient
g_beta = - beta ./ 100;

% log(sv) prior gradient
g_lsv = -2*a + 2*b*exp(-2*lsv);

grad = [g_beta; g_lsv];


%function lgp = lg_pr_kmnrl(theta)
%UNTITLED Summary of this function goes here

%k = length(theta)-1;
%lgp_b  = log(mvnpdf(theta(1:end-1)',zeros(1,k),eye(k)*100)); %p to musi być wektor leżący!!
%lgp_sv = log(invgampdf(exp(2*theta(end)), 0.00005, 0.00005)*2*exp(2*theta(end)));

%lgp = lgp_b + lgp_sv;
end