function [nL, ng] = nlgl_re_b(theta, X, y, n, T)
% Random-effects panel model with parametrization B
% theta = [beta ; log(sigma2) ; logit(gamma)]
% sigma2 = sv^2 + su^2
% gamma  = su^2 / sigma2
% so:
% sv^2 = (1-gamma)*sigma2
% su^2 = gamma*sigma2

k = size(X,2);

% Parameters
beta = theta(1:k);
logs2 = theta(k+1);

%stability #1
logs2 = max(min(logs2,30),-30);
sigma2 = exp(logs2);
lgam  = theta(k+2);
%stability #2
%gamma  = 1 ./ (1 + exp(-lgam));
if lgam >= 0
    gamma = 1 / (1 + exp(-lgam));
else
    elgam = exp(lgam);
    gamma = elgam / (1 + elgam);
end
gamma = min(max(gamma,1e-12),1-1e-12);

A = (1-gamma)*sigma2;   % sv^2
B = gamma*sigma2;       % su^2

% Residuals
e = y - X*beta;
E = reshape(e,T,n);     % T x N

% Useful scalars
d = A + T*B;
a = 1/A;
c = B/(A*d);
sum_e2 = sum(E.^2,1);   % 1 x N
sum_e  = sum(E,1);      % 1 x N
quad_i = a*sum_e2 - c*(sum_e.^2);
quad   = sum(quad_i);
logdetOmega = (T-1)*log(A) + log(d);

% neg Log-likelihood
nL = -(-0.5*n*T*log(2*pi) -0.5*n*logdetOmega -0.5*quad);

if nargout > 1
    % Gradient wrt beta
    WE = a*E - c*ones(T,1)*sum_e;
    score_beta = X' * WE(:);
    
    % Scores wrt log(A), log(B)
    % where A = sv^2, B = su^2
    
    dlogdet_dlogA = (T-1) + A/d;
    dlogdet_dlogB = T*B/d;
    
    dq_dlogA_i = -a*sum_e2 + c*(1 + A/d).*(sum_e.^2);
    
    dq_dlogB_i = -c*(1 - T*B/d).*(sum_e.^2);
    
    score_logA = -0.5*n*dlogdet_dlogA -0.5*sum(dq_dlogA_i);
    
    score_logB = -0.5*n*dlogdet_dlogB -0.5*sum(dq_dlogB_i);
    
    % Chain rule to parametrization B
    % log(A) = log(sigma2) + log(1-gamma)
    % log(B) = log(sigma2) + log(gamma)
    % d log(A) / d log(sigma2) = 1
    % d log(B) / d log(sigma2) = 1
    % d log(A) / d logit(gamma) = -gamma
    % d log(B) / d logit(gamma) = 1-gamma
    
    score_logs2 = score_logA + score_logB;
    score_lgam = -gamma*score_logA + (1-gamma)*score_logB;
    % FINAL GRADINT
    ng = -[score_beta; score_logs2; score_lgam];
end


end