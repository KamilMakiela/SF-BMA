function model = sfa_fit_rep(X,y, n, T, sfa_opt, dec_crit, if_mdd)
% sfa_opt = 0 CNLRM
% sfa_opt = 1 SF normal-exponential
% sfa_opt = 2 SF normal-half-normal
% sfa_opt = 3 panel SF normal-exponential
% sfa_opt = 4 panel SF normal-half-normal
% sfa_opt = 5 panel RE
% dec_crit = 0 BIC based
% dec_crit = 1 mdd based (i.e., -mdd); mdd is integrated likelihood
% dec_crit = 2 AIC based (for whatever reason ;) )
% if_mdd = 0, mdd approximated with BIC 
% if_mdd = 1, exact mdd calculation

arguments
    X
    y
    n
    T=1
    sfa_opt=0
    dec_crit=0
    if_mdd=0
end
nT=n*T;
%starting params
b_ml = X\y; %equivalent of inv(X'X)X'y
s_ml = sqrt((y-X*b_ml)'*(y-X*b_ml) / nT);
% swithcing to log(sigma)
k = size(X,2)+2; %because by default I have two more params than columns in X
% keeping indices for scale parameters
sig_idx = k-1:k;
switch sfa_opt
    case 0 %cnlrm
        %this case is mostly analitical
        %in parametrization A from the start!!
        model.name = 'cnlrm';
        k = k-1;
        sig_idx = k;
        theta_l_max = [b_ml; log(s_ml)];
        theta_start = theta_l_max;
        theta_l_max_a = theta_l_max;
        [l_max, ~] = lgl_kmnrl(theta_l_max, X, y);
        Var_a = zeros(k,k);
        R = chol(X' * X);
        Var_a(1:end-1,1:end-1) = s_ml^2 * (R \ (R' \ eye(k-1)));
        Var_a(end,end) = 1/(2*nT);
        funopt_post = @(theta)(nlgMAP_kmnrl(theta, X, y));
        fun_MAP_a = @(theta)(lgMAP_kmnrl(theta, X, y));
        scores = @(theta)(scores_nonsf(theta,X,y));
        prior_hess = @(theta)(prior_hes_nonsf(theta));
    case 1 %sf-nex
        model.name = 'nex';
        %theta_start = [b_mnk; log(s_mnk/2); log(s_mnk/2)];
        theta_start = [b_ml; 2*log(s_ml); 0];
        funopt = @(theta)(nlgl_nex_b(theta, X, y));
        funopt_post = @(theta)(nlgMAP_nex_b(theta,X,y));
        fun_MAP_a = @(theta)(lgMAP_nex_a(theta,X,y));
        scores = @(theta)(scores_nex_a(theta,X,y));
        prior_hess = @(theta)(prior_hes_nex_a(theta));
    case 2 %sf-nhn
        model.name = 'nhn';
        theta_start = [b_ml; 2*log(s_ml); 0];
        funopt = @(theta)(nlgl_nhn_b(theta, X, y));
        funopt_post = @(theta)(nlgMAP_nhn_b(theta, X, y));
        fun_MAP_a = @(theta)(lgMAP_nhn_a(theta,X,y));
        scores = @(theta)(scores_nhn_a(theta,X,y));
        prior_hess = @(theta)(prior_hes_nhn_a(theta));
    case 3 %sf-nex with persitant inefficiency for panel data
        model.name = 'nexp';
        theta_start = [b_ml; 2*log(s_ml); 0];
        funopt = @(theta)(nlgl_nexP_b(theta, X, y, n, T));
        funopt_post = @(theta)(nlgMAP_nexP_b(theta,X,y,n,T));
        fun_MAP_a = @(theta)(lgMAP_nexP_a(theta,X,y,n,T));
        scores = @(theta)(scores_nexP_a(theta,X,y,n,T));
        prior_hess = @(theta)(prior_hes_nex_a(theta));
    case 4 %sfa-nhn with persistent ineff
        model.name = 'nhnp';
        theta_start = [b_ml; 2*log(s_ml); 0];
        funopt = @(theta)(nlgl_nhnP_b(theta, X, y, n, T));
        funopt_post = @(theta)(nlgMAP_nhnP_b(theta,X,y,n,T));
        fun_MAP_a = @(theta)(lgMAP_nhnP_a(theta,X,y,n,T));
        scores = @(theta)(scores_nhnP_a(theta,X,y,n,T));
        prior_hess = @(theta)(prior_hes_nhn_a(theta));
    case 5 %panel data RE model
        model.name = 'RE';
        theta_start = [b_ml;2*log(s_ml); 0];
        funopt = @(theta)(nlgl_re_b(theta, X, y, n, T));
        funopt_post = @(theta)(nlgMAP_re_b(theta, X, y, n, T));
        fun_MAP_a = @(theta)(lgMAP_re_a(theta,X,y,n,T));
        scores = @(theta)(scores_re_a(theta,X,y,n,T));
        prior_hess = @(theta)(prior_hes_re_a(theta));
end

options = optimoptions('fminunc', ...,
        'Algorithm', 'quasi-newton', ...
        'SpecifyObjectiveGradient', true, ...
        'MaxFunctionEvaluations', 5000, ...
        'MaxIterations', 5000, ...
        'OptimalityTolerance', 1e-12, ...
        'StepTolerance', 1e-12, ...
        'FunctionTolerance', 1e-12, ...
        'Display', 'off');
    
if sfa_opt ~= 0 %otherwise cnlrm already computed in the case statment
    [theta_l_max, l_max, ~,~,~,~] = fminunc(funopt,theta_start, options);
    l_max = -l_max;
    theta_l_max_a = theta_l_max;
    if sfa_opt ~= 0
        [theta_l_max_a(end-1), theta_l_max_a(end)] = sv_su_from_sigma_gamma(theta_l_max_a(end-1),theta_l_max_a(end));
    end
    G_lik = scores(theta_l_max_a);
    hes = G_lik'*G_lik;
    hes = (hes + hes')/2;
    [V,D] = eig(hes);
    d = diag(D);
    %numerical stability check
    d(d < 1e-10) = 1e-10;
    hes = V * diag(d) * V';
    Var_a = inv(hes);
end

% MODEL STATS
% stats based on ML
model.bic = k*log(nT) - 2*l_max;
model.aic = 2*k - 2*l_max;
model.l_max = l_max;

% stats based on integrated likelihood (mdd)
if if_mdd == 1
    %disp(['Initial logposterior: ', num2str(-funopt_post(theta_l_max),10)]);
    theta_l_max_hlp = theta_l_max;
    theta_l_max_hlp(end) = min(max(theta_l_max(end),-2.197224577), 2.197224577);
    try
    	[theta_p_max, ~, ~,~,~,~] = fminunc(funopt_post,theta_l_max_hlp, options);
    catch
        disp('Starting point from ML not good for MAP. Using theta_start');
        [theta_p_max, ~, ~,~,~,~] = fminunc(funopt_post,theta_start, options);
    end
    %Going back to parametrization in A to avoid skewness problem in MAP
    theta_p_a = theta_p_max;
    if sfa_opt ~= 0
        [theta_p_a(end-1), theta_p_a(end)] = sv_su_from_sigma_gamma(theta_p_max(end-1),theta_p_max(end));
    end
    p_max_a = fun_MAP_a(theta_p_a);
    G_a = scores(theta_p_a);
    H_pr_a = prior_hess(theta_p_a);
    hes_a = G_a'*G_a - H_pr_a;
    hes_a = (hes_a + hes_a')/2;
    [V,D] = eig(hes_a);
    d = diag(D);
    d(d < 1e-10) = 1e-10;
    hes_a = V * diag(d) * V';
	logdetHa = 2*sum(log(diag(chol(hes_a))));
	model.mdd = p_max_a + 0.5*k*log(2*pi) - 0.5*logdetHa;
    model.p_max = p_max_a;
else
    model.p_max = l_max;
    model.mdd = -0.5*model.bic;
end

switch dec_crit
    case 0
        model.inf_cr = model.bic;
        model.mdd = -0.5*model.bic;
    case 1
        model.inf_cr = -model.mdd;
    case 2
        model.inf_cr = model.aic;
        model.mdd = -0.5*model.bic;
end

% more stats
model.theta_Var_ml = Var_a;
%theta
model.theta_ml = theta_l_max_a;
model.theta_ml_se = sqrt(diag(Var_a));
model.theta_b = theta_l_max;
%params
model.params_ml = theta_l_max_a;
model.params_ml_se = model.theta_ml_se;
model.params_ml(sig_idx) = exp(theta_l_max_a(sig_idx));
model.params_ml_se(sig_idx) = model.params_ml(sig_idx) .* model.theta_ml_se(sig_idx);
model.T = T;
model.n = n;

if if_mdd == 1
    %posterior covariance - quick inversion scheeme
    R = chol(hes_a);
    model.bayes.theta_Var_post = R \ (R \ eye(k));
    %posterior theta
    model.bayes.theta_post = theta_p_a;
    model.bayes.theta_post_se = sqrt(diag(model.bayes.theta_Var_post));
    %posterior beta
    model.bayes.param_post = theta_p_a;
    model.bayes.param_post_se = model.bayes.theta_post_se;
    model.bayes.param_post(sig_idx) = exp(theta_p_a(sig_idx));
    hlp = model.bayes.theta_post_se(sig_idx);
    model.bayes.param_post_se(sig_idx) = model.bayes.param_post(sig_idx).*hlp;
end
end