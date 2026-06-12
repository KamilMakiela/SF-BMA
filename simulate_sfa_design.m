function datasets = simulate_sfa_design(n, beta1, k2_grid, sv_grid, R_grid)
% SIMULATE_SFA_DESIGN
% One Monte Carlo replication: 27 datasets by default.
%
% Dimensions:
%   k1 = length(beta1) true regressors
%   k2 in k2_grid irrelevant regressors
%   sv in sv_grid
%   R  in R_grid
%
% DGP:
%   y = 1 + X*beta + v - u
%   v ~ N(0, sv^2)
%   u ~ Exp(mean = su)
%
% Output:
%   datasets(ik2,isv,iR)

arguments
	n
    beta1 = [0.5; 1; 1.5; 2]
    k2_grid = [4 8 10]
    %k2_grid = [4 5]
    sv_grid = [0.1 0.4 0.8]
    R_grid = [0.25 1 4]
end

beta1 = beta1(:);
su = 0.4;

k1 = length(beta1);
k2_max = max(k2_grid);
k_max = k1 + k2_max;

% One maximal X draw reused across all k2 cases

% ===== baseline case
X_all = randn(n, k_max);

% ===== correlated case
%rho = 0.4;
%Sigma = (1-rho)*eye(k_max) + rho*ones(k_max);
%X_all = randn(n,k_max) * chol(Sigma);


% Signal variance depends only on true regressors
VX = var(X_all(:,1:k1) * beta1, 1);

n_k2 = length(k2_grid);
n_sv = length(sv_grid);
n_R  = length(R_grid);

datasets(n_k2,n_sv,n_R) = struct( ...
    'X', [], 'y', [], 'beta', [], ...
    'k1', [], 'k2', [], 'k', [], ...
    'sv', [], 'su', [], 'R', [], 'c', []);

for ik2 = 1:n_k2
    
    k2 = k2_grid(ik2);
    k  = k1 + k2;
    
    X_base = X_all(:,1:k);
    beta   = [beta1; zeros(k2,1)];
    
    for isv = 1:n_sv
        
        sv = sv_grid(isv);
        Vvu = sv^2 + su^2;
        
        for iR = 1:n_R
            
            R = R_grid(iR);
            
            % Deterministic scaling
            c = sqrt(R * Vvu / VX);
            
            % Scale regressors only, not intercept
            X_scaled = c * X_base;
            
            % Add intercept for estimation
            X = [ones(n,1), X_scaled];
            
            % Full beta including intercept equall one!
            beta_full = [1; beta];
            
            % Draw disturbance components
            v = randn(n,1);
            % correcting for minor imperfections in finite sample
            %v = (v-mean(v))/std(v,1);
            v = sv * v;
            u = exprnd(su,n,1); 
            % correcting for minor imperfections in finite sample
            %u = u * (su / mean(u));
            
            % Generate y
            y = X * beta_full + v - u;
            
            % Store
            datasets(ik2,isv,iR).X    = X;
            datasets(ik2,isv,iR).y    = y;
            datasets(ik2,isv,iR).beta = beta_full;
            
            datasets(ik2,isv,iR).k1 = k1;
            datasets(ik2,isv,iR).k2 = k2;
            datasets(ik2,isv,iR).k  = k;
            
            datasets(ik2,isv,iR).sv = sv;
            datasets(ik2,isv,iR).su = su;
            datasets(ik2,isv,iR).R  = R;
            datasets(ik2,isv,iR).c  = c;
            
        end
    end
end
end