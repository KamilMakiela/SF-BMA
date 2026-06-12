function output = simulation_design()

%Simulation script

%rng(0);
n=500;

datasets = simulate_sfa_design(n); % generate once

[Res0, sum0] = run_one_mc_summary2(datasets, 0);  % classic BMA
[Res1, sum1] = run_one_mc_summary2(datasets, 1);  % SFA-BMA

%% Averages per simulation draw
es_d_avg = table;
es_d_avg.MS_loss_TPR = -nanmean(Res1.es_tpr - Res0.es_tpr);
es_d_avg.MS_loss_FPR = nanmean(Res1.es_fpr - Res0.es_fpr);
es_d_avg.MS_InCorrect = -nanmean(Res1.es_correct - Res0.es_correct);

es_d_avg.MA_beta_rmse = nanmean(Res1.es_beta_rmse - Res0.es_beta_rmse);
es_d_avg.MA_beta_mae = nanmean(Res1.es_beta_mae - Res0.es_beta_mae);
es_d_avg.MA_beta_bias = nanmean(abs(Res1.es_beta_bias) - abs(Res0.es_beta_bias));
es_d_avg.MA_PIP_brier = nanmean(Res1.es_pip_brier - Res0.es_pip_brier);
es_d_avg.MA_PIP_loss_true = -nanmean(Res1.es_pip_true - Res0.es_pip_true);
es_d_avg.MA_PIP_gain_noise = nanmean(Res1.es_pip_noise - Res0.es_pip_noise);
es_d_avg.MA_PIP_sign_InAcc = -nanmean(Res1.es_sign_acc - Res0.es_sign_acc);

%% Table of differences:
es_d = table;

% identifiers (keep for interpretation)
es_d.k2     = Res0.k2;
es_d.lambda = Res0.lambda;
es_d.SS     = Res0.SS;

% ===== Model selection =====
%d_es.k       = Res1.es_k       - Res0.es_k;
%d_es.loss_tp      = Res1.es_tp      - Res0.es_tp;
%d_es.fp      = Res1.es_fp      - Res0.es_fp;

es_d.loss_tpr     = -(Res1.es_tpr     - Res0.es_tpr);
es_d.incr_fpr     = Res1.es_fpr     - Res0.es_fpr;
es_d.incorrect = -(Res1.es_correct - Res0.es_correct);

% ===== Model averaging =====
es_d.beta_rmse = Res1.es_beta_rmse - Res0.es_beta_rmse;
es_d.beta_mae  = Res1.es_beta_mae  - Res0.es_beta_mae;
es_d.beta_bias = abs(Res1.es_beta_bias) - abs(Res0.es_beta_bias);

es_d.pip_brier = Res1.es_pip_brier - Res0.es_pip_brier;
es_d.pip_loss_true  = -(Res1.es_pip_true  - Res0.es_pip_true);
es_d.pip_gain_noise = Res1.es_pip_noise - Res0.es_pip_noise;

es_d.sign_InAcc  = -(Res1.es_sign_acc  - Res0.es_sign_acc);

%%output
output.res0 = Res0;
output.res1 = Res1;
output.es_d = es_d;
output.es_d_avg = es_d_avg;
output.sum0 = sum0;
output.sum1 = sum1;
%output.datasets = datasets; 

end