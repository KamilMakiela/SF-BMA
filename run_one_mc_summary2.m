function [Res, summary] = run_one_mc_summary2(datasets, sfa_opt)

% sample size
n = length(datasets(1,1,1).y);

% Settings
Tpanel   = 1;
init_var = 1;   % ✅ intercept only
dec_cr   = 1;
k1 = 4;   % number of true regressors

% Dimensions
dims = size(datasets);
n_cases = prod(dims);

% Preallocate struct array
rows(n_cases,1) = struct();

row = 0;
summary = cell(n_cases,6);
for ik2 = 1:dims(1)
    for isv = 1:dims(2)
        for iR = 1:dims(3)
            row = row + 1;
            data = datasets(ik2,isv,iR);
            X = data.X;
            y = data.y;
            k  = data.k;
            k2 = data.k2;

            % True model (excluding intercept)
            true_model = false(1,k);
            true_model(1:k1) = true;

            % True beta (excluding intercept)
            beta_true = data.beta(2:(k+1))';

            % FBS search - not reaaly needed anymore
            tic;
            out_fbs = full_fbs_unique(X, y, n, Tpanel, sfa_opt, init_var, dec_cr);
            time_fbs = toc;
            fbs_model = out_fbs.best_model.included(2:(k+1));
            [fbs_tp, fbs_fp, fbs_tpr, fbs_fpr, fbs_correct] = eval_model(fbs_model, true_model);
            fbs_bma = eval_bma_summary(out_fbs, beta_true, true_model, k);

            % Exhaustive search
            tic;
            out_es = fast_ES(array2table(X), y, n, Tpanel, sfa_opt, init_var, dec_cr);
            time_es = toc;
            es_model = out_es.best_model.included(2:(k+1));
            [es_tp, es_fp, es_tpr, es_fpr, es_correct] = eval_model(es_model, true_model);
            es_bma = eval_bma_summary(out_es, beta_true, true_model, k);

            % Store scenario info
            summary{row,1} = k2;
            summary{row,2} = data.su / data.sv;
            summary{row,3} = data.R;
            summary{row,4} = out_es.summary;
            summary{row,5} = out_fbs.summary;
            summary{row,6} = [out_es.all_id out_es.all_model_prob out_es.all_bic];
            rows(row).k2 = k2;
            rows(row).k  = k;
            rows(row).sv     = data.sv;
            rows(row).su     = data.su;
            rows(row).lambda = data.su / data.sv;
            rows(row).SS     = data.R;
            rows(row).c      = data.c;

            % ES model selection
            rows(row).es_k       = sum(es_model);
            rows(row).es_tp      = es_tp;
            rows(row).es_fp      = es_fp;
            rows(row).es_tpr     = es_tpr;
            rows(row).es_fpr     = es_fpr;
            rows(row).es_correct = es_correct;
            rows(row).es_mdd     = out_es.best_model.mdd;
            rows(row).es_lmax    = out_es.best_model.l_max;
            rows(row).es_pmax    = out_es.best_model.p_max;
            rows(row).es_nmodels = numel(out_es.all_mdd);
            rows(row).es_time    = time_es;

            % ES BMA metrics
            rows(row).es_beta_rmse = es_bma.beta_rmse;
            rows(row).es_beta_mae  = es_bma.beta_mae;
            rows(row).es_beta_bias = es_bma.beta_bias;
            rows(row).es_pip_brier = es_bma.pip_brier;
            rows(row).es_pip_true  = es_bma.pip_true_mean;
            rows(row).es_pip_noise = es_bma.pip_noise_mean;
            rows(row).es_sign_acc  = es_bma.sign_acc_true;

            % FBS model selection - not really needed anymore
            rows(row).fbs_k       = sum(fbs_model);
            rows(row).fbs_tp      = fbs_tp;
            rows(row).fbs_fp      = fbs_fp;
            rows(row).fbs_tpr     = fbs_tpr;
            rows(row).fbs_fpr     = fbs_fpr;
            rows(row).fbs_correct = fbs_correct;
            rows(row).fbs_mdd     = out_fbs.best_model.mdd;
            rows(row).fbs_lmax    = out_fbs.best_model.l_max;
            rows(row).fbs_pmax    = out_fbs.best_model.p_max;
            rows(row).fbs_nmodels = numel(out_fbs.all_mdd);
            rows(row).fbs_time    = time_fbs;

            % FBS BMA metrics - not really needed anymore
            rows(row).fbs_beta_rmse = fbs_bma.beta_rmse;
            rows(row).fbs_beta_mae  = fbs_bma.beta_mae;
            rows(row).fbs_beta_bias = fbs_bma.beta_bias;
            rows(row).fbs_pip_brier = fbs_bma.pip_brier;
            rows(row).fbs_pip_true  = fbs_bma.pip_true_mean;
            rows(row).fbs_pip_noise = fbs_bma.pip_noise_mean;
            rows(row).fbs_sign_acc  = fbs_bma.sign_acc_true;
        end
    end
end

Res = struct2table(rows);

end

function [tp, fp, tpr, fpr, correct] = eval_model(model, true_model)

tp = sum(model & true_model);
fp = sum(model & ~true_model);
fn = sum(~model & true_model);
tn = sum(~model & ~true_model);
tpr = tp / (tp + fn);
fpr = fp / (fp + tn);
correct = isequal(model, true_model);

end

function M = eval_bma_summary(out, beta_true, true_model, k)

Tab = out.summary;
pip = Tab.PIP(2:(k+1))';
est = Tab.Est(2:(k+1))';
true_model = logical(true_model);
noise_model = ~true_model;
err = est - beta_true;

% Parameter accuracy
M.beta_rmse = sqrt(mean(err.^2));
M.beta_mae  = mean(abs(err));
M.beta_bias = mean(err);

% PIP quality
M.pip_brier = mean((pip - double(true_model)).^2);
M.pip_true_mean  = mean(pip(true_model));
M.pip_noise_mean = mean(pip(noise_model));

% Sign accuracy
M.sign_acc_true = mean(sign(est(true_model)) == sign(beta_true(true_model)));

end