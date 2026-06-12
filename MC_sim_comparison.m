%MC sim_comparison
%[fbs_es_0, fbs_es_av0] = compare_fbs_to_es(mc_res{1,1}.res0);   % classic BMA
%[fbs_es_1, fbs_es_av1] = compare_fbs_to_es(mc_res{1,1}.res1);   % SFA-BMA
draws = numel(mc_res);

% storage
FBS_ES_0    = cell(draws,1);
FBS_ES_AV0  = cell(draws,1);

FBS_ES_1    = cell(draws,1);
FBS_ES_AV1  = cell(draws,1);

% Loop over MC draws
for b = 1:draws
    %fprintf('Comparing FBS vs ES for draw %d out of %d\n', b, draws);
    % Classic BMA
    [FBS_ES_0{b}, FBS_ES_AV0{b}] = compare_fbs_to_es(mc_res{b}.res0);
    % SFA-BMA
    [FBS_ES_1{b}, FBS_ES_AV1{b}] = compare_fbs_to_es(mc_res{b}.res1);

end


% Stack scenario-level tables
T0_all = vertcat(FBS_ES_0{:});
T1_all = vertcat(FBS_ES_1{:});

% Scenario identifiers
id_vars = {'k2','lambda','SS'};

% Metric variables
all_vars = T0_all.Properties.VariableNames;
metric_vars = setdiff(all_vars, id_vars, 'stable');

% CLASSIC BMA
[G0, scen0] = findgroups(T0_all(:,id_vars));

% ---- Means
FBS_ES_SCEN_MEAN0 = scen0;

for j = 1:numel(metric_vars)

    v = metric_vars{j};

    FBS_ES_SCEN_MEAN0.(v) = ...
        splitapply(@(x) mean(x,'omitnan'), T0_all.(v), G0);

end

% ---- Stds
FBS_ES_SCEN_STD0 = scen0;

for j = 1:numel(metric_vars)

    v = metric_vars{j};

    FBS_ES_SCEN_STD0.(v) = ...
        splitapply(@(x) std(x,'omitnan'), T0_all.(v), G0);

end

% SFA-BMA
[G1, scen1] = findgroups(T1_all(:,id_vars));

% ---- Means
FBS_ES_SCEN_MEAN1 = scen1;

for j = 1:numel(metric_vars)

    v = metric_vars{j};

    FBS_ES_SCEN_MEAN1.(v) = ...
        splitapply(@(x) mean(x,'omitnan'), T1_all.(v), G1);

end

% ---- Std
FBS_ES_SCEN_STD1 = scen1;

for j = 1:numel(metric_vars)

    v = metric_vars{j};

    FBS_ES_SCEN_STD1.(v) = ...
        splitapply(@(x) std(x,'omitnan'), T1_all.(v), G1);

end

% Stack average tables
T0 = vertcat(FBS_ES_AV0{:});
T1 = vertcat(FBS_ES_AV1{:});

% MC averages
FBS_ES_AVG0 = varfun(@(x) mean(x,'omitnan'), T0);
FBS_ES_AVG1 = varfun(@(x) mean(x,'omitnan'), T1);

% Optional: cleaner names
FBS_ES_AVG0.Properties.VariableNames = T0.Properties.VariableNames;
FBS_ES_AVG1.Properties.VariableNames = T1.Properties.VariableNames;

% MC standard deviations
FBS_ES_STD0 = varfun(@(x) std(x,'omitnan'), T0);
FBS_ES_STD1 = varfun(@(x) std(x,'omitnan'), T1);
FBS_ES_STD0.Properties.VariableNames = T0.Properties.VariableNames;
FBS_ES_STD1.Properties.VariableNames = T1.Properties.VariableNames;
res0_avg = aggregate_mc_table(mc_res, 'res0');
res1_avg = aggregate_mc_table(mc_res, 'res1');
diff_avg = aggregate_mc_table(mc_res, 'es_d');


% Ranking comparison over MC draws
% Scenarios 19:27

MC_draws = numel(mc_res);
scen_grid = 19:27;
n_scen = numel(scen_grid);
RANK_CELL = cell(MC_draws * n_scen, 1);
idx = 1;

for b = 1:MC_draws

    fprintf('Processing MC draw %d out of %d\n', b, MC_draws);
    for a = scen_grid

        id_SFA   = mc_res{b}.sum1{a,6}(:,1);
        prob_SFA = mc_res{b}.sum1{a,6}(:,2);
        BIC_SFA  = mc_res{b}.sum1{a,6}(:,3);
        id_classic  = mc_res{b}.sum0{a,6}(:,1);
        BIC_classic = mc_res{b}.sum0{a,6}(:,3);

        ranking = compare_model_rankings( ...
            id_SFA, ...
            id_classic, ...
            prob_SFA, ...
            BIC_SFA, ...
            BIC_classic, ...
            1000);

        tmp = ranking.summary;
        tmp.MC = b;
        tmp.scen = a;
        RANK_CELL{idx} = tmp;
        idx = idx + 1;
    end
end

% Stack all MC-scenario rows
rank_summary = vertcat(RANK_CELL{:});

% Average over MC draws by scenario
id_vars = {'scen'};
all_vars = rank_summary.Properties.VariableNames;
metric_vars = setdiff(all_vars, [id_vars, {'MC'}], 'stable');
[G, scen_tbl] = findgroups(rank_summary(:, id_vars));
rank_mean = scen_tbl;
rank_std  = scen_tbl;

for j = 1:numel(metric_vars)
    v = metric_vars{j};
    rank_mean.(v) = splitapply(@(x) mean(x, 'omitnan'), rank_summary.(v), G);
    rank_std.(v) = splitapply(@(x) std(x, 'omitnan'), rank_summary.(v), G);
end