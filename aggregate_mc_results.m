function out = aggregate_mc_results(mc_res)

    draws = numel(mc_res);

    % Stack all scenario-level difference tables
    ES_D = cell(draws,1);

    for b = 1:draws
        ES_D{b} = mc_res{b}.es_d;
        ES_D{b}.MC = repmat(b, height(ES_D{b}), 1);
    end

    T_all = vertcat(ES_D{:});

    % Scenario identifiers
    id_vars = {'k2','k','sv','su','lambda','SS','c'};
    id_vars = id_vars(ismember(id_vars, T_all.Properties.VariableNames));

    % Variables to aggregate
    all_vars = T_all.Properties.VariableNames;
    metric_vars = setdiff(all_vars, [id_vars, {'MC'}]);

    % Group by scenario
    [G, scenario_tbl] = findgroups(T_all(:, id_vars));

    % MEAN
    T_mean = scenario_tbl;

    for j = 1:numel(metric_vars)
        v = metric_vars{j};
        T_mean.(v) = splitapply(@(x) mean(x,'omitnan'), T_all.(v), G);
    end

    % STD
    T_std = scenario_tbl;

    for j = 1:numel(metric_vars)
        v = metric_vars{j};
        T_std.(v) = splitapply(@(x) std(x,'omitnan'), T_all.(v), G);
    end

    % SCORE (replaces win)
    T_score = scenario_tbl;

    for j = 1:numel(metric_vars)
        v = metric_vars{j};
        T_score.(v) = splitapply(@(x) ...
            mean(x < 0,'omitnan') + 0.5 * mean(x == 0,'omitnan'), ...
            T_all.(v), G);
    end

    % OVERALL
    T_overall_mean  = table();
    T_overall_std   = table();
    T_overall_score = table();

    for j = 1:numel(metric_vars)
        v = metric_vars{j};

        x = T_all.(v);

        T_overall_mean.(v)  = mean(x,'omitnan');
        T_overall_std.(v)   = std(x,'omitnan');
        T_overall_score.(v) = ...
            mean(x < 0,'omitnan') + 0.5 * mean(x == 0,'omitnan');
    end

    % OUTPUT
    out = struct();
    out.all_draws      = T_all;
    out.mean_by_scen   = T_mean;
    out.std_by_scen    = T_std;
    out.score_by_scen  = T_score;
    out.overall_mean   = T_overall_mean;
    out.overall_std    = T_overall_std;
    out.overall_score  = T_overall_score;

end