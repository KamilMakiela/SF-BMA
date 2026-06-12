function out = aggregate_mc_table(mc_res, fieldname)

    draws = numel(mc_res);
    Tcell = cell(draws,1);

    for b = 1:draws
        Tcell{b} = mc_res{b}.(fieldname);
        Tcell{b}.MC = repmat(b, height(Tcell{b}), 1);
    end

    T_all = vertcat(Tcell{:});

    % Scenario identifiers: fixed by design
    id_vars = {'k2','k','sv','su','lambda','SS'};
    id_vars = id_vars(ismember(id_vars, T_all.Properties.VariableNames));

    % Numeric variables only
    all_vars = T_all.Properties.VariableNames;
    is_num = varfun(@isnumeric, T_all, 'OutputFormat', 'uniform');
    numeric_vars = all_vars(is_num);

    % Exclude identifiers and MC
    metric_vars = setdiff(numeric_vars, [id_vars, {'MC'}], 'stable');

    % Group by scenario
    [G, scenario_tbl] = findgroups(T_all(:, id_vars));

    % Means and standard deviations
    T_mean = scenario_tbl;
    T_std  = scenario_tbl;

    for j = 1:numel(metric_vars)
        v = metric_vars{j};

        T_mean.(v) = splitapply(@(x) mean(x,'omitnan'), T_all.(v), G);
        T_std.(v)  = splitapply(@(x) std(x,'omitnan'),  T_all.(v), G);
    end

    % Overall means/stds
    T_overall_mean = table();
    T_overall_std  = table();

    for j = 1:numel(metric_vars)
        v = metric_vars{j};

        T_overall_mean.(v) = mean(T_all.(v),'omitnan');
        T_overall_std.(v)  = std(T_all.(v),'omitnan');
    end

    out = struct();
    out.all_draws     = T_all;
    out.mean_by_scen  = T_mean;
    out.std_by_scen   = T_std;
    out.overall_mean  = T_overall_mean;
    out.overall_std   = T_overall_std;

end