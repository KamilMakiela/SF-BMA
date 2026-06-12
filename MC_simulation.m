% MC simulation from Makieła (2026)
% I reset the random seed so that the results are replicable
rng(0);
n=500;
draws=50;
mc_res = cell(draws,1);

% switching off unnecessary warnings
warning('off','MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:singularMatrix');
warning('off','MATLAB:illConditionedMatrix');

if isempty(gcp('nocreate'))
    parpool;
end

% back on
pctRunOnAll warning('off','MATLAB:nearlySingularMatrix');
pctRunOnAll warning('off','MATLAB:singularMatrix');
pctRunOnAll warning('off','MATLAB:illConditionedMatrix');

t1 = tic;
t_start = datetime('now');
fprintf('Simulation started at: %s \n', string(t_start, 'yyyy-MM-dd HH:mm:ss'));
for i = 1:draws
    fprintf('Runing MC draw %.0f out of %.0f.\n', i, draws);
    %try
        mc_res{i,1} = simulation_design(n);
    %catch err
    %   fprintf('Simulation %.0f failed.\n', i, draws);
    %   disp(err.message);
    %   keyboard;
    %   mc_res(i,1) = NaN;
    %end
    toc(t1);
end 

warning('on', 'MATLAB:nearlySingularMatrix');
warning('on', 'MATLAB:singularMatrix');
warning('on','MATLAB:illConditionedMatrix');

%% caclulate stats

mc_tables = aggregate_mc_results(mc_res);
table_summary = build_MC_summary_table(mc_tables);
