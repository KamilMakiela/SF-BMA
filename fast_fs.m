function output = fast_fs(tabela, dep, n, T, sfa, init_var, dec_cr)

%init_var = 2; %number of fixed vars including the constant
% dec_cr = 0; % BIC based + ML
% dec_crit = 1 mdd + bayesian estimates 
% dec_crit = 2 aic + ML
% opt = 0; opt = 1 is mdd;
% sfa - rodzaj modelu
% init_var - ile zmiennych x ma być na siłę uwzględnione w modelu
%n
%T

if dec_cr == 1
    if_mdd = 1;
else
    if_mdd = 0;
end

% data prep
if istable(tabela)
    %kowersja do tablicy
    X = table2array(tabela);
    nazwy = tabela.Properties.VariableNames;
    %typowanie zmiennych ważnych; tu zakładam x0 (zawsze) oraz x5
else
    X = tabela;
end

if istable(dep)
    %kowersja do tablicy
    y = table2array(dep);
    %y_lab = dep.Properties.VariableNames;
    %typowanie zmiennych ważnych; tu zakładam x0 (zawsze) oraz x5
else
    y = dep;
end

%na początku są tylko zmienne które muszą być w modelu
X0 = X(:,1:init_var);
X_cand = X(:,init_var+1:end);
p = size(X_cand,2);

r = corr(X_cand, y);
[~, idx] = sort(abs(r), 'descend');
included = zeros(1,p);
remaining = idx;       % ordered candidate list
current_vars = [];

%initial model:
mdl = sfa_fit_rep(X0,y,n,T,sfa,dec_cr,if_mdd);
current_BIC = mdl.inf_cr;

cykli = 0;
%%
tic;
while ~isempty(remaining)

    best_BIC = Inf;
    best_var = -1;

    % Try each remaining variable (in correlation order)
    for t = 1:length(remaining)
        cykli = cykli+1;
        j = remaining(t);

        vars = [current_vars j];
        Xtemp = [X0 X_cand(:,vars)];

        mdl_curr = sfa_fit_rep(Xtemp,y,n,T,sfa,dec_cr,if_mdd);
        BIC = mdl_curr.inf_cr;

        if BIC < best_BIC
            best_BIC = BIC;
            best_var = j;
        end

    end

    % Check improvement
    if best_BIC < current_BIC
        current_BIC = best_BIC;
        current_vars = [current_vars best_var];
        included(best_var) = 1;

        % remove from remaining
        remaining(remaining == best_var) = [];
    else
        %this is were we break, so not a full FS!
        break
    end

end

X_best = [X0, X_cand(:,logical(included))];
mdl_best = sfa_fit_rep(X_best,y,n,T,sfa,dec_cr,if_mdd);

if istable(tabela)
    labels_best = nazwy(logical([ones(1,init_var) included]));
else
    labels_best = [ones(1,init_var) included];
end

%output- strukura z wynikami: 
%output.best_model - mdl
%output.included - zm. sztuczne z uwględnionymi zmiennymi
%output.labels_bets - etykiety tych zmiennych 
output.best_model = mdl_best;
output.included = [ones(1,init_var) included];
output.labels_best = labels_best;

end

