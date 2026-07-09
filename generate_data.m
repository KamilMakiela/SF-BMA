%n = 100;
%T = 10;
nT = n*T;
reg = 15; %number of parameters beta, so p=reg-1
etykiety = "x" + string(0:reg-1); 
dane = [ones(nT,1), randn(nT,reg-1)];

exp_v  = array2table(dane, 'VariableNames', etykiety);

%beta = [1; 2; 3; 2; 1.5; 0.5; 1; 1.5];

beta = [1; 1.5; 2; 2.5; -3];
lista = ["x0","x1","x2","x6","x7"];

X = exp_v{:, lista}; %przechodzę z tabeli na tablicę
er = randn(nT,1);
er = 0.4.*(er - mean(er))/std(er,1);
gen = sfa_opt;

sig_idx = reg+1:reg+2;
% generuje y:
switch gen
    case 0
        y = X*beta + er;
        sig_idx = reg+1; %bo tylko jedna sigma
    case 1
        u = 0.4.*exprnd(1,nT,1); %implies s_u = mean(u) = 1
        y = X*beta + er - u;
    case 2
        u = kmdraw2(0,1,nT);
        y = X*beta + er - u;
    case 3
        u = ones(T,1) * exprnd(1,1,n); %implies s_u = mean(u) = 1
        y = X*beta + er - u(:);        
    case 4
        u = ones(T,1) * kmdraw2(1,1,n)';
        y = X*beta + er - u(:);
    case 5
        %for panel data RE model
        %inividual effects
        er2 = ones(T,1) * randn(1,n);
        y = X*beta + er + er2(:);
    otherwise
        disp('unknown option for data generation. Shutting down.')
        return;
end
dep_v = table(y);
