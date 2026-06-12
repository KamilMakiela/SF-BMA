function G = scores_nhnP_a(theta,X,y,n,T)
%---------------------------------------------------------------
% Score matrix for panel Normal-Half-Normal SFA (time-invariant u_i)
%
% Parametrization A:
% theta = [beta ; log(s_v) ; log(s_u)]
%
% Model:
% y_it = x_it'*beta + v_it - u_i
% v_it ~ N(0,s_v^2)
% u_i ~ |N(0,s_u^2)|
%
% Returns:
% G : n x (k+2)
%     row i = score contribution of panel unit i
%
% Then:
% negH_lik = G' * G;
%---------------------------------------------------------------

k = size(X,2);

b   = theta(1:k);
lv  = theta(end-1);
lu  = theta(end);

sv  = exp(lv);
su  = exp(lu);

s2v = sv^2;
s2u = su^2;

D = s2v + T*s2u;      % scalar

% residuals
e = y - X*b;

% stacked by unit: unit1 all t, then unit2 all t ...
ee  = reshape(e,T,n);

r   = sum(ee,1)';      % n x 1
sse = sum(ee.^2,1)';   % n x 1

a = s2u / D;

% quadratic piece
Q = sse - a .* r.^2;

% z for Phi term
z = - r .* su ./ (sv * sqrt(D));

% stable logPhi
logPhi = zeros(n,1);
idx = z < -10;

logPhi(idx)  = log(0.5) + log(erfcx(-z(idx)./sqrt(2))) - 0.5*z(idx).^2;
logPhi(~idx) = log(0.5 * erfc(-z(~idx)./sqrt(2)));

% lambda = phi(z)/Phi(z)
lambda = exp(-0.5*z.^2 - 0.5*log(2*pi) - logPhi);

% reshape X
XX = reshape(X,T,n,k);

Xsum = zeros(n,k);
EXsum = zeros(n,k);

for j = 1:k
    Xj = XX(:,:,j);
    Xsum(:,j)  = sum(Xj,1)';          % sum_t x_itj
    EXsum(:,j) = sum(ee .* Xj,1)';    % sum_t e_it x_itj
end

G = zeros(n,k+2);

% beta scores
% dQ/db = -2 EXsum + 2 a r Xsum
% dz/db = (su/(sv*sqrt(D))) Xsum
c = su / (sv * sqrt(D));

for j = 1:k
    G(:,j) = (EXsum(:,j) - a .* r .* Xsum(:,j)) ./ s2v + lambda .* c .* Xsum(:,j);
end

% score wrt log(s_v)

da_dlv = -2*s2u*s2v / D^2;
dQ_dlv = - da_dlv .* r.^2;
dz_dlv = z .* (-1 - s2v/D);

G(:,k+1) = -(T-1) - s2v/D -0.5 * dQ_dlv ./ s2v + Q ./ s2v + lambda .* dz_dlv;

% score wrt log(s_u)
da_dlu = 2*s2u*s2v / D^2;
dQ_dlu = - da_dlu .* r.^2;
dz_dlu = z .* (1 - T*s2u/D);

G(:,k+2) = - T*s2u/D -0.5 * dQ_dlu ./ s2v + lambda .* dz_dlu;

end