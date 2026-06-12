function [nL, ngrad] = nlgl_nexP_b(theta, X, y, n, T)

% theta = [beta; log_sigma2; logit_gamma]

k = size(X,2);

b   = theta(1:k);
logs2 = theta(end-1);     % log(sigma^2)
eta = theta(end);       % logit(gamma)

S  = exp(logs2);
% stability #1
%gamma  = 1 / (1 + exp(-eta));
if eta >= 0
    gm = 1 / (1 + exp(-eta));
else
    eeta = exp(eta);
    gm = eeta / (1 + eeta);
end
gm = min(max(gm,1e-12),1-1e-12);
%gm = 1 ./ (1 + exp(-eta));

s2u = gm * S;
s2v = (1 - gm) * S;

s_u = sqrt(s2u);
s_v = sqrt(s2v);

e = y - X*b;

ee   = reshape(e,T,n);
e_sr = mean(ee,1)';
sse  = sum(ee.^2,1)';

% X as 3D: t x n x k
XX = reshape(X,T,n,k);
Xbar = squeeze(mean(XX,1));     % n x k

if k == 1
    Xbar = Xbar(:);
end

EXbar = zeros(n,k);
for j = 1:k
    Xj = XX(:,:,j);
    EXbar(:,j) = mean(ee .* Xj,1)';
end

c = s2v/(T*s_u);
B = e_sr + c;

%stable Phi
%logPhi = log(0.5 * erfc(-a ./ sqrt(2)));
a = -sqrt(T)*e_sr/s_v - s_v/(sqrt(T)*s_u);
logPhi = zeros(size(a));
idxPhi = a < -10;
logPhi(idxPhi) = log(0.5) + log(erfcx(-a(idxPhi)./sqrt(2))) - 0.5*a(idxPhi).^2;
logPhi(~idxPhi) = log(0.5 * erfc(-a(~idxPhi)./sqrt(2)));

L1 = - n*log(s_u) - 0.5*n*(T-1)*log(2*pi*s2v) - 0.5*n*log(T);
L2 = - (T/(2*s2v)) * sum(sse./T - B.^2);
L3 = sum(logPhi);

nL = -(L1 + L2 + L3);

if nargout > 1
    
    % inverse Mills ratio: phi(a)/Phi(a)
    % stable lambda
    %lambda = exp(-0.5*a.^2 - 0.5*log(2*pi) - logPhi);
    logphi = -0.5*a.^2 - 0.5*log(2*pi);
    logdelta = logphi - logPhi;
    delta = zeros(size(a));
    idx = a < -10;
    delta(~idx) = exp(min(logdelta(~idx), log(1e12)));
    aa = a(idx);
    delta(idx) = -aa - 1./aa + 2./(aa.^3);
    delta(~isfinite(delta)) = 1e12;
    delta = min(delta, 1e14);
    
    % beta gradient
    grad_b = sum((T/s2v) .* (EXbar - B .* Xbar) + (sqrt(T)/s_v) .* delta .* Xbar, 1)';
    
    % gradients w.r.t. s_v and s_u first
    d = sse./T - B.^2;
    dL_ds2v = sum(T .* d ./ (2*s2v^2) + B ./ (s2v*s_u) );
    
    dL_dsv = 2*s_v*dL_ds2v - n*(T-1)/s_v + sum(delta .* (sqrt(T)*e_sr/s_v^2 - 1/(sqrt(T)*s_u)));
    dL_dsu = - n/s_u - sum(B ./ s_u^2) + sum(delta .* (s_v/(sqrt(T)*s_u^2)));
    
    % chain rule to parametrization B
    % wrt log_sigma2
    dsv_dlogs2 = 0.5*s_v;
    dsu_dlogs2 = 0.5*s_u;
    
    % wrt eta = logit_gamma
    dsv_deta = -0.5*s_v*gm;
    dsu_deta =  0.5*s_u*(1-gm);
    
    grad_logs2 = dL_dsv*dsv_dlogs2 + dL_dsu*dsu_dlogs2;
    grad_eta   = dL_dsv*dsv_deta   + dL_dsu*dsu_deta;
    
    ngrad = -[grad_b; grad_logs2; grad_eta];
end

end