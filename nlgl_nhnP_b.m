function [L, grad] = nlgl_nhnP_b(theta, X, y, n, T)

% Panel normal-half-normal SFA
% Parametrization B:
% theta = [beta; log(sigma2); eta]
% sigma2 = s2u + s2v
% eta = logit(gamma)
% gamma = s2u / (s2u + s2v)

k = size(X,2);

b = theta(1:k);

logsigma2 = theta(end-1);
eta       = theta(end);

sigma2 = exp(logsigma2);

if eta >= 0
    gamma = 1 / (1 + exp(-eta));
else
    eeta = exp(eta);
    gamma = eeta / (1 + eeta);
end
gamma = min(max(gamma,1e-12),1-1e-12);
%gamma  = 1 ./ (1 + exp(-eta));

s2u = gamma * sigma2;
s2v = (1 - gamma) * sigma2;

s_u = sqrt(s2u);
s_v = sqrt(s2v);

D = s2v + T*s2u;

e = y - X*b;

ee  = reshape(e,T,n);
r   = sum(ee,1)';          % n x 1, sum_t e_it
sse = sum(ee.^2,1)';       % n x 1

a = s2u / D;

Q = sse - a * r.^2;

z = - r .* s_u ./ (s_v * sqrt(D));

% stable log Phi
logPhi = zeros(n,1);
idx = z < -10;

logPhi(idx)  = log(0.5) + log(erfcx(-z(idx)./sqrt(2))) - 0.5*z(idx).^2;
logPhi(~idx) = log(0.5 * erfc(-z(~idx)./sqrt(2)));

L1 = n*log(2) - 0.5*n*T*log(2*pi) - 0.5*n*(T-1)*log(s2v) - 0.5*n*log(D);
L2 = -0.5 * sum(Q ./ s2v);
L3 = sum(logPhi);

L = -(L1 + L2 + L3);


if nargout > 1

    % inverse Mills ratio phi(z)/Phi(z)
        % stable inverse Mills ratio phi(z)/Phi(z)
    logphi = -0.5*z.^2 - 0.5*log(2*pi);
    logdelta = logphi - logPhi;
    delta = zeros(size(z));
    idxdel = z < -10;
    delta(~idxdel) = exp(min(logdelta(~idxdel), log(1e12)));
    zz = z(idxdel);
    delta(idxdel) = -zz - 1./zz + 2./(zz.^3);
    delta(~isfinite(delta)) = 1e12;
    delta = min(delta, 1e14);
    
    XX = reshape(X,T,n,k);

    Xsum = zeros(n,k);
    EXsum = zeros(n,k);

    for j = 1:k
        Xj = XX(:,:,j);
        Xsum(:,j)  = sum(Xj,1)';
        EXsum(:,j) = sum(ee .* Xj,1)';
    end

    grad_b = zeros(k,1);

    c = s_u / (s_v * sqrt(D));

    for j = 1:k
        grad_b(j) = sum( ...
            (EXsum(:,j) - a*r.*Xsum(:,j)) ./ s2v ...
            + delta .* c .* Xsum(:,j) );
    end

    % derivatives w.r.t. s2v and s2u
    dz_ds2v = z .* (-0.5./s2v - 0.5./D);
    dz_ds2u = z .* ( 0.5./s2u - 0.5*T./D);

    dL_ds2v = -0.5*(T-1)./s2v -0.5./D -0.5 .* (s2u .* r.^2) ./ (D.^2 .* s2v) + 0.5 .* Q ./ (s2v.^2) + delta .* dz_ds2v;

    dL_ds2u = -0.5*T./D +0.5 .* r.^2 ./ (D.^2) + delta .* dz_ds2u;

    % chain rule to parametrization B
    ds2v_dlogsigma2 = s2v;
    ds2u_dlogsigma2 = s2u;

    ds2v_deta = -gamma * s2v;
    ds2u_deta = (1 - gamma) * s2u;
    grad_logsigma2 = sum(dL_ds2v .* ds2v_dlogsigma2 + dL_ds2u .* ds2u_dlogsigma2);
    grad_eta = sum(dL_ds2v .* ds2v_deta + dL_ds2u .* ds2u_deta);
    
    % FINAL GRADIENT 
    grad = -[grad_b; grad_logsigma2; grad_eta];

end
end