function L = lgl_nhnP_a(theta, X, y, n ,T)
% likelihood for normal-half-normal with persitent efficiency
% based on Pitt and Lee (1981, p. 60 and 61)
%numerical stability
theta(end-1:end) = max(min(theta(end-1:end), 20), -20);
s_v = exp(theta(end-1)); 
s_u = exp(theta(end)); 
s2v = s_v^2;
s2u = s_u^2;
sigs2 = s2v + T*s2u;

%e = y - X*theta(1:end-2);
%ee = reshape(e,T,n);
%e_sr = mean(ee,1)';

e = y - X*theta(1:end-2);
ee = reshape(e,T,n);
e_sr = mean(ee,1)';
sse  = sum(ee.^2,1)';
esum = sum(ee,1)';

%tt = ones(T,T);         % ll' from p. 60 
%A = eye(T) - tt.*(s2u/sigs2);

arg1 = n*log(2) - 0.5*n*T*log(2*pi) - 0.5*n*(T-1)*log(s2v) - 0.5*n*log(sigs2);
arg2 = -(0.5/s2v) * sum(sse - (s2u/sigs2).*esum.^2);

%arg2 = - e'*kron(eye(n),A)*e./(2*s2v);
%arg2 = - 0.5 * sum(sum((A * ee) .* ee))/s2v;
%arg2 = - 0.5 * sum(ee.*(A*ee), 'all')/s2v;
%arg3 = sum(log(normcdf(-e_sr.*(t*s_u/(s_v*sqrt(sigs2))))));
%stability 
a = -e_sr .* (T*s_u/(s_v*sqrt(sigs2)));
logPhi = zeros(size(a));
idx = a < -10;
logPhi(idx) = log(0.5) + log(erfcx(-a(idx)./sqrt(2))) - 0.5*a(idx).^2;
logPhi(~idx) = log(0.5 * erfc(-a(~idx)./sqrt(2)));

%logPhi = log(0.5 * erfc(-z ./ sqrt(2)));
%logPhi(~isfinite(logPhi)) = -1e12;
arg3 = sum(logPhi);

L = arg1 + arg2 + arg3;

end
