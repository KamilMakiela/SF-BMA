function L = lgl_re_a(theta, X, y, n ,T)
% it is assumed that observations are stacked in "object->time" order
%numerical stability
theta(end-1:end) = max(min(theta(end-1:end), 20), -20);
s2v = exp(2*theta(end-1));
s2u = exp(2*theta(end));
sig2 = s2v + T*s2u;

%e = y - X*theta(1:(end-2));
%ee = reshape(e,T,n);

e = y - X*theta(1:end-2);
ee = reshape(e,T,n);

sse  = sum(ee.^2,1)';
esum = sum(ee,1)';

arg1 = -0.5*n*T*log(2*pi) -0.5*n*(T-1)*log(s2v) -0.5*n*log(sig2);

arg2 = -0.5/s2v * sum(sse - (s2u/sig2).*esum.^2);

L = arg1 + arg2;
end

%{

%e_sr = mean(ee,1)';

tt = ones(T,T);         % J_T from p. 42 
%A = eye(t) - tt.*(s2u/sigs2);
om_inv = (eye(T) - (s2u/sig2).*tt)./s2v; %p.38

%arg1 = n*log(2) - 0.5*n*t*log(2*pi) - 0.5*n*(t-1)*log(s2v) - 0.5*n*log(sigs2);
arg1  =          - 0.5*n*T*log(2*pi) - 0.5*n*(T-1)*log(s2v) - 0.5*n*log(sig2) ;
%arg1 = -0.5*n*t*log(2*pi) - 0.5*n*log(sigs2*s2v^(t-1));

%arg2 = - e'*kron(eye(n),A)*e./(2*s2v);
% pomyśleć czy się nie da szybciej bo go spowalnia to strasznie
%arg2= - 0.5 * e'*kron(eye(n),A)*e.;
%arg2 = - 0.5 * (e'*kron(eye(n),om_inv)*e);
arg2 = - 0.5 * sum(sum((om_inv * ee) .* ee));
%arg3 = sum(log(1-normcdf(e_sr.*(t*s_u/(s_v*sqrt(sigs2))))));

L = arg1 + arg2;
%}

