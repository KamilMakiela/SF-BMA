function L = lgl_nexP_a(theta, X, y, n ,t)
% based on van den Broack, Koop, Osiewalski and Steel (1994)
%for numerical stability
theta(end-1:end) = max(min(theta(end-1:end), 20), -20);
s_v = exp(theta(end-1));
s_u = exp(theta(end));
s2v = s_v^2;
e = y - X*theta(1:(end-2));
%sigs2 = s_v^2 + t*s_u^2;
%s2u = s_u^2;
%e_sr = y_sr - X_sr*b;
%tt = ones(t,t);         % ll' from p. 60 

%disp(size(e_sr));

ee = reshape(e,t,n);
e_sr = mean(ee,1)';
sse  = sum(ee.^2, 1)';
%sse = diag(ee'*ee); 

% suma kwadratow reszt dla danego obiektu
%disp(size(sse));
%A = eye(t) - tt.*(s2u/sigs2);

%arg1 = n*log(2) - 0.5*n*t*log(2*pi) - 0.5*n*(t-1)*log(s2v) - 0.5*n*log(sigs2);
%arg2 = - n*s_v^2/(2*s_u^2) - sum(- e - s_v^2/s_u)/s_u;
%arg3 = sum(log(normcdf(-e_sr.*(t*s_u/(s_v*sqrt(sigs2))))));

arg1 = - n*log(s_u) - 0.5*n*(t-1)*log(2*pi*s2v) - 0.5*n*log(t);
arg2 = - (t/(2*s2v))*sum(sse./t - (e_sr + s2v/(t*s_u)).^2);
%a = -sqrt(t)*e_sr/s_v - s_v/(sqrt(t)*s_u);
%for numerical stability
a = -sqrt(t)*e_sr/s_v - s_v/(sqrt(t)*s_u);
arg3 = zeros(size(a));
idx = a < -10;
arg3(idx) = log(0.5) + log(erfcx(-a(idx)./sqrt(2))) - 0.5*a(idx).^2;
arg3(~idx) = log(0.5 * erfc(-a(~idx)./sqrt(2)));

%arg3 = log(0.5 * erfc(-a ./ sqrt(2)));
%arg3(~isfinite(arg3)) = -1e12;
%arg3 = sum(log(normcdf(-e_sr.*(t^0.5/s_v) -  s_v*(t^-0.5)/s_u)));

%{
spradzenie dla arg2
ec = reshape(e,t,n);
ECC = zeros(n,1);
for i = 1:n
    ECC(i) = ec(:,i)'*A*ec(:,i);
end
arg22 = sum(ECC)/(2*s2v);
%}
L = arg1 + arg2 + sum(arg3);

%disp(arg1);
%disp(arg2);
%disp(arg3);
end

