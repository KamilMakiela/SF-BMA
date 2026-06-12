function model = pre_screen(X,y,n)

b_ml = X\y; %equivalent of inv(X'X)X'y
s2_ml = (y-X*b_ml)'*(y-X*b_ml) / n;
k = numel(b_ml)+1; %heoretically more accurate
model.inf_cr = n*log(s2_ml) + k *log(n)+ n*(1+log(2*pi));