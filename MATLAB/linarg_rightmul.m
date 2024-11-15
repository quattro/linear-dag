function y = linarg_rightmul(A, S, M, y)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

y = M * y;
y = (speye(size(A)) - A) \ y;
y = S * y;

end