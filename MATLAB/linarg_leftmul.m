function y = linarg_leftmul(A, S, M, y)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

y = y * S;
y = y / (speye(size(A)) - A);
y = y * M;

end