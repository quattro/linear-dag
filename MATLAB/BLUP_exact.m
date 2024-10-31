function BLUP = BLUP_exact(X, y, SigmaG, SigmaE)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
ns = size(X,1);
GRM = X * SigmaG * X';
z = (GRM + SigmaE*speye(ns)) \ y;
BLUP = X * (SigmaG * (X' * z));

end