function indices = ProjectionToIndices(P)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[i,j] = find(P);
[~, order] = sort(i);
indices = j(order);
end