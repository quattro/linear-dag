function A = readMatrixMarket(filepath)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

A = readmatrix(filepath, 'FileType', 'text', "CommentStyle", "%");
noVertices = A(1,1:2);
A = sparse(A(2:end,1), A(2:end,2), A(2:end,3),noVertices(1),noVertices(2));


end