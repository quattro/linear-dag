function [A, samples, mutations] = read_linarg(filename)
%Read a linear ARG from a trio of files

A = mmread([filename, '.mtx']);
samples = readtable([filename, '.samples.txt']);
mutations = readtable([filename, '.mutations.txt']);

end