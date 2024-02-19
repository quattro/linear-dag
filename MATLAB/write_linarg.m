function write_linarg(A, samples, mutations, filename)
%Write a linear ARG together with tables for samples + mutations to a
%file

assert(istriu(A), 'Input matrix should be upper-triangular')

writetable(samples,[filename, '.samples.txt']);
writetable(mutations,[filename, '.mutations.txt']);
mmwrite([filename, '.mtx'], A);


end