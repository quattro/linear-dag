A = readMatrixMarket('~/Dropbox/linearARG/data/one_summed_dag_011623_small.mtx');
nn = length(A);
A = A';

samples = readmatrix('~/Dropbox/linearARG/data/samples_011623_small.txt') + 1;
mutations = readmatrix('~/Dropbox/linearARG/data/mutations_011623_small.txt') + 1;

assert(all(sum(A(:,samples)) == 0), 'Sample haplotypes should be leaves')
ns = length(samples);
nm = length(mutations);
nn = length(A);

% Genotype matrix
Xa = inv(speye(nn) - A);
Xs = Xa(samples,mutations);

% Check that X is 0-1 valued
assert(numel(unique(nonzeros(Xs))) == 1)

% Check that samples are leaves
assert(all(sum(A(:,samples)==0)))

% Check that there are common variants
af = mean(Xs); maf = min(af,1-af);
assert(max(maf) > .4)

X = Xs(:, maf > 0);
mutations = mutations(maf > 0);
disp('Ratio of number of nonzeros:')
disp(nnz(Xs)/nnz(A))

% Re-compute linarg from the genotype matrix 
flip_minor_alleles = true;
[A1, flip, samples1, mutations1] = infer_linarg_local(X,1,flip_minor_alleles);
disp('Ratio of number of nonzeros:')
disp(nnz(X)/nnz(A1))

X1 = inv(speye(size(A1)) - A1);
% x -> flip * (1 + x) maps 0->1, 1->0 if flip==1
X1 = flip + X1(samples1,mutations1) .* ((-1).^flip);

% Check that inferred A agrees with Xs
assert(all(X1 == X, 'all'))




