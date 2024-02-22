function [report, A, flip, samples, mutations, X] = infer_linarg(in_file, out_file, ...
    flip_minor_alleles, make_triangular, rsq_threshold, nmax)
%Infer a linear ARG from a genotype matrix X
% Inputs: in_file: a .genos file
% out_file: three files will be saved: *.mtx, *.samples.txt, *.mutations.txt
% flip_minor_alleles: whether or not to flip alleles so that ref allele =
% major allele (default true)
% make_triangular: re-order rows/cols s.t. A is upper-triangular (default
% false)
% rsq_threshold: for imputation accuracy (default 0.8)
%
% Outputs: A: linear ARG of size n + m + recombinations
% flip: which alleles were flipped
% samples: indices of samples in A
% mutations: indices of the variants in A

if nargin < 3
    flip_minor_alleles = true;
end
if nargin < 4
    rsq_threshold = 0.8;
end
if nargin < 5
    make_triangular = false;
end
if nargin < 6
    nmax = 1e7;
end
geno_tol = 0;

tic;

T = readtable([in_file,'.snpinfo.csv']);
X = readmatrix([in_file,'.txt'],'OutputType','single',...
    'Range',sprintf('2:%d',nmax+1));
X = X(:,2:end);
m_init = size(X,2);
assert(~any(isnan(X),'all'))

% Flip so that 0 = minor allele
ploidy = max(X(:));
assert(ploidy==1 | ploidy==2)
af = mean(X)/ploidy;
if flip_minor_alleles
    flip = af>.5;
    X(:,flip) = ploidy - X(:,flip);
else
    flip = zeros(1,m_init);
end
T.flip = flip';
X(:,flip) = ploidy - X(:,flip);
X = sparse(double(X));

disp('Finished preprocessing data')

% Throw out poorly imputed SNPs
rsq_dosage = zeros(m_init,1);
for ii = 1:m_init
    if any(unique(X(:,ii)) ~= round(unique(X(:,ii))))
        rsq_dosage(ii) = corr(X(:,ii), round(X(:,ii)))^2;
    end
end
nnzX_init = nnz(X);
X = round(X(:, rsq_dosage >= rsq_threshold));
T = T(rsq_dosage >= rsq_threshold,:);

clear Xr
[n,m] = size(X);
nnzX = nnz(X);
T.index = (n+1:n+m)';

% Haplotype matrix is defined s.t. haplotype i has minor allele j if every
% carrier of i is a carrier of j, and every homozygous carrier of i is a
% homozygous carrier of j.
Xhaplo = true;
for min_alleles = 1:double(ploidy)
    X_carrier = sparse(X >= min_alleles);
    R_carrier = X_carrier' * X_carrier;
    Xhaplo = Xhaplo & (R_carrier >= diag(R_carrier));
end
clear X_carrier R_carrier

% Break ties (i.e., variants in perfect LD) arbitrarily
ties = Xhaplo & Xhaplo';
Xhaplo = Xhaplo - triu(ties,1);
Ahaplo = speye(m) - inv(Xhaplo);
clear ties Xhaplo

% Initial linarg
Asample = X * (speye(m) - Ahaplo);
if nargout < 6
    clear X
end
A = [sparse(m+n,n) [Asample; Ahaplo]];
nnzA_mut = nnz(A);
disp('Finished making initial linarg')

% Simplified linarg
A = find_recombinations(A);
nnzA = nnz(A);
ratio = nnzX/nnzA;
recombinations = length(A) - n - m;

samples = 1:n;
mutations = n+1:n+m;

if make_triangular
    G = digraph(A(n+1:end,n+1:end));
    order = toposort(G);
    invorder(order) = 1:length(order);
    mutations = invorder(1:m) + n;
    A = A([samples, order+n],[samples, order+n]);
end
T.index = mutations';

if nargin > 1
    if ~isempty(out_file)
        writetable(T,[out_file,'.mutations.txt']);
        writematrix(samples',[out_file,'.samples.txt'])
        mmwrite([out_file,'.mtx'], A);
    end
end

time = toc;

if nargout > 0
    in_file = {in_file};
    report = table(in_file,rsq_threshold,geno_tol,n,m,m_init,ratio,nnzX,...
        nnzX_init,nnzA,nnzA_mut,recombinations,time);
end

end