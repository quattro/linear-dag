function [A, flip, samples, mutations, Xhaplo_init, Ainit] = ...
    infer_linarg_local(X, rsq_threshold,...
    flip_minor_alleles, Xhaplo)
%Infer a linear ARG from a genotype matrix X
% Inputs: X: an n x m genotype matrix, haploid or diploid
% rsq_threshold: threshold for throwing out poorly-imputed SNPs
% flip_minor_alleles: whether or not to flip alleles so that ref allele =
% major allele
% Xhaplo: testing option
%
% Outputs: A: linear ARG of size n + m + recombinations
% flip: which alleles were flipped
% samples: indices of samples in A
% mutations: indices of the variants in A

if nargin < 2
    rsq_threshold = 0.8;
end
if nargin < 3
    flip_minor_alleles = true;
end

assert(~any(isnan(X),'all'))

% Flip so that 0 = minor allele
ploidy = max(X(:));
assert(ploidy==1 | ploidy==2)
af = mean(X)/ploidy;
if flip_minor_alleles
    flip = af>.5;
    X(:,flip) = ploidy - X(:,flip);
else
    flip = 0;
end
X = sparse(double(X));

% Throw out poorly imputed SNPs
m_init = size(X,2);
rsq_dosage = ones(m_init,1);
for ii = 1:m_init
    if any(unique(X(:,ii)) ~= round(unique(X(:,ii))))
        rsq_dosage(ii) = corr(X(:,ii), round(X(:,ii)))^2;
    end
end
nnzX_init = nnz(X);
X = round(X(:, rsq_dosage >= rsq_threshold));

clear Xr
[n,m] = size(X);
nnzX = nnz(X);

% Haplotype matrix is defined s.t. haplotype i has minor allele j if every
% carrier of i is a carrier of j, and every homozygous carrier of i is a
% homozygous carrier of j.
if ~exist('Xhaplo')
    Xhaplo = true;
    for min_alleles = 1:ploidy
        X_carrier = sparse(X >= min_alleles);
        R_carrier = X_carrier' * X_carrier;
        Xhaplo = Xhaplo & (R_carrier >= diag(R_carrier));
    end
    if nargout > 4
        Xhaplo_init = Xhaplo;
    end
    clear X_carrier R_carrier
end
% Break ties (i.e., variants in perfect LD) arbitrarily
ties = Xhaplo & Xhaplo';
Xhaplo = Xhaplo - triu(ties,1);
clear ties

% Initial linarg
Ahaplo = speye(m) - inv(Xhaplo);
Asample = X * (speye(m) - Ahaplo);
if nargout < 4
    clear X
end
Ainit = [sparse(m+n,n) [Asample; Ahaplo]];

% Simplified linarg
A = find_recombinations(Ainit);

samples = 1:n;
mutations = n+1:n+m;

end