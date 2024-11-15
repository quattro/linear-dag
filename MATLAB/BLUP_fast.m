function [BLUP, beta_BLUP, pcg_flag, pcg_num_iter] = BLUP_fast(A, y, S, M, SigmaG, SigmaE, pcg_tol)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

if nargin < 7
    pcg_tol = 1e-3;
end
pcg_max_iter = 100;

ns = size(S,1);
nn = size(A,1);

epsilon = 1e-9 * SigmaG;
sigmasq = epsilon * ones(nn,1);
sigmasq(any(S,1)) = diag(SigmaE);
sigmasq(any(M,2)) = diag(SigmaG);
Omega = diag(sparse(1./sigmasq));
% Sigma = diag(sparse(sigmasq));
IminusA = speye(nn) - A;


T = speye(nn);
T = T(~any(S,1), :); % complement of S

% For debugging
% B = IminusA' * Omega * IminusA;
% s = ProjectionToIndices(S);
t = ProjectionToIndices(T);
IminusA_t = IminusA(:,t);
IminusA_tt = IminusA(t,t);


z = IminusA' * (Omega * (IminusA * (S' * y)));
x = T * z; % B_ts * y

for i = 1:size(x,2)
    [x(:,i), pcg_flag(i), ~, pcg_num_iter(i)]  = pcg(@Mfun, x(:,i), ...
        pcg_tol, pcg_max_iter, IminusA_tt', Omega(t,t) * IminusA_tt);
end

% x = B(t,t) \ x;
x = S * IminusA' * (Omega * (IminusA_t * x)); % B_st * (B_tt \ (B_ts * y))

beta_BLUP = SigmaG * (M' * (IminusA' \ (S' * (S*z - x))));
BLUP = S * ( IminusA \ (M*beta_BLUP));
% beta_BLUP = M' * beta_BLUP;

    function Mx = Mfun(x)
        % Mx = B(t,t) * x;
        Mx = IminusA_t' * (Omega * (IminusA_t * x));
    end
end