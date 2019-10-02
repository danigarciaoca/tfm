function [ D ] = GetLimStationStateVisitProb( P )
%GetLimStationStateVisitProb This function returns a diagonal matrix D with
%   diagonal elements the limiting stationary state visitation probability
%   distributions associated to the states of a Markov chain with state-transition
%   probability matrix P

[V,D] = eig(P'); % diagonal matrix D of eigenvalues and matrix V whose columns are the corresponding right eigenvectors
[~, ind_col] = find(D == 1); % eigenvector corresponding to an eigenvalue of 1. Sometimes the eigenvalues have a bit of noise (maybe is 1+1e-17 instead of 1)
if isempty(ind_col) % When eigenvalue is noisy, we round all of them to the fourth decimal and then aquire the column index corresponding to an eigenvalue of 1
    D = round(D*10000)/10000;
    [~, ind_col] = find(D == 1);
end
d_aux = V(:,ind_col);
d = abs(d_aux)/sum(abs(d_aux)); % normalization to get stationary visit probability
D = diag(d);
end

