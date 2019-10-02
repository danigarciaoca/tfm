function phi_sa = GetStateActionFeatures(phi_s, a, NA)
%GETSTATEACTIONFEATURES Summary of this function goes here
%   Detailed explanation goes here

A = zeros(NA,1); A(a) = 1;
phi_sa = kron(A,phi_s);
end

