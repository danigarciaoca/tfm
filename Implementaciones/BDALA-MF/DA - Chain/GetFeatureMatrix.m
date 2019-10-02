function [phi] = GetFeatureMatrix( S, N_features, GetStateFeatures, env)
%GETFEATUREMATRIX Summary of this function goes here
%   Detailed explanation goes here

phi = zeros(S, N_features);
for s = 1:S
    phi(s,:) = GetStateFeatures(s, env);
end

end

