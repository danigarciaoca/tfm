function [ diffV_norm, epi_convergence] = getConvergence( Vs_acumulada, umbral_convergencia )
%GETCONVERGENCE Summary of this function goes here
%   Detailed explanation goes here

diffV_norm = zeros([size(Vs_acumulada,3), size(Vs_acumulada,2)-1]);
epi_convergence = zeros([1 size(Vs_acumulada,3)]);
for it_ex=1:size(Vs_acumulada,3)
    Vs_acumulada(:,:,it_ex);
    Vs_acumulada_desp = [Vs_acumulada(:,2:end,it_ex) zeros(size(Vs_acumulada,1),1)];
    diffV = sum(sqrt((Vs_acumulada(:,:,it_ex)-Vs_acumulada_desp).^2),1); diffV = diffV(1:end-1);
    denom = sum(sqrt((Vs_acumulada(:,:,it_ex)).^2),1); denom = denom(1:end-1);
    diffV_norm(it_ex,:) = diffV./denom;
    a = find(diffV_norm(it_ex,:) <= umbral_convergencia); epi_convergence(it_ex) = a(1);
end
end

