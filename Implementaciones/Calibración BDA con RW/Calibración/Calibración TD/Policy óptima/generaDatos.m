% clear all
% close all
% clc
% alf_epi_mse_G = [];
% alfas_prueba = linspace(0.03,1,21)
alf_epi_mse_G = [alf_epi_mse_G; 1.1 epi_convergence_mean mse_v_convergence_exact mean(G(:,round(epi_convergence_mean)))]
alfas_prueba(15:end)
save('alf_epi_mse_G_opt.mat','alf_epi_mse_G')