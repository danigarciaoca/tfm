% alf_epi_mse_G = []
alf_epi_mse_G = [0.2 epi_convergence_mean mse_v_convergence_exact mean(G(:,round(epi_convergence_mean))); alf_epi_mse_G]
save('alf_epi_mse_G_optGreedy0.2.mat','alf_epi_mse_G')