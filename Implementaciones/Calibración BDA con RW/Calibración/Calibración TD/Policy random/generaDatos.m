% alf_epi_mse_G = [];
alf_epi_mse_G = [alf_epi_mse_G; 0.286 epi_convergence_mean mse_v_convergence_exact mean(G(:,round(epi_convergence_mean)))]
save('alf_epi_mse_G_rand.mat','alf_epi_mse_G')