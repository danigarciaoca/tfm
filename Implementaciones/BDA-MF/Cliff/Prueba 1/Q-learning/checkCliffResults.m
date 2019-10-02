function checkCliffResults(env, q_opt_format, Qsa_and_policy_opt)
grid_result = gridPolicyRepresentation(env, Qsa_and_policy_opt.pi_lineal);
disp('Grid final para Q-learning (1=izquierda, 2 = arriba, 3 = derecha, 4 = abajo)'), grid_result

disp('Valor función Q en el estado inicial (teórica y obtenida)')
[q_opt_format(37,:); Qsa_and_policy_opt.Qsa(37,:)]
end