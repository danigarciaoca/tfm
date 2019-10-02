function [ v_opt, q_opt, d_opt_norm ] = getOptimalPolicy( game )
%GETOPTIMALPOLICY Obtiene la optimal policy a partir de los valores de q
%�ptimos.

% Optimum values
v_opt = inv(eye(game.N_states)-game.gamma*game.pi_opt*game.P)*game.pi_opt*game.R;
q_opt = inv(eye(game.N_states*game.N_actions)-game.gamma*game.P*game.pi_opt)*game.R;
% Valor �ptimo de d (pol�tica en forma vector)
d_opt_norm = getPolicyVector(q_opt, game); % policy �ptima
end

