function [ v_opt, q_opt, d_norm ] = getRandomPolicy( game )
%GETRANDOMPOLICY Obtiene la política random que se generó previamente, así
%como las funciones valor asociadas.

load d_aleatoria.mat % policy random
d_norm = getPolicyVectorFromD(d, game);

% Get policy matrix
A = eye(game.N_states);
B = [1; 0];
mult1 = kron(A,B);
B = [0; 1];
mult2 = kron(A,B);
policy_by_action = reshape(d_norm', [game.N_actions,game.N_states])';
policy_matrix = diag(policy_by_action(:,1))*mult1' + diag(policy_by_action(:,2))*mult2';

% Optimum values
v_opt = inv(eye(game.N_states)-game.gamma*policy_matrix*game.P)*policy_matrix*game.R;
q_opt = inv(eye(game.N_states*game.N_actions)-game.gamma*game.P*policy_matrix)*game.R;
% Valor óptimo de d (política en forma vector)
end

