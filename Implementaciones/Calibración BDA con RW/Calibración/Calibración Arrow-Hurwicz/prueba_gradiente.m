% % % % ENTORNO:
N_states = 7;
transitionType = 'det';
% transitionType = 'rand';
rewardType = 'det';
% rewardType = 'rand';
game = random_walk_set_up_mod(N_states, transitionType, rewardType);

S = game.N_states; % Número de estados
A = game.N_actions; % Número de acciones
game.mu(:) = (1/S)/(S-1); game.mu(2) = 1-(1/S); mu = game.mu; % Distribución inicial de probabilida de los estados
P = game.P; % Matriz de transiciones
R = game.R; % Vector de rewards
% d = rand(S*A,1);
d = [0.3723 0.4817 0.7768 0.9599 0.0158 0.6020 0.0473 0.1675 0.6476 0.2098 0.4108 0.3288 0.1067 0.5251]';

vGrad_analit = zeros(S,1);
for nextState=1:S
    Ptrasp = P';
    vGrad_analit(nextState) = (1-game.gamma)*mu(nextState) +  game.gamma*Ptrasp(nextState,:)*d - sum(d(((nextState-1)*A)+1:nextState*A));
end
vGrad_vect = (1-game.gamma)*mu + (game.gamma*P' - game.marg)*d;

[vGrad_analit vGrad_vect]