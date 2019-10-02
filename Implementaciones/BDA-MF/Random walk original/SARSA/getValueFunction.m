function Vs = getValueFunction(Qsa, policy, game)
%GETVALUEFUNCTION Recupera la función valor de estado óptima a partir de la
%política y de la función valor de estado-acción óptima

Qsa_lineal = reshape(Qsa', [game.N_states*game.N_actions 1]);
Vs = policy*Qsa_lineal;

end

