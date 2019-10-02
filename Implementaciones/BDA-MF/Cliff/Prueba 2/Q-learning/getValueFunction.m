function Vs = getValueFunction(Qsa, policy, game)
%GETVALUEFUNCTION Recupera la función valor de estado óptima a partir de la
%política y de la función valor de estado-acción óptima

Qsa_lineal = reshape(Qsa', [game.numStates*game.numActions 1]);
Vs = policy*Qsa_lineal;

end

