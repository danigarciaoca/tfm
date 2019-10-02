function Vs = getValueFunction(Qsa, policy, game)
%GETVALUEFUNCTION Recupera la funci�n valor de estado �ptima a partir de la
%pol�tica y de la funci�n valor de estado-acci�n �ptima

Qsa_lineal = reshape(Qsa', [game.N_states*game.N_actions 1]);
Vs = policy*Qsa_lineal;

end

