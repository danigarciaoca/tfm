function [Qsa, Qsa_acumulada] = getStateActionValueFunction(Vs_acumulada, game)
%GETSTATEACTIONVALUEFUNCTION Recupera la función valor de acción-estado óptima
%a partir de la política y de la función valor de estado óptima

Qsa_acumulada = zeros(game.numStates*game.numActions, size(Vs_acumulada,2), size(Vs_acumulada,3));
for i = 1:size(Vs_acumulada,2)
    for j = 1:size(Vs_acumulada,3)
        Qsa_acumulada(:,i,j) = game.P*(game.R + game.gamma*Vs_acumulada(:,i,j));
    end
end
Qsa = mean(squeeze(Qsa_acumulada(:,end,:)),2);

end

