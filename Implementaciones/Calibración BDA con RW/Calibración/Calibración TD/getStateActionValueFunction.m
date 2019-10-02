function [Qsa, Qsa_acumulada] = getStateActionValueFunction(Vs_acumulada, game)
%GETSTATEACTIONVALUEFUNCTION Recupera la función valor de acción-estado óptima
%a partir de la política y de la función valor de estado óptima

Qsa_acumulada = zeros(game.N_states*game.N_actions, size(Vs_acumulada,2), size(Vs_acumulada,3));
for i = 1:size(Vs_acumulada,2)
    for j = 1:size(Vs_acumulada,3)
        Qsa_acumulada(:,i,j) = game.R + game.gamma*game.P*Vs_acumulada(:,i,j);
    end
end
Qsa = mean(squeeze(Qsa_acumulada(:,end,:)),2);

end

