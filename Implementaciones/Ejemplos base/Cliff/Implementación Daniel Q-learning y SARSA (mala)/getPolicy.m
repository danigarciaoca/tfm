function [ Qsa_optima, policy, grid_result ] = getPolicy( Qsa, game )
%GETPOLICY Función para obtener la policy y la función Q óptimas
%   grid_result -> representación visual del tablero y la acción tomada
%   en cada estado. De nuevo, las celdas con valor NaN representan el
%   acantilado.
%   policy -> greedy policy de la función Q(s,a) pasada por argumento.
%   Qsa -> función Qsa de la que se quiere conocer la greedy policy.

[Qsa_optima, policy] = max(Qsa,[],2);

% La variable grid_result es únicamente para que el usuario pueda ver si
% los resultados obtenidos tienen sentido
grid_result = reshape(policy(1:end-2)', [game.numColTablero game.numFilTablero-1])';
grid_result(game.numFilTablero, :) = game.board(end,:);
grid_result(game.numFilTablero, 1) = policy(end-1);
% En el último estado, Q(s,a) = 0 para cualquier acción ya que es el estado
% terminal. Dado que no tiene sentido decir que tomamos la acción 1, 2, 3 o
% 4, ponemos este campo a 0 indicando que no se toma ninguna acción.
grid_result(game.numFilTablero, end) = 0;

end

