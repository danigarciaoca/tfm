function [ Qsa_optima, policy, grid_result ] = getPolicy( Qsa, game )
%GETPOLICY Funci�n para obtener la policy y la funci�n Q �ptimas
%   grid_result -> representaci�n visual del tablero y la acci�n tomada
%   en cada estado. De nuevo, las celdas con valor NaN representan el
%   acantilado.
%   policy -> greedy policy de la funci�n Q(s,a) pasada por argumento.
%   Qsa -> funci�n Qsa de la que se quiere conocer la greedy policy.

[Qsa_optima, policy] = max(Qsa,[],2);

% La variable grid_result es �nicamente para que el usuario pueda ver si
% los resultados obtenidos tienen sentido
grid_result = reshape(policy(1:end-2)', [game.numColTablero game.numFilTablero-1])';
grid_result(game.numFilTablero, :) = game.board(end,:);
grid_result(game.numFilTablero, 1) = policy(end-1);
% En el �ltimo estado, Q(s,a) = 0 para cualquier acci�n ya que es el estado
% terminal. Dado que no tiene sentido decir que tomamos la acci�n 1, 2, 3 o
% 4, ponemos este campo a 0 indicando que no se toma ninguna acci�n.
grid_result(game.numFilTablero, end) = 0;

end

