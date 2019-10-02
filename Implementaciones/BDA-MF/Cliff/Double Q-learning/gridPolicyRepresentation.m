function grid_result = gridPolicyRepresentation(game, policy_lineal)
% grid_result -> representación visual del tablero y la acción tomada

% La variable grid_result es únicamente para que el usuario pueda ver si
% los resultados obtenidos tienen sentido
policy_lineal(game.cliff) = NaN; % En las casillas que había cliff ponemos NaN (por representación visual sólo)
grid_result = reshape(policy_lineal', [game.numColTablero game.numFilTablero])';
% En el último estado, Q(s,a) = 0 para cualquier acción ya que es el estado
% terminal. Dado que no tiene sentido decir que tomamos la acción 1, 2, 3 o
% 4, ponemos este campo a 0 indicando que no se toma ninguna acción.
grid_result(game.numFilTablero, end) = 0;
end