function grid_result = gridPolicyRepresentation(game, policy_lineal)
% grid_result -> representaci�n visual del tablero y la acci�n tomada

% La variable grid_result es �nicamente para que el usuario pueda ver si
% los resultados obtenidos tienen sentido
policy_lineal(game.cliff) = NaN; % En las casillas que hab�a cliff ponemos NaN (por representaci�n visual s�lo)
grid_result = reshape(policy_lineal', [game.numColTablero game.numFilTablero])';
% En el �ltimo estado, Q(s,a) = 0 para cualquier acci�n ya que es el estado
% terminal. Dado que no tiene sentido decir que tomamos la acci�n 1, 2, 3 o
% 4, ponemos este campo a 0 indicando que no se toma ninguna acci�n.
grid_result(game.numFilTablero, end) = 0;
end