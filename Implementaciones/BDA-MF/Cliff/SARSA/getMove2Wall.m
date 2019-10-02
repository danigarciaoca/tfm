function [ move2wall ] = getMove2Wall( board, numActions, wall )
%GETMOVE2WALL Esta funci�n permite determinar qu� movimiento nos va a
%llevar a rebotar contra una pared. Dado que no tiene sentido evaluar esta
%condici�n para todos los posibles estados, la evaluamos para aquellos que
%sabemos que s� tienen ocasi�n de toparse con una pared, esto es, los
%"estados pared" o wall.
% 
%   board: tablero de juego
%   numActions: n�mero de acciones permitidas en el juego. Cada n�mero est�
%   asociado a un movimiento (1 = izquierda, 2 = arriba, 3 = derecha, 4 =
%   abajo)
%   wall: paredes o "estados pared"

% Inicializamos la matriz que contendr� qu� movimientos nos llevan a chocar
% con una pared. Reservamos la �ltima columna para poder referenciar desde
% qu� estado estamos realizando el movimiento (los llamados "estados pared")
move2wall = zeros(numActions+1, size(wall,2));

% Obtenemos los �ndeces de las paredes de nuestro tablero
[row,col] = find(ismember(board,wall)); 

% Para esos "estados pared", determinamos qu� movimiento les va a hacer
% chocar con la pared. Cada fila hace referencia al movimiento tomado
move2wall(1,:) = (col-1 == 0); % Movimiento a la izquierda (1)
move2wall(2,:) = (row-1 == 0); % Movimiento arriba (2)
move2wall(3,:) = (col+1 == size(board,2)+1); % Movimiento a la derecha (3)
move2wall(4,:) = (row+1 == size(board,1)+1); % Movimiento abajo (4)

% La �ltima fila contendr� los estados que tienen pared para poder hacer el
% mapping "estado-movimiento que lleva a pared"
move2wall(numActions+1,:) = board(sub2ind(size(board), row, col))';

move2wall = sortrows(move2wall',numActions+1)'; % Ordenamos los datos seg�n los estados
end

