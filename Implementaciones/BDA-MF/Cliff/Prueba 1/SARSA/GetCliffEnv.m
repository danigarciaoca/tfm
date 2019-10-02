function env = GetCliffEnv( name, numFilTablero, numColTablero, rewards, initType , transitionType )
%DEFINEPROBLEM Crea el entorno del juego en funci�n de los argumentos que
%recibe.
% 
%   name -> nombre del juego
%   numFilTablero -> numero de filas que tendr� el tablero sobre el que se
%       representa el juego.
%   numColTablero -> numero de columnas que tendr� el tablero sobre el que se
%       representa el juego.
%   numActions -> n�mero de acciones que tiene permitido cada estado del
%       juego.
%   rewards -> recompensas instant�neas en cada transici�n, definidas al
%       comienzo del juego.
%   initType -> permite definir el estado inicial de manera fija o
%       aleatoria (ver initPos.m)

% Creamos el tablero de juego
tabAux = 1:numFilTablero*numColTablero;
board = reshape(tabAux, [numColTablero numFilTablero])';
cliffValue = board(end,2:numColTablero-1);

% Definimos el n�mero de estados y de acciones que habr�
numStates = board(end); % Habr� 36 estados normales, 1 de inicio y otro de fin y 10 de cliff
numActions = 4; % izquierda, arriba, derecha, abajo

% Definimos d�nde estar�n las paredes (de manera gen�rica, ser�n las
% paredes circundantes al tablero)
wall0 = board(1,2:end-1); wall0 = wall0(:); % de 2 a end-1 para que no se repitan en las siguientes "paredes" el primer y el �ltimo "estados pared"
wall1 = board(:,1); wall1 = wall1(:);
wall2 = board(:,end); wall2 = wall2(:);
wall = [wall1' wall0' wall2'];
move2wall = getMove2Wall(board, numActions, wall);

DoAction = @DoActionCliff;

% Creamos el juego
field1 = 'name';  value1 = name;
field2 = 'numFilTablero';  value2 = numFilTablero;
field3 = 'numColTablero';  value3 = numColTablero;
field4 = 'board';  value4 = board;
field5 = 'numStates';  value5 = numStates;
field6 = 'numActions';  value6 = numActions;
field7 = 'initState';  value7 = initPos(initType, board, numStates);
field8 = 'finalState';  value8 = board(end,end);
field9 = 'cliff';  value9 = cliffValue;
field10 = 'walls';  value10 = move2wall;
field11 = 'rewardOnTransition';  value11 = rewards(1);
field12 = 'rewardOnCliff';  value12 = rewards(2);

env = struct(field1,value1,field2,value2,field3,value3,field4,value4,field5,value5,field6,value6,field7,value7,field8,value8,field9,value9,field10,value10,field11,value11,field12,value12);

env.DoAction = DoAction; % handler to DoAction function
env.gamma = 0.99; % discount factor

% Create rewards vector R(s,a) and transition matrix P
[R, P] = createRandP(numStates, numActions, env, transitionType);
env.R = R;
env.P = P;

% Auxiliar matrix for constructing policy vector d
A = eye(numStates);
B = ones(numActions,1);
duplicate = kron(A,B);
env.duplicate = duplicate;
end

