% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Nombre del fichero: q_learning.m                                  %
% Autor: Daniel M. Garc�a-Oca�a Hern�ndez                           %
% Fecha de creaci�n: 28/11/2016                                     %
% Implementaci�n de Q-learning: an off-policy TD control algorithm  %
% (p.137 de Reinforcement Learning: An Introduction)                %
% Ejemplo: Cliff Walking, p.138 Example 6.6                         %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% clear all, close all, clc

% % % % ENTORNO:
name = 'Cliff';
numFilTablero = 4;
numColTablero = 12;
numActions = 4; % izquierda, arriba, derecha, abajo
rewards = [-1, -100]; % -1 en todas las transiciones, -100 si cae al cliff

% Inicializamos el juego
initType = 'fixed'; % fixed: empieza en la esquina inferior izquierda; random: empieza en cualquier estado
game = defEnvironment(name, numFilTablero, numColTablero, numActions, rewards, initType);

% % % % AGENTE:
numEpisodes = 500; % N�mero de episodios de cada experimento
numRep = 10; % N�mero de experimentos sobre el que se promedia
G = zeros(numRep,numEpisodes); % Reward por episodio
epsilon = 0.1;
alfa = 0.5;
gamma = 1;

for k = 1:numRep
    % Inicializamos Q para cada experimento
    Qsa = zeros(game.numStates, game.numActions);
    
    for i = 1:numEpisodes
        % Inicializamos S
        game.initState = initPos( 'random', game.board, game.numStates ); % Comentar esta l�nea si queremos empezar desde la esquina (es el que viene por defecto)
        currentState = game.initState;
        
        while true % siempre que el episodio no haya terminado
            % Escogemos A (currentAction) de S (currentState) seg�n la e-greedy policy.
            currentAction = e_greedy(Qsa, epsilon, currentState, game.numActions);
            
            % Tomamos la acci�n A (currentAction), observamos la recompensa R
            % (reward) y el siguiente estado S' (nextState).
            [nextState, reward] = getNextState(game, currentState, currentAction);
            G(k,i) = reward+gamma*G(k,i);
            
            % Actualizamos el valor de Q(s,a)
            nextAction = greedy(Qsa, currentState);
            Qsa(currentState, currentAction) = Qsa(currentState, currentAction) + alfa*(reward + gamma*Qsa(nextState, nextAction) - Qsa(currentState, currentAction));
            
            % Actualizamos los valores
            currentState = nextState;
            
            % Evaluaci�n de si el episodio ha terminado o no
            if currentState == game.finalState % Si el estado actual es el terminal
                break; % terminamos el episodio
            end
        end
    end
end

% getPolicy obtiene la policy para el �ltimo experimento. Si queremos, podemos
% obtenerla por cada experimento meti�ndola dentro del bucle for k = 1:numRep,
% pero cuando numRep es muy engorroso (en cuanto a presentaci�n visual)
[ maxim, policy, grid_result ] = getPolicy( Qsa, game );

% % % REPRESENTACI�N DE RESULTADOS:
% Resultados obtenidos para numRep experimentos de numEpisodes episodios
Gmean = mean(G,1); % Media de los experimentos realizados
order = 10; long = 55;
Gsmooth = sgolayfilt(Gmean,order,long); % Suavizamos los resultados

field1 = 'policy';  value1 = policy;
field2 = 'Qopt';  value2 = maxim;
field3 = 'finalGrid';  value3 = grid_result;
field4 = 'Gmean';  value4 = Gmean;
field5 = 'Gsmooth';  value5 = Gsmooth;
field6 = 'numRep';  value6 = numRep;
field7 = 'numEpisodes';  value7 = numEpisodes;

Qlearning_res = struct(field1,value1,field2,value2,field3,value3,field4,value4,field5,value5,field6,value6,field7,value7);

plot((1:Qlearning_res.numEpisodes), Qlearning_res.Gsmooth, 'LineWidth', 1.5), ylim([-100 0])
grid, legend('Q-learning', 'Location','SouthEast')
xlabel('Episode'), ylabel('Reward per episode'), title('Evoluci�n (curva suavizada)')

disp('Grid final para Q-learning (1=izquierda, 2 = arriba, 3 = derecha, 4 = abajo)'), Qlearning_res.finalGrid