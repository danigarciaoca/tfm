function [Qsa, Vs, G] = sarsa_core(env, Qsa, epsilon, alpha, maxNumStepsPerEpisode)

numEpisodes = 1; % sólo corremos un episodio para evaluar la política a la que hemos convergido
DoAction = env.DoAction;
S = env.numStates; % Número de estados
A = env.numActions; % Número de acciones

% Return por episodio
G = zeros(1,numEpisodes);

for i = 1:numEpisodes
    % Inicializamos S
    % currentState = game.centralState;
    s_t = env.initState; % empezar en el de la izquierda
    stepPerEpisode = 0; % stepPerEpisode counts the number of steps taken in ONE of the numEpisodes episodes simulated
    
    % Escogemos A (currentAction) de S (currentState) según la e-greedy policy.
    a_t = e_greedy(Qsa, epsilon, s_t, A);
    
    while true % siempre que el episodio no haya terminado
        
        % Tomamos la acción A (currentAction), observamos la recompensa R
        % (reward) y el siguiente estado S' (nextState).
        [ s_t1, reward, terminal ] = DoAction(a_t, s_t, env);
        
        % G(exp,i) = reward+game.gamma*G(exp,i); % return following the initial state (si el estado inicial fuese aleatorio, habría que crear una G para cada vez que e pasa por s por primera vez [p. 105 de Sutton])
        G(i) = (env.gamma^stepPerEpisode)*reward + G(i);
        
        % Escogemos A' (nextAction) de S' (nextState) según la e-greedy policy.
        a_t1 = e_greedy(Qsa, epsilon, s_t1, A);
        
        % Actualizamos el valor de Q(s,a)
        %Qsa(s_t, a_t) = Qsa(s_t, a_t) + alpha*(reward + env.gamma*Qsa(s_t1, nextAction) - Qsa(s_t, a_t));
        
        % Actualizamos los valores
        s_t = s_t1;
        a_t = a_t1;
        stepPerEpisode = stepPerEpisode + 1;
        
        % Evaluación de si el episodio ha terminado o no
        if terminal || stepPerEpisode == maxNumStepsPerEpisode % Si el estado actual es el terminal
            terminal = true;
            %[~, ~, G_eps0] = q_learning_core(env, Qsa, 0, alpha, maxNumStepsPerEpisode);
            %G_eps0_each_epi(exp,i) = G_eps0;
            break; % terminamos el episodio
        end
    end
    
    % Acumulamos el valor de la función V(s)
    policy = getPolicy(Qsa, env);
    Vs = getValueFunction(Qsa, policy, env);
end
end

