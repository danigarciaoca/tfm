function [Qsa, Vs, G] = q_learning_core(env, Qsa, epsilon, alpha, maxNumStepsPerEpisode)

numEpisodes = 1; % sólo corremos un episodio para evaluar la política a la que hemos convergido
DoAction = env.DoAction;
S = env.numStates; % Número de estados
A = env.numActions; % Número de acciones

% Return por episodio
G = zeros(1,numEpisodes);

for i = 1:numEpisodes
    % Inicializamos S
    s_t = env.initState;
    stepPerEpisode = 0; % stepPerEpisode counts the number of steps taken in ONE of the numEpisodes episodes simulated
    
    while true % siempre que el episodio no haya terminado
        % Escogemos A (currentAction) de S (currentState) según la e-greedy policy.
        a_t = e_greedy(Qsa, epsilon, s_t, A);
        
        % Tomamos la acción A (currentAction), observamos la recompensa R
        % (reward) y el siguiente estado S' (nextState).
        [ s_t1, reward, terminal ] = DoAction(a_t, s_t, env);
        
        G(i) = (env.gamma^stepPerEpisode)*reward + G(i);
        
        % Actualizamos el valor de Q(s,a)
        a_t1 = greedy(Qsa, s_t1);
        %Qsa(s_t, a_t) = Qsa(s_t, a_t) + alpha*(reward + env.gamma*Qsa(s_t1, nextAction) - Qsa(s_t, a_t));
        
        % Actualizamos los valores
        stepPerEpisode = stepPerEpisode + 1;
        s_t = s_t1;
        
        % Evaluación de si el episodio ha terminado o no
        if terminal || stepPerEpisode == maxNumStepsPerEpisode % Si el estado actual es el terminal
            terminal = true;
            break; % terminamos el episodio
        end
        
    end
    % Acumulamos el valor de la función V(s)
    policy = getPolicy(Qsa, env);
    Vs = getValueFunction(Qsa, policy, env);
end
end

