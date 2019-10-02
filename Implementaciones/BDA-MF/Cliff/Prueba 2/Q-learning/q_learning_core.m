function [Qsa, Vs, G] = q_learning_core(env, Qsa, epsilon, alfa, numEpisodes, max_steps)
% Asignaciones por comodidad
DoAction = env.DoAction;
S = env.numStates;
A = env.numActions;
gamma = env.gamma;
% Return por episodio
G = zeros(1,numEpisodes);

for i = 1:numEpisodes
    % Initialize episode
    terminal = false;
    step = 0;
    s_t = env.initState; %env.initState = initPos( 'random', env.board, S ); % Comentar esta línea si queremos empezar desde la esquina (es el que viene por defecto)
    
    while ~terminal % siempre que el episodio no haya terminado
        % Escogemos A (currentAction) de S (currentState) según la e-greedy policy.
        a_t = e_greedy(Qsa, epsilon, s_t, A);
        
        % Tomamos la acción 'a', observamos la recompensa 'r' y el siguiente estado s'.
        [s_t1, reward, terminal] = DoAction( a_t, s_t, env );
        G(i) = G(i) + gamma^step * reward;
        
        % Actualizamos el valor de Q(s,a)
        a_t1 = greedy(Qsa, s_t1);
        Qsa(s_t, a_t) = Qsa(s_t, a_t) + alfa*(reward + gamma*Qsa(s_t1, a_t1) - Qsa(s_t, a_t));
        
        % Actualizamos los valores
        s_t = s_t1;
        step = step +1;
        
        % Terminate the episode
        if step == max_steps
            break
        end
    end
    % Obtenemos la función valor V(s)
    [ ~, policy_matrix] = getPolicy(Qsa, env);
    Vs = getValueFunction(Qsa, policy_matrix, env);
end
end