function [Qsa1, Qsa2, Vs1, Vs2, G] = double_q_learning_core(env, Qsa1, Qsa2, epsilon, alfa, numEpisodes, max_steps)
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
        Qsa = (Qsa1+Qsa2)./2;
        a_t = e_greedy(Qsa, epsilon, s_t, A);
        
        % Tomamos la acción 'a', observamos la recompensa 'r' y el siguiente estado s'.
        [s_t1, reward, terminal] = DoAction( a_t, s_t, env );
        G(i) = G(i) + gamma^step * reward;
        
        % Actualizamos el valor de Q(s,a) de acuerdo a double Q-learning
        flip_coin = binornd(1,0.5); % Con probabilidad 0.5 hay exito (1); con probabilidad  1-0.5=0.5 hay fracaso (0)
        if flip_coin == 0
            a_t1 = greedy(Qsa1, s_t1);
            Qsa1(s_t, a_t) = Qsa1(s_t, a_t) + alfa*(reward + gamma*Qsa2(s_t1, a_t1) - Qsa1(s_t, a_t));
        elseif flip_coin == 1
            a_t1 = greedy(Qsa2, s_t1);
            Qsa2(s_t, a_t) = Qsa2(s_t, a_t) + alfa*(reward + gamma*Qsa1(s_t1, a_t1) - Qsa2(s_t, a_t));
        end
        
        % Actualizamos los valores
        s_t = s_t1;
        step = step + 1;
        
        % Terminate the episode
        if step == max_steps
            break
        end
    end
    % Acumulamos el valor de la función V(s)
    [ ~, policy_matrix] = getPolicy(Qsa1, env);
    Vs1 = getValueFunction(Qsa1, policy_matrix, env);
    [ ~, policy_matrix] = getPolicy(Qsa2, env);
    Vs2 = getValueFunction(Qsa2, policy_matrix, env);
end
end