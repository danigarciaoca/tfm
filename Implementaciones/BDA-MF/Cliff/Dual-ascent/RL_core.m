function [d, Vs, G] = RL_core(env, d, v, epsilon, alphaTD, alphaD, numRep, numEpisodes, maxNumStepsPerEpisode, mult)
numEpisodes = 1; % s�lo corremos un episodio para evaluar la pol�tica a la que hemos convergido
numRep = 1;
% Asignaciones por comodidad
DoAction = env.DoAction;
S = env.numStates;
A = env.numActions;
gamma = env.gamma;
% Return por episodio
G = zeros(1,numRep*numEpisodes);
Vs = zeros(S,numRep*numEpisodes);

episodeCountV = 0; % episodeCount counts the number of episodes taken in all numRep (para el bucle de V)
episodeCountD = 0; % episodeCount counts the number of episodes taken in all numRep (para el bucle de D)
for k = 1:numRep
    s_a_sNext = []; % Vector que almacena la secuencia de estados recorridos en un episodio
    totalStepsPerRep = 1; % totalStepsPerRep counts the number of steps taken in ALL numEpi episodes simulated in ONE numRep
    for n = 1:numEpisodes % This loop sets the update frequency of v (every numEpi episodes)
        s_t = env.initState;
        terminal = false; % true when episode finish, false otherwise
        step = 0; % step counts the number of steps taken in ONE of the numEpi episodes simulated
        
        % Normalize d
        d_norm = getPolicyVectorFromD( d, env );
        
        % Get policy matrix
        policy_matrix = zeros(S,S*A);
        policy_by_action = reshape(d_norm', [A,S])';
        for i=1:A
            policy_matrix = policy_matrix + diag(policy_by_action(:,i))*mult(:,:,i)';
        end
        while ~terminal
            % Escogemos A (currentAction) de S (currentState) seg�n la e-greedy policy.
            a_t = e_greedy(d_norm, epsilon, s_t, A);
            
            % Tomamos la acci�n a (currentAction), observamos la recompensa
            % r (reward) y el siguiente estado s' (nextState).
            [s_t1, reward, terminal] = DoAction( a_t, s_t, env );
            
            G((k-1)*numEpisodes+n) = gamma^step * reward + G((k-1)*numEpisodes+n); % return following the initial state (si el estado inicial fuese aleatorio, habr�a que crear una G para cada vez que e pasa por s por primera vez [p. 105 de Sutton])
            
            % Update de v(s)
            % TEMPORAL DIFFERENCE
            v(s_t) = v(s_t) + alphaTD*(reward + gamma*v(s_t1) - v(s_t)); % policy evaluation
            
            % Almacenamos las transiciones del episodio
            s_a_sNext(totalStepsPerRep,:) = [s_t a_t s_t1 reward step terminal];
            
            % Actualizamos valores
            step = step + 1;
            totalStepsPerRep = totalStepsPerRep + 1;
            s_t = s_t1;
            
            % Evaluaci�n de si el episodio ha terminado o no
            if terminal || step == maxNumStepsPerEpisode % Si el estado actual es el terminal
                episodeCountV = episodeCountV + 1;
                Vs(:, episodeCountV) = v;
                % disp(['Fin' num2str(n) ' y ' num2str(stepPerEpisode)])
                break;
            end
        end
    end
%     totalStepsPerRep = totalStepsPerRep-1; % Compensamos el que se increment� de m�s
%     
%     for i = 1:totalStepsPerRep
%         % Recover saved episodes
%         [s_t, s_t1, reward, step, s_a_index, terminal] = recoverSavedEpisode(s_a_sNext, A, i);
%         
%         % Policy (or d) update
%         d(s_a_index) = d(s_a_index) + alphaD*(reward + gamma*v(s_t1) - v(s_t));
%         d_orig = d;
%         d(d<0)=0; % Projection of d over positives
%         
%         % Normalize d
%         d_norm = getPolicyVectorFromD( d, env );
%         
%         % Evaluaci�n de si el episodio ha terminado o no para guardar el error en la policy (save policy error)
%         if any(isnan(d_norm))
%             % Fix de la d original (que pod�a tener n�meros negativos)
%             d_orig(isnan(d_norm) & d_orig<0) = abs(d_orig(isnan(d_norm) & d_orig<0));
%             d = d_orig;
%             d(d<0)=0; % Projection of d over positives
%             d_norm = getPolicyVectorFromD( d, env );
%             % disp('nan!')
%             % not_error = false;
%             % break;
%         end
%         if terminal || step == maxNumStepsPerEpisode-1 % Si el estado siguiente es el terminal
%             episodeCountD = episodeCountD + 1;
%             % Calculate norm-2 of policy error
%             % errorD(exp, episodeCountD) = norm(abs(d_norm(A+1:end-A) - d_opt_norm(A+1:end-A)),2);
%             % d_norm_acumulada(:,episodeCountD, exp) = d_norm;
%             % d_acumulada(:,episodeCountD, exp) = d_orig;
%             % [d_norm; episodeCountD]
%             % reshape(d_norm', [2 21])'
%             % [~, policy_lineal] = max(reshape(d_norm,[A S])',[],2);
%             % checkCliffResults(env, policy_lineal)
%         end
%     end
end
end