function [ v, G ] = RL_chain_buffer_core(env, d, phi, theta, epsilon, alphaD, numRep, numEpisodes, maxStepsEpisode, mult1, mult2)
%RL_CHAIN_BUFFER_CORE Summary of this function goes here
%   Detailed explanation goes here

numRep = 1;
numEpisodes = 1;
% Asignaciones por comodidad
DoAction = env.DoAction;
GetStateFeatures = env.GetStateFeatures;
S = env.S; % number of states
A = env.num_actions; % number of actions
gamma = env.gamma;
% Return por episodio
G = zeros(1,numRep*numEpisodes);

episodeCountV = 0; % episodeCount counts the number of episodes taken in all numRep (para el bucle de V)
episodeCountD = 0; % episodeCount counts the number of episodes taken in all numRep (para el bucle de D)

for k = 1:numRep
    s_a_sNext = []; % Vector que almacena la secuencia de estados recorridos en un episodio
    totalStepsPerRep = 1; % totalStepsPerRep counts the number of steps taken in ALL numEpi episodes simulated in ONE numRep
    
    for n = 1:numEpisodes % This loop sets the update frequency of v (every numEpi episodes)
        % currentState = game.centralState; % estado inicial el central
        % currentState = randi([2 game.centralState]);
        s = env.initial_state; % empezar en el de la izquierda
        terminal = false; % true when episode finish, false otherwise
        stepPerEpisode = 0; % stepPerEpisode counts the number of steps taken in ONE of the numEpi episodes simulated
        
        % Normalize d
        d_norm = getPolicyVectorFromD( d, env );
        % Get policy matrix
        policy_by_action = reshape(d_norm', [A,S])';
        policy_matrix = diag(policy_by_action(:,1))*mult1' + diag(policy_by_action(:,2))*mult2';
        %             for i=1:A
        %                 policy_matrix2 = policy_matrix2 + diag(policy_by_action(:,i))*mult(:,:,i)';
        %             end
        
        while true
            % Escogemos A (currentAction) de S (currentState) según la e-greedy policy.
            a = e_greedy(d_norm, epsilon, s, A);
            
            % Tomamos la acción 'a', observamos la recompensa 'r' y el siguiente estado s'.
            [s_t1, r, terminal] = DoAction( a, s, env );
            %[s_t1, r] = getNextState(env, currentState, currentAction);
            G((k-1)*numEpisodes+n) = (gamma^stepPerEpisode)*r + G((k-1)*numEpisodes+n); % return following the initial state (si el estado inicial fuese aleatorio, habría que crear una G para cada vez que e pasa por s por primera vez [p. 105 de Sutton])
            
            
            % Actualizamos valores
            stepPerEpisode = stepPerEpisode + 1;
            totalStepsPerRep = totalStepsPerRep + 1;
            s = s_t1;
            
            % Evaluate if episode has finished or not
            if (stepPerEpisode == maxStepsEpisode) || (terminal == true) % if maximum numbero of steps per episode or terminal state reached
                episodeCountV = episodeCountV + 1;
                break
            end
        end
    end
    
    % estimate value function
    v = phi*theta;

end

end

