function [ G ] = RL_mount_sample_core(env, d, theta, epsilon, alphaD, numRep, numEpisodes, maxStepsEpisode, mult)
%RL_CHAIN_BUFFER_CORE Summary of this function goes here
%   Detailed explanation goes here

numRep = 1;
numEpisodes = 1;
% Asignaciones por comodidad
DoAction = env.DoAction;
GetStateFeatures = env.GetStateFeatures;
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
        s_t = env.initial_state; % empezar en el de la izquierda
        terminal = false; % true when episode finish, false otherwise
        stepPerEpisode = 0; % stepPerEpisode counts the number of steps taken in ONE of the numEpi episodes simulated
        
        % Normalize d
        d_norm = getPolicyVectorFromD( d, env );
        
        while true
            % Escogemos A (currentAction) de S (currentState) según la e-greedy policy.
            s_t_disc = GetDiscretizedState(s_t, env.xy_disc);
            a = e_greedy(d_norm, epsilon, s_t_disc, A);
            
            % Tomamos la acción 'a', observamos la recompensa 'r' y el siguiente estado s'.
            [s_t1, r, terminal] = DoAction( a, s_t, env );
            G((k-1)*numEpisodes+n) = (gamma^stepPerEpisode)*r + G((k-1)*numEpisodes+n); % return following the initial state (si el estado inicial fuese aleatorio, habría que crear una G para cada vez que e pasa por s por primera vez [p. 105 de Sutton])
            
            % Almacenamos las transiciones del episodio
            s_a_sNext(totalStepsPerRep,:) = [s_t a s_t1 r stepPerEpisode];
            
            % Actualizamos valores
            stepPerEpisode = stepPerEpisode + 1;
            totalStepsPerRep = totalStepsPerRep + 1;
            s_t = s_t1;
            
            % Evaluate if episode has finished or not
            if (stepPerEpisode == maxStepsEpisode) || (terminal == true) % if maximum numbero of steps per episode or terminal state reached
                episodeCountV = episodeCountV + 1;
                break
            end
        end
    end
   
end