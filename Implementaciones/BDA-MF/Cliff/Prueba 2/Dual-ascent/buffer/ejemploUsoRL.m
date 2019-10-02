clear all, close all, clc

% % % % ENTORNO:
name = 'Cliff';
numFilTablero = 4;
numColTablero = 12;
rewards = [-1, -100]; % -1 en todas las transiciones, -100 si cae al cliff
initType = 'fixed'; % fixed: empieza en la esquina inferior izquierda; random: empieza en cualquier estado
env = GetCliffEnv(name, numFilTablero, numColTablero, rewards, initType);

% Asignaciones por comodidad
DoAction = env.DoAction;
S = env.numStates; % Número de estados
A = env.numActions; % Número de acciones
env.mu(:) = (1/S)/(S-1); env.mu(env.initState) = 1-(1/S); mu = env.mu; % Distribución inicial de probabilida de los estados
P = env.P; % Matriz de transiciones
R = env.R; % Vector de rewards
gamma = env.gamma;

% mult es un tensor auxiliar para crear la matriz de la política
aux1 = eye(S);
aux2 = eye(A);
for i=1:A
    mult(:,:,i) = kron(aux1,aux2(:,i));
end

% % % % AGENTE:
numExperiments = 20; % Número de experimentos de numRep*numEpisodes episodios
numRep = 25; % Número de repeticiones de cada set de episodios
numEpisodes = 20; % Número de episodios de cada repeticion
maxNumStepsPerEpisode = 1000; % Número máximo de pasos en cada episodio
G = zeros(numExperiments, numRep*numEpisodes); % Return por episodio
G_eps0_acumulada = zeros(numExperiments, numRep*numEpisodes); % Return por episodio cuando epsilon = 0
epsilon = 0.1; % e-greedy value (entre 0.05 y 0.2) %con 0.4 tira bien para el cliff
alphaD = 0.05; % Stepsize para la iteración de la variable dual d % 0.1 valor original
alphaTD = 0.2; % Stepsize para la iteración de la variable primal v

% % Optimum values
% v_opt = inv(eye(S)-gamma*env.pi_opt*P)*env.pi_opt*R;
% q_opt = inv(eye(S*A)-gamma*P*env.pi_opt)*R;

% Variable que acumulará la funcion V y el error en la política al final de cada episodio
Vs_acumulada = nan(S, numRep*numEpisodes, numExperiments);

load('v_opt_q-learning.mat')
MINIBATCH_DIM = 1000;
BUFF_DIM = 5000;
data = CreateCircularBuffer(BUFF_DIM, 6); % 6 columnas para los 6 parámetros a guardar
for exp = 1:numExperiments
    % Inicializamos D y V
    d = rand(S*A,1);
%     v = rand(S,1);
%     terminal_states = [env.cliff env.finalState];
%     v(terminal_states) = 0;
    v = Vs;
    % Inicializamos el buffer circular
    data = RestoreBuffer(data);
    
    exp
    
    episodeCountV = 0; % episodeCount counts the number of episodes taken in all numRep (para el bucle de V)
    episodeCountD = 0; % episodeCount counts the number of episodes taken in all numRep (para el bucle de D)
    for k = 1:numRep
        
        %s_a_sNext = []; % Vector que almacena la secuencia de estados recorridos en un episodio
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
                % Escogemos A (currentAction) de S (currentState) según la e-greedy policy.
                a_t = e_greedy(d_norm, epsilon, s_t, A);
                
                % Tomamos la acción a (currentAction), observamos la recompensa
                % r (reward) y el siguiente estado s' (nextState).
                [s_t1, reward, terminal] = DoAction( a_t, s_t, env );
                
                G(exp, (k-1)*numEpisodes+n) = gamma^step * reward + G(exp,(k-1)*numEpisodes+n); % return following the initial state (si el estado inicial fuese aleatorio, habría que crear una G para cada vez que e pasa por s por primera vez [p. 105 de Sutton])
                
                % Update de v(s)
                % TEMPORAL DIFFERENCE
                % v(s_t) = v(s_t) + alphaTD*(reward + gamma*v(s_t1) - v(s_t)); % policy evaluation
                % EXACTA (BELLMAN)
                %v = (inv(eye(S)-gamma*policy_matrix*P))*policy_matrix*R;
                v = Vs;
                % Almacenamos las transiciones del episodio
                %s_a_sNext(totalStepsPerRep,:) = [s_t a_t s_t1 reward step terminal];
                data = AddItem(data, [s_t a_t s_t1 reward step terminal]);
                
                % Actualizamos valores
                step = step + 1;
                totalStepsPerRep = totalStepsPerRep + 1;
                s_t = s_t1;
                
                % Evaluación de si el episodio ha terminado o no
                if terminal || step == maxNumStepsPerEpisode % Si el estado actual es el terminal
                    episodeCountV = episodeCountV + 1;
                    Vs_acumulada(:, episodeCountV, exp) = v;
                    % disp(['Fin' num2str(n) ' y ' num2str(stepPerEpisode)])
                    break;
                end
            end
        end
        totalStepsPerRep = totalStepsPerRep-1; % Compensamos el que se incrementó de más
        
        if sum(sum(data.s_a_sNext(data.index:end,:))) == 0 % Si aún no se ha llenado el buffer
            real_buffer = data.index-1;
            s_a_sNext = data.s_a_sNext(1:real_buffer,:);
        else
            real_buffer = MINIBATCH_DIM;
            s_a_sNext = data.s_a_sNext(randperm(BUFF_DIM, MINIBATCH_DIM),:);
        end
        for i = 1:real_buffer
            % Recover saved episodes
            [s_t, s_t1, reward, step, s_a_index, terminal] = recoverSavedEpisode(s_a_sNext, A, i);
            
            % Policy (or d) update
            d(s_a_index) = d(s_a_index) + alphaD*(reward + gamma*v(s_t1) - v(s_t));
            d_orig = d;
            d(d<0)=0; % Projection of d over positives
            
            % Normalize d
            d_norm = getPolicyVectorFromD( d, env );
            
            % Evaluación de si el episodio ha terminado o no para guardar el error en la policy (save policy error)
            if any(isnan(d_norm))
                % Fix de la d original (que podía tener números negativos)
                d_orig(isnan(d_norm) & d_orig<0) = abs(d_orig(isnan(d_norm) & d_orig<0));
                d = d_orig;
                d(d<0)=0; % Projection of d over positives
                d_norm = getPolicyVectorFromD( d, env );
                % disp('nan!')
                % not_error = false;
                % break;
            end
            if terminal || step == maxNumStepsPerEpisode-1 % Si el estado siguiente es el terminal
                episodeCountD = episodeCountD + 1;
                % Calculate norm-2 of policy error
                % errorD(exp, episodeCountD) = norm(abs(d_norm(A+1:end-A) - d_opt_norm(A+1:end-A)),2);
                % d_norm_acumulada(:,episodeCountD, exp) = d_norm;
                % d_acumulada(:,episodeCountD, exp) = d_orig;
                % [d_norm; episodeCountD]
                % reshape(d_norm', [2 21])'
%                 [~, policy_lineal] = max(reshape(d_norm,[A S])',[],2);
%                 checkCliffResults(env, policy_lineal)
            end
        end
    end
    [~, ~, G_eps0] = RL_core(env, d_norm, v, 0, alphaTD, alphaD, numRep, numEpisodes, maxNumStepsPerEpisode, mult, BUFF_DIM, MINIBATCH_DIM);
    G_eps0_acumulada(exp,:) = G_eps0;
end

Vs_mean = mean(squeeze(Vs_acumulada(:,end,:)),2);
% [Qsa_mean, Qsa_acumulada] = getStateActionValueFunction(Vs_acumulada, env);

% % % REPRESENTACIÓN DE RESULTADOS:
% Resultados obtenidos para numExperiments experimentos de numRep*numEpisodes episodios
Gmean = mean(G,1); % Media de los experimentos realizados
Gmean_eps0 = mean(G_eps0_acumulada,1); % Media de los experimentos realizados cuando epsilon = 0
figure, hold on
plot((1:numRep*numEpisodes), Gmean, 'b', 'LineWidth', 2)
% plot((1:numRep*numEpisodes), Gmean_eps0, 'LineWidth', 2)
plot((1:numRep*numEpisodes), mean(Gmean_eps0)*ones(size(Gmean_eps0)),'--r', 'LineWidth', 2)
hold off, title(['Return per episode (always starting at s=' ,num2str(env.initState), ')'])
xlabel('Episode'), ylabel('G')
legend({['E_{exp}[G] || \epsilon=' num2str(epsilon)], ['E_{epi}[E_{exp}[G]]=' num2str(mean(Gmean_eps0)) ' || \epsilon=0']},'Location','southeast','FontSize',12)

% aux1 = sum(Qsa_acumulada);
% aux2 = squeeze(aux1)';
% sumQ_mean = mean(aux2);
% figure, plot((1:numRep*numEpisodes), sumQ_mean, 'LineWidth', 2)
% title('\Sigmaq(s,a) for each episode')
% xlabel('Episode'), ylabel('\Sigmaq(s,a)')
%
% aux1 = sum(Vs_acumulada);
% aux2 = squeeze(aux1)';
% sumV_mean = mean(aux2);
% figure, plot((1:numRep*numEpisodes), sumV_mean, 'LineWidth', 2)
% title('\Sigmav(s) for each episode')
% xlabel('Episode'), ylabel('\Sigmav(s)')
%
% aux1 = sum((Qsa_acumulada(game.N_actions+1:end-game.N_actions,:,:)-q_opt(game.N_actions+1:end-game.N_actions,:)).^2 , 1);
% aux2 = squeeze(aux1)';
% mse_q = sqrt(mean(aux2));
% figure, plot((1:numRep*numEpisodes), mse_q, 'LineWidth', 2)
% title('Mean-squared error of q(s,a) for each episode')
% xlabel('Episode'), ylabel('MSE q(s,a)')
%
% aux1 = sum((Vs_acumulada(2:end-1,:,:)-v_opt(2:end-1,:)).^2 , 1);
% aux2 = squeeze(aux1)';
% mse_v = sqrt(mean(aux2));
% figure, plot((1:numRep*numEpisodes), mse_v, 'LineWidth', 2)
% title('Mean-squared error of v(s) for each episode')
% xlabel('Episode'), ylabel('MSE v(s)')

% errorD_mean = mean(errorD,1); % Media de los experimentos realizados
% figure, plot((1:numRep*numEpisodes), errorD_mean, 'LineWidth', 2)
% title('Policy error')
% xlabel('Episode'), ylabel('d-d_{opt}')