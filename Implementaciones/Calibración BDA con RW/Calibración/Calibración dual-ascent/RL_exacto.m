clear all, clc

% % % % ENTORNO:
% save('juego_test.mat','game')
load juego_test.mat

A = eye(game.N_states);
B = [1; 0];
mult1 = kron(A,B);
B = [0; 1];
mult2 = kron(A,B);

% % % % AGENTE:
numExperiments = 100; % Número de experimentos de numRep*numEpisodes episodios
numRep = 40; % Número de repeticiones de cada set de episodios
numEpisodes = 1; % Número de episodios de cada repeticion
maxNumStepsPerEpisode = 500; % Número máximo de pasos en cada episodio
G = zeros(numExperiments, numRep*numEpisodes); % Reward por episodio
epsilon = 0; % e-greedy value (entre 0.05 y 0.2)
alphaD = 0.1; % Stepsize para la iteración de la variable dual d
% [0.1 0.5 1 1.5 2 3]
alphaTD = 0.4; % Stepsize para la iteración de la variable primal v
% con alphaTD = 0.5 converge más rápido la return

S = game.N_states; % Número de estados
A = game.N_actions; % Número de acciones
game.mu(:) = (1/S)/(S-1); game.mu(2) = 1-(1/S); mu = game.mu; % Distribución inicial de probabilida de los estados
P = game.P; % Matriz de transiciones
R = game.R; % Vector de rewards
terminal = false; % flag used when terminal state reached

% Optimum values
v_opt = inv(eye(game.N_states)-game.gamma*game.pi_opt*game.P)*game.pi_opt*game.R;
q_opt = inv(eye(game.N_states*game.N_actions)-game.gamma*game.P*game.pi_opt)*game.R;
% Valor óptimo de d (política en forma vector)
d_opt_norm = getPolicyVector(q_opt, game); % policy óptima

% Variable que acumulará la funcion V y el error en la política al final de cada episodio
Vs_acumulada = nan(game.N_states, numRep*numEpisodes, numExperiments);
errorD = nan(numExperiments, numRep*numEpisodes);
d_norm_acumulada = nan(game.N_states*game.N_actions, numRep*numEpisodes, numExperiments);
% d_acumulada = nan(game.N_states*game.N_actions, numRep*numEpisodes, numExperiments);

for exp = 1:numExperiments
    not_error = true;
    % Inicializamos D y V
    d = rand(S*A,1);
    v = rand(S,1);
    v(1) = 0; v(end) = 0;
    
    exp
    
    episodeCountV = 0; % episodeCount counts the number of episodes taken in all numRep (para el bucle de V)
    episodeCountD = 0; % episodeCount counts the number of episodes taken in all numRep (para el bucle de D)
    for k = 1:numRep
        s_a_sNext = []; % Vector que almacena la secuencia de estados recorridos en un episodio
        totalStepsPerRep = 1; % totalStepsPerRep counts the number of steps taken in ALL numEpi episodes simulated in ONE numRep
        
        % currentState = game.centralState; % estado inicial el central
        % currentState = randi([2 game.centralState]);
        currentState = game.centralState; % empezar en el de la izquierda
        terminal = false; % true when episode finish, false otherwise
        stepPerEpisode = 0; % stepPerEpisode counts the number of steps taken in ONE of the numEpi episodes simulated
        
        % Normalize d
        d_norm = getPolicyVectorFromD( d, game );
        % Get policy matrix
        policy_by_action = reshape(d_norm', [A,S])';
        policy_matrix = diag(policy_by_action(:,1))*mult1' + diag(policy_by_action(:,2))*mult2';
        

        % Update de v(s)
        v = (inv(eye(S)-game.gamma*policy_matrix*P))*policy_matrix*R;

        % Evaluación de si el episodio ha terminado o no
        
        terminal = true;
        episodeCountV = episodeCountV + 1;
        Vs_acumulada(:, episodeCountV, exp) = v;
        % disp(['Fin' num2str(n) ' y ' num2str(stepPerEpisode)])
        

        % Policy (or d) update
        d = d + alphaD*(R + game.gamma*P*v - game.duplicar*v);
        %d(s_a_index) = d(s_a_index) + alphaD*(reward + game.gamma*game.P(s_a_index,:)*v - v(currentState));
        d_orig = d;
        d(d<0)=0; % Projection of d over positives
        
        % Normalize d
        d_norm = getPolicyVectorFromD( d, game );
        
        % Evaluación de si el episodio ha terminado o no para guardar el error en la policy (save policy error)
        if any(isnan(d_norm))
            % Fix de la d original (que podía tener números negativos)
            d_orig(isnan(d_norm) & d_orig<0) = abs(d_orig(isnan(d_norm) & d_orig<0));
            d = d_orig;
            d(d<0)=0; % Projection of d over positives
            d_norm = getPolicyVectorFromD( d, game );
            % disp('nan!')
            % not_error = false;
            % break;
        end
        
        episodeCountD = episodeCountD + 1;
        % Calculate norm-2 of policy error
        errorD(exp, episodeCountD) = norm(abs(d_norm(game.N_actions+1:end-game.N_actions) - d_opt_norm(game.N_actions+1:end-game.N_actions)),2);
        d_norm_acumulada(:,episodeCountD, exp) = d_norm;
        d_acumulada(:,episodeCountD, exp) = d_orig;
        % [d_norm; episodeCountD]
        % reshape(d_norm', [2 21])'
        
        
    end
end

% Apaño para cuando ha aparecido un NaN. Replicamos el último valor bueno
% para el resto de episodios
% for i = 1:size(errorD,1)
%     ind_nan = find(isnan(errorD(i,:))==1);
%     if ~isempty(ind_nan)
%         errorD(i,ind_nan) = errorD(i,ind_nan(1)-1);
%     end
% end
% Vs_mean = mean(squeeze(Vs_acumulada(:,end,:)),2);
% [Qsa_mean, Qsa_acumulada] = getStateActionValueFunction(Vs_acumulada, game);

% % % % REPRESENTACIÓN DE RESULTADOS:
% % Resultados obtenidos para numExperiments experimentos de numRep*numEpisodes episodios
% Gmean = mean(G,1); % Media de los experimentos realizados
% figure, plot((1:numRep*numEpisodes), Gmean, 'LineWidth', 2)
% title(['Reward per episode (always starting at s=' ,num2str(game.centralState), ')'])
% xlabel('Episode'), ylabel('G')

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

tam_fuent_title = 14;
tam_fuente_label = 14;
tam_fuente_legend = 12;
errorD_mean = mean(errorD,1); % Media de los experimentos realizados
figure, plot((1:numRep*numEpisodes), errorD_mean, 'LineWidth', 2)
title('Error en la política','FontSize',tam_fuent_title)
xlabel('Iteración','FontSize',tam_fuente_label), ylabel('${\left\Vert \pi_{d_{k}}-\pi^{*}\right\Vert _{2}^{2}}$','FontSize',tam_fuente_label,'Interpreter','LaTex')
h_legend=legend(['\alpha_D=' num2str(alphaD)]);
set(h_legend,'FontSize',tam_fuente_legend);