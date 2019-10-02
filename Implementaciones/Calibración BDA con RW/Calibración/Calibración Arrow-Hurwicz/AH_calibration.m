clear all, clc

% % % % % ENTORNO:
% % save('juego_test.mat','game')
% load juego_test.mat
% % Policy �ptima
% [v_opt, q_opt, d_norm] = getOptimalPolicy(game);

% % % % ENTORNO:
N_states = 7;
transitionType = 'det';
% transitionType = 'rand';
rewardType = 'det';
% rewardType = 'rand';
game = random_walk_set_up_mod(N_states, transitionType, rewardType);
% Valores �ptimos
[v_opt, q_opt, d_norm] = getRandomPolicy(game);
load d_aleatoria.mat % carga la d con la que se gener� d_norm
            
A = eye(game.N_states);
B = [1; 0];
mult1 = kron(A,B);
B = [0; 1];
mult2 = kron(A,B);

% % % % AGENTE:
numExperiments = 2; % N�mero de experimentos de numRep*numEpisodes episodios
numEpisodes = 10000; % N�mero de episodios de cada repeticion
maxNumStepsPerEpisode = 500; % N�mero m�ximo de pasos en cada episodio
G = zeros(numExperiments, numEpisodes); % Reward por episodio
epsilon = 0; % e-greedy value (entre 0.05 y 0.2)
alphaV = 0.0001; % Stepsize para la iteraci�n de la variable primal v

S = game.N_states; % N�mero de estados
A = game.N_actions; % N�mero de acciones
game.mu(:) = (1/S)/(S-1); game.mu(4) = 1-(1/S); mu = game.mu; % Distribuci�n inicial de probabilida de los estados
P = game.P; % Matriz de transiciones
R = game.R; % Vector de rewards
terminal = false; % flag used when terminal state reached

% Variable que acumular� la funcion V y el error en la pol�tica al final de cada episodio
Vs_acumulada = nan(game.N_states, numEpisodes, numExperiments);

for exp = 1:numExperiments
    % Inicializamos V
%     v = rand(S,1); v(1) = 0; v(end) = 0;
    v = zeros(S,1);
    
    episodeCountV = 0; % episodeCount counts the number of episodes taken in all numRep (para el bucle de V)
    episodeCountD = 0; % episodeCount counts the number of episodes taken in all numRep (para el bucle de D)
    
    totalStepsPerExp = 1; % totalStepsPerRep counts the number of steps taken in ALL numEpi episodes simulated in ONE numRep
    
    for n = 1:numEpisodes % This loop sets the update frequency of v (every numEpi episodes)
        currentState = game.centralState; % estado inicial el central % currentState = discretesample(mu, 1);
        terminal = false; % true when episode finish, false otherwise
        stepPerEpisode = 0; % stepPerEpisode counts the number of steps taken in ONE of the numEpi episodes simulated
        
        while ~terminal
            % Escogemos A (currentAction) de S (currentState) seg�n la e-greedy policy.
            currentAction = e_greedy(d_norm, epsilon, currentState, game.N_actions);
            
            % Tomamos la acci�n a (currentAction), observamos la recompensa
            % r (reward) y el siguiente estado s' (nextState).
            [nextState, realCurrentAction, reward] = getNextState(game, currentState, currentAction);
            G(exp,n) = (game.gamma^stepPerEpisode)*reward+G(exp,n); % return following the initial state (si el estado inicial fuese aleatorio, habr�a que crear una G para cada vez que e pasa por s por primera vez [p. 105 de Sutton])
            
            % Update de v(s)
            % GRADIENTE ESTOC�STICO (Arrow-Hurwicz): policy evaluation
            s_a = ((currentState-1)*A)+currentAction;
            vGrad_vect = (1-game.gamma)*mu + (game.gamma*P' - game.marg)*d;
            v = v - alphaV*vGrad_vect;
            % v(nextState) = v(nextState) - alphaV*((1-game.gamma)*mu(nextState) +  game.gamma*d(s_a) - sum(d(((nextState-1)*A)+1:nextState*A))); % policy evaluation
            
            % Actualizamos valores
            stepPerEpisode = stepPerEpisode + 1;
            totalStepsPerExp = totalStepsPerExp + 1;
            currentState = nextState;
            
            % Evaluaci�n de si el episodio ha terminado o no
            if sum(currentState == game.finalState) == 1 || stepPerEpisode == maxNumStepsPerEpisode % Si el estado actual es el terminal
                terminal = true;
                episodeCountV = episodeCountV + 1;
                Vs_acumulada(:, episodeCountV, exp) = v;
                [v(game.centralState) v_opt(game.centralState);exp exp]
            end
        end
    end
    totalStepsPerExp = totalStepsPerExp-1; % Compensamos el que se increment� de m�s
end
%%
Vs_mean = mean(squeeze(Vs_acumulada(:,end,:)),2);
[Qsa_mean, Qsa_acumulada] = getStateActionValueFunction(Vs_acumulada, game);

% % % REPRESENTACI�N DE RESULTADOS:
% Resultados obtenidos para numExperiments experimentos de numRep*numEpisodes episodios
Gmean = mean(G,1); % Media de los experimentos realizados
figure, plot((1:numEpisodes), Gmean, 'LineWidth', 2), ylim([min([0,Gmean]) max(Gmean)+10])
title(['Reward per episode (always starting at s=' ,num2str(game.centralState), ')'])
xlabel('Episode'), ylabel('G')

aux1 = sum((Vs_acumulada(2:end-1,:,:)-v_opt(2:end-1,:)).^2 , 1);
aux2 = sqrt(squeeze(aux1)');
denom1 = sum((Vs_acumulada(2:end-1,:,:)).^2 , 1);
denom2 = sqrt(squeeze(denom1)');
mse_v = mean(aux2./denom2);
% plot((1:numEpisodes),aux2./denom2); % con esto vemos si hay alg�n experimento "desviado"

figure, plot((1:numEpisodes), mse_v, 'LineWidth', 2)
title('Mean-squared error of v(s) for each episode')
xlabel('Episode'), ylabel('MSE v(s)')

% Detectamos en qu� episodio converge
umbral_convergencia = 0.01;
[ diffV_norm, epi_convergence] = getConvergence(Vs_acumulada, umbral_convergencia);
figure, plot(mean(diffV_norm))
epi_convergence_mean = mean(epi_convergence)

% Sabiendo el episodio en el que converge, comprobamos si cumple el
% criterio 2 de convergencia (el mse en el estado central menor que 0.2)
aux1 = sum((Vs_acumulada(game.centralState,:,:)-v_opt(game.centralState,:)).^2 , 1);
aux2 = sqrt(squeeze(aux1)');
denom1 = sum((Vs_acumulada(game.centralState,:,:)).^2 , 1);
denom2 = sqrt(squeeze(denom1)');
mse_v_convergence = mean(aux2./denom2);
figure, plot([1:numEpisodes], mse_v_convergence)
mse_v_convergence_exact = mse_v_convergence(round(epi_convergence_mean))
% figure, plot(100*diffV_norm(1,:))

% save(['alhpaTD = ',num2str(alphaTD),'. Policy optima. epsilon=0.2.mat'], 'G', 'Vs_acumulada', 'game', 'd_norm', 'epi_convergence_mean', 'mse_v_convergence_exact')