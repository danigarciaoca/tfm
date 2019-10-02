import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from ChainWalkSetUp import ChainWalkSetUp, GetPolyFeatures
from GetPolicyVector import GetPolicyVector
from GetFeatureMatrix import GetFeatureMatrix
from GetPolicyVectorFromD import GetPolicyVectorFromD
from E_greedy import E_greedy
from GetNextState import GetNextState
from GetStochParam import GetStochParam
from RecoverSavedEpisode import RecoverSavedEpisode

# # # # ENTORNO:
game = ChainWalkSetUp()

A = np.eye(game.N_states)
B = np.array([[1], [0]])
mult1 = np.kron(A, B)
B = np.array([[0], [1]])
mult2 = np.kron(A, B)

# # # # AGENTE:
numExperiments = 50  # Número de experimentos de numRep*numEpisodes episodios
numRep = 10  # Número de repeticiones de cada set de episodios
numEpisodes = 10  # Número de episodios de cada repeticion
maxNumStepsPerEpisode = 20  # Número máximo de pasos en cada episodio
G = np.zeros((numExperiments, numRep * numEpisodes))  # Reward por episodio
epsilon = 0.1  # e-greedy value (entre 0.05 y 0.2)
alphaD = 0.1  # Stepsize para la iteración de la variable dual d
alphaTD = 0.4  # Stepsize para la iteración de la variable primal v

S = game.N_states  # Número de estados
A = game.N_actions  # Número de acciones
N = game.N_features  # state's feature vector length
mu = game.mu  # Distribución inicial de probabilida de los estados
P = game.P  # Matriz de transiciones
R = game.R  # Vector de rewards
gamma = game.gamma  # Discount rate/factor
terminal = False  # flag used when terminal state reached

# Optimum values
v_opt = np.dot(np.linalg.inv(np.eye(S) - game.gamma * np.dot(game.pi_opt, P)), np.dot(game.pi_opt, R))
q_opt = np.dot(np.linalg.inv(np.eye(S * A) - game.gamma * np.dot(P, game.pi_opt)), R)
# Valor óptimo de d (política en forma vector)
d_opt_norm = GetPolicyVector(q_opt, game)  # policy óptima

# Variable que acumulará la funcion V y el error en la política al final de cada episodio
Vs_acumulada = np.empty((game.N_states, numRep * numEpisodes, numExperiments)) * np.nan
errorD = np.empty((numExperiments, numRep * numEpisodes)) * np.nan
d_norm_acumulada = np.empty((game.N_states * game.N_actions, numRep * numEpisodes, numExperiments)) * np.nan

phi = GetFeatureMatrix(S, N)

for exp in range(0, numExperiments):
    not_error = True
    # Inicializamos D
    d = np.random.rand(S * A, 1)  # d >= 0
    d = d / sum(d)  # sum(d) = 1

    episodeCountV = 0  # episodeCount counts the number of episodes taken in all numRep (para el bucle de V)
    episodeCountD = 0  # episodeCount counts the number of episodes taken in all numRep (para el bucle de D)
    for k in range(0, numRep):
        s_a_sNext = np.empty((numEpisodes * maxNumStepsPerEpisode, 5)) * np.nan  # 5 parameter to accumulate. Vector que almacena la secuencia de estados recorridos en un episodio
        totalStepsPerRep = 0  # totalStepsPerRep counts the number of steps taken in ALL numEpi episodes simulated in ONE numRep
        phi_t = np.empty((numEpisodes * maxNumStepsPerEpisode, N)) * np.nan
        phi_t1 = np.empty((numEpisodes * maxNumStepsPerEpisode, N)) * np.nan
        reward_t = np.empty((numEpisodes * maxNumStepsPerEpisode, 1)) * np.nan

        for n in range(0, numEpisodes):  # This loop sets the update frequency of v (every numEpi episodes)
            currentState = game.initial_state  # empezar en el de la izquierda
            terminal = False  # true when episode finish, false otherwise
            stepPerEpisode = 0  # stepPerEpisode counts the number of steps taken in ONE of the numEpi episodes simulated

            # Normalize d
            d_norm = GetPolicyVectorFromD(d, game)
            # Get policy matrix
            policy_by_action = np.reshape(d_norm, (S, A))
            policy_matrix = np.dot(np.diag(policy_by_action[:, 0]), mult1.transpose()) + np.dot(np.diag(policy_by_action[:, 1]), mult2.transpose())

            while not terminal:
                # Escogemos A (currentAction) de S (currentState) según la e-greedy policy.
                currentAction = E_greedy(d_norm, epsilon, currentState, A)

                # Tomamos la acción a (currentAction), observamos la recompensa
                # r (reward) y el siguiente estado s' (nextState).
                [nextState, reward] = GetNextState(game, currentState, currentAction)
                # G(exp, (k-1)*numEpisodes+n) = reward+game.gamma*G(exp,(k-1)*numEpisodes+n) # return following the initial state (si el estado inicial fuese aleatorio, habría que crear una G para cada vez que e pasa por s por primera vez [p. 105 de Sutton])
                G[exp, k * numEpisodes + n] = (game.gamma ** stepPerEpisode) * reward + G[exp, k * numEpisodes + n]

                # Update de phi
                # LEAST-SQUARES TEMPORAL DIFFERENCE (I)
                phi_t[totalStepsPerRep, :] = GetPolyFeatures(currentState)
                phi_t1[totalStepsPerRep, :] = GetPolyFeatures(nextState)
                reward_t[totalStepsPerRep, :] = reward

                # EXACTA (APROXIMACIÓN DE FUNCIONES)
                # theta = GetExactParamOpt( policy_matrix*R, policy_matrix*P, phi, gamma);
                # v = phi*theta;

                # Almacenamos las transiciones del episodio
                s_a_sNext[totalStepsPerRep, :] = [currentState, currentAction, nextState, reward, stepPerEpisode]
                # s_a_sNext = np.vstack([s_a_sNext, [currentState, currentAction, nextState, reward, stepPerEpisode]])

                # Actualizamos valores
                stepPerEpisode += 1
                totalStepsPerRep += 1
                currentState = nextState

                # Evaluación de si el episodio ha terminado o no
                if stepPerEpisode == maxNumStepsPerEpisode:  # Si el estado actual es el terminal
                    terminal = True
                    episodeCountV += 1

        totalStepsPerRep = totalStepsPerRep  # En Matlab restabamos 1 aquí para compensar la última iteración. En python, dado que totalStepsPerRep se usará en el bucle for de alante y este va de 0 a totalStepsPerRep-1, no nos hace falta compensar
        # LEAST-SQUARES TEMPORAL DIFFERENCE (II)
        theta = GetStochParam(reward_t, phi_t, phi_t1, gamma)
        v = np.dot(phi, theta)

        for i in range(0, totalStepsPerRep):
            # Recover saved episodes
            [currentState, nextState, reward, stepPerEpisode, s_a_index] = RecoverSavedEpisode(s_a_sNext, game.N_actions, i)

            # Policy (or d) update
            # d(s_a_index) = d(s_a_index) + alphaD*(reward + game.gamma*P(s_a_index,:)*phi*theta - phi(currentState,:)*theta);
            d[s_a_index] = d[s_a_index] + alphaD * (reward + game.gamma * v[nextState] - v[currentState])
            d_orig = deepcopy(d)
            d[d < 0] = 0  # Projection of d over positives

            # Normalize d
            d_norm = GetPolicyVectorFromD(d, game)

            # Evaluación de si el episodio ha terminado o no para guardar el error en la policy (save policy error)
            if any(np.isnan(d_norm)):
                # Fix de la d original (que podía tener números negativos)
                d_orig[np.logical_and(np.isnan(d_norm), d_orig.transpose()[0] < 0)] = abs(d_orig[np.logical_and(np.isnan(d_norm), d_orig.transpose()[0] < 0)])
                d = deepcopy(d_orig)
                d[d < 0] = 0  # Projection of d over positives
                d_norm = GetPolicyVectorFromD(d, game)

            if stepPerEpisode == maxNumStepsPerEpisode - 1:  # Si el estado siguiente es el terminal
                # Calculate norm-2 of policy error
                errorD[exp, episodeCountD] = np.linalg.norm(abs(d_norm - d_opt_norm), 2)
                d_norm_acumulada[:, episodeCountD, exp] = d_norm
                #print("d normalizado: \n", d_norm.transpose())
                episodeCountD += 1

# # # # REPRESENTACIÓN DE RESULTADOS:
# Resultados obtenidos para numExperiments experimentos de numRep*numEpisodes episodios
Gmean = np.mean(G, 0)  # Media de los experimentos realizados
plt.figure(), plt.plot(np.arange(1, numRep * numEpisodes + 1), Gmean, linewidth=2)
plt.title("Reward per episode (always starting at s=%s)" % game.initial_state)
plt.xlabel('Episode'), plt.ylabel('G')
#plt.show()

errorD_mean = np.mean(errorD, 0)  # Media de los experimentos realizados
plt.figure(), plt.plot(np.arange(1, numRep * numEpisodes + 1), errorD_mean, linewidth=2)
plt.title('Policy error')
plt.xlabel('Episode'), plt.ylabel('d-d_{opt}')
plt.show()
