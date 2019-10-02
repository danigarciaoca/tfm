import numpy as np
import matplotlib.pyplot as plt
import gym

from copy import deepcopy
from ChainWalkSetUp import ChainWalkSetUp, GetPolyFeatures
from GetPolicyVector import GetPolicyVector
from GetPolicyVectorFromD import GetPolicyVectorFromD
from E_greedy import E_greedy
from GetNextState import GetNextState
from RecoverSavedEpisode import RecoverSavedEpisode
from BufferCircular import BufferCircular
from NeuralNetworkOperations import MyNeuralNetwork

# # # # ENTORNO:
game = ChainWalkSetUp()

# # # # AGENTE:
numExperiments = 50  # Número de experimentos de numRep*numEpisodes episodios
numRep = 20  # Número de repeticiones de cada set de episodios
numEpisodes = 100  # Número de episodios de cada repeticion
maxNumStepsPerEpisode = 50  # Número máximo de pasos en cada episodio
G = np.zeros((numExperiments, numRep * numEpisodes))  # Reward por episodio
epsilon = 0  # e-greedy value (entre 0.05 y 0.2)
alphaD = 0.1  # Stepsize para la iteración de la variable dual d

S = game.N_states  # Número de estados
A = game.N_actions  # Número de acciones
P = game.P  # Matriz de transiciones
R = game.R  # Vector de rewards
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

NN1_pred = MyNeuralNetwork()
NN2_fit = MyNeuralNetwork()
trainingset = BufferCircular(numEpisodes * maxNumStepsPerEpisode,2) # 2 is de number of columns to store: x_train (1 col, states) and y_train (1 col, predicted bellman eq. r+gamma*prediction(NN1))

for exp in range(0, numExperiments):
    # Initialize vector 'd' representing random initial policy
    d = np.random.rand(S * A, 1)  # d >= 0
    d = d / sum(d)  # sum(d) = 1

    # Initialize NN theta parameters
    NN1_pred.init_NN_params()
    NN2_fit.init_NN_params()

    # Initialize training set
    trainingset.restore_buffer()

    # Initialize debugging episode counter
    episodeCountV = 0  # episodeCount counts the number of episodes taken in all numRep (for V loop)
    episodeCountD = 0  # episodeCount counts the number of episodes taken in all numRep (for D loop)
    for k in range(0, numRep):
        s_a_sNext = np.empty((numEpisodes * maxNumStepsPerEpisode, 5)) * np.nan  # 5 parameter to accumulate. Vector que almacena la secuencia de estados recorridos en un episodio
        totalStepsPerRep = 0  # totalStepsPerRep counts the number of steps taken in ALL numEpi episodes simulated in ONE numRep

        for n in range(0, numEpisodes):  # This loop sets the update frequency of v (every numEpi episodes)
            currentState = game.initial_state  # empezar en el de la izquierda
            terminal = False  # true when episode finish, false otherwise
            stepPerEpisode = 0  # stepPerEpisode counts the number of steps taken in ONE of the numEpi episodes simulated

            # Normalize d
            # d_norm = GetPolicyVectorFromD(d, game)
            d_norm = d_opt_norm
            while not terminal:
                # Escogemos A (currentAction) de S (currentState) según la e-greedy policy.
                currentAction = E_greedy(d_norm, epsilon, currentState, A)

                # Tomamos la acción a (currentAction), observamos la recompensa r (reward) y el siguiente estado s' (nextState).
                [nextState, reward] = GetNextState(game, currentState, currentAction)

                # Obtain return
                G[exp, k * numEpisodes + n] = (game.gamma ** stepPerEpisode) * reward + G[exp, k * numEpisodes + n] # return following the initial state (si el estado inicial fuese aleatorio, habría que crear una G para cada vez que e pasa por s por primera vez [p. 105 de Sutton])
                # Obtain value function prediction (NN 1)
                pred_nn1 = NN1_pred.predict(np.array([currentState]))

                # Almacenamos las transiciones del episodio y los datos del set de entrenamiento
                s_a_sNext[totalStepsPerRep, :] = [currentState, currentAction, nextState, reward, stepPerEpisode]
                trainingset.add((currentState, reward + game.gamma*pred_nn1))
                # s_a_sNext = np.vstack([s_a_sNext, [currentState, currentAction, nextState, reward, stepPerEpisode]])

                # Actualizamos valores
                stepPerEpisode += 1
                totalStepsPerRep += 1
                currentState = nextState

                # Evaluación de si el episodio ha terminado o no
                if stepPerEpisode == maxNumStepsPerEpisode:  # Si el estado actual es el terminal
                    terminal = True
                    episodeCountV += 1

        NN2_fit.fit(trainingset.data[:,0], trainingset.data[:,1], epochs=200, batch_size=128, verbose=2)
        updated_weights = NN2_fit.get_weights()
        NN1_pred.set_weights(updated_weights)
        v = NN1_pred.predict(np.array([0,1,2,3]))
        print(v)
        print(k)
        # Theta parameters update (fit NN2 and
        # fit NN2
        # theta = ...
        # update theta in NN1 with NN2
        # predict V with NN1 to use in d loop

        '''for i in range(0, totalStepsPerRep):
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
'''
