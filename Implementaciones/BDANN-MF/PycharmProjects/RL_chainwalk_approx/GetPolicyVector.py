import numpy as np
import random


def GetPolicyVector(Qsa_lineal_acumulada, game):
    if len(Qsa_lineal_acumulada.shape) == 1:
        # expand dimension in order to make it suitable for use of general cases (more than one experiment)
        Qsa_lineal_acumulada = np.expand_dims(Qsa_lineal_acumulada, axis=1)
        Qsa_lineal_acumulada = np.expand_dims(Qsa_lineal_acumulada, axis=2)

    policy_vector = np.zeros(
        (game.N_actions * game.N_states, np.size(Qsa_lineal_acumulada, 1), np.size(Qsa_lineal_acumulada, 2)))
    for i in range(0, np.size(Qsa_lineal_acumulada, 2)):  # choose experiment
        for j in range(0, np.size(Qsa_lineal_acumulada, 1)):  # choose episode
            Qsa = np.reshape(Qsa_lineal_acumulada[:, j, i], (game.N_states, game.N_actions))
            Qsa = np.round(Qsa * 10000000000) / 10000000000  # Python's precision is too much. Round to the tenth decimal (necessary for finding act_max)
            Qsa_optima = np.max(Qsa, axis=1)

            for st in range(0, game.N_states):
                act_max = np.where((Qsa_optima[st] == Qsa[st, :]) == 1)[0]
                if np.size(act_max) > 1:  # Si hay más de una acción que maximiza, escoger una aleatoriamente
                    act_max = random.choice(act_max);
                policy_vector[(st * game.N_actions) + act_max, j, i] = 1;

    return policy_vector.transpose()[0][0]
