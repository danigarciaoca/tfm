import numpy as np


def GetNextState(game, currentState, action):
    # GETNEXTSTATE Devuelve el siguiente estado y la recompensa instantánea.
    #   En base al estado actual (currentState) y a la acción tomada (action),
    #   devuelve el siguiente estado y la recompensa asociada a dicha
    #   transición, así como la verdadera acción tomada en función de la matriz
    #   de transición P

    nextState = np.random.choice(game.N_states, 1, p=game.P[currentState * game.N_actions + action, :])[0]
    reward = game.Rs[nextState]

    return (nextState, reward)
