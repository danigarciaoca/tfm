import numpy as np


def GetPolicyVectorFromD(d, game):
    # GETPOLICYVECTORFROMD Obtains the current policy from vector d of joint
    # probability distribution

    sumDoverA_aux = np.sum(np.reshape(d, (game.N_states, game.N_actions)).transpose(), 0)
    sumDoverA = np.sum(np.dot(game.duplicar, np.diag(sumDoverA_aux)), 1)
    policy = d.transpose() / sumDoverA

    return policy[0]

#Comprobación del RuntimeWarning
#if any(sumDoverA==0):
#        print(sumDoverA)
