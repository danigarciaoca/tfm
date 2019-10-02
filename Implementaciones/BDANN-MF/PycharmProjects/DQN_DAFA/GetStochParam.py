import numpy as np


def GetStochParam(reward_t, phi_t, phi_t1, gamma):
    # GETEXACTPARAMOPT This function returns the exact and optimum calculation
    #   of vector of state parameters theta, given the reward vector R, transition
    #   probability matrix P and features/basis functions matrix phi

    theta_opt = np.dot(np.dot(np.linalg.inv(np.dot(phi_t.transpose(), phi_t) - gamma * np.dot(phi_t.transpose(), phi_t1)), phi_t.transpose()), reward_t)

    return theta_opt
