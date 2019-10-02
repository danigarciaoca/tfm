import numpy as np

GetPolyFeatures = lambda s: np.array([1, (s + 1), (s + 1) ** 2])


class ChainWalkSetUp:
    # Expected reward per state
    Rs = np.array([0, 1, 1, 0])
    # Expected reward per state-action (s,a)
    R = np.array([0, 1, 0, 1, 1, 0, 1, 0])

    # Probability transition matrix
    P = np.array([[0.9, 0.1, 0, 0],
                  [0.1, 0.9, 0, 0],
                  [0.9, 0, 0.1, 0],
                  [0.1, 0, 0.9, 0],
                  [0, 0.9, 0, 0.1],
                  [0, 0.1, 0, 0.9],
                  [0, 0, 0.9, 0.1],
                  [0, 0, 0.1, 0.9]])

    N_states = 4  # number of states
    initial_state = 1  # initial state (1 in Python is equivalent to 2 in Matlab; Python start indexing by 0
    final_state = np.array([0, N_states-1])
    # initial states distribution
    mu = ((1 / N_states) / (N_states - 1)) * np.ones((N_states, 1))
    mu[initial_state] = 1 - (1 / N_states)

    N_actions = 2  # number of actions
    gamma = .9  # discount factor

    # Features
    GetStateFeatures = GetPolyFeatures  # state features
    N_features = 3  # number of state features
    M = N_features * N_actions  # number of state-action features

    # Optimal policy
    pi_opt = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0]])

    # Auxiliar matrix for constructing policy vector d
    A = np.eye(N_states)
    B = np.array([[1], [1]])
    duplicar = np.kron(A, B)

# def __init__(self, N_states, N_actions, P, R, Rs, gamma, GetStateFeatures, N_features, initial_state, M, final_state, mu, pi_opt, duplicar):
#        self.N_states = N_states
#        self.N_actions = N_actions
#       self.P = P
#       self.R = R
#        self.Rs = Rs
#        self.gamma = gamma
#        self.GetStateFeatures = GetStateFeatures
#        self.N_features = N_features
#        self.initial_state = initial_state
#        self.M = M
#        self.final_state = final_state
#        self.mu = mu
#        self.pi_opt = pi_opt
#        self.duplicar = duplicar
