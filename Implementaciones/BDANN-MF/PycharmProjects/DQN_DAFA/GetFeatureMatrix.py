import numpy as np
from ChainWalkSetUp import GetPolyFeatures


def GetFeatureMatrix(S, N_features):
    phi = np.zeros((S, N_features))
    for s in range(0, S):
        phi[s, :] = GetPolyFeatures(s)
    return phi
