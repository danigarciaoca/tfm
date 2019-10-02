def RecoverSavedEpisode(s_a_sNext, A, i):
    # RECOVERSAVEDEPISODE Recover s,a,s' tuple from saved episode (or hsitory of
    # episodes).

    currentState = s_a_sNext[i, 0]
    currentAction = s_a_sNext[i, 1]
    nextState = s_a_sNext[i, 2]
    reward = s_a_sNext[i, 3]
    stepPerEpisode = s_a_sNext[i, 4]
    s_a = (currentState * A) + currentAction

    return (int(currentState), int(nextState), reward, int(stepPerEpisode), int(s_a))
