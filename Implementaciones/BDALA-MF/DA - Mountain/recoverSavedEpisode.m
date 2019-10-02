function [ currentState, nextState, reward, stepPerEpisode, s_a ] = recoverSavedEpisode( s_a_sNext, A, i , env)
%RECOVERSAVEDEPISODE Recover s,a,s' tuple from saved episode (or hsitory of
%episodes).

if size(s_a_sNext,2) == 5
    currentState = s_a_sNext(i,1);
    currentAction = s_a_sNext(i,2);
    nextState = s_a_sNext(i,3);
    reward = s_a_sNext(i,4);
    stepPerEpisode = s_a_sNext(i,5);
    if rem(currentState,1) == 0 % then it's an integer
        s_a = ((currentState-1)*A)+currentAction;
    else % it is not an integer, we have to acquire the discretized index
        s = GetDiscretizedState(currentState, env.xy_disc);
        s_a = ((s-1)*A)+currentAction;
    end
elseif size(s_a_sNext,2) == 7
    currentState = s_a_sNext(i,1:2);
    currentAction = s_a_sNext(i,3);
    nextState = s_a_sNext(i,4:5);
    reward = s_a_sNext(i,6);
    stepPerEpisode = s_a_sNext(i,7);
    if rem(currentState,1) == 0 % then it's an integer
        s_a = ((currentState-1)*A)+currentAction;
    else % it is not an integer, we have to acquire the discretized index
        s = GetDiscretizedState(currentState, env.xy_disc);
        s_a = ((s-1)*A)+currentAction;
    end
    
end

