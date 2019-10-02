function [ currentState, nextState, reward, stepPerEpisode, s_a ] = recoverSavedEpisode( s_a_sNext, A, i )
%RECOVERSAVEDEPISODE Recover s,a,s' tuple from saved episode (or hsitory of
%episodes).

currentState = s_a_sNext(i,1);
currentAction = s_a_sNext(i,2);
nextState = s_a_sNext(i,3);
reward = s_a_sNext(i,4);
stepPerEpisode = s_a_sNext(i,5);
s_a = ((currentState-1)*A)+currentAction;
end

