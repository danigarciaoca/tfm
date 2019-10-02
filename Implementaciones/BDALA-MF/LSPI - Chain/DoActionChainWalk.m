function [snew, reward, terminal] = DoActionChainWalk( a, s, env )
%DoActionChainWalk: executes the action (a) into the chain walk environment
% a: is left or right

Pssa = env.Pssa;
Rs = env.Rs;

snew = SampleDiscrete(Pssa(s,:,a)'); % Observe new state

reward = Rs(snew); % Observe reward for the new state

terminal = false; % Infinite horizon problem


end % function DoActionMountainCar
