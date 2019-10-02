function [snew, reward, terminal] = DoActionChainWalk( a, s, env )
%DoActionChainWalk: executes the action (a) into the chain walk environment
% a: is left(1) or right(2)
% It returns next state and the reward asociated to this transition

Pssa = env.Pssa;
Rs = env.Rs;

snew = discretesample(Pssa((s-1)*env.num_actions + a,:), 1); % Observe new state

reward = Rs(snew); % Observe reward for the new state

terminal = false; % Infinite horizon problem

end
