function [ actions, policy ] = generateRandomPolicy( game )
%GETRANDOMPOLICY Summary of this function goes here
%   Detailed explanation goes here

policy = zeros(game.N_states*game.N_actions,1);

actions = randi(2,[1 game.N_states]);
s_a = (([1:game.N_states]-1)*game.N_actions)+actions;
policy(s_a) = 1;
end

