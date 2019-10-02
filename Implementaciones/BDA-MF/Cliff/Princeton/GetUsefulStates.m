function [ useful_states, useful_states_sz ] = GetUsefulStates( n, terminal_states )
%GETUSEFULSTATES Summary of this function goes here
%   Detailed explanation goes here

states = 1:n;
useful_states = states(~ismember(states, terminal_states))';
useful_states_sz = size(useful_states, 1);
end

