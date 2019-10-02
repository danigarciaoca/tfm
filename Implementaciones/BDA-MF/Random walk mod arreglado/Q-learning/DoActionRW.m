function [ snew, reward, terminal ] = DoActionRW( action, currentState, env )
%GETNEXTSTATE Devuelve el siguiente estado y la recompensa instantánea.
%   En base al estado actual (currentState) y a la acción tomada (action),
%   devuelve el siguiente estado y la recompensa asociada a dicha
%   transición, así como la verdadera acción tomada en función de la matriz
%   de transición P

terminal = false; % flag which indicates if the episode has finished
Pssa = env.P;
R = env.R;

snew = discretesample(Pssa((currentState-1)*env.numActions + action,:), 1); % Observe new state

reward = R(snew); % Observe reward for the new state

isTerminal = (length(find(snew == env.terminal_states)) == 1);
if isTerminal
    terminal = true;
end

end