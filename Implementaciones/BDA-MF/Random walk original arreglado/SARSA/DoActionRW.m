function [ snew, reward, terminal ] = DoActionRW( action, currentState, env )
%GETNEXTSTATE Devuelve el siguiente estado y la recompensa instant�nea.
%   En base al estado actual (currentState) y a la acci�n tomada (action),
%   devuelve el siguiente estado y la recompensa asociada a dicha
%   transici�n, as� como la verdadera acci�n tomada en funci�n de la matriz
%   de transici�n P

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