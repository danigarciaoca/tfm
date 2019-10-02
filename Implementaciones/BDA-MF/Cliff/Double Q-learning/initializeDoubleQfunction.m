function [ Qsa1, Qsa2 ] = initializeDoubleQfunction( S, A, terminal_states )
%INITIALIZEDOUBLEQFUNCTION inicializa las dos funciones Q(s,a) necesarias
%para double Q-learning. Todos los valores a rand excepto los de los estados
%terminales.

Qsa1 = rand(S, A);
Qsa1(terminal_states,:) = 0;

Qsa2 = rand(S, A);
Qsa2(terminal_states,:) = 0;
end

