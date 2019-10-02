function position = initPos( initType, board, numStates )
%INITPOS Define cuál será la posición/estado inicial
%   La inicialización depende del valor que tome initType
%   Si initType = fixed -> Inicializamos de manera fija al estado que se
%       consideró inicial (de acuerdo a Fig. 6.8 de p.138 de Reinforcement 
%       Learning: An Introduction).
%   Si initType = random -> Inicializamos de manera aleatoria a cualquier
%       estado que no sea ni el acantilado ni el estado final.

initType = lower(initType);
initState_default = board(end,1); % Estado considerado como inicial por defecto

switch initType
    case 'fixed'
        position = initState_default; % Inicializamos de manera fija al estado que se consideró inicial
    case 'random'
        position = randi(numStates-1); % Inicializamos de manera aleatoria de entre los posibles estados (excepto el final, por eso -1)
    otherwise
        disp('Introduce ''fixed'' o ''random'' ')
end

end

