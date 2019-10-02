figure

tam_fuent_title = 14;
tam_fuente_label = 14;
tam_fuente_legend = 12;

xlabel('Episodio','FontSize',tam_fuente_label), ylabel('G','FontSize',tam_fuente_label)
title('Curva de aprendizaje','FontSize',tam_fuent_title)
title('Curva de explotación durante aprendizaje','FontSize',tam_fuent_title)
xlim([0 1500])

h_legend=legend('SARSA', 'Q-learning', 'BDA-MF');
set(h_legend,'FontSize',tam_fuente_legend);

% Camino safe
r=-1*ones(1,17)%16
g=[0.99].^(0:16)%15
g*r'

r=-1*ones(1,16)%16
g=[0.99].^(0:15)%15
g*r'

% Camino óptimo
r=-1*ones(1,13) %12
g=[0.99].^(0:12) %11
g*r'

r=-1*ones(1,12) %12
g=[0.99].^(0:11) %11
g*r'

% Camino medio
r=-1*ones(1,15) %14
g=[0.99].^(0:14) %13
g*r'

r=-1*ones(1,14) %14
g=[0.99].^(0:13) %13
g*r'
% RL-> solución intermedia entre óptimo y medio
% SARSA-> solución óptima, pero al ser una política e-greedy, a veces se empeora y hace que
% caigamos al acantilado, lo cual reduce notablemente el prmedio porque la recompensa es de
% un orden de magnitud mayor (-100 vs -12)