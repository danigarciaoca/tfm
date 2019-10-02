clear all, close all, clc

q_learning;
clearvars -except Qlearning_res, close all, clc
sarsa;
clearvars -except Qlearning_res SARSA_res, close all, clc

figure, hold on
plot((1:Qlearning_res.numEpisodes), Qlearning_res.Gmean, 'LineWidth', 1.5), ylim([-100 0])
plot((1:SARSA_res.numEpisodes), SARSA_res.Gmean, 'LineWidth', 1.5), ylim([-100 0])
hold off
grid, legend('Q-learning', 'SARSA', 'Location','SouthEast')
xlabel('Episode'), ylabel('Reward per episode'), title('Evolución (curva sin suavizar)')

figure, hold on
plot((1:Qlearning_res.numEpisodes), Qlearning_res.Gsmooth, 'LineWidth', 1.5), ylim([-100 0])
plot((1:SARSA_res.numEpisodes), SARSA_res.Gsmooth, 'LineWidth', 1.5), ylim([-100 0])
hold off
grid, legend('Q-learning', 'SARSA', 'Location','SouthEast')
xlabel('Episode'), ylabel('Reward per episode'), title('Evolución (curva suavizada)')

disp('Grid final para Q-learning (1=izquierda, 2 = arriba, 3 = derecha, 4 = abajo)'), Qlearning_res.finalGrid
disp('Grid final para SARSA (1=izquierda, 2 = arriba, 3 = derecha, 4 = abajo)'), SARSA_res.finalGrid