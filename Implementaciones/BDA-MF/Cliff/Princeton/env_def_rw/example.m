clear all, close all, clc
addpath('env_def_rw\')

% % % % ENVIRONMENT:
name = 'RandomWalk';
N_states = 13;
initType = 'leftCorner'; % {'center', 'leftCorner'}
transitionType = 'det'; % {'det', 'rand'}
rewardType = 'det'; % {'det', 'rand'} || if rand, extra argument: finalReward=50
env = GetRndWalkEnv(name, N_states, initType, transitionType, rewardType);

% Optimum value (value iteration)
N_steps = 1000;
[v_opt, q_opt, q_opt_format] = value_iteration(env, N_steps, env.terminal_states);
[~, pi_lineal] = max(q_opt_format,[],2)