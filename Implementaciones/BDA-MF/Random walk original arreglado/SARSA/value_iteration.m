function [v_opt, q_opt, q_format_opt] = value_iteration(env, N_steps, terminal_states)

% Policy evaluation. V / Q functions
v = rand(env.numStates,1); v(terminal_states) = 0;
q = env.P*(env.R + env.gamma*v);

for k=1:N_steps-1
    q_format = reshape(q,[env.numActions env.numStates])';
    v = max(q_format, [] ,2);
    q = env.P*(env.R + env.gamma*v);
end
q_opt = q;
q_format_opt = reshape(q,[env.numActions env.numStates])';
v_opt=v;
end


