% Random Walk problem
N_states=7;
central_state = round((N_states+1)/2);
N_actions=2;
gamma=.9;
alpha=.02;
v_ini=zeros(N_states,1);
q_ini=zeros(N_states*N_actions,1);
R = zeros(N_states*N_actions, 1);
R(end-3) = 1;
R=[0 0 0 0 0 0 0 0 0 0 1 0 0 0]';
mu = (1/N_states)*ones(N_states,1); % initial distribution over states
P = zeros(N_states*N_actions, N_states);
P(1:2,1) = [1; 1];
P(end-1:end,end) = [1; 1];
for s = 2:N_states-1
    fila1 = [zeros(1,s) 1 zeros(1,N_states-(s+1))];
    fila2 = [zeros(1,s-2) 1 zeros(1,N_states-(s-2+1))];
    P((((s-1)*N_actions)+1):s*N_actions, :) = [fila1; fila2];
end

A = eye(N_states);
B = [1 0];
pi_opt = kron(A,B);

A = eye(N_states);
B = [1 1];
marg = kron(A,B);

duplicar = [1 0 0 0 0 0 0;
    1 0 0 0 0 0 0;
    0 1 0 0 0 0 0;
    0 1 0 0 0 0 0;
    0 0 1 0 0 0 0;
    0 0 1 0 0 0 0;
    0 0 0 1 0 0 0;
    0 0 0 1 0 0 0;
    0 0 0 0 1 0 0;
    0 0 0 0 1 0 0;
    0 0 0 0 0 1 0;
    0 0 0 0 0 1 0;
    0 0 0 0 0 0 1;
    0 0 0 0 0 0 1];
mult1 = [ 1 0 0 0 0 0 0,
    0 0 0 0 0 0 0,
    0 1 0 0 0 0 0,
    0 0 0 0 0 0 0,
    0 0 1 0 0 0 0,
    0 0 0 0 0 0 0,
    0 0 0 1 0 0 0,
    0 0 0 0 0 0 0,
    0 0 0 0 1 0 0,
    0 0 0 0 0 0 0,
    0 0 0 0 0 1 0,
    0 0 0 0 0 0 0,
    0 0 0 0 0 0 1,
    0 0 0 0 0 0 0];
mult2 = [ 0 0 0 0 0 0 0,
    1 0 0 0 0 0 0,
    0 0 0 0 0 0 0,
    0 1 0 0 0 0 0,
    0 0 0 0 0 0 0,
    0 0 1 0 0 0 0,
    0 0 0 0 0 0 0,
    0 0 0 1 0 0 0,
    0 0 0 0 0 0 0,
    0 0 0 0 1 0 0,
    0 0 0 0 0 0 0,
    0 0 0 0 0 1 0,
    0 0 0 0 0 0 0,
    0 0 0 0 0 0 1];

R_W=struct('N_states',N_states,'N_actions',N_actions,'P',P,'R',R,'pi_rp',...
    pi_rp,'gamma',gamma,'pi_opt',pi_opt,'v_ini',v_ini,'q_ini',q_ini,...
    'alpha',alpha)
