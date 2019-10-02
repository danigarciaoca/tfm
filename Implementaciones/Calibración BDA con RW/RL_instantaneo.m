clear all, close all, clc
random_walk_set_up
S = R_W.N_states;
A = R_W.N_actions;
convergence = false;
stop = false;

d_old = rand(A*S,1); % Almacena el estado actual d(i) y actualizado d(i+1)
v_old = zeros(S,1);
v_new = zeros(S,1);
load v_opt.mat
v_opt = v; 

sumDoverA_aux = sum(reshape(d_old,[A,S]),1);
sumDoverA = sum(duplicar*diag(sumDoverA_aux),2);
policy = d_old./sumDoverA;
policy_by_action = reshape(policy', [A,S])';
policy_matrix = diag(policy_by_action(:,1))*mult1' + diag(policy_by_action(:,2))*mult2';

alphaD = 0.5; % Stepsize para la iteración de la variable dual d
alphaT = 0.1; % Stepsize para la iteración de la variable dual d
iter=0;

while ~stop == true
    s = randi([2 6],1,1); % estado inicial elegido de manera uniforme (excluimos los estados 1 y 7 que corresponden a los terminales)
    convergence = false;
    while ~convergence == true
        [~, act_max] = max(d_old((((s-1)*A)+1):s*A));
        % act_max = discretesample(d_old((((s-1)*A)+1):s*A), 1); % da error si d_old es un array de ceros
        s_a = ((s-1)*A)+act_max;
        s_next = find(P(s_a,:)==1);
        
        v_new(s) = v_old(s) + alphaT*(R(s_a) + gamma*v_old(s_next) - v_old(s)); % policy evaluation
        d_new = d_old + alphaD*(R + gamma*P*v_new - marg'*v_new); % policy update
        
        d_new(d_new<0)=0;
        sumDoverA_aux = sum(reshape(d_new,[A,S]),1);
        sumDoverA = sum(duplicar*diag(sumDoverA_aux),2);
        policy = d_new./sumDoverA;
        policy_by_action = reshape(policy', [A,S])';
        policy_matrix = diag(policy_by_action(:,1))*mult1' + diag(policy_by_action(:,2))*mult2';
        
        if v_old == v_new
            convergence = true;
        end
        
        d_old = d_new;
        v_old = v_new;
        s = s_next;
        iter = iter+1;
    end
    
    if abs(v_old - v_opt) < 1e-5 %comparamos la v del episodio anterior con la óptima.
        stop = true;
    end
end
v_old
policy_matrix
iter