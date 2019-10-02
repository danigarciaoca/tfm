clear all, close all, clc
random_walk_set_up
S = R_W.N_states;
A = R_W.N_actions;
convergence = false;

d_old = rand(A*S,1); % Almacena el estado actual d(i) y actualizado d(i+1)

A = eye(N_states);
B = [1; 0];
mult1 = kron(A,B);
A = eye(N_states);
B = [0; 1];
mult2 = kron(A,B);

sumDoverA_aux = sum(reshape(d_old,[A,S]),1);
sumDoverA = sum(duplicar*diag(sumDoverA_aux),2);
policy = d_old./sumDoverA;
policy_by_action = reshape(policy', [A,S])';
policy_matrix = diag(policy_by_action(:,1))*mult1' + diag(policy_by_action(:,2))*mult2';

alphaD = 1e-3; % Stepsize para la iteración de la variable dual d
iter=0;
while ~convergence == true
    v = (inv(eye(S)-gamma*policy_matrix*P))*policy_matrix*R;
    d_new = d_old + alphaD*(R + gamma*P*v - marg'*v); % maximizar en d
    
    d_new(d_new<0)=0;
    sumDoverA_aux = sum(reshape(d_new,[A,S]),1);
    sumDoverA = sum(duplicar*diag(sumDoverA_aux),2);
    policy = d_new./sumDoverA;
    policy_by_action = reshape(policy', [A,S])';
    policy_matrix = diag(policy_by_action(:,1))*mult1' + diag(policy_by_action(:,2))*mult2';
    
    if mod(iter,1000)==0
%         policy_by_action
        v
        %         v_new
        % %         ((1-gamma)*mu'*v_new + d'*(R + gamma*P*v_new - marg'*v_new))
        % %         (R + gamma*P*v_new - marg'*v_new)
    end
    
    d_old = d_new;
    iter = iter+1;
end