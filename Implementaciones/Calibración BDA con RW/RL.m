clear all, close all, clc
random_walk_set_up;
S = R_W.N_states;
A = R_W.N_actions;
terminal = false;
stop = false;

maxNumSteps = 30;
numEpi = 10;
numK = 1000;

mu = [0 1/5 1/5 1/5 1/5 1/5 0]; % distribución inicial de estados (pdf)

errorD = nan(1,numK*numEpi*maxNumSteps);
d = rand(S*A,1);
v = rand(S,1);
load v_opt.mat;
load d_opt.mat;

alphaD = 0.1; % Stepsize para la iteración de la variable dual d
alphaT = 0.1; % Stepsize para la iteración de la variable primal v

r=.2; % e-greedy value

% while ~stop == true
iterEpi = 0; % iterGlobal counts the number of steps taken in all numEpi episodes simulated, each of them numK times
for k = 1:numK
    s_a_sNext = [];
    iter = 0; % iter counts the number of steps taken in all numEpi episodes simulated
    for n = 1:numEpi % This loop sets the update frequency of v (every numEpi episodes)
        s = 4; % estado inicial el central
        % s = discretesample(mu, 1);
        terminal = false;
        step = 0; % step counts the number of steps taken in one of the numEpi episodes simulated
        while ~terminal && sum(d((((s-1)*A)+1):s*A)) ~= 0
            % [~, act_max] = max(d_old((((s-1)*A)+1):s*A));
            % Normalize d
            sumDoverA_aux = sum(reshape(d,[A,S]),1);
            sumDoverA = sum(duplicar*diag(sumDoverA_aux),2);
            d_norm = d./sumDoverA;
            
            % % Select an action according to policy (d normalized)
            % act_max = discretesample(d_norm((((s-1)*A)+1):s*A), 1);
            
            % Select an action according to epsilon-policy (d normalized)
            u=rand(1,1);
            if u<r
                act_max = randi(A);
            else
                act_max = discretesample(d_norm((((s-1)*A)+1):s*A), 1);
            end
            
            s_a = ((s-1)*A)+act_max;
            s_next = find(P(s_a,:)==1);
            
            v(s) = v(s) + alphaT*(R(s_a) + gamma*v(s_next) - v(s)); % policy evaluation
            
            iter = iter+1;
            s_a_sNext(iter,:) = [s act_max s_next];
            s = s_next;
            step = step + 1;
            
            if s == 1 || s == 7
                terminal = true;
            end
        end
    end
    
    numIter = size(s_a_sNext,1);
    for i = 1:numIter
        % Recover saved episodes
        s = s_a_sNext(i,1);
        a = s_a_sNext(i,2);
        s_next = s_a_sNext(i,3);
        s_a = ((s-1)*A)+a;
        
        % Policy (or d) update
        d(s_a) = d(s_a) + alphaD*(R(s_a) + gamma*P(s_a,s_next)*v(s_next) - v(s)); % policy update
        
        % Projection of d over positives
        d(d<0)=0;
        
        % Normalize d
        sumDoverA_aux = sum(reshape(d,[A,S]),1);
        sumDoverA = sum(duplicar*diag(sumDoverA_aux),2);
        d_norm = d./sumDoverA;
        
        % Save policy error
        if s_next == 1 || s_next == 7
            iterEpi = iterEpi + 1;
            % Calculate norm-2 of policy error (d_opt_norm was obtained by means of linear programming)
            errorD(iterEpi) = norm(abs(d_norm(3:12) - d_opt_norm(3:12)),2);
        end
        
        
    end
    
    % if abs(v - v_opt) < 1e-5 %comparamos la v del episodio anterior con la óptima.
    %    stop = true;
    % end
    [d;errorD(iterEpi)]
    
end

plot([1:numK*numEpi*maxNumSteps], errorD, 'LineWidth',2)
a = find(errorD==0);

if ~isempty(a)
    title(['Convergencia en ' num2str(a(1)) ' iteraciones'])
else
    title(['No converge'])
end

xlabel('Número de iteraciones')
ylabel('|| d - d_{opt} ||')

% sumDoverA_aux = sum(reshape(d,[A,S]),1);
% sumDoverA = sum(duplicar*diag(sumDoverA_aux),2);
% policy = d./sumDoverA;
% policy_by_action = reshape(policy', [A,S])';
% policy_matrix = diag(policy_by_action(:,1))*mult1' + diag(policy_by_action(:,2))*mult2';