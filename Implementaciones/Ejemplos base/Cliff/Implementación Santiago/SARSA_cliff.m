function [q_ql] = SARSA_cliff(problem,N_epi)
% Implementation of the Q-Learning algorithm for the cliff problem

q_ql=zeros(problem.N_states*problem.N_actions,N_epi);
q_ql(:,1)=problem.R;
r=.1;                   % e-greedy value       
for k=1:N_epi-1
    S_ql=zeros(500,1);
    current_state=1;        % always start from 1
    S_ql(1)=current_state;
    in=1;
    cont=1;
    q_ql(:,k+1)=q_ql(:,k);
    aux_ql=q_ql((S_ql(cont)-1)*problem.N_actions+1:S_ql(cont)*problem.N_actions,k);
    current_action = e_greedy(aux_ql,r,problem.N_actions);
    while in==1
        aux_P=problem.P((S_ql(cont)-1)*problem.N_actions+1:S_ql(cont)*problem.N_actions,:);
        candidates_next_states=zeros(problem.N_actions,1);
        for kk=1:problem.N_actions
            candidates_next_states(kk,1)=find(aux_P(kk,:)==1);
        end
        next_state=candidates_next_states(current_action,1);
        next_reward=problem.R((S_ql(cont)-1)*problem.N_actions+current_action);
        if length(find(next_state==[2:12]))==1,in=0;end
        cont=cont+1;
        S_ql(cont)=next_state;
        aux_ql=q_ql((S_ql(cont)-1)*problem.N_actions+1:S_ql(cont)*problem.N_actions,k);
        next_action = e_greedy(aux_ql,r,problem.N_actions);
        q_ql((S_ql(cont-1)-1)*problem.N_actions+current_action,k+1)=...
            q_ql((S_ql(cont-1)-1)*problem.N_actions+current_action,k+1)+problem.alpha*...
            (next_reward+problem.gamma*q_ql((S_ql(cont)-1)*problem.N_actions+next_action,k+1)...
            -q_ql((S_ql(cont-1)-1)*problem.N_actions+current_action,k+1));
        current_action=next_action;
    end
end

end






