function [q_ql] = QL_R_W(problem,N_epi)
% Implementation of the Q-Learning algorithm for the cliff problem

q_ql=zeros(problem.N_states*problem.N_actions,N_epi);
q_ql(:,1)=problem.R;
r=.2;                   % e-greedy value       
for k=1:N_epi-1
    S_ql=zeros(100,1);
    current_state=4;        % always start from 4
    S_ql(1)=current_state;
    in=1;
    cont=1;
    q_ql(:,k+1)=q_ql(:,k);
    while in==1
        aux_P=problem.P((S_ql(cont)-1)*problem.N_actions+1:S_ql(cont)*problem.N_actions,:);
        candidates_next_states=zeros(problem.N_actions,1);
        for kk=1:problem.N_actions
            candidates_next_states(kk,1)=find(aux_P(kk,:)==1);
        end
        aux_ql=q_ql((S_ql(cont)-1)*problem.N_actions+1:S_ql(cont)*problem.N_actions,k);
%         current_action=randi(problem.N_actions,1);      % Random policy
        current_action = e_greedy(aux_ql,r,problem.N_actions);
        next_state=candidates_next_states(current_action,1);
        next_reward=problem.R((S_ql(cont)-1)*problem.N_actions+current_action);
        if next_state==7,in=0;end
        if next_state==1,in=0;end
        %[S_ql(cont) current_action next_state next_reward],pause
        cont=cont+1;
        S_ql(cont)=next_state;
        %aux=problem.P((S_ql(cont)-1)*problem.N_actions+1:S_ql(cont)*problem.N_actions,:);
        aux_ql=q_ql((S_ql(cont)-1)*problem.N_actions+1:S_ql(cont)*problem.N_actions,k);
        next_action = e_greedy(aux_ql,0,problem.N_actions);
%         q_ql((S_ql(cont-1)-1)*problem.N_actions+current_action,k+1)=...
%             q_ql((S_ql(cont-1)-1)*problem.N_actions+current_action,k+1)+problem.alpha*...
%             (next_reward+problem.gamma*q_ql((S_ql(cont)-1)*problem.N_actions+next_action,k+1)...
%             -q_ql((S_ql(cont-1)-1)*problem.N_actions+current_action,k+1));
        q_ql((S_ql(cont-1)-1)*problem.N_actions+current_action,k+1)=...
            q_ql((S_ql(cont-1)-1)*problem.N_actions+current_action,k+1)+problem.alpha*...
            (next_reward+problem.gamma*q_ql((S_ql(cont)-1)*problem.N_actions+next_action,k+1)...
            -q_ql((S_ql(cont-1)-1)*problem.N_actions+current_action,k+1));
    end
end

end






