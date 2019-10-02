function [q_ql v_out] = QL_cliff(problem,N_epi)
% Implementation of the Q-Learning algorithm for the cliff problem

q_ql=zeros(problem.N_states*problem.N_actions,N_epi);
v_out=zeros(1,N_epi);
q_ql(:,1)=problem.R;
r=.1;                   % e-greedy value       
for k=1:N_epi-1
	k
    S_ql=zeros(500,1);
    current_state=1;        % always start from 1
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
        
        if length(find(next_state==[2:12]))==1,
            in=0;
            v_out(k)=max(aux_ql);
        end
        cont=cont+1;
        S_ql(cont)=next_state;
        %aux=problem.P((S_ql(cont)-1)*problem.N_actions+1:S_ql(cont)*problem.N_actions,:);
        aux_ql=q_ql((S_ql(cont)-1)*problem.N_actions+1:S_ql(cont)*problem.N_actions,k);
        next_action = e_greedy(aux_ql,0,problem.N_actions);
        q_ql((S_ql(cont-1)-1)*problem.N_actions+current_action,k+1)=...
            q_ql((S_ql(cont-1)-1)*problem.N_actions+current_action,k+1)+problem.alpha*...
            (next_reward+problem.gamma*q_ql((S_ql(cont)-1)*problem.N_actions+next_action,k+1)...
            -q_ql((S_ql(cont-1)-1)*problem.N_actions+current_action,k+1));
    end
   % v_out(k)=max(q_ql((S_ql(cont-1)-1)*problem.N_actions:(S_ql(cont-1))*problem.N_actions,k+1));
    v_out(k)
end

end






