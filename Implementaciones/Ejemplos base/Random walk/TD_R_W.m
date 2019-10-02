function [v_td] = TD_R_W(problem,N_epi)
% Works for the random policy

v_td=zeros(problem.N_states,N_epi);
for k=1:N_epi-1
	S_td=zeros(100,1);
    S_td(1)=4;           % always start from 4
    in=1;
    cont=1;
    v_td(:,k+1)=v_td(:,k);
    while in==1
        aux=problem.P((S_td(cont)-1)*problem.N_actions+1:S_td(cont)*problem.N_actions,:);
        candidates_next_states=zeros(problem.N_actions,1);
        for kk=1:problem.N_actions
            candidates_next_states(kk,1)=find(aux(kk,:)==1);
        end
        current_action=randi(problem.N_actions,1);      % Random policy
        next_state=candidates_next_states(current_action,1);
        next_reward=problem.R((S_td(cont)-1)*problem.N_actions+current_action);
        if next_state==7,in=0;end
        if next_state==1,in=0;end
        cont=cont+1;
        S_td(cont)=next_state;
        v_td(S_td(cont-1),k+1)=v_td(S_td(cont-1),k+1)+problem.alpha*...
            (next_reward+problem.gamma*v_td(S_td(cont),k+1)-v_td(S_td(cont-1),k+1));
%         if in==0
%          v_td(S_td(cont),k+1)=v_td(S_td(cont),k+1)+problem.alpha*...
%             (0+problem.gamma*v_td(1,k+1)-v_td(S_td(cont),k+1));
%         end
    end
end

end






