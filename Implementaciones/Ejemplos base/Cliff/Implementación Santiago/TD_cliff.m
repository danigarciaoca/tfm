function [v_td] = TD_cliff(problem,N_epi)
% Works for the random policy

v_td=zeros(problem.N_states,N_epi);
for k=1:N_epi-1
    k;
	S_td=zeros(500,1);
    current_state=1;        % always start from 1
    S_td(1)=current_state;
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
%         [S_td(cont) current_action next_state next_reward]
%         pause
        if length(find(next_state==[2:12]))==1,in=0;end
        cont=cont+1;
        S_td(cont)=next_state;
        v_td(S_td(cont-1),k+1)=v_td(S_td(cont-1),k+1)+problem.alpha*...
            (next_reward+problem.gamma*v_td(S_td(cont),k+1)-v_td(S_td(cont-1),k+1));
%         if in==0
%          v_td(S_td(cont),k+1)=v_td(S_td(cont),k+1)+problem.alpha*...
%             (0+problem.gamma*v_td(1,k+1)-v_td(S_td(cont),k+1));
 %       end
    end
end

end






