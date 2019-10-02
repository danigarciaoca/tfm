function [v_pi q_pi] = policy_improvement(problem,N_steps)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
% Policy evaluation. V / Q functions
% Policy improvement. Start from uniform distribution

pi_ite=problem.pi_rp;
q_pi=zeros(problem.N_states*problem.N_actions,N_steps);
v_pi=zeros(problem.N_states,N_steps);
q_pi(:,1)=problem.R;
N_steps_pe=3;
pp=zeros(problem.N_states,problem.N_actions);
for k=1:N_steps-1
      aux=zeros(problem.N_states*problem.N_actions,N_steps);
      aux(:,1)=q_pi(:,k);
      for kk=1:N_steps_pe-1
    	 aux(:,kk+1)=(problem.R+problem.gamma*problem.P*pi_ite*aux(:,kk));   
      end
      q_pi(:,k+1)=aux(:,N_steps_pe);
%problem.p_rp=pi_ite;
%     problem.q_ini=q_pi(:,k);
%     [vv qq]=value_iteration(problem,N_steps_pe);
%     [m n]=size(qq);
%     q_pi(:,k+1)=qq(:,n);
    for kk=1:problem.N_states
        v_pi(kk,k+1)=max(q_pi((kk-1)*problem.N_actions+1:kk*problem.N_actions,k+1));
        aux=q_pi((kk-1)*problem.N_actions+1:kk*problem.N_actions,k+1);
        aux2=find(aux==max(aux));
        sol=zeros(1,problem.N_actions);
        sol(aux2)=ones(1,length(aux2))/length(aux2);    
        pi_ite(kk,(kk-1)*problem.N_actions+1:kk*problem.N_actions)=sol;
        pp(kk,:)=sol;
    end
end
%cliff_set_up
end



