% Analysis cliff problem
clear all
cliff_set_up
N_steps_value_ite=500;
% Optimum value
v_opt=inv(eye(cliff.N_states)-cliff.gamma*cliff.pi_opt*cliff.P)*cliff.pi_opt*cliff.R;
q_opt=inv(eye(cliff.N_states*cliff.N_actions)-cliff.gamma*cliff.P*cliff.pi_opt)*cliff.R;

% Solution of the Bellman linear equations
% Solution of the V/Q equations
[v_rp q_rp]=bellman_linear(cliff);
[v_pe_bell q_pe_bell]=policy_evaluation(cliff,N_steps_value_ite);

% We plot the results
[v_rp(37:48)';v_rp(25:36)';v_rp(13:24)';v_rp(1:12)']
[v_pe_bell(37:48,N_steps_value_ite)';v_pe_bell(25:36,N_steps_value_ite)';v_pe_bell(13:24,N_steps_value_ite)';v_pe_bell(1:12,N_steps_value_ite)']
aux=repmat(v_rp,[1 N_steps_value_ite])-v_pe_bell;
plot(1:N_steps_value_ite,10*log10(diag(aux'*aux)),'b','LIneWidth',5),grid
hold
aux=repmat(q_rp,[1 N_steps_value_ite])-q_pe_bell;
plot(1:N_steps_value_ite,10*log10(diag(aux'*aux)),'r','LIneWidth',5)
xlabel('Steps')
ylabel('Error')
title('Value / Q iteration performance'),hold off

% Value iteration
[v_vi q_vi] = value_iteration(cliff,N_steps_value_ite);
[v_opt(37:48)';v_opt(25:36)';v_opt(13:24)';v_opt(1:12)']
[v_vi(37:48)';v_vi(25:36)';v_vi(13:24)';v_vi(1:12)']

% Policy iteration
%cliff_set_up
[v_pi q_pi] = policy_improvement(cliff,N_steps_value_ite);
[v_vi v_pi(:,N_steps_value_ite)];
[v_opt(37:48)';v_opt(25:36)';v_opt(13:24)';v_opt(1:12)']
[v_pi(37:48,500)';v_pi(25:36,500)';v_pi(13:24,500)';v_pi(1:12,500)']
aux=repmat(v_vi,[1 N_steps_value_ite])-v_pi;
plot(1:N_steps_value_ite,10*log10(diag(aux'*aux)))
xlabel('Steps')
ylabel('Policy improvement')

% TD V function. Random policy
cliff.alpha=.03;
N_epi=N_steps_value_ite*200;
[v_td] = TD_cliff(cliff,N_epi);
[v_rp(37:48)';v_rp(25:36)';v_rp(13:24)';v_rp(1:12)']
[v_td(37:48,N_epi)';v_td(25:36,N_epi)';v_td(13:24,N_epi)';v_td(1:12,N_epi)']
aux=repmat(v_rp,[1 N_epi])-v_td;
plot(1:N_epi,10*log10(sum(aux.^2,1)),'b','LIneWidth',5),grid


% QL algorithm
cliff.alpha=.1;
[q_ql v_sal] = QL_cliff(cliff,N_steps_value_ite*100);
aux=repmat(q_opt,[1 N_steps_value_ite*100])-q_ql;
plot(1:N_steps_value_ite*100,10*log10(sum(aux.^2,1)),'b','LIneWidth',5),grid
q_ql_last=q_ql(:,N_steps_value_ite*100);
v_ql_last=zeros(cliff.N_states,1);
for k=1:48,
    v_ql_last(k)=max(q_ql_last((k-1)*cliff.N_actions+1:k*cliff.N_actions,:));
    %pause,
end
[v_opt(37:48)';v_opt(25:36)';v_opt(13:24)';v_opt(1:12)']
[v_ql_last(37:48)';v_ql_last(25:36)';v_ql_last(13:24)';v_ql_last(1:12)']
[v_opt(37:48)';v_opt(25:36)';v_opt(13:24)';v_opt(1:12)']-[v_ql_last(37:48)';v_ql_last(25:36)';v_ql_last(13:24)';v_ql_last(1:12)']

for k=1:cliff.N_states
    aa=q_ql_last((k-1)*cliff.N_actions+1:k*cliff.N_actions,:)
    [m a]=max(aa);
    k,b=find(aa==m)
    aa(b)
    pause
end


% SARSA algorithm
cliff.alpha=.1;
[q_sarsa] = SARSA_cliff(cliff,N_steps_value_ite*100);
aux=repmat(q_opt,[1 N_steps_value_ite*100])-q_sarsa;
plot(1:N_steps_value_ite*100,10*log10(sum(aux.^2,1)),'b','LIneWidth',5),grid
q_sarsa_last=q_sarsa(:,N_steps_value_ite*100);
v_sarsa_last=zeros(cliff.N_states,1);
for k=1:48,
    v_sarsa_last(k)=max(q_sarsa_last((k-1)*cliff.N_actions+1:k*cliff.N_actions,:));
    %pause,
end
[v_opt(37:48)';v_opt(25:36)';v_opt(13:24)';v_opt(1:12)']
[v_sarsa_last(37:48)';v_sarsa_last(25:36)';v_sarsa_last(13:24)';v_sarsa_last(1:12)']
for k=1:cliff.N_states
    a_sarsa=q_sarsa_last((k-1)*cliff.N_actions+1:k*cliff.N_actions,:);
    a_ql=q_ql_last((k-1)*cliff.N_actions+1:k*cliff.N_actions,:);
    [a_ql a_sarsa]
    [m_ql a1]=max(a_ql);
    [m_sarsa a2]=max(a_sarsa);
    [k a1 a2]
    pause
end

t_ql=zeros(1,100);t_ql(1)=1;
t_sarsa=zeros(1,100);t_sarsa(1)=1;
current=1;
for k=1:cliff.N_states
    a_ql=q_ql_last((t_ql(k)-1)*cliff.N_actions+1:t_ql(k)*cliff.N_actions,:);
    [m_ql in_ql]=max(a_ql);
    a_sarsa=q_sarsa_last((t_sarsa(k)-1)*cliff.N_actions+1:t_sarsa(k)*cliff.N_actions,:);
    [m_sarsa in_sarsa]=max(a_sarsa);
    if in_ql==1,t_ql(k+1)=t_ql(k)+12;end
    if in_ql==2,t_ql(k+1)=t_ql(k)+1;end
    if in_ql==3,t_ql(k+1)=t_ql(k)-12;end
    if in_ql==4,t_ql(k+1)=t_ql(k)-1;end
    if t_ql(k+1)==12,in_m_ql=k+1;end
    if in_sarsa==1,t_sarsa(k+1)=t_sarsa(k)+12;end
    if in_sarsa==2,t_sarsa(k+1)=t_sarsa(k)+1;end
    if in_sarsa==3,t_sarsa(k+1)=t_sarsa(k)-12;end
    if in_sarsa==4,t_sarsa(k+1)=t_sarsa(k)-1;end
    if t_sarsa(k+1)==12,in_m_sarsa=k+1;break,end
    
end
t_ql(1:14)
t_sarsa(1:in_m_sarsa)





