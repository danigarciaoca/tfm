function output = e_greedy(v_input,r,N_actions)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[ma output]=max(v_input);
aux=find(v_input==ma);
output=aux(randi(length(aux)));
u=rand(1,1);
if u<r,aux=find([1:N_actions]~=output);
    output=aux(randi(N_actions-1));end
end
% if u<r,
%     output=randi(N_actions);end
% end
