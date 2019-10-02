function [ l_k ] = DualProjection_dMDP( l_k, xi, gamma, m, n )
%DUALPROJECTION_DMDP Projection of dual iterates for discounted MDP (dMDP) case.
%   l_k: dual variable at k-th iteration
%   xi: arbitrary vector with positive entries
%   gamma: discount factor
%   m: number of actions
%   n: number of states

l_k(sum(l_k,2)< xi,:) = xi(sum(l_k,2)< xi,:).*(l_k(sum(l_k,2)< xi,:)./sum(l_k(sum(l_k,2)< xi,:),2)); % POR LA PRECISIÓN DE MATLAB, A VECES ALGUNOS VALORES SE DETECTAN COMO SI SIGUISESN SIENDO MENORES TRAS LA PROYECCIÓN

l_k(l_k < 0) = 0;

l_k_aux = reshape(l_k', [n*m 1]);
if norm(l_k_aux, 1) ~= norm(xi, 1)/(1-gamma)
    l_k_aux = (l_k_aux./sum(l_k_aux))*(norm(xi,1)/(1-gamma));
end
l_k = reshape(l_k_aux, [m n])';
end

