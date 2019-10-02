function best_a = Greedy(theta, phi_s, num_actions)

    best_q = -inf;
    best_a = nan;
    
    for a = 1:num_actions
        phi_sa = GetStateActionFeatures(phi_s, a, num_actions);
        q = phi_sa'*theta;
        if q >= best_q
            best_q = q;
            best_a = a;
        end
    end

end