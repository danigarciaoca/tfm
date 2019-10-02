function idx = GetDiscretizedState(s, xy_disc)
    s_rep = ones(size(xy_disc,1),1)*s;
    [~, idx] = min(sqrt(sum(abs(s_rep-xy_disc).^2,2)));
end