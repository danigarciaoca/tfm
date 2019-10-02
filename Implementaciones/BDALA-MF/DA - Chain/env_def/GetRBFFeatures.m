function phi = GetRBFFeatures(xy, env)

N = env.N;

mu_x = env.mu_x;
mu_y = env.mu_y;

sigma = diag(0.8*[mu_x(2)-mu_x(1), mu_y(2)-mu_y(1)]);
% sigma = 1/N*diag([dimx(2)-dimx(1), dimy(2)-dimy(1)]);


phi = nan(N.^2, 1);

k = 1;
for k1 = 1:length(mu_y)
    for k2 = 1:length(mu_x)
        phi(k) = mvnpdf(xy, [mu_x(k2), mu_y(k1)], sigma);
        k = k+1;
    end
end


end