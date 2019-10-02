function phi = GetRBFFeatures(xy, env)

dimx = env.dimx;
dimy = env.dimy;
N = env.N;


mu_x = linspace(dimx(1), dimx(2), N+2);
mu_x = mu_x(2:end-1);
mu_y = linspace(dimy(1), dimy(2), N+2);
mu_y = mu_y(2:end-1);

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