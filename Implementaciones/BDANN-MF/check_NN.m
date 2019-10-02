% x = linspace(-10,10,100);
% y = linspace(-10,10,100);
% [X,Y] = meshgrid(x,y);
% 
% F = cos(X).^2+sin(X.*Y);
% surf(X,Y,F), shading flat
% 
% xy = combvec(x,y);
% x = xy(1,:)
% y = xy(2,:)
% 
% f = cos(x).^2+sin(x.*y)
% hold on, plot3(x,y,f,'rx')

syms x y
f = symfun(cos(x).^2+sin(x.*y), [x y])
fsurf(f)

x_train = csvread('xtrain.csv');
y_train = csvread('ytrain.csv');
hold on, plot3(x_train(:,1),x_train(:,2),y_train,'rx')