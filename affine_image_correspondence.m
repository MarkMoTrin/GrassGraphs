clear; clc; close all; 

% Read in the image. 
I = imread('cameraman.tif');
points = detectSURFFeatures(I);
I_center = size(I)/2;

imshow(I); movegui(gcf, 'southwest')

% Get the surf features. 
X = points.Location;
X_center = mean(X);

hold on; 
plot(X(:,1), X(:,2), 'r*')
plot(X_center(1), X_center(2), 'co');
plot(I_center(1), I_center(2), 'yo')

diff_vec = X_center - I_center;

%% Define the warp
sigma = pi/4;  % Rotation angle 2.
a = 1;      % Scale in the x direction. 
b = 2;      % Scale in the y direction. 
theta = 0;      % Rotation angle 1
tx = 0;     % Translation in x. 
ty = 0;     % Translation in y. 
tVec = [tx, ty];

% Generate an affine transformation. 
A = affineTransformation2D_Clean(theta, sigma, tVec, a, b);
tform = affine2d(A);

%% Warp the image.

B = imwarp(I, tform);
B_size = size(B);
B_center = B_size/2;

f_warp = figure; movegui(f_warp, 'south'); imshow(B);

X_temp = X;
mean_X_temp = mean(X_temp,1);
X_temp = bsxfun(@minus, X_temp, mean(X_temp,1));

% Transform the changed X. 
Xh = [X_temp ones(size(X,1),1)]; % Homogeneous representation of X. 
Y = Xh*A; % Apply the affine to create Y. 
Y = Y(:,1:2);  % Keep only the first 2 dimensions. 

Y = bsxfun(@minus, Y, mean(Y,1));
Y = bsxfun(@plus, Y, B_center);

diff_vec_h = [diff_vec, 1];
diff_vec = diff_vec_h * A; 
diff_vec = diff_vec(:, 1:2);

Y = bsxfun(@plus, Y, diff_vec);

figure(f_warp); hold on; 
plot(Y(:,1), Y(:,2), 'g*');
% plot(B_center(1), B_center(2), 'g*');


