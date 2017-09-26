clear; close all; clc;
% Use SURF features to find corresponding points between two images
% rotated and scaled with respect to each other
I1 = imread('cameraman.tif');

%% Define the warp
sigma = pi/4;  % Rotation angle 2.
a = 0;      % Scale in the x direction. 
b = 0;      % Scale in the y direction. 
theta = 0;      % Rotation angle 1
tx = 0;     % Translation in x. 
ty = 0;     % Translation in y. 
tVec = [tx, ty];

% Generate an affine transformation. 
A = affineTransformation2D_Clean(theta, sigma, tVec, a, b)
tform = affine2d(A)
tform.T

%% Warp the image.

I2 = imwarp(I1, tform);
im_size = size(I2)
im_center = im_size/2;

points1 = detectSURFFeatures(I1);
points2 = detectSURFFeatures(I2);
% 
% [f1, vpts1] = extractFeatures(I1, points1);
% [f2, vpts2] = extractFeatures(I2, points2);
% 
% indexPairs = matchFeatures(f1, f2) ;
% matchedPoints1 = vpts1(indexPairs(:, 1));
% matchedPoints2 = vpts2(indexPairs(:, 2));

X = points1.Location;
mean_X = mean(X,1);

Y_temp = bsxfun(@minus, points1.Location, mean_X);
% Transform the changed X. 
Xh = [Y_temp ones(size(X,1),1)]; % Homogeneous representation of X. 
Y = Xh*A; % Apply the affine to create Y. 
Y = Y(:,1:2);  % Keep only the first 2 dimensions. 

mean_Y = mean(Y_temp,1);
Y = bsxfun(@minus, Y, mean_Y);
mean_Y
Y = bsxfun(@plus, Y, im_center);

% matchedPoints2.Location

figure; ax = axes;
% Visualize putative matches
% showMatchedFeatures(I1,I2,X,Y, 'montage', 'Parent',ax);
showMatchedFeatures(I1,I2,points1,points2, 'montage', 'Parent',ax);

title('Putative point matches');
legend('matchedPts1','matchedPts2');

% figure(gcf); hold on; plot(200, 100, 'bo')
