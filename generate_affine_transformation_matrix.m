% Setup an affine transformation. 
sigma = -pi/4;   % Rotation angle 2.
a = -2;      % Scale in the x direction. 
b = 1;      % Scale in the y direction. 

theta = 0;      % Rotation angle 1
tx = 5;     % Translation in x. 
ty = 6;     % Translation in y. 

tVec = [tx, ty];

% Generate an affine transformation. 
A = affineTransformation2D_Clean(theta, sigma, tVec, a, b)

tform = affine2d(A)

tform.T