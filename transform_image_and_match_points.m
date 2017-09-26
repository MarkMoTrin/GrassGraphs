% Define a function to take in an image and an affine transformation and
% return a transformed image, the transformed pointset on the image and an
% image showing the correspondence. 
function [ax , f_matched] = transform_image_and_match_points(im, A, params)

im_orig = im;

size_im = size(im);
if (length(size_im) == 3)
    im = rgb2gray(im);
end    

%% Get surf features.
% Get the sift feature points of the image.
surf_points = detectSURFFeatures(im);
X = double(surf_points.Location);

%% Reduce number of features.
num_X_points = size(X,1);
if (params.num_X_points < num_X_points)
    rand_vec = randperm(num_X_points);
    X = X(rand_vec(1:params.num_X_points), :);
end

if (params.plot_imgs == 1)
    imshow(im); movegui(gcf, 'southwest'); hold on;
    plot(X(:,1), X(:,2), 'r*');
end

% Resize the shape to a square grid so that the same epsilon works across
% all shapes. 
X_orig = X;
X = resizeShapesToSquareGrid(X, params.resize_val);

%% Warp image, get image sizes and centers.

% Transform the image with A. 
tform = affine2d(A);
warped_im = imwarp(im, tform);
warped_im_orig = imwarp(im_orig, tform);

% Get the center coordinate of the original image. 
size_im = size(im);
im_center = fliplr(size_im)/2;

% Get the center coordinate of the transformed image. Remember the row
% number is actually the y coordinate and the columns are the x coordinate.
% For square images this would not make a difference but this is not the
% case for non-square images.
size_warped_im = size(warped_im);
warped_im_center = fliplr(size_warped_im)/2;

% Get the mean of the extracted points X. 
mean_X = mean(X_orig,1);
diff_im_X = mean_X - im_center;

%% Transform the resized X.

% Subtract the mean of X from X.
centered_X = bsxfun(@minus, X, mean(X,1));

% Transform X with A to create Y.
centered_X_h = [centered_X, ones(size(centered_X,1), 1)];
Y = centered_X_h * A; 
Y = Y(:,1:2); 
Y = Y(randperm(size(Y,1)),:); % Shuffle Y. 

%% Recover the affine from the transformed resized pointset.

% plotU = 1;
% Get the correspondences of Y.
[~, rA] = grassGraphsMatching(centered_X, Y, params.plotU, params);

%% Create the new transformed features using the recovered affine.

% Transform the original mean subtracted X with the recovered affine
% transformation. 
centered_X_orig = bsxfun(@minus, X_orig, mean(X_orig,1));

centered_X_orig_h = [centered_X_orig, ones(size(centered_X_orig,1), 1)];
corr_Y = centered_X_orig_h * rA; 
corr_Y = corr_Y(:,1:2); 

%% Setup the transformed pointset. 

% Transform the distance vector.
diff_im_X_h = [diff_im_X, 1];
diff_im_Y = diff_im_X_h * rA;
diff_im_Y = diff_im_Y(:,1:2);

disp(['Error in rA and A: ', num2str(norm(rA - A, 'fro'))]);

% Subtract the mean of correspondence Y from correspondence Y itself. 
mean_corr_Y = mean(corr_Y,1);
corr_Y = bsxfun(@minus, corr_Y, mean_corr_Y);

% Add the center of the transformed image to the subtracted mean
% correspondence Y. 
% Add the affine transformed distance vector of X and the original image to
% the above pointset. 
corr_Y = bsxfun(@plus, corr_Y, warped_im_center + diff_im_Y - rA(3,1:2));

% Display the image. 
f_matched = figure; 
ax = axes;
% showMatchedFeatures(im, warped_im, X_orig, corr_Y, 'montage', 'Parent',ax);
showMatchedFeatures(im_orig, warped_im_orig, X_orig, corr_Y, 'montage', 'Parent',ax);
