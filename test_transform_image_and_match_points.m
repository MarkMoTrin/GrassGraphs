% test transform_image_and_match_points(im, A, params)
clear; clc; close all; 

%% Load an image. 
im_fold = 'C:\Users\mmoyou\Dropbox\Pami_revision\data\affine_image_exps\';
save_fold = 'C:\Users\mmoyou\Dropbox\Pami_revision\feature_matching_imgs\';

dir_cont = dir(im_fold);
num_imgs = length(dir_cont);

%% Create an affine. 

affine_number = 36;

sigma = pi/0.97;  % Rotation angle 2.
a = -2.3;      % Scale in the x direction. 
b = -0.5;      % Scale in the y direction. 
theta = 0;      % Rotation angle 1
tx = 0;     % Translation in x. 
ty = 0;     % Translation in y. 

tVec = [tx, ty];
% Generate an affine transformation. 
A = affineTransformation2D_Clean(theta, sigma, tVec, a, b);

%% Define the parameters. 
p = grassGraphsParams_Clean; % GrassGraphs parameters.

p.s = 0.7
% p.Epsilon = 0.12;
p.Epsilon = 0.7;
p.num_X_points = 75;
p.resize_val = 10;

p.plotU = 0;
p.plot_imgs = 0;
save_imgs_flag = 1;

%% Run the transform_image_and_match_points function. 

for i = 4 : num_imgs
    curr_im_name = dir_cont(i).name;
    im = imread([im_fold, curr_im_name]);    
    [ax, f_matched] = transform_image_and_match_points(im, A, p);

    name_to_save = curr_im_name(1 : end-4);
    
    name_to_save = [name_to_save, '_aff_', num2str(affine_number), '_np_', num2str(p.num_X_points)];
    full_name_to_save = [save_fold, name_to_save];
%     pause;
    
    if (save_imgs_flag == 1)
        print(full_name_to_save,'-depsc','-tiff');
        saveas(f_matched, [full_name_to_save, '.jpg'])
    end 
    close(f_matched)
end