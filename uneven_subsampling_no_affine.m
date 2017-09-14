% Script for unequal points correspondence original shapes equal. 
clc; clear; close all;

shape_folder = './Datasets/2D/';
shape_name = 'airplane.mat';
im_fold = 'C:\Users\mmoyou\Dropbox\Pami_revision\imgs_unequal_points\uneven_subsampling_no_affine\';

%% ----------------------------

p.fhkt = 100;                   % For the full graph case, not used. 
p.Epsilon = 0.012;              % Eapsilon value. 
p.hkt = 175;                    % Heat kernel value.
p.GraphCase = 'epsilon';
p.GraphLapType = 'heatKer';
p.CorrMethod = 'minDistThrowOut';
p.CondFac = 500;        
p.GraphLapType = 'heatKer';
p.NumLBOEvec = 5;               % Number of eigenvectors to compute. 
p.EvecToMatch = [1, 2, 3];      % Eigenvectors used to match in the LBO space.
p.DistScFac = 1e-4; 
p.ScaleDist = 0; % May not be used. 
p.ScType = 'maxScaling'; % Scaling type for distance matrix.

% Resolving eigenvector flipping. 
% Now we have to check which sign flip of the target shape eigenvectors
% correponds to the closest of the eigenvectors of the source shape.
% The extra -1 is because the number of eigenvectors chosen has the zero
% vector contained in it so we remove it. The second -1 is to used to
% generate the different permutations. This should genereate 8 different
% combination of the three eigenvector flips.
numEigVecForPerm = numel(p.EvecToMatch);
p.AllPerms = dec2bin(2^(numEigVecForPerm) - 1: -1 : 0) - '0'; % All permutations.
p.AllPerms(p.AllPerms == 0) = -1;   % Changing the zeros to -1s. 
p.NumPerms = size(p.AllPerms,1);    % Number of permutations. 
p.eigSc = 0;        % Scale the eigenvectors with the eigenvalues.
p.CorrThresh = 3;
p.ScoreType = 'NumCorr';
%% Print im flags.

print_flag_orig = 1; 
print_flag_UVecs = 1; 
print_flag_LBO = 1; 
print_flag_LBO_unscaled = 1; 

dispEigVals = 0;
plot_U_vecs = 1;
plotLBO = 1;
plotLBO_unscaled = 1;

view_tuple_x = 65;
view_tuple_y = 24;
view_LBO_x = 1;
view_LBO_y = 2;

view_LBO_x = -172;
view_LBO_y = 60;

X_p = 0.0012;

subsampling_rate = 95;
Y_p = 0.0014
Y_scale_fac = 1;

%% Loading and plotting the shapes. 

% Load the shape.
X = load([shape_folder, shape_name]); 
X = X.x;
X = resizeShapesToSquareGrid(X, 10);
Y = X;

num_subsampled_points = round(size(Y,1)*subsampling_rate/100);
Y = Y(1 : num_subsampled_points,:);
Y = Y(randperm(size(Y,1)),:);

% Number of points to subsample by.
num_points_shape = size(X,1);
num_points_Y = size(Y,1);

%% Plotting Original. 
fOrig = figure;
movegui(gcf, 'northwest');
plot(X(:, 1), X(:, 2), 'ro', 'Markersize',3); hold on;
plot(Y(:, 1), Y(:, 2), 'b.', 'Markersize',5)
title('Original Shapes');
legend('X', 'Y');

if (print_flag_orig == 1)
    fOrig_name = ['Orig_Sub_rate_', num2str(subsampling_rate), '_X_', num2str(num_points_shape), '_Y_',num2str(num_points_Y)];
    print([im_fold, fOrig_name],'-depsc','-tiff');
    saveas(fOrig, [im_fold, fOrig_name, '.jpg'])
end   

%% Mean subtraction cases. 
[UX, UY] = grassmannianRepresentation(X, Y);

%% Display U eigenvector output. 

% Plot U eigenvectors.
if (plot_U_vecs == 1)
    plot2D3DShapes_Clean(UX,UY, {'UX', 'UY'}, 'U Eigenvectors', 'north');
end
view([view_tuple_x, view_tuple_y]);

if (print_flag_UVecs == 1)
    fUVec_name = ['UVec_Sub_rate_', num2str(subsampling_rate), '_X_', num2str(num_points_shape), '_Y_',num2str(num_points_Y)];
    print([im_fold, fUVec_name],'-depsc','-tiff');
    saveas(gcf, [im_fold, fUVec_name, '.jpg'])
end 

%% Graph Laplacian. 

% Form the epsilon-graph and its graph Laplacian.
p.Epsilon = X_p;              % Epsilon value. 
LX = graphLaplacian_Clean(UX, p);
p.Epsilon = Y_p;              % Epsilon value. 
LY = graphLaplacian_Clean(UY, p); 

%% Eigendecomposition of the LBO.

try % Error checking for eigenvector computation. 
    
    [XEvec, XEvals] = eigenDecompositionLBO_Clean(LX,p);
    [YEvec, YEvals] = eigenDecompositionLBO_Clean(LY,p);
catch
    % Increase the conditioning number. 
    prevCond = p.CondFac;
    p.CondFac = 1e5; 
    p.Epsilon = X_p;              % Epsilon value. 
    [XEvec, XEvals] = eigenDecompositionLBO_Clean(LX,p);
    p.Epsilon = Y_p;              % Epsilon value. 
    [YEvec, YEvals] = eigenDecompositionLBO_Clean(LY,p);

    p.CondFac = prevCond; % Replace the value with the original. 
end

numEvecsUsed = numel(p.EvecToMatch);

% Display the eigenvalues of UX. 
XEvals = diag(XEvals);           % Pull the eigenvalues off the diagonal. 
XEvals(end) = [];                % Remove the smallest or zeroth eigenvalue. 
XEvals = flipud(XEvals);         % Put the eigenvalues in ascending order. 
XEvals = XEvals(1:numEvecsUsed); % Choose a subset of the eigenvectors. 

if (dispEigVals == 1)
    disp(['Eigenvalues of LBO of UX = ' num2str(XEvals')]);
end

% Display the eigenvalues of UY. 
YEvals = diag(YEvals);           % Pull the eigenvalues off the diagonal.
YEvals(end) = [];                % Remove the smallest or zeroth eigenvalue.
YEvals = flipud(YEvals);         % Put the eigenvalues in ascending order. 
YEvals = YEvals(1:numEvecsUsed); % Choose a subset of the eigenvectors. 

YEvec_orig = YEvec;
YEvec = Y_scale_fac * YEvec;

if (dispEigVals == 1)
    disp(['Eigenvalues of LBO of UY = ' num2str(YEvals')]);
end

% Plot the LBO shapes.
if (plotLBO == 1)
    plot2D3DShapes_Clean(XEvec, YEvec, {'XEvec','YEvec'},...
        'LBO Eigenvectors Scaled', 'west');  
    view(view_LBO_x, view_LBO_y);
end

if (print_flag_LBO == 1)
    fLBO_name = ['LBO_sc_Sub_rate_', num2str(subsampling_rate), '_X_', num2str(num_points_shape), '_Y_',num2str(num_points_Y)];
    print([im_fold, fLBO_name],'-depsc','-tiff');
    saveas(gcf, [im_fold, fLBO_name, '.jpg']);
end 

% Plot the LBO shapes unscaled.
if (plotLBO_unscaled == 1)
    plot2D3DShapes_Clean(XEvec, YEvec_orig, {'XEvec','YEvec'},...
        'LBO Eigenvectors Unscaled', 'west');  
    view(view_LBO_x, view_LBO_y);
end

if (print_flag_LBO_unscaled == 1)
    fLBO_unsc_name = ['LBO_unsc_Sub_rate_', num2str(subsampling_rate), '_X_', num2str(num_points_shape), '_Y_',num2str(num_points_Y)];
    print([im_fold, fLBO_unsc_name],'-depsc','-tiff');
    saveas(gcf, [im_fold, fLBO_unsc_name, '.jpg']);
end 