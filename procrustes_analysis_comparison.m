% Script for unequal points correspondence original shapes equal. 
clc; clear; close all;

shape_folder = './Datasets/2D/';
shape_name = 'airplane.mat';
im_fold = 'C:\Users\mmoyou\Dropbox\Pami_revision\imgs_unequal_points\equal\';

%% ----------------------------

X_p = 0.0012;
Y_p = 0.0012;
X_scale_fac = 1;

p.fhkt = 100;                   % For the full graph case, not used. 
p.Epsilon = 0.012;              % Epsilon value. 
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
p.s = 0.7
%% Print im flags.
print_flag_orig = 0; 
print_flag_UVecs = 0; 
print_flag_LBO = 0; 

dispEigVals = 0;
plot_U_vecs = 1;
plotLBO = 1;

affine_level = 'small';
% affine_level = 'medium';
% affine_level = 'large';

affine_number = 1;

view_tuple_x = 131;
view_tuple_y = -68;
view_LBO_x = 1;
view_LBO_y = 2;

%% Loading and plotting the shapes. 

% Load the shape.
X = load([shape_folder, shape_name]); 
X = X.x;
X = resizeShapesToSquareGrid(X, 10);

% Setup an affine transformation. 
theta = 0;      % Rotation angle 1. 
sigma = pi/2;   % Rotation angle 2.
a = 0;      % Scale in the x direction. 
b = 0;      % Scale in the y direction. 
tx = 0;     % Translation in x. 
ty = 0;     % Translation in y. 

tVec = [tx, ty];

% Generate an affine transformation. 
A = affineTransformation2D_Clean(theta, sigma, tVec, a, b);

Xh = [X ones(size(X,1),1)]; % Homogeneous representation of X. 
Y = Xh*A; % Apply the affine to create Y. 
Y = Y(:,1:2);  % Keep only the first 2 dimensions. 

% Number of points to subsample by.
num_points_shape = size(X,1);
num_points_Y = size(Y,1);

fOrig = figure;
movegui(gcf, 'northwest');
plot(X(:, 1), X(:, 2), 'ro', 'Markersize',3); hold on;
plot(Y(:, 1), Y(:, 2), 'b.', 'Markersize',5)
title('Original Shapes');
legend('X', 'Y');

if (print_flag_orig == 1)
    fOrig_name = ['equal_points_', affine_level '_', num2str(affine_number)];
    print([im_fold, fOrig_name],'-depsc','-tiff');
    saveas(fOrig, [im_fold, fOrig_name, '.jpg'])
end   

%% Mean subtraction cases. 
[UX, UY] = grassmannianRepresentation(X, Y);

% Apply the affine to UX.
thetas = [pi/2, 0, pi/3];
sigmas = zeros(3,1);
tVec = sigmas;
sc = sigmas;
A = affineTransformation3D_Clean(thetas, sigmas, tVec, sc)

UXh = [UX ones(size(X,1),1)]; % Homogeneous representation of X. 
UY = UXh*A; % Apply the affine to create Y. 
UY = UY(:,1:3);  % Keep only the first 2 dimensions. 
UY = UY(randperm(size(UY,1)),:);

f = figure; movegui(f, 'south');
plot2D3DShapes_Single_hold_on(UX, 'g*', f)
plot2D3DShapes_Single_hold_on(UY, 'ro', f)

% Recover the transformation from UX to UY.
[d,Z,transform] = procrustes(UY,UX);

XX = transform.b*UX*transform.T + transform.c;
plot2D3DShapes_Single_hold_on(XX, 'b.', f)

legend('UX', 'UY', 'XX')

transform.T
norm(A(1:3,1:3) - transform.T, 'fro')

% 
% 
% %% Display U eigenvector output. 
% 
% % Plot U eigenvectors.
% if (plot_U_vecs == 1)
%     plot2D3DShapes_Clean(UX,UY, {'UX', 'UY'}, 'U Eigenvectors', 'north');
% end
% view([view_tuple_x, view_tuple_y]);
% 
% if (print_flag_UVecs == 1)
%     fUVec_name = ['UVecs_', affine_level '_', num2str(affine_number)];
%     print([im_fold, fUVec_name],'-depsc','-tiff');
%     saveas(gcf, [im_fold, fUVec_name, '.jpg'])
% end 
% 
% 
% %% Graph Laplacian. 
% 
% % Form the epsilon-graph and its graph Laplacian.
% p.Epsilon = X_p;              % Epsilon value. 
% LX = graphLaplacian_Clean(UX, p);
% p.Epsilon = Y_p;              % Epsilon value. 
% LY = graphLaplacian_Clean(UY, p); 

% %% Eigendecomposition of the LBO.
% 
% try % Error checking for eigenvector computation. 
%     
%     [XEvec, XEvals] = eigenDecompositionLBO_Clean(LX,p);
%     [YEvec, YEvals] = eigenDecompositionLBO_Clean(LY,p);
% catch
%     % Increase the conditioning number. 
%     prevCond = p.CondFac;
%     p.CondFac = 1e5; 
%     p.Epsilon = X_p;              % Epsilon value. 
%     [XEvec, XEvals] = eigenDecompositionLBO_Clean(LX,p);
%     p.Epsilon = Y_p;              % Epsilon value. 
%     [YEvec, YEvals] = eigenDecompositionLBO_Clean(LY,p);
% 
%     p.CondFac = prevCond; % Replace the value with the original. 
% end
% 
% numEvecsUsed = numel(p.EvecToMatch);
% 
% XEvec = X_scale_fac * XEvec;
% 
% % Display the eigenvalues of UX. 
% XEvals = diag(XEvals);           % Pull the eigenvalues off the diagonal. 
% XEvals(end) = [];                % Remove the smallest or zeroth eigenvalue. 
% XEvals = flipud(XEvals);         % Put the eigenvalues in ascending order. 
% XEvals = XEvals(1:numEvecsUsed); % Choose a subset of the eigenvectors. 
% 
% if (dispEigVals == 1)
%     disp(['Eigenvalues of LBO of UX = ' num2str(XEvals')]);
% end
% 
% % Display the eigenvalues of UY. 
% YEvals = diag(YEvals);           % Pull the eigenvalues off the diagonal.
% YEvals(end) = [];                % Remove the smallest or zeroth eigenvalue.
% YEvals = flipud(YEvals);         % Put the eigenvalues in ascending order. 
% YEvals = YEvals(1:numEvecsUsed); % Choose a subset of the eigenvectors. 
% 
% if (dispEigVals == 1)
%     disp(['Eigenvalues of LBO of UY = ' num2str(YEvals')]);
% end
% 
% % Plot the LBO shapes.
% if (plotLBO == 1)
%     plot2D3DShapes_Clean(XEvec, YEvec, {'XEvec','YEvec'},...
%         'LBO eigenvectors ', 'west');  
%     view(view_LBO_x, view_LBO_y);
% end
% 
% if (print_flag_LBO == 1)
%     fLBO_name = ['LBO_', affine_level '_', num2str(affine_number)];
%     print([im_fold, fLBO_name],'-depsc','-tiff');
%     saveas(gcf, [im_fold, fLBO_name, '.jpg']);
% end
