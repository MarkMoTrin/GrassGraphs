% Script for unequal points correspondence. 
clc; clear; close all;

shape_folder = './Datasets/2D/';
shape_name = 'airplane.mat';
im_fold = 'C:\Users\mmoyou\Dropbox\Pami_revision\imgs_unequal_points\';

%% Print im flags.
print_flag_orig = 1; 
print_flag_UVecs = 0; 

apply_affine = 1;

generate_Y = 1;
partial_Y = 0;

num_partial_points = 100;

sub_sampling_rate = 1;
start_ind_sub_samp = 1;

%% Loading and plotting the shapes. 

% Load the shape.
X = load([shape_folder, shape_name]); 
X = X.x;
X = resizeShapesToSquareGrid(X, 10);

% Number of points to subsample by.
num_points_shape = size(X,1);

if (partial_Y == 1)
    Y = X(start_ind_sub_samp : start_ind_sub_samp + sub_sampling_rate, :);
    num_points_Y = size(Y,1);
else
    % Subsample the shape.
    Y = X(1 : sub_sampling_rate : end,:);
    num_points_Y = size(Y,1);
end

fOrig = figure;
movegui(gcf, 'northwest');
plot(X(:, 1), X(:, 2), 'ro', 'Markersize',6); hold on;
plot(Y(:, 1), Y(:, 2), 'b.', 'Markersize',5)
title(['|X| = ', num2str(num_points_shape), ', |Y| = ', num2str(size(Y,1))]);
legend('X', 'Y');
% plot2D3DShapes_Clean(Y, X, {'Y', 'X'}, 'Original shapes', 'northwest');
% Get the Laplacian eigenvectors of length X_n and Y_n.

if (print_flag_orig == 1)
    fOrig_name = ['Subsample_X_', num2str(num_points_shape), '_Y_',num2str(num_points_Y)];
    print([im_fold, fOrig_name],'-depsc','-tiff');
    saveas(fOrig, [im_fold, fOrig_name, '.jpg'])
end    
%% ----------------------------------

if (apply_affine == 1)
    % Setup an affine transformation. 
    theta = 0;      % Rotation angle 1. 
    sigma = pi/2;   % Rotation angle 2.
    a = 3;      % Scale in the x direction. 
    b = 1;      % Scale in the y direction. 
    tx = 0;     % Translation in x. 
    ty = 0;     % Translation in y. 

    tVec = [tx, ty];
    % Generate an affine transformation. 
    A = affineTransformation2D_Clean(theta, sigma, tVec, a, b);
    
    if (generate_Y == 1)
        num_shape_points = size(X,1);
        Xh = [X ones(size(X,1),1)]; % Homogeneous representation of X. 
        Y = Xh*A; % Apply the affine to create Y. 
        Y = Y(:,1:2);  % Keep only the first 2 dimensions.
    else
        Yh = [Y ones(size(Y,1),1)]; % Homogeneous representation of X. 
        Y = Yh*A; % Apply the affine to create Y. 
        Y = Y(:,1:2);  % Keep only the first 2 dimensions. 
    end
end
%% ----------------------------

X_p = 0.012;
Y_p = 0.012;
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

% ------------------------------------------------

% Check if X and Y are N x D. 
[numXRow, numXCol] = size(X);
if (numXRow < numXCol)
    X = X';
end

[numYRow, numYCol] = size(Y);
if (numYRow < numYCol)
    Y = Y';
end

%% Flags.
plotLBO = 1;   % Display the LBO eigenvectors.
plotUVecs = 1;
plotRecShape = 1;
debugMode = 0; % Shows the individual eigenvector flips. 
dispEigVals = 0;

%% Mean subtraction cases. 
[UX, UY] = grassmannianRepresentation(X, Y);


%% Display U eigenvector output. 

% Plot U eigenvectors.
if (plotUVecs == 1)
    plot2D3DShapes_Clean(UX,UY, {'UX', 'UY'}, 'U Eigenvectors', 'north');
end

view([-1, 90]);
if (print_flag_UVecs == 1)
    fUVec_name = ['UVecs', num2str(num_points_shape), '_Y_',num2str(num_points_Y)];
    print([im_fold, fUVec_name],'-depsc','-tiff');
end 

%% Graph Laplacian. 

% Form the epsilon-graph and its graph Laplacian.
p.Epsilon = X_p;              % Epsilon value. 
LX = graphLaplacian_Clean(UX, p);
p.Epsilon = Y_p;              % Epsilon value. 
LY = graphLaplacian_Clean(UY, p);     

% norm(LX - LY, 'fro')
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

XEvec = X_scale_fac * XEvec;

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

if (dispEigVals == 1)
    disp(['Eigenvalues of LBO of UY = ' num2str(YEvals')]);
end

% Plot the LBO shapes.
if (plotLBO == 1)
    plot2D3DShapes_Clean(XEvec, YEvec, {'XEvec','YEvec'},...
        'LBO eigenvectors ', 'west');            
end

%% Correspondence Matching.

% Score between the source and permutated target shape. The minimum
% of these scores is used to choose the best permutation of the
% eigenvector sign flips.
totalScoreVec = zeros(p.NumPerms,1);

% Cell that stores the indices retrieved from the distance
% computation of the source shape to the permuted target shape. 
% These indices are used to re-order the original point sets. Note 
% that the distance computation may assign the same point in the 
% target shape to multiple points in the source shape. In this 
% case we need to eliminate those correspondences from the target 
% shape. The indices of the multiple correspondences in the target
% shape will be stored in singCorrIndVecCell.
corrIndCell = cell(p.NumPerms, 1); % Correspondence index cell. 

% The indices are the row indices that have a single correspondence.
% Storing the vector for each permutation.
singCorrIndVecCell = corrIndCell; 

% Debuggin the correspondence curve. 
if (debugMode == 1)
    % Make a big figure.
    figure('units','normalized','outerposition',[0 0 1 1]);
end

% Correspondence matching. We need to loop through the permutations 
% of the eigenvector to see which one gives us a minimum score. 
for k = 1 : p.NumPerms
    
    % Compute the permuted version of the target shape (modified 
    % target.  
    modY = bsxfun(@times, YEvec, p.AllPerms(k,:));
    
    % Compute the distance between the source and target shape.
    % There may be a scaling on the distance matrix depending on
    % the user's choice.
    [totalScoreVec(k), singCorrIndVecCell{k}, corrIndCell{k}] =...
                             doubleCorrespondences_Clean(XEvec,modY, p);
    
    % Plot each set of eigenshapes for the permutation.
    if (debugMode == 1)
        subplot(4,4,k);
        plot3(XEvec(:,1), XEvec(:,2), XEvec(:,3), 'ro'); hold on;
        plot3(modY(:,1), modY(:,2), modY(:,3), 'bs'); 

        % Number of correspondences. 
        numCorr = sum(singCorrIndVecCell{k}); 
        disp([numCorr totalScoreVec(k)]);
        title(['Scores:' num2str(numCorr) ' T score:'...
                    num2str(totalScoreVec(k))]);
    end        
end           

%% Correspondence Matching.

% Score between the source and permutated target shape. The minimum
% of these scores is used to choose the best permutation of the
% eigenvector sign flips.
totalScoreVec = zeros(p.NumPerms,1);

% Cell that stores the indices retrieved from the distance
% computation of the source shape to the permuted target shape. 
% These indices are used to re-order the original point sets. Note 
% that the distance computation may assign the same point in the 
% target shape to multiple points in the source shape. In this 
% case we need to eliminate those correspondences from the target 
% shape. The indices of the multiple correspondences in the target
% shape will be stored in singCorrIndVecCell.
corrIndCell = cell(p.NumPerms, 1); % Correspondence index cell. 

% The indices are the row indices that have a single correspondence.
% Storing the vector for each permutation.
singCorrIndVecCell = corrIndCell; 

% Debuggin the correspondence curve. 
if (debugMode == 1)
    % Make a big figure.
    figure('units','normalized','outerposition',[0 0 1 1]);
end

% Correspondence matching. We need to loop through the permutations 
% of the eigenvector to see which one gives us a minimum score. 
for k = 1 : p.NumPerms
    
    % Compute the permuted version of the target shape (modified 
    % target.  
    modY = bsxfun(@times, YEvec, p.AllPerms(k,:));
    
    % Compute the distance between the source and target shape.
    % There may be a scaling on the distance matrix depending on
    % the user's choice.
    [totalScoreVec(k), singCorrIndVecCell{k}, corrIndCell{k}] =...
                             doubleCorrespondences_Clean(XEvec,modY, p);
    
    % Plot each set of eigenshapes for the permutation.
    if (debugMode == 1)
        subplot(4,4,k);
        plot3(XEvec(:,1), XEvec(:,2), XEvec(:,3), 'ro'); hold on;
        plot3(modY(:,1), modY(:,2), modY(:,3), 'bs'); 

        % Number of correspondences. 
        numCorr = sum(singCorrIndVecCell{k}); 
        disp([numCorr totalScoreVec(k)]);
        title(['Scores:' num2str(numCorr) ' T score:'...
                    num2str(totalScoreVec(k))]);
    end        
end           

% Determine the optimal eigenvector sign flip and return the 
% correpondences.
[corr.X, corr.Y, ~]...
 = correspondenceData_Clean(singCorrIndVecCell, totalScoreVec,...
      corrIndCell, X, Y, p); 
    
rA = recoveredAffine_Clean(corr.X, corr.Y); % Recover the affine.

%% Recover the affine and correspondences using the GrassGraphs algorithm.
Xh = [X ones(size(X,1),1)]; % Homogeneous representation of X. 

recY = Xh*rA; % Form the recovered shape. 
recY = recY(:,1:2); % Remove the extra homogeneous dimension. 

% Norm between the recovered affine and the true affine. 
% normAff = norm(rA - A, 'fro') ;

if (plotRecShape == 1)
    plot2D3DShapes_Clean(Y,recY,{'Y', 'Recovered Y'},...
        ['Recovered Shape'], 'south');
end
% Set a range of scale values and search for correspondence.

% Look at the scaled shapes and see if playing with s produces easier
% slopes

% Check the scaling of each eigenvector.

% How do the number of connections in the graph affect the eigenvectors. 

% Change the level of subsampling and show eigvectors. 

