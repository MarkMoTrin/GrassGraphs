%
% {GrassGraph Algorithm, used to perform affine invariant feature matching.}
%     Copyright (C) {2016}  {Mark Moyou}
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
%   
%     mmoyou@my.fit.edu
%     Information Characterization and Exploitation (ICE) Lab
% ------------------------------------------------------------------------------
function [corr, rA] = grassGraphsMatching(X, Y, plotUVecs, p)
% [corr, rA] = grassGraphsMatching(X, Y, p)

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
plotLBO = 0;   % Display the LBO eigenvectors.

debugMode = 0; % Shows the individual eigenvector flips. 

dispEigVals = 0;

%% Mean subtraction cases. 
[UX, UY] = grassmannianRepresentation(X, Y);


%% Display U eigenvector output. 

% Plot U eigenvectors.
if (plotUVecs == 1)
    plot2D3DShapes_Clean(UX,UY, {'UX', 'UY'}, 'U Eigenvectors', 'northeast');
end

%% Graph Laplacian. 

% Form the epsilon-graph and its graph Laplacian.
LX = graphLaplacian_Clean(UX, p);
LY = graphLaplacian_Clean(UY, p);        


%% Eigendecomposition of the LBO.

try     % Error checking for eigenvector computation. 
    [XEvec, XEvals] = eigenDecompositionLBO_Clean(LX,p);
    [YEvec, YEvals] = eigenDecompositionLBO_Clean(LY,p);
catch
    % Increase the conditioning number. 
    prevCond = p.CondFac;
    p.CondFac = 1e5; 
    [XEvec, XEvals] = eigenDecompositionLBO_Clean(LX,p);
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

% Determine the optimal eigenvector sign flip and return the 
% correpondences.
[corr.X, corr.Y, ~]...
 = correspondenceData_Clean(singCorrIndVecCell, totalScoreVec,...
      corrIndCell, X, Y, p); 
    
rA = recoveredAffine_Clean(corr.X, corr.Y); % Recover the affine.

        
