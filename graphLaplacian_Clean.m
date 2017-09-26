%--------------------------------------------------------------------------
% Function:    graphLaplacian_Clean
% Description: This function constructs an epsilon graph for a particular
%              epsilon and then computes the graph Laplacian of this graph.
%              The graph Laplacian can either be a weighted graph or a
%              simple adjacency computations. The weighted graphs can be
%              either a heat kernel weighting or just the Euclidean
%              distance weights. 
% 
% Inputs: 
%
%   data            - Can be any NxD data matrix, for this framework we 
%                     form the graph on the U eigenvectors from the SVD 
%                     decomposition of the affine representation of the 
%                     shape.  
%
%   p.epsilon       - Epsilon for epsilon neighborhood graph. 
%
%   p.graphType     - Flag for the type of graph to construct. 
%                     'eucDist' - Euclidean distances as the weights. 
%                     'heatKer' - Heat kernel weights. 
%                     'connect' - Just the connectivity graph using the
%                                 adjacency matrix. 
%
%   p.hkt            - Value for the heat kernel. 
%
% Outputs.
%
%   L                - Graph Laplacian.
% 
% Usage: Used in the affine invariance framework.
%
% -------------------------------------------------------------------------
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
function L = graphLaplacian_Clean(data, p)

A = pdist2(data, data).^2;    % Squared distance matrix.
% figure; hist(A(:)); movegui(gcf,'west');

epsilon = adaptive_histogram_epsilon(A, p.s);
p.Epsilon = epsilon;

A(A > p.Epsilon) = 0;       % Create the epsilon graph. 
% figure; surf(A); view(0,90);
A = sparse(A);              % Euclidean distance weighted adjacency matrix. 
connPos = logical(A == 0);  % Connection positions.

switch p.GraphLapType
    case 'eucDist'
        
        D = diag(sum(A,2)); % Diagonal sum of weights.  
        L = D - A;          % Graph Laplacian.         
        
    case 'heatKer'  % Heat Kernel weights.
        
        A = exp(-A/p.hkt);      % Weighted adjacency matrix.
        A(connPos) = 0;
        D = diag(sum(A,2)); % Diagonal sum of weights.  
        L = D - A;          % Graph Laplacian. 
    
    case 'connect'  % Connectivity only.
        
        A = A > 0;          % Connectivity adjacency matrix. 
        D = diag(sum(A,2)); % Diagonal sum of weights.  
        L = D - A;          % Graph Laplacian. 
        
end

        
