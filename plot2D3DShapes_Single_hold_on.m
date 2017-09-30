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
function plot2D3DShapes_Single_hold_on(X,col_str,f)

% plot2D3DShapes(X,Y,varargin)
% position = varargin{1};
% titleStr = varargin{2};

% Check if X and Y are N x D. 
[numXRow, numXCol] = size(X);
if (numXRow < numXCol)
    X = X';
end

[~, nDim] = size(X); 

% Show the shape.
if (nDim == 2)
    % 2D plots. 
    figure(f);
    plot(X(:,1), X(:,2), col_str); hold on; 
else
    % 3D plots. 
    figure(f); 
    plot3(X(:,1), X(:,2), X(:,3), col_str, 'Linewidth', 1); 
    
end
hold on;
