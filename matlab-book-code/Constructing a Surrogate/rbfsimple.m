function rbf
% Estimates the parameters of a Radial Basis Function model (all other
% information about the model is expected to exist in the ModelInfo global
% variable).
%
% Copyright 2007 A Sobester
%
% This program is free software: you can redistribute it and/or modify  it
% under the terms of the GNU Lesser General Public License as published by
% the Free Software Foundation, either version 3 of the License, or any
% later version.
% 
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
% General Public License for more details.
% 
% You should have received a copy of the GNU General Public License and GNU
% Lesser General Public License along with this program. If not, see
% <http://www.gnu.org/licenses/>.

global ModelInfo

if ModelInfo.Code < 4
	% Fixed basis function, only w needs estimating
	rbfestimw
	
function rbfestimw
% Estimates the basis function weights w if there is no other parameter
% (sigma), or this is known already and included in ModelInfo

global ModelInfo


% This should have the following fields defined:
% Training data: ModelInfo.X
%                ModelInfo.y
% Basis function type:
%                ModelInfo.Code
% If basis function type > 3
% additional basis function parameter
%                ModelInfo.Sigma


if (ModelInfo.Code > 3) && (~isfield(ModelInfo,'Sigma'))
    error('The basis function in ModelInfo.Code requires a Sigma.')
end

%Number of points
n = length(ModelInfo.y);

% Build Gram matrix
d = zeros(n,n);
for i=1:n
   for j=1:i
      d(i,j) = norm(ModelInfo.X(i,:)-ModelInfo.X(j,:),2);      
      d(j,i) = d(i,j);
   end
end

% Constructing the PHI matrix
ModelInfo.Phi = zeros(n,n);
for i = 1:n
   for j = 1:i
      if isfield(ModelInfo,'Sigma')
          ModelInfo.Phi(i,j) = basis(ModelInfo.Code, d(i,j), ModelInfo.Sigma);
      else
          ModelInfo.Phi(i,j) = basis(ModelInfo.Code, d(i,j));
      end
      ModelInfo.Phi(j,i) = ModelInfo.Phi(i,j);
   end
end

% Calculating the weights of the radial basis function surrogate
% Cholesky factorization used if Gaussian or inverse mq required
% LU decomposition otherwise

if ModelInfo.Code==4 ||  ModelInfo.Code==6
    [U,p] = chol(ModelInfo.Phi);
    if p==0
        ModelInfo.Weights = U\(U'\ModelInfo.y);
        ModelInfo.Success = 1;
    else
        display('Cholesky factorization failed.');
        display('Two points may be too close together.');
        ModelInfo.Weights = [];
        ModelInfo.Success = 0; 
    end
else
    ModelInfo.Weights = ModelInfo.Phi\ModelInfo.y;
    if isempty(ModelInfo.Weights)
        ModelInfo.Success = 0;
    else
        ModelInfo.Success = 1;
    end
end
