function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for i=1:size(idx,1) % For each example
    
     distancesVector = sum(((centroids - X(i,:)).^2),2); % currently 3x1 each row stores the distance b/w example and the centroid
     closestCentroidIndex = find(distancesVector == min(distancesVector)); % If an example is at the same distance to more than 1 centroid then it will be vector. Choose the first element.       
     idx(i) = closestCentroidIndex(1);    

end


% =============================================================

end

