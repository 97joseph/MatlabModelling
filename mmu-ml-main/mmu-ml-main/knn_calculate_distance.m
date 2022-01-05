% TODO: add validation?
% Description: calculate the Euclidean distance between any two
% points
%
% Inputs:
% p: an array containing the coordinates of the first point
% q: an array containing the coordinates of the second point
% 
% Outputs:
% d: a numeric value holding the straight-line distance
% between the two points
function d = knn_calculate_distance(p, q)
    % square (this helps remove negative values) each of the differences and sum them then compute
    % the square root to get the unsquared value
    d = sqrt(sum((p-q) .^ 2));
end