function epsilon = adaptive_histogram_epsilon(A, s)

% Get the histogram and the bins.
[h, x] = hist(A(:));

% Get the sum of the historgram.
sum_h = sum(h);

% Get the cumsum of the histogram.
cumsum_h = cumsum(h);

% Find the indices of x where the cumsum of h is less than s*sum_h.
x_indices = find(cumsum_h < sum_h*s);

% Get the end index. 
x_end_index = x_indices(end);

% Use this index to get an epsilon.
epsilon = x(x_end_index);