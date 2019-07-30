% LDA function

function [LDA_output] = LDA(feature)

% Step 1: Computing the d-dimensional mean vectors

for i = 1:1536
    for k = 1:108
        mean_vec(i,k) = ...
        (feature(i,k) + feature(i,k+108) + feature(i,k+216))/3;
    end

% Step 2: Computing the Scatter Matrices

% Within class matrix sw

sw = zeros(1536,1536);

for k = 1:108
    w1 = [feature(:,k), feature(:,k+108), feature(:,k+216)];
    w2 = w1 - mean_vec(:,k);
    sw = sw + w2 * w2';
end

% Between class matrix sb

sb = zeros(1536,1536);

for k = 1:108
    b1 = mean_vec(:,k) - overall_mean(i,1);
    b2 = 3 * b1 * b1';
    sb = sb + b2;
end

% Step 3: Solving the generalized eigenvalue problem

target_mat = inv(sw) * sb; % (sw)^-1 * sb
[vec, val] = eig(target_mat); 
% vec is the eigenvector matrix and val is the eigenvalue matrix

% Step 4: Selecting linear discriminants for the new feature subspace

W = vec(:, 1:107); % select 107 eigenvectors which have larger eigenvalues

% Step 5: Transforming the samples onto the new subspace

LDA_output = W' * feature; % f = W^t * f
end