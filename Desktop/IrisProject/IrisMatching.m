% IrisMatching

function [d1,d2,d3] = IrisMatching(feature_test, feature_train)
 
for i = 1:324
    for j = 1:432
    d1(j,i) = sum(abs(feature_test(:,j) - feature_train(:,i)));
    % L1 distance between each pair of test image and train image
    end
end

for i = 1:324
    for j = 1:432
    d2(j,i) = (feature_test(:,j) - feature_train(:,i))'* ...
              (feature_test(:,j) - feature_train(:,i));
    % L2 distance between each pair of test image and train image
    end
end

for i = 1:324
    for j = 1:432
    d3(j,i) = 1 - feature_test(:,j)' * feature_train(:,i)/...
              (norm(feature_test(:,j)) * norm(feature_train(:,i)));
    % cosine similarity between each pair of test image and train image
    end
end

end