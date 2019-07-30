  % Feature Extracion

function [feature] = FeatureExtraction(enhancement)

height = 4;
width = 4;

for y = -height:height-1
    for x = -width:width-1
        G1(height + y + 1, width + x + 1) = ...
            1/(2*pi*3*1.5) * exp((-1/2)*(x^2/(3^2) + y^2/(1.5^2))) * ...
            cos(2*pi*(1/1.5) * sqrt(x^2 + y^2));
        G2(height + y + 1, width + x + 1) = ...
            1/(2*pi*4.5*1.5) * exp((-1/2)*(x^2/(4.5^2) + y^2/(1.5^2))) * ...
            cos(2*pi*(1/1.5) * sqrt(x^2 + y^2));
    end
end
% 2 8*8 Gabor filters G1 and G2

enhance = enhancement;
ROI1 = conv2(enhance(1:48, :), G1, 'same'); % filtered image 1
ROI2 = conv2(enhance(1:48, :), G2, 'same'); % filtered image 2
% spatial domain convolution

feature = zeros(1536,1);

m = 1;
k = 1;
while m <= 48
    n = 1;
    while n <= 512
          mat1m = ROI1(m:m + 7, n:n + 7);
          val1 = mean(abs(mat1m(:)));
          feature(k,1) = val1;
          k = k + 1;
          % the mean of the 8*8 submatrix
          mat1s = abs(abs(mat1m(:)) - val1);
          feature(k,1) = mean(mat1s(:));
          k = k + 1;
          n = n + 8;
          % the average absolute deviation of the 8*8 submatrix
    end
    m = m + 8;
end
% feature of the 1st filtered image

m = 1;
k = 769;
while m <= 48
    n = 1;
    while n <= 512
          mat2m = ROI2(m:m + 7, n:n + 7);
          val2 = mean(abs(mat2m(:)));
          feature(k,1) = val2;
          k = k + 1;
          % the mean of the 8*8 submatrix
          mat2s = abs(abs(mat2m(:)) - val2);
          feature(k,1) = mean(mat2s(:));
          k = k + 1;
          n = n + 8;
          % the average absolute deviation of the 8*8 submatrix
    end
    m = m + 8;
end
% feature of the 2nd filtered image

end