% Enhancement

% Background illumination
% m = 1;
% mat16output = zeros(4,32);
% while m <= 64
%     n = 1;
%     while n <= 512
%           mat = output2(m:m + 15, n:n + 15);
%           mat16mean = mean(mat(:));
%           mat16output((m + 15)/16, (n + 15)/16) = mat16mean;
%           n = n + 16;
%     end
%     m = m + 16;
% end
% 
% output3 = round(imresize(mat16output, 16, 'bicubic'));
% output3 = uint8(output3);

% I get the background illumination but don't know how to use it

% Enhancement
function [enhance] = ImageEnhancement(output2)
enhance = zeros(64,512);
m = 1;
while m <= 64
    n = 1;
    while n <= 512
          mat = output2(m:m + 31, n:n + 31); % 32*32 subimages
          enhance(m:m + 31, n:n + 31) = histeq(mat); % enhance the result
          n = n + 32;
    end
    % enhance all 32*32 subimages
    m = m + 32;
end
enhance = uint8(enhance);
% imshow(enhance);
end