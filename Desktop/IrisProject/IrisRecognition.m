% Read images

% Read train images
photo = {}; % create a empty cell
a = 1;
while a <= 108 % number of folders
    b = 1;
    while b <= 3 % number of images in each subfolder      
if a < 10
photo(a, b) = {imread(['/Users/Tass/Desktop/Data/Iris/00',num2str(a),'/1/00',...
           num2str(a),'_1_',num2str(b),'.bmp'])}; % read images in folders 00x
elseif a < 100
photo(a, b) = {imread(['/Users/Tass/Desktop/Data/Iris/0',num2str(a),'/1/0',...
           num2str(a),'_1_',num2str(b),'.bmp'])}; % read images in folders 0xx
else
photo(a, b) = {imread(['/Users/Tass/Desktop/Data/Iris/',num2str(a),'/1/',...
           num2str(a),'_1_',num2str(b),'.bmp'])}; % read images in folders xxx
end
    b = b + 1;
    end
    a = a + 1;
end

% Read test images
a = 1;
while a <= 108
    b = 1;
    while b <= 4       
if a < 10
photo(a, b + 3) = {imread(['/Users/Tass/Desktop/Data/Iris/00',num2str(a),'/2/00',...
           num2str(a),'_2_',num2str(b),'.bmp'])};
elseif a < 100
photo(a, b + 3) = {imread(['/Users/Tass/Desktop/Data/Iris/0',num2str(a),'/2/0',...
           num2str(a),'_2_',num2str(b),'.bmp'])};
else
photo(a, b + 3) = {imread(['/Users/Tass/Desktop/Data/Iris/',num2str(a),'/2/',...
           num2str(a),'_2_',num2str(b),'.bmp'])};
end
    b = b + 1;
    end
    a = a + 1;
end

% Localization

for c = 1:108
    for d = 1:7
    [output1{c, d}, rp{c, d}, ri{c, d}, cp{c, d}, ci{c, d}] = ...
    IrisLocalization(cell2mat(photo(c, d)));
    end
end

% apply function Localization for all 108*7 images
% and save output in a 108*7 cell

% Normalization

for c = 1:108
    for d = 1:7
    output2{c, d} = IrisNormalization(cell2mat(output1(c, d)), ...
                    cell2mat(rp(c, d)), cell2mat(ri(c, d)), ...
                    cell2mat(cp(c, d)), cell2mat(ci(c, d)));
    end
end

% apply function Normalization for all 108*7 images
% and save output in a 108*7 cell

% Enhancement

for c = 1:108
    for d = 1:7
    enhancement{c, d} = ImageEnhancement(cell2mat(output2(c, d)));
    end
end

% apply function Normalization for all 108*7 images
% and save output in a 108*7 cell

% Feature Extraction
for c = 1:108
    for d  = 1:3
    feature_train(:, 108*(d - 1) + c) = FeatureExtraction(cell2mat(enhancement(c, d)));
    end
end

% apply function Feature Extraction for 108*3 train images
% and save output in a 1536*324 matrix
% i.e. for folder 001, 1_1 is the 1st column, 1_2 is the 109th column
% 1_3 is the 217th column

for c = 1:108
    for d  = 4:7
    feature_test(:, 108*(d - 4) + c) = FeatureExtraction(cell2mat(enhancement(c, d)));
    end
end

% apply function Feature Extraction for 108*4 test images
% and save output in a 1536*432 matrix
% i.e. for folder 001, 1_1 is the 1st column, 1_2 is the 109th column
% 1_3 is the 217th column, 1_4 is the 325th column

feature = [feature_train, feature_test];

% combine above two matrices together

% IrisMatching
% LDA_test = LDA(feature_test);
% LDA_train = LDA(feature_train);

% The LDA transform. Something is wrong during the process on the website
% provided, complex numbers in the LDA output 

[d1, d2, d3] = IrisMatching(feature_test, feature_train);

% apply function IrisMatching to the 1536*756 matrix
% and calculate L1 distance, L2 distance and cosine similarity

% Performance Evaluation
[CRR1, CRR2, CRR3] = PerformanceEvaluation(d1, d2, d3);

% apply function Performance Evaluation to 3 distance matrices
% and calculate CRR