% Performance Evaluation

function [CRR1, CRR2, CRR3] = PerformanceEvaluation(d1, d2, d3)

% L1 distance measure
k = 0;
for i = 1:432
    pos1(i) = find(d1(i,:) == min(d1(i,:))); 
    % find the minimum distance in a row(a test image) and record it
    if pos1(i) == mod(i,108) % if the shortest distance is xxx_1_1
       k = k + 1; 
    elseif pos1(i) == mod(i,108) + 108 % if the shortest distance is xxx_1_2
       k = k + 1;
    elseif pos1(i) == mod(i,108) + 216 % if the shortest distance is xxx_1_3
       k = k + 1;
    elseif pos1(i) == mod(i,108) + 324 % consider when the mod equals 0
       k = k + 1;
    end
end

CRR1 = k/432; % calculate L1 CRR

% L2 distance meansure
k = 0;
for i = 1:432
    pos2(i) = find(d2(i,:) == min(d2(i,:)));
    if pos2(i) == mod(i,108) % if the shortest distance is xxx_1_1
       k = k + 1; 
    elseif pos2(i) == mod(i,108) + 108 % if the shortest distance is xxx_1_2
       k = k + 1;
    elseif pos2(i) == mod(i,108) + 216 % if the shortest distance is xxx_1_3
       k = k + 1;
    elseif pos2(i) == mod(i,108) + 324 % consider when the mod equals 0
       k = k + 1;
    end
end

CRR2 = k/432; % calculate L2 CRR

% cosine similarity measure
k = 0;
for i = 1:432
    pos3(i) = find(d3(i,:) == min(d3(i,:)));
    if pos3(i) == mod(i,108) % if the shortest distance is xxx_1_1
       k = k + 1; 
    elseif pos3(i) == mod(i,108) + 108 % if the shortest distance is xxx_1_2
       k = k + 1;
    elseif pos3(i) == mod(i,108) + 216 % if the shortest distance is xxx_1_3
       k = k + 1;
    elseif pos3(i) == mod(i,108) + 324 % consider when the mod equals 0
       k = k + 1;
    end
end

CRR3 = k/432; % calculate L3 CRR

end