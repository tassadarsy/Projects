% Normalization

function [output2] = IrisNormalization(output1, rp, ri, cp, ci)

rp = round(rp); % we need rp be positive integer in this function
ri = round(ri); % we need ri be positive integer in this function

M = 64; % number of rows in the output
N = 512; % number of columns in the output
theta = 2*pi/N; % split the circle to N pieces

% By columns
for j = 1:N
    point_A_x(1,j) = cp(2) + rp*cos(j*theta);
    point_A_y(1,j) = cp(1) + rp*sin(j*theta); % Inner circle location
    point_B_x(1,j) = ci(2) + ri*cos(j*theta);
    point_B_y(1,j) = ci(1) + ri*sin(j*theta); % Outer circle location
end

% By rows
for i = 1:M
    point_x(i,:) = round(point_A_x(1,:) + (point_B_x(1,:) - point_A_x(1,:))*i/M);
    point_y(i,:) = round(point_A_y(1,:) + (point_B_y(1,:) - point_A_y(1,:))*i/M);
end

% Fill the new M*N matrix
for i = 1:M
    for j = 1:N
if point_x(i,j) > 280
   point_x(i,j) = 280;
elseif point_x(i,j) < 1
       point_x(i,j) = 1;
end
    end
end
% deal with outer circle indices exceed the boundary along the x direction

for i = 1:M
    for j = 1:N
if point_y(i,j) > 320
   point_y(i,j) = 320;
elseif point_y(i,j) < 1
       point_y(i,j) = 1;
end
    end
end
% deal with outer circle indices exceed the boundary along the y direction

for i = 1:M
    for j = 1:N
        output2(i,j) = output1(point_x(i,j),point_y(i,j));
    end
end
% fill the output matrix

% imshow(output2);
end
