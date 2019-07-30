% Localization

function [output1, rp, ri, cp, ci] = IrisLocalization(photo)
pupil_area = photo(60:220,80:240); % select only the center area

mode_acc = zeros(1,65);
v = 1;
while v <= 65
    for i = 1:161
        for j = 1:161
            if pupil_area(i,j) == v
               mode_acc(1,v) = mode_acc(v) + 1;
            end
        end
    end
    v = v + 1;
end
mode_cal = find(mode_acc == max(mode_acc)); % mode under 65 in the pupil area

mode_photo = mode(pupil_area(:)); % find the mode in the pupil area

% for i = 60:220
%     for j = 100:220
%         if photo(i, j) == mode_photo && photo(i+1, j+1) == mode_photo
%             x1 = i;
%             y1 = j;
%             break
%         end
%     end
% end

% search from the left-top to find the first pixel which equals the pupil

% for i = 220:-1:60
%     for j = 220:-1:100
%         if photo(i, j) == mode_photo && photo(i-1, j-1) == mode_photo
%             x2 = i;
%             y2 = j;
%             break
%         end
%     end
% end

% search from the right-bottom to find the first pixel which equals the pupil

% xc = round((x1 + x2)/2);
% yc = round((y1 + y2)/2);

% the approximated center of the pupil 

for i = 1:280
    mode_row(i) = sum(photo(i,:) == mode_cal);
end
xc = find(mode_row == max(mode_row));
xc = mean(xc);
xc = round(xc);

% find the row has most values equal the mode

for j = 1:320
    mode_col(j) = sum(photo(:,j) == mode_cal);
end
yc = find(mode_col == max(mode_col));
yc = mean(yc);
yc = round(yc);

% find the column has most values equal the mode

% Binarize the surrounding area
%
% bw = im2bw(photo((xc-60):(xc+60),(yc-60):(yc+60)), 0.3);
% 
% for i = 1:121
%     mean_row(i) = mean(bw(i,:));
% end
% xc2 = find(mean_row == min(mean_row));
% xc = xc + (xc2 - 61);
% xc = mean(xc);
% xc = round(xc);
% 
% for j = 1:121
%     mean_col(j) = mean(bw(:,j));
% end
% yc2 = find(mean_col == min(mean_col));
% yc = yc + (yc2 - 61);
% yc = mean(yc);
% yc = round(yc);

% Geometric method

% r = sqrt((x1 - xc)^2 + (y1 - yc)^2);
% alpha = 0:pi/20:2 * pi;
% x = r * cos(alpha) + yc;
% y = r * sin(alpha) + xc;
% imshow(photo)
% hold on
% plot(x,y)

% use the geometric method to find the inner circle

% imhist(photo);

% histogram of the whole photo which can help us find the mode (pupil) value

Pupil = im2bw(photo(round(xc-70):round(xc+70), round(yc-70):round(yc+70))); %#ok<IM2BW>
% select a surrounding area of the pupil and transform to binary
BWP = edge(Pupil, 'canny');
% edge detection
[cpv,rpv] = imfindcircles(BWP,[40 58],'Sensitivity', 0.990);
% Hough transform to find the inner circle
% imshow(BWP);

cp = round(cpv(1,:)); % select the best fitted center in the subimage
cp(1) = cp(1)-70+yc; 
cp(2) = cp(2)-70+xc; % calculate the original coordinates
rp = rpv(1); % select the best fitted radius

dist_p = sqrt((cp(1)-71)^2+(cp(2)-71)^2); 
% the distance between the best fitted center and the geometric center
if dist_p >= 5
    cp(1) = yc;
    cp(2) = xc;
end
% if the distance is too large we perfer the geometric center
    
% imshow(photo)
% hold on
% viscircles(cp,rp,'EdgeColor','b');

Iris = im2bw(photo(round(xc-85):round(xc+85), round(yc-110):round(yc+95))); %#ok<IM2BW>
% select a surrounding area of the iris and transform to binary
BWI = edge(Iris, 'canny');
% edge detection
BWI(1:60, :) = 0;
BWI(40:125, 35:140) = 0;
% add some masks to easily find the outer circle
[civ,riv] = imfindcircles(BWI,[95 120],'Sensitivity', 0.999);
% Hough transform to find the outer circle
% imshow(BWI);

ival = find(sqrt((civ(1) - cp(1))^2 + (civ(2) - cp(2))^2) == ...
       min(sqrt((civ(1) - cp(1))^2 + (civ(2) - cp(2))^2)));
% find the minimal distance between the best fitted center and the pupil center
ci = round(civ(1,:));
% select the best fitted center in the subimage
ci(1) = ci(1)-110+yc;
ci(2) = ci(2)-85+xc;
% calculate the original coordinates
ri = riv(1);
% select the best fitted radius

dist_i = sqrt((ci(1)-111)^2+(ci(2)-86)^2);
% the distance between the best fitted center and the geometric center
if dist_i >= 5
    ci(1) = yc;
    ci(2) = xc;
end
% if the distance is too large we perfer the geometric center

% imshow(photo)
% hold on
% viscircles(cp,rp,'EdgeColor','b')
% viscircles(ci,ri,'EdgeColor','b');

for i = 1:280
    for j = 1:320
        diso = sqrt((i - ci(2)).^2 + (j - ci(1)).^2);
        if diso > ri 
            photo(i,j) = 0;
        end
    end
end
% black inside area of the inner circle

for i = 1:280
    for j = 1:320
        disi = sqrt((i - cp(2)).^2+(j - cp(1)).^2);
        if disi < rp
            photo(i,j) = 0;
        end
    end
end
% black outside area of the outer circle

output1 = photo;
% imshow(output1);
end