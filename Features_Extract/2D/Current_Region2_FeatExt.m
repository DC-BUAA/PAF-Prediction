function [Region2Features] = Current_Region2_FeatExt(Current_X,Current_Y,MM,numChannels)

M=sqrt(numChannels);
N=M;
scaleFactor=255/(N-1); 

[MM_region1,MM_region2,MM_region3,MM_region4] = Matrix_Divide_Region256(MM);
[Current_X_region1,Current_X_region2,Current_X_region3,Current_X_region4] = Matrix_Divide_Region256(Current_X);
[Current_Y_region1,Current_Y_region2,Current_Y_region3,Current_Y_region4] = Matrix_Divide_Region256(Current_Y);
MM_regions = {MM_region1, MM_region2, MM_region3, MM_region4};
Current_X_regions = {Current_X_region1, Current_X_region2, Current_X_region3, Current_X_region4};
Current_Y_regions = {Current_Y_region1, Current_Y_region2, Current_Y_region3, Current_Y_region4};

for i = 1:4
    MMs = MM_regions{i};Current_Xs = Current_X_regions{i};Current_Ys = Current_Y_regions{i};
    MM_max = max(MMs(:));  
    MM_min = min(MMs(:));   
    MM_mean = mean(MMs(:)); 
    MM_std = std(MMs(:));  
    [Max_X,Max_Y]=find(MMs==MM_max); 
    MCV = [Current_Xs(Max_X,Max_Y),Current_Ys(Max_X,Max_Y)];
    MCV_Amplitude = MM_max;
    MCV_Angle = atan2(MCV(1,1),MCV(1,2))*180/pi;
    MCV_Perimeter = 2*(abs(MCV(1,1))/scaleFactor+abs(MCV(1,2))/scaleFactor);
    MCV_Area = abs(MCV(1,1)/scaleFactor)*abs(MCV(1,2)/scaleFactor);

    result{i}=[MCV_Amplitude,MCV_Angle,MCV_Perimeter,MCV_Area,MM_min,MM_mean,MM_std];
    clear MM_max Max_X Max_Y MCV MCV_Amplitude MCV_Angle MCV_Perimeter MCV_Area MM_min MM_mean MM_std
end

Region2Features = [result{1}(1),result{1}(2),result{1}(3),result{1}(4),result{1}(5),result{1}(6),result{1}(7),...
                   result{2}(1),result{2}(2),result{2}(3),result{2}(4),result{2}(5),result{2}(6),result{2}(7),...
                   result{3}(1),result{3}(2),result{3}(3),result{3}(4),result{3}(5),result{3}(6),result{3}(7),...
                   result{4}(1),result{4}(2),result{4}(3),result{4}(4),result{4}(5),result{4}(6),result{4}(7)];

end


function [region1,region2,region3,region4] = Matrix_Divide_Region256(MM)

[rows, cols] = size(MM); 
midRow = round(rows / 2);
midCol = round(cols / 2);
region1 = MM(1:midRow, 1:midCol);        
region2 = MM(1:midRow, midCol+1:end);    
region3 = MM(midRow+1:end, 1:midCol);     
region4 = MM(midRow+1:end, midCol+1:end); 

end

