function [Region1Features] = Current_Region1_FeatExt(Current_X,Current_Y,MM,numChannels)

M=sqrt(numChannels);
N=M;
scaleFactor=255/(N-1); % 缩放因子

[MM_LAD,MM_LCX,MM_RCA] = Artery_Divide_Region256(MM);
[Current_X_LAD,Current_X_LCX,Current_X_RCA] = Artery_Divide_Region256(Current_X);
[Current_Y_LAD,Current_Y_LCX,Current_Y_RCA] = Artery_Divide_Region256(Current_Y);

[MM_max_LAD,MM_max_LAD_idx] = max(MM_LAD);
MM_min_LAD = min(MM_LAD);
MM_mean_LAD = mean(MM_LAD);
MM_std_LAD = std(MM_LAD);
MCV_LAD = [Current_X_LAD(MM_max_LAD_idx),Current_Y_LAD(MM_max_LAD_idx)];
MCV_Amplitude_LAD = MM_max_LAD;
MCV_Angle_LAD = atan2(MCV_LAD(1,1),MCV_LAD(1,2))*180/pi;
MCV_Perimeter_LAD = 2*(abs(MCV_LAD(1,1))/scaleFactor+abs(MCV_LAD(1,2))/scaleFactor);
MCV_Area_LAD = abs(MCV_LAD(1,1)/scaleFactor)*abs(MCV_LAD(1,2)/scaleFactor);

[MM_max_LCX,MM_max_LCX_idx] = max(MM_LCX);
MM_min_LCX = min(MM_LCX);
MM_mean_LCX = mean(MM_LCX);
MM_std_LCX = std(MM_LCX);
MCV_LCX = [Current_X_LCX(MM_max_LCX_idx),Current_Y_LCX(MM_max_LCX_idx)];
MCV_Amplitude_LCX = MM_max_LCX;
MCV_Angle_LCX = atan2(MCV_LCX(1,1),MCV_LCX(1,2))*180/pi;
MCV_Perimeter_LCX = 2*(abs(MCV_LCX(1,1))/scaleFactor+abs(MCV_LCX(1,2))/scaleFactor);
MCV_Area_LCX = abs(MCV_LCX(1,1)/scaleFactor)*abs(MCV_LCX(1,2)/scaleFactor);

[MM_max_RCA,MM_max_RCA_idx] = max(MM_RCA);
MM_min_RCA = min(MM_RCA);
MM_mean_RCA = mean(MM_RCA);
MM_std_RCA = std(MM_RCA);
MCV_RCA = [Current_X_RCA(MM_max_RCA_idx),Current_Y_RCA(MM_max_RCA_idx)];
MCV_Amplitude_RCA = MM_max_RCA;
MCV_Angle_RCA = atan2(MCV_RCA(1,1),MCV_RCA(1,2))*180/pi;
MCV_Perimeter_RCA = 2*(abs(MCV_RCA(1,1))/scaleFactor+abs(MCV_RCA(1,2))/scaleFactor);
MCV_Area_RCA = abs(MCV_RCA(1,1)/scaleFactor)*abs(MCV_RCA(1,2)/scaleFactor);

Region1Features = [MCV_Amplitude_LAD,MCV_Angle_LAD,MCV_Perimeter_LAD,MCV_Area_LAD,MM_min_LAD,MM_mean_LAD,MM_std_LAD,...
                   MCV_Amplitude_LCX,MCV_Angle_LCX,MCV_Perimeter_LCX,MCV_Area_LCX,MM_min_LCX,MM_mean_LCX,MM_std_LCX,...
                   MCV_Amplitude_RCA,MCV_Angle_RCA,MCV_Perimeter_RCA,MCV_Area_RCA,MM_min_RCA,MM_mean_RCA,MM_std_RCA];

end

function [MM_LAD,MM_LCX,MM_RCA] = Artery_Divide_Region256(MM)

MM1_1 = MM(129:170, 44:170);
MM1_2 = MM(171:256, 44:213);
MM1_1 = MM1_1(:);
MM1_2 = MM1_2(:);
MM_LAD = vertcat(MM1_1, MM1_2);

MM2 = MM(44:128, 129:256);
MM_LCX = MM2(:);

MM3 = MM(44:128, 1:128);
MM_RCA = MM3(:);

end



