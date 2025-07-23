function [TCVFeatures] = Current_TCV_FeatExt(Current_X,Current_Y,numChannels)

M=sqrt(numChannels);
N=M;
scaleFactor=255/(N-1); 

Total_X = sum(Current_X,'all'); 
Total_Y = sum(Current_Y,'all'); 
TCV=[Total_X,Total_Y];         
TCV_Amplitude = sqrt(Total_X.^2+Total_Y.^2); 
TCV_Angle = atan2(TCV(1,1),TCV(1,2))*180/pi;
TCV_Perimeter = 2*(abs(TCV(1,1))/scaleFactor+abs(TCV(1,2))/scaleFactor);
TCV_Area = abs(TCV(1,1)/scaleFactor)*abs(TCV(1,2)/scaleFactor);


TCVFeatures=[TCV_Amplitude,TCV_Angle,TCV_Perimeter,TCV_Area];

end


