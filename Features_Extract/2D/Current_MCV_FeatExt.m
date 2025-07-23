function [MCVFeatures] = Current_MCV_FeatExt(Current_X,Current_Y,MM,numChannels)

M=sqrt(numChannels);
N=M;
 
MM_max=max(max(MM));            
[Max_X,Max_Y]=find(MM==MM_max); 
scaleFactor=255/(N-1); 

MCV = [Current_X(Max_X,Max_Y),Current_Y(Max_X,Max_Y)];
MCV_Amplitude = MM_max;
MCV_Angle = atan2(MCV(1,1),MCV(1,2))*180/pi;
MCV_Perimeter = 2*(abs(MCV(1,1))/scaleFactor+abs(MCV(1,2))/scaleFactor);
MCV_Area = abs(MCV(1,1)/scaleFactor)*abs(MCV(1,2)/scaleFactor);
MCV_X = Max_X/scaleFactor + 1;
MCV_Y = Max_Y/scaleFactor + 1;
MCV_Channal = round(MCV_Y) + (round(MCV_X)-1)*N;

MCVFeatures=[MCV_Amplitude,MCV_Angle,MCV_Perimeter,MCV_Area,MCV_X,MCV_Y,MCV_Channal];

end


