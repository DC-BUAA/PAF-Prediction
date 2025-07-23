function [DiPoleFeatures] = ISO_DiPole_FeatExt(zzt,numChannels)

M=sqrt(numChannels);
N=M;

PositiveValue=max(max(zzt));
NegativeValue=min(min(zzt));
[x1,y1]=find(zzt==PositiveValue);
[x2,y2]=find(zzt==NegativeValue);
scaleFactor=255/(N-1); 
DiPole_Distance = (sqrt((x1-x2)^2+(y1-y2)^2))/scaleFactor;
DiPole_Angle = (-atan((y2-y1)/(x2-x1)))*180/pi;
DiPole_Perimeter = 2*(abs(x1-x2)/scaleFactor + abs(y1-y2)/scaleFactor);
DiPole_Area = (abs(x1-x2)/scaleFactor) * (abs(y1-y2)/scaleFactor);

Positive_X = x1/scaleFactor + 1;
Positive_Y = y1/scaleFactor + 1;
Negative_X = x2/scaleFactor + 1;
Negative_Y = y2/scaleFactor + 1;

Positive_Channal=round(Positive_Y) + (round(Positive_X)-1)*N;
Negative_Channal=round(Negative_Y) + (round(Negative_X)-1)*N;

DiPoleFeatures=[PositiveValue,NegativeValue,DiPole_Distance,DiPole_Angle,DiPole_Perimeter,DiPole_Area,...
    Positive_X,Positive_Y,Positive_Channal,Negative_X,Negative_Y,Negative_Channal];

end


