function [xq,yq,zzt,poleDistance,poleAngle,maxZheng,maxFu,areaZheng,areaFu]=ISO_Generate_256(signal,time,numChannels,Title,debug)


if nargin<5
    debug=0; 
end
signal = signal';
M=sqrt(numChannels);
N=M;
[xq,yq]=meshgrid(linspace(1,N,256),linspace(1,M,256));
timeSignal=reshape(signal(:,time),N,M)';
zzt=interp2(1:N,1:M,timeSignal,xq,yq,'spline');


maxFu=min(min(zzt));maxZheng=max(max(zzt));
[x1,y1]=find(zzt==maxFu);[x2,y2]=find(zzt==maxZheng);
scaleFactor=255/(N-1);
poleDistance=(sqrt((x1-x2)^2+(y1-y2)^2))/scaleFactor;
poleAngle=(-atan((y2-y1)/(x2-x1)))*180/pi;
areaZheng=sum(sum(zzt>0.8*maxZheng))/numel(zzt);
areaFu=sum(sum(zzt<0.8*maxFu))/numel(zzt);



end


