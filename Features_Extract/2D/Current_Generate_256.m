function [Current_X,Current_Y,MM]=Current_Generate_256(signal,time,numChannels,debug)


if nargin<4
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


dx = 1/scaleFactor;
dy = 1/scaleFactor;

dBx = zeros(256,256);
for j=1:256
    dBx(j,1)=(zzt(j,2)-zzt(j,1))/dx;
    dBx(j,256)=(zzt(j,256)-zzt(j,256-1))/dx;
    for k=2:256-1
        dBx(j,k)=(zzt(j,k+1)-zzt(j,k-1))/(2*dx);
    end
end

dBy = zeros(256,256);
for k=1:256
    dBy(1,k)=(zzt(2,k)-zzt(1,k))/dy;
    dBy(256,k)=(zzt(256,k)-zzt(256-1,k))/dy;
    for j=2:256-1
        dBy(j,k)=(zzt(j+1,k)-zzt(j-1,k))/(2*dy);
    end
end

MM = sqrt(dBx.^2+dBy.^2); 
Current_X = dBx;
Current_Y = -dBy;




end


