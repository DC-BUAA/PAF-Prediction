function [GravityFeatures] = Current_Gravity_FeatExt(MM,numChannels)
M=sqrt(numChannels);
N=M;


A_1= []; A_2= []; 
B_1= []; B_2= []; 
C_1= []; C_2= []; 
D_1= []; D_2= []; 
q=1;w=1;          
for i=1:size(MM,1)
    for j=1:size(MM,2)
        if MM(i,j) > 0
            A_1(q,1)= MM(i,j)*i; 
            A_2(q,1)= MM(i,j);  
            B_1(q,1)= MM(i,j)*j; 
            B_2(q,1)= MM(i,j);   
            q=q+1;
        end
        if MM(i,j) < 0
            C_1(w,1)= MM(i,j)*i; 
            C_2(w,1)= MM(i,j);   
            D_1(w,1)= MM(i,j)*j; 
            D_2(w,1)= MM(i,j);   
            w=w+1;
        end                      
    end
end

Positive_Gravity_X = 0; 
Positive_Gravity_Y = 0; 
Negative_Gravity_X = 0; 
Negative_Gravity_Y = 0; 

if sum(A_2) ~=  0
    Positive_Gravity_X = sum(A_1)/sum(A_2); 
    Positive_Gravity_Y = sum(B_1)/sum(B_2); 
end
if sum(C_2) ~=  0
    Negative_Gravity_X = sum(C_1)/sum(C_2); 
    Negative_Gravity_Y = sum(D_1)/sum(D_2); 
end

scaleFactor=255/(N-1); 

Gravity_Distance = (sqrt((Positive_Gravity_X-Negative_Gravity_X)^2+(Positive_Gravity_Y-Negative_Gravity_Y)^2))/scaleFactor;
Gravity_Angle = (-atan((Negative_Gravity_Y-Positive_Gravity_Y)/(Negative_Gravity_X-Positive_Gravity_X)))*180/pi;
Gravity_Perimeter = 2*(abs(Positive_Gravity_X-Negative_Gravity_X)/scaleFactor + abs(Positive_Gravity_Y-Negative_Gravity_Y)/scaleFactor);
Gravity_Area = (abs(Positive_Gravity_X-Negative_Gravity_X)/scaleFactor) * (abs(Positive_Gravity_Y-Negative_Gravity_Y)/scaleFactor);

Positive_Gravity_X = Positive_Gravity_X/scaleFactor + 1;  
Positive_Gravity_Y = Positive_Gravity_Y/scaleFactor + 1;
Negative_Gravity_X = Negative_Gravity_X/scaleFactor + 1;
Negative_Gravity_Y = Negative_Gravity_Y/scaleFactor + 1;
Positive_Gravity_Channal=round(Positive_Gravity_Y) + (round(Positive_Gravity_X)-1)*N;
Negative_Gravity_Channal=round(Negative_Gravity_Y) + (round(Negative_Gravity_X)-1)*N;


GravityFeatures=[Positive_Gravity_X,Positive_Gravity_Y,Positive_Gravity_Channal];


end


