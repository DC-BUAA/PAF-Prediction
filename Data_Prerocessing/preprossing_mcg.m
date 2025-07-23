function [baseDate_denoised] = preprossing_mcg(data,fs,x1,x2)

y = data(:,x1:x2);
cc_y_imf = [];
mseb_imf = [];
ycoef_former = [];
C2_assemble = [];
y_test_later = [];
y_test_former = [];

amp_cos = 1;
maxPhase0 = 12;
numSift = 12; 
numImf = 9;

    % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
imf = SAM_UPEMD(y,1,numImf,numSift,maxPhase0,amp_cos);
[m2,n2]=size(imf);
y_f = y;
% % % ------------------------------------------------------------------------- %

    [m2,n2]=size(imf);
    for i=1:m2
        a2=corrcoef(imf(i,:),y_f); 
        xg2(i)=a2(1,2);
    end
    cc_y_imf = [cc_y_imf; xg2];
    for i=1:m2-1

        mse2(i)=mean(imf(i,:).^2,2)-mean(imf(i,:),2).^2; %计算方差
    end;
    mmse2=sum(mse2);
    for i=1:m2-1
 
        mseb2(i)=mse2(i)/mmse2*100;

    end
    %-----------------------------------------------------------------------%    
   y_denoised =  0*imf(4,:)+ 0*imf(9,:)+ 1*imf(5,:)+imf(6,:)+ 1*imf(7,:)+1*imf(8,:);


    order = 100;
    b = fir1(order, 40/(fs/2), 'low');  
    

    baseDate_denoised = zeros(size(y_denoised));
    
  
    for i = 1:size(y_denoised, 1)
        baseDate_denoised(i, :) = filtfilt(b, 1, y_denoised(i, :));
    end
end
