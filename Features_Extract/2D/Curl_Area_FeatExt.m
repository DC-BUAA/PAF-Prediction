function [AreaFeatures] = Curl_Area_FeatExt(Curlz)
m=1; n=1;
Positive_Field = []; Negative_Field = [];
for i=1:size(Curlz,1)
    for j=1:size(Curlz,2)
        if Curlz(i,j) > 0
            Positive_Field(m,1) = Curlz(i,j);
            m=m+1;
        end
       if Curlz(i,j) < 0
            Negative_Field(n,1) = Curlz(i,j);
            n=n+1;
       end                      
    end
end
Amplitude_Ratio =   (sum(Positive_Field,'all')) / abs(sum(Negative_Field,'all'));    
Amplitude_Difference =  (sum(Positive_Field,'all')) - abs(sum(Negative_Field,'all'));


Positive_Field_SerialNumber= find(Curlz>0);
Area_positive_Field=length(Positive_Field_SerialNumber);
Negative_Field_SerialNumber= find(Curlz<0);
Area_negative_Field=length(Negative_Field_SerialNumber);
Area_Ratio = Area_positive_Field / Area_negative_Field;     
Area_Difference = Area_positive_Field - Area_negative_Field;


Amplitude_Ratio =  double(Amplitude_Ratio); 
Amplitude_Difference =  double(Amplitude_Difference); 
Area_Ratio = double(Area_Ratio); 
Area_Difference = double(Area_Difference); 
AreaFeatures = [Amplitude_Ratio,Amplitude_Difference,Area_Ratio,Area_Difference];
end

