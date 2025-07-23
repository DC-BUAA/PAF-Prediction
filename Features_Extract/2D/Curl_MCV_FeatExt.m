function [MCVFeatures] = Curl_MCV_FeatExt(MM,Curlz)


MM_max=max(max(MM));        
[Max_X,Max_Y]=find(MM==MM_max); 

MCVFeatures = Curlz( int8(mean(Max_X)) , int8(mean(Max_Y)) );


end


