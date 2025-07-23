function [PoleNumberFeatures] = ISO_PoleNumber_FeatExt(zzt)
 

max_fu = min(min(zzt));       
max_zheng = max(max(zzt));    
PN_zheng = 0;                
PN_fu = 0;                    
isDipole = false;            
positiveAreaRatio = 0;       
negativeAreaRatio = 0;      
areaRatio = 0;               
areaDifference = 0;          

totalArea = 256 * 256;       


if max_zheng > 0 && max_fu < 0

    A_zheng_index = (zzt >= 0.8 * max_zheng);

    Area_index = zeros(256, 256);
    Area_index(A_zheng_index) = 1;
    L = bwlabel(Area_index, 8);     
    stats = regionprops(L, 'Area'); 
    Ar = cat(1, stats.Area);        
    

    if length(Ar) == 1 
        PN_zheng = 1;                         
        max_zheng_area = Ar;                   
        positiveAreaRatio = Ar / totalArea;     
    else

        max_zheng_area = 0;  
    end
    

    A_fu_index = (zzt <= 0.8 * max_fu);

    Area_index = zeros(256,256);
    Area_index(A_fu_index) = 1;
    L = bwlabel(Area_index, 8);     
    stats = regionprops(L, 'Area'); 
    Ar = cat(1, stats.Area);        
 

    if length(Ar) == 1 
        PN_fu = 1;                        
        max_fu_area = Ar;                     
        negativeAreaRatio = Ar / totalArea;  
    else

        max_fu_area = 0; 
    end


    if max_zheng_area > 0 && max_fu_area > 0
        areaRatio = max_zheng_area / max_fu_area;       
        areaDifference = abs(max_zheng_area - max_fu_area); 
    end


    if PN_zheng == 1 && PN_fu == 1 && max_zheng_area > 0 && max_fu_area > 0
        isDipole = true; 
    end

    PN = PN_zheng + PN_fu;
else
 
    PN = 0;
end


PoleNumberFeatures = [isDipole,PN,positiveAreaRatio,negativeAreaRatio,areaRatio,areaDifference];
end
