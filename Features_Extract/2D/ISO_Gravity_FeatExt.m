function [GravityFeatures] = ISO_Gravity_FeatExt(zzt,numChannels)


    M = sqrt(numChannels);
    N = M;
    

    [X, Y] = meshgrid(1:size(zzt,2), 1:size(zzt,1));
    
 
    pos_mask = zzt > 0;
    if any(pos_mask(:))
        pos_weights = zzt(pos_mask);
        Positive_Gravity_X = sum(X(pos_mask) .* pos_weights) / sum(pos_weights);
        Positive_Gravity_Y = sum(Y(pos_mask) .* pos_weights) / sum(pos_weights);
    else
        Positive_Gravity_X = 0;
        Positive_Gravity_Y = 0;
    end
    

    neg_mask = zzt < 0;
    if any(neg_mask(:))
        neg_weights = -zzt(neg_mask); 
        Negative_Gravity_X = sum(X(neg_mask) .* neg_weights) / sum(neg_weights);
        Negative_Gravity_Y = sum(Y(neg_mask) .* neg_weights) / sum(neg_weights);
    else
        Negative_Gravity_X = 0;
        Negative_Gravity_Y = 0;
    end
    
    scaleFactor = 255/(N-1);
    
 
    dx = (Positive_Gravity_X - Negative_Gravity_X)/scaleFactor;
    dy = (Positive_Gravity_Y - Negative_Gravity_Y)/scaleFactor;
    
    Gravity_Distance = sqrt(dx^2 + dy^2);
    Gravity_Angle = (-atan2(dy, dx)) * 180/pi;
    Gravity_Perimeter = 2*(abs(dx) + abs(dy));
    Gravity_Area = abs(dx) * abs(dy);
    
    GravityFeatures = [Gravity_Distance, Gravity_Angle, Gravity_Perimeter, Gravity_Area];
end


