function [BoundaryFeatures] = ISO_Boundary_FeatExt(zzt,numChannels)


    M = sqrt(numChannels);
    N = M;
    scaleFactor = 255/(N-1);

    binaryImg = zzt > 0;
    filledImg = imfill(binaryImg, 'holes');
    filledImg = imclose(filledImg, strel('disk',1));
    boundaryImg = bwperim(filledImg); 

    [y, x] = find(boundaryImg);
    if numel(x) < 2
        length_chain = 0;
    else
        [sortedX, sortedY] = sortBoundaryPoints(x, y);

        shiftedX = circshift(sortedX, -1);
        shiftedY = circshift(sortedY, -1);
        distances = sqrt((shiftedX-sortedX).^2 + (shiftedY-sortedY).^2);
        length_chain = sum(distances) / scaleFactor;
    end


    boxSizes = round(2.^(1:0.5:8));
    maxSize = min(size(boundaryImg));
    boxSizes = boxSizes(boxSizes <= maxSize);
    counts = zeros(size(boxSizes));
    

    pyramid = cell(length(boxSizes),1);
    for k = 1:length(boxSizes)
        boxSize = boxSizes(k);
      
        scale = 1/boxSize;
        if scale < 1
            pyramid{k} = imresize(boundaryImg, scale, 'nearest');
        else
            pyramid{k} = boundaryImg;
        end
    end
    
    
    parfor k = 1:length(boxSizes)
        counts(k) = sum(pyramid{k}(:)) > 0;
    end
    
    
    validIdx = counts > 0;
    if sum(validIdx) >= 5 
        p = polyfit(log(1./boxSizes(validIdx)), log(counts(validIdx)), 1);
        fractalDim = p(1);
    else
        fractalDim = 0;
    end

    
    stats = regionprops(filledImg, 'Area', 'Perimeter');
    if ~isempty(stats)
        compactness = (4 * pi * sum([stats.Area])) / (sum([stats.Perimeter])^2);
    else
        compactness = 0;
    end

    BoundaryFeatures = [length_chain, fractalDim, compactness];
    
end


function [sortedX, sortedY] = sortBoundaryPoints(x, y)
    if isempty(x)
        sortedX = [];
        sortedY = [];
        return;
    end
    centroid = [mean(x), mean(y)];         
    angles = atan2(y - centroid(2), x - centroid(1)); 
    [~, order] = sort(angles);            
    sortedX = x(order);                   
    sortedY = y(order);                   
end

