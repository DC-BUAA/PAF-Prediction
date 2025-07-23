function [Region2Features] = ISO_Region2_FeatExt(zzt,numChannels)

M=sqrt(numChannels);
N=M;
scaleFactor=255/(N-1); 

[region1,region2,region3,region4] = Matrix_Divide_Region256(zzt);
regions = {region1, region2, region3, region4};

for r = 1:4
    region = regions{r};
    max_value = max(region(:));   
    min_value = min(region(:));   
    mean_value = mean(region(:));

 
    binaryImg = region > 0; 
    filledImg = imfill(binaryImg, 'holes'); 
    filledImg = imclose(filledImg, strel('disk',1));
    boundaryImg = bwperim(filledImg);     
  
    [y, x] = find(boundaryImg); 
    if numel(x) < 2
        length_chain = 0; 
    else
        [sortedX, sortedY] = sortBoundaryPoints(x, y);
        dx = diff([sortedX; sortedX(1)]); 
        dy = diff([sortedY; sortedY(1)]);   
        distances = sqrt(dx.^2 + dy.^2);   
        length_chain = sum(distances) / scaleFactor; 
    end
  
    boxSizes = 2.^(1:0.5:8);   
    counts = zeros(size(boxSizes));
    for k = 1:length(boxSizes)
        boxSize = round(boxSizes(k));

        gridX = 1:boxSize:size(boundaryImg,2); 
        gridY = 1:boxSize:size(boundaryImg,1); 
        count = 0;
      
        for i = 1:length(gridX)-1
            for j = 1:length(gridY)-1
                xRange = gridX(i):min(gridX(i+1), size(boundaryImg,2));
                yRange = gridY(j):min(gridY(j+1), size(boundaryImg,1));
                subImg = boundaryImg(yRange, xRange);
                if any(subImg(:)), count = count + 1; end
            end
        end
        counts(k) = count;
    end

    validIdx = counts > 0 & boxSizes <= size(boundaryImg,1); 
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
  
    min_points_for_curvature = 5; 

    if ~exist('sortedX', 'var') || numel(sortedX) < min_points_for_curvature
        meanCurvature = 0;
        stdCurvature = 0;
    else
        try
            curvature = zeros(numel(sortedX),1);
            validCurvatureCount = 0;
            for k = 2:numel(sortedX)-1
                x_prev = sortedX(k-1); y_prev = sortedY(k-1);
                x_curr = sortedX(k);   y_curr = sortedY(k);
                x_next = sortedX(k+1); y_next = sortedY(k+1);
                if isColinear(x_prev,y_prev, x_curr,y_curr, x_next,y_next)
                    continue; 
                end
                [~, R] = Circumcircle([x_prev, y_prev; x_curr, y_curr; x_next, y_next]);
                if isfinite(R) && R > 0
                    curvature(k) = 1/R;
                    validCurvatureCount = validCurvatureCount + 1;
                end
            end
            if validCurvatureCount >= 3
                curvature = curvature(curvature ~= 0);
                meanCurvature = mean(curvature);
                stdCurvature = std(curvature);
            else
                meanCurvature = 0;
                stdCurvature = 0;
            end
        catch
            meanCurvature = 0;
            stdCurvature = 0;
        end
    end

    result{r}=[max_value,min_value,mean_value,length_chain,fractalDim,compactness,meanCurvature];
    clear sortedX sortedY max_value min_value mean_value length_chain fractalDim compactness meanCurvature
end

Region2Features = [result{1}(1),result{1}(2),result{1}(3),result{1}(4),result{1}(5),result{1}(6),result{1}(7),...
                   result{2}(1),result{2}(2),result{2}(3),result{2}(4),result{2}(5),result{2}(6),result{2}(7),...
                   result{3}(1),result{3}(2),result{3}(3),result{3}(4),result{3}(5),result{3}(6),result{3}(7),...
                   result{4}(1),result{4}(2),result{4}(3),result{4}(4),result{4}(5),result{4}(6),result{4}(7)];

end

function [region1,region2,region3,region4] = Matrix_Divide_Region256(zzt)

[rows, cols] = size(zzt); 
midRow = round(rows / 2);
midCol = round(cols / 2);
region1 = zzt(1:midRow, 1:midCol);        
region2 = zzt(1:midRow, midCol+1:end);    
region3 = zzt(midRow+1:end, 1:midCol);     
region4 = zzt(midRow+1:end, midCol+1:end); 

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


function flag = isColinear(x1,y1, x2,y2, x3,y3)
   
    area = 0.5 * abs( (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) );
    flag = area < 1e-6; 
end

