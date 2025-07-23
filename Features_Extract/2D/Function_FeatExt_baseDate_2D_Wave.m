function [Features_baseDate_2D_Wave] = Function_FeatExt_baseDate_2D_Wave(baseDate_transform_bfd, P_info, R_info, T_info, numChannels, Fs, prefix)
    fields = fieldnames(baseDate_transform_bfd);

    waveSegments = {
        'P_start_P_peak',    'P_start', 'P_peak';   
        'P_start_P_end',     'P_start', 'P_end';    
        'P_peak_P_end',     'P_peak',  'P_end';     
        'P_end_T_start',    'P_end',   'T_start';   
        'T_start_T_peak',   'T_start', 'T_peak';    
        'T_start_T_end',    'T_start', 'T_end';     
        'T_peak_T_end',     'T_peak',  'T_end'      
    };
    
    for f = 1:numel(fields)
        currentField = fields{f};
        bfd_current = baseDate_transform_bfd.(currentField);
        
        
        eventPoints = struct(...
            'P_start',  P_info.(currentField).Start, ...
            'P_peak',   P_info.(currentField).Position, ...
            'P_end',    P_info.(currentField).End, ...
            'T_start',  T_info.(currentField).Start, ...
            'T_peak',   T_info.(currentField).Position, ...
            'T_end',    T_info.(currentField).End);
        

        ISOWavefeatContainer = struct();
        CurrentWavefeatContainer = struct();
        CurlWavefeatContainer = struct();


        for w = 1:size(waveSegments, 1)
            segmentName = waveSegments{w, 1};
            startEvent = waveSegments{w, 2};
            endEvent = waveSegments{w, 3};
            
           
            startTime = eventPoints.(startEvent);
            endTime = eventPoints.(endEvent);
            

            

            [dipoleData, gravityData, boundaryData] = initializeContainers_ISO();
            [MCVData, TCVData] = initializeContainers_Current();
            [MCV2Data, DiPole2Data, AreaData] = initializeContainers_Curl();

            for t = startTime:endTime

                [zzt,Current_X,Current_Y,MM,Curlz]=zzt_Current_Curl_Generate_256(bfd_current,t,numChannels);


                dipoleFeatures = ISO_DiPole_FeatExt(zzt, numChannels);
                dipoleData.PositiveValue(end+1)    = dipoleFeatures(1);
                dipoleData.NegativeValue(end+1)    = dipoleFeatures(2);
                dipoleData.DiPole_Distance(end+1)  = dipoleFeatures(3);
                dipoleData.DiPole_Angle(end+1)     = dipoleFeatures(4);
                dipoleData.DiPole_Perimeter(end+1) = dipoleFeatures(5);
                dipoleData.DiPole_Area(end+1)      = dipoleFeatures(6);
                gravityFeatures = ISO_Gravity_FeatExt(zzt, numChannels);
                gravityData.Gravity_Distance(end+1)  = gravityFeatures(1);
                gravityData.Gravity_Angle(end+1)     = gravityFeatures(2);
                gravityData.Gravity_Perimeter(end+1) = gravityFeatures(3);
                gravityData.Gravity_Area(end+1)      = gravityFeatures(4);
                boundaryFeatures = ISO_Boundary_FeatExt(zzt, numChannels);
                boundaryData.length_chain(end+1)  = boundaryFeatures(1);
                boundaryData.fractalDim(end+1)    = boundaryFeatures(2);
                boundaryData.compactness(end+1)   = boundaryFeatures(3);



                MCVFeatures = Current_MCV_FeatExt(Current_X,Current_Y,MM,numChannels);
                MCVData.Amplitude(end+1) = MCVFeatures(1);
                MCVData.Angle(end+1)     = MCVFeatures(2);
                MCVData.Perimeter(end+1) = MCVFeatures(3);
                MCVData.Area(end+1)      = MCVFeatures(4);
                TCVFeatures = Current_TCV_FeatExt(Current_X,Current_Y,numChannels);
                TCVData.Amplitude(end+1) = TCVFeatures(1);
                TCVData.Angle(end+1)     = TCVFeatures(2);
                TCVData.Perimeter(end+1) = TCVFeatures(3);
                TCVData.Area(end+1)      = TCVFeatures(4);

      

                MCV2Features = Curl_MCV_FeatExt(MM,Curlz);
                MCV2Data.CurlValue(end+1) = MCV2Features(1);
                DiPole2Features = Curl_DiPole_FeatExt(Curlz,numChannels);
                DiPole2Data.PositiveValue(end+1)    = DiPole2Features(1);
                DiPole2Data.NegativeValue(end+1)    = DiPole2Features(2);
                DiPole2Data.DiPole_Distance(end+1)  = DiPole2Features(3);
                DiPole2Data.DiPole_Angle(end+1)     = DiPole2Features(4);
                DiPole2Data.DiPole_Perimeter(end+1) = DiPole2Features(5);
                DiPole2Data.DiPole_Area(end+1)      = DiPole2Features(6);
                AreaFeatures = Curl_Area_FeatExt(Curlz);
                AreaData.Amplitude_Ratio(end+1)      = AreaFeatures(1);
                AreaData.Amplitude_Difference(end+1) = AreaFeatures(2);
                AreaData.Area_Ratio(end+1)           = AreaFeatures(3);
                AreaData.Area_Difference(end+1)      = AreaFeatures(4);
            end
            
            ISOWavefeatContainer.(segmentName).DiPole = calcFeaturesForGroup(dipoleData);
            ISOWavefeatContainer.(segmentName).Gravity = calcFeaturesForGroup(gravityData);
            ISOWavefeatContainer.(segmentName).Boundary = calcFeaturesForGroup(boundaryData);
            CurrentWavefeatContainer.(segmentName).MCV = calcFeaturesForGroup(MCVData);
            CurrentWavefeatContainer.(segmentName).TCV = calcFeaturesForGroup(TCVData);
            CurlWavefeatContainer.(segmentName).MCV = calcFeaturesForGroup(MCV2Data);
            CurlWavefeatContainer.(segmentName).DiPole = calcFeaturesForGroup(DiPole2Data);
            CurlWavefeatContainer.(segmentName).Area = calcFeaturesForGroup(AreaData);
        end
        

        [~, verticalISOFeat] = Struct_to_Vertical(ISOWavefeatContainer, 'ISO_Wave', '');
        [~, verticalCurrentFeat] = Struct_to_Vertical(CurrentWavefeatContainer, 'Current_Wave', '');
        [~, verticalCurlFeat] = Struct_to_Vertical(CurlWavefeatContainer, 'Curl_Wave', '');
        verticalFeat = vertcat(verticalISOFeat,verticalCurrentFeat,verticalCurlFeat);

        if f == 1
            featureList = verticalFeat;
        else
            featureList = [featureList, verticalFeat(:,2)]; 
        end

 
        [~, verticalCurlFeat] = Struct_to_Vertical(CurlWavefeatContainer, 'Curl_Wave', '');

        if f == 1
            featureList = verticalCurlFeat;
        else
            featureList = [featureList, verticalCurlFeat(:,2)]; 
        end

        
    end
    

    header = ['Features', fields'];
    Features_baseDate_2D_Wave = [header; featureList];
end


function [dipole, gravity, boundary] = initializeContainers_ISO()
    dipole = struct(...
        'PositiveValue', [], 'NegativeValue', [], 'DiPole_Distance', [], ...
        'DiPole_Angle', [], 'DiPole_Perimeter', [], 'DiPole_Area', []);
    gravity = struct('Gravity_Distance', [], 'Gravity_Angle', [], 'Gravity_Perimeter', [], 'Gravity_Area', []);
    boundary = struct('length_chain', [], 'fractalDim', [], 'compactness', []);
end
function [MCVData, TCVData] = initializeContainers_Current()
    MCVData = struct('Amplitude', [], 'Angle', [], 'Perimeter', [], 'Area', []);
    TCVData = struct('Amplitude', [], 'Angle', [], 'Perimeter', [], 'Area', []);
end
function [MCV2Data, DiPole2Data, AreaData] = initializeContainers_Curl()
    DiPole2Data = struct(...
        'PositiveValue', [], 'NegativeValue', [], 'DiPole_Distance', [], ...
        'DiPole_Angle', [], 'DiPole_Perimeter', [], 'DiPole_Area', []);
    MCV2Data = struct('CurlValue', []);
    AreaData = struct('Amplitude_Ratio', [], 'Amplitude_Difference', [], 'Area_Ratio', [], 'Area_Difference', []);
end


function featureGroup = calcFeaturesForGroup(data)
    fields = fieldnames(data);
    featureGroup = struct();
    for i = 1:numel(fields)
        featureName = fields{i};
        featureValues = data.(featureName);
        featureGroup.(featureName) = CalculateFeatures(featureValues);
    end
end

function Features = CalculateFeatures(data)
    if isempty(data)
        Features = struct('Max', NaN, 'Mean', NaN, 'Std', NaN, 'Cv', NaN, ...
            'Skewness', NaN, 'Kurtosis', NaN, 'Entropy', NaN);
        return;
    end
    Features.Max = max(data);
    Features.Mean = mean(data);
    Features.Std = std(data);
    Features.Cv = Features.Std / Features.Mean;
    Features.Skewness = skewness(data);
    Features.Kurtosis = kurtosis(data);
    

    dataNonNegative = data - min(data) + eps; 
    probabilities = histcounts(dataNonNegative, 'Normalization', 'probability');
    Features.Entropy = -sum(probabilities .* log2(probabilities + eps));
end