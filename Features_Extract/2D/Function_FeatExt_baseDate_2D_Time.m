function [Features_baseDate_2D_Time] = Function_FeatExt_baseDate_2D_Time(baseDate_transform_bfd,P_info,R_info,T_info,numChannels,Fs,prefix)
    fields = fieldnames(baseDate_transform_bfd);
    
    for f = 1:numel(fields)
        bfd_current = baseDate_transform_bfd.(fields{f});
        eventPoints = struct(...
            'P_start',  P_info.(fields{f}).Start, ...
            'P_peak',   P_info.(fields{f}).Position, ...
            'P_end',    P_info.(fields{f}).End, ...
            'R_peak',   R_info.(fields{f}).Position, ...
            'T_start',  T_info.(fields{f}).Start, ...
            'T_peak',   T_info.(fields{f}).Position, ...
            'T_end',    T_info.(fields{f}).End);
   
        ISOfeatContainer = struct();  
        CurrentfeatContainer = struct();
        CurlfeatContainer = struct();

        eventTypes = fieldnames(eventPoints);
        for j = 1:numel(eventTypes)
            eventName = eventTypes{j};
            eventLoc = eventPoints.(eventName);
            
            
            [~,~,zzt,~,~,~,~,~,~] = ISO_Generate_256(bfd_current, eventLoc, numChannels, prefix, 0);
            ISOfeatContainer.([eventName '_PoleNumber']) = ISO_PoleNumber_FeatExt(zzt);
            ISOfeatContainer.([eventName '_DiPole'])     = ISO_DiPole_FeatExt(zzt, numChannels);
            ISOfeatContainer.([eventName '_Gravity'])    = ISO_Gravity_FeatExt(zzt, numChannels);
            ISOfeatContainer.([eventName '_Boundary'])   = ISO_Boundary_FeatExt(zzt, numChannels);
            ISOfeatContainer.([eventName '_Region1'])    = ISO_Region1_FeatExt(zzt);
            ISOfeatContainer.([eventName '_Region2'])    = ISO_Region2_FeatExt(zzt, numChannels);
            
           
            [Current_X,Current_Y,MM] = Current_Generate_256(bfd_current, eventLoc, numChannels, 0);
            CurrentfeatContainer.([eventName '_MCV'])     = Current_MCV_FeatExt(Current_X,Current_Y,MM,numChannels);
            CurrentfeatContainer.([eventName '_TCV'])     = Current_TCV_FeatExt(Current_X,Current_Y,numChannels);
            CurrentfeatContainer.([eventName '_Gravity']) = Current_Gravity_FeatExt(MM,numChannels);
            CurrentfeatContainer.([eventName '_SVD'])     = Current_SVD_FeatExt(MM);
            CurrentfeatContainer.([eventName '_Region1']) = Current_Region1_FeatExt(Current_X,Current_Y,MM,numChannels);
            CurrentfeatContainer.([eventName '_Region2']) = Current_Region2_FeatExt(Current_X,Current_Y,MM,numChannels);

           
            [MM2,Curlz ] = Curl_Generate_256(bfd_current,eventLoc,numChannels);
            CurlfeatContainer.([eventName '_MCV'])    = Curl_MCV_FeatExt(MM2,Curlz);
            CurlfeatContainer.([eventName '_DiPole']) = Curl_DiPole_FeatExt(Curlz,numChannels);
            CurlfeatContainer.([eventName '_Area'])   = Curl_Area_FeatExt(Curlz);
        end

        [~, verticalISOFeat] = Struct_to_Vertical(ISOfeatContainer, 'ISO_Time', '');
        [~, verticalCurrentFeat] = Struct_to_Vertical(CurrentfeatContainer, 'Current_Time', '');
        [~, verticalCurlFeat] = Struct_to_Vertical(CurlfeatContainer, 'Curl_Time', '');
        verticalFeat = vertcat(verticalISOFeat,verticalCurrentFeat,verticalCurlFeat);
        
        if f == 1
            
            featureList(:,1) = verticalFeat(:,1);  
           
            featureList(:,f+1) = verticalFeat(:,2);  
        else
           
            featureList(:,f+1) = verticalFeat(:,2);
        end
       

    end

    numSegments = numel(fields);
    header = cell(1, numSegments + 1);
    header{1, 1} = 'Features';
    for seg = 1:numSegments
        header{1, seg + 1} = ['Segment' num2str(seg)];
    end
    Features_baseDate_2D_Time = [header; featureList];
    
end



