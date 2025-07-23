clc;
clear;
close all;
tic

Path_AllMCGData = "F:\";      
SelPath = uigetdir(Path_AllMCGData); 
if SelPath == 0                                       
    fprintf('Please Select a New Folder!\n');
    error('No path selected!')
else
    RawFile_1 = dir(SelPath); 
end
disp( strcat('Path for reading data--',SelPath) ) 


subFolders = dir(SelPath);
subFolderNames = {subFolders([subFolders.isdir]).name};
Names = subFolderNames(~ismember(subFolderNames, {'.', '..'})); 
Number_MCGData = size(RawFile_1,1)-2; 
NameTitle = [{''}, Names];
Fs = 1000; 
disp('================================================================');
Features_baseDate_2D_Wave_Total = {};
batch_counter = 0;
batch_size = 20; 
output_path = 'E:\';

for i = 1 : Number_MCGData
    batch_counter = batch_counter + 1;
    disp(strcat( 'processing...') );
    patientPath = fullfile(SelPath, RawFile_1(i+2).name);
    RawFile_2 = dir(patientPath);

    validSubFolders = RawFile_2([RawFile_2.isdir]);
    validSubFolders = validSubFolders(~ismember({validSubFolders.name}, {'.', '..'}));

    for j = 1:numel(validSubFolders)

        timeFolder = validSubFolders(j).name;
        timeFolderPath = fullfile(patientPath, timeFolder);
        [~, end_name] = fileparts(timeFolder);
        end_path = timeFolderPath;

        try                          
            fid_1=fopen([end_path,'\',end_name,'.PK']); 
            pk = fread(fid_1,'int');
            fclose(fid_1);
        catch
            warning('file open failure');
        end 

        try                          
            fid_2=fopen([end_path,'\',end_name,'.HRS']); 
            HRS = fread(fid_2,'int');
            fclose(fid_2); 
        catch
            warning('file open failure');
        end 
    

        try                                          
            fid_3 = fopen([end_path,'\',end_name,'.BFD']); 
            bfd = fread(fid_3,'float');
            fclose(fid_3);
            if mod(length(bfd),1000) 
            bfd = bfd(513:end);   
            end
        catch
            warning('file open failure');
        end

        if length(bfd)>=64*1000
             numChannels=64;
        else 
             numChannels=36;
        end

        try                          
            fid=fopen([end_path,'\',end_name,'.baseDate']);
            base = fread(fid,'float');
            fclose(fid);
            a=mod(length(base),1000);
            basedate=zeros;
            if a==512
                basedate=base(513:length(base),:);   
            else
                basedate=base;
            end
        catch
            warning('file open failure');
        end   

        try                          
            load([end_path, '\processed_segments.mat']);
        catch
            warning('file open failure');
        end 

        prefix = Names{i};


        [baseDate_transform_bfd, P_info, R_info, T_info] = Superimposed_Average(baseDate_denoised,Fs,0);
      
        [Features_baseDate_2D_Wave] = Function_FeatExt_baseDate_2D_Wave(baseDate_transform_bfd,P_info,R_info,T_info,numChannels,Fs,prefix);

        
       
        % if i == 1 && j == 1
        %     % Features_baseDate_2D_Time(1,2:end) = cellfun(@(x) [prefix num2str(j) '_' x], Features_baseDate_2D_Time(1,2:end), 'UniformOutput', false);
        %     % Features_baseDate_2D_Time_Total = Features_baseDate_2D_Time;
        %     Features_baseDate_2D_Wave(1,2:end) = cellfun(@(x) [prefix num2str(j) '_' x], Features_baseDate_2D_Wave(1,2:end), 'UniformOutput', false);
        %     Features_baseDate_2D_Wave_Total = Features_baseDate_2D_Wave;
        % else
        %     
        %     % Features_baseDate_2D_Time(1,:) = cellfun(@(x) [prefix num2str(j) '_' x], Features_baseDate_2D_Time(1,:), 'UniformOutput', false);
        %     % Features_baseDate_2D_Time_Data = Features_baseDate_2D_Time(:, 2:end);
        %     Features_baseDate_2D_Wave(1,:) = cellfun(@(x) [prefix num2str(j) '_' x], Features_baseDate_2D_Wave(1,:), 'UniformOutput', false);
        %     Features_baseDate_2D_Wave_Data = Features_baseDate_2D_Wave(:, 2:end);
        %     
        %     % Features_baseDate_2D_Time_Total = [Features_baseDate_2D_Time_Total, Features_baseDate_2D_Time_Data];
        %     Features_baseDate_2D_Wave_Total = [Features_baseDate_2D_Wave_Total, Features_baseDate_2D_Wave_Data];
        % end


        if isempty(Features_baseDate_2D_Wave_Total)
            
            Features_baseDate_2D_Wave(1,2:end) = cellfun(@(x) [prefix num2str(j) '_' x], Features_baseDate_2D_Wave(1,2:end), 'UniformOutput', false);
            Features_baseDate_2D_Wave_Total = Features_baseDate_2D_Wave;
        else
            
            Features_baseDate_2D_Wave(1,:) = cellfun(@(x) [prefix num2str(j) '_' x], Features_baseDate_2D_Wave(1,:), 'UniformOutput', false);
            Features_baseDate_2D_Wave_Data = Features_baseDate_2D_Wave(:, 2:end);
           
            Features_baseDate_2D_Wave_Total = [Features_baseDate_2D_Wave_Total, Features_baseDate_2D_Wave_Data];
        end
        
       
        if mod(batch_counter, batch_size) == 0 || (i == Number_MCGData && j == numel(validSubFolders))
           
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            filename = sprintf('Features_baseDate_2D_Wave_Total_batch_1_to_%d_%s.xlsx', ...
                               batch_counter, timestamp);
            
          
            writecell(Features_baseDate_2D_Wave_Total, fullfile(output_path, filename));
           
            

        end
    

        clear Features_baseDate_2D_Time Features_baseDate_2D_Wave
        clear pk HRS bfd base processed_segments
        toc
        disp('================================================================');
    end
end

%%
output_path = 'E:\';
writecell(Features_baseDate_2D_Wave_Total, fullfile(output_path,'Features_baseDate_2D_Wave_Total.xlsx'));



if mod(batch_counter, batch_size) ~= 0
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    filename = sprintf('Features_baseDate_2D_Wave_Total_%d_to_%d_%s.xlsx', ...
                      floor(batch_counter/batch_size)*batch_size + 1, batch_counter, timestamp);
    
    writecell(Features_baseDate_2D_Wave_Total, fullfile(output_path, filename));

end

fprintf('Processing completed！');
fprintf('The characteristic data is stored in：%s\n',output_path);

toc
































