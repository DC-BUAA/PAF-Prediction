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


for i = 1 : Number_MCGData
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

        numChannels=36;


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
      

        [Features_baseDate_1D] = Function_FeatExt_baseDate_1D(baseDate_denoised,Fs);

        if i == 1 && j == 1
            Features_baseDate_1D(1,2:end) = cellfun(@(x) [prefix num2str(j) '_' x], Features_baseDate_1D(1,2:end), 'UniformOutput', false);
            Features_baseDate_1D_Total = Features_baseDate_1D;
        else
            Features_baseDate_1D(1,:) = cellfun(@(x) [prefix num2str(j) '_' x], Features_baseDate_1D(1,:), 'UniformOutput', false);
            Features_baseDate_1D_Data = Features_baseDate_1D(:, 2:end);
            Features_baseDate_1D_Total = [Features_baseDate_1D_Total, Features_baseDate_1D_Data];
        end
    
        clear Features_baseDate_1D
        clear pk HRS bfd base processed_segments
        toc
        disp('================================================================');
    end
end

%%
output_path = 'E:\';
writecell(Features_baseDate_1D_Total, fullfile(output_path,'Features_baseDate_1D_Total.xlsx'));
fprintf('Processing completed！');
fprintf('The features data is stored in：%s\n',output_path);

toc
































