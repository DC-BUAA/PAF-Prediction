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
segment_length = 5 * Fs;   
channels_to_skip = [1,6,31,36]; 
discard_points = 1000;

for i = 1 : Number_MCGData
    disp(strcat( 'Processing...:'  , Names{i}) );
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
            fid_1 = fopen(fullfile(end_path, [end_name, '.BFD'])); % 读取 .BFD 文件
            bfd = fread(fid_1, 'float');
            fclose(fid_1);
            if mod(length(bfd), 1000)
                bfd = bfd(513:end);
            end
            bfd_all(:, i) = bfd;
        catch
            warning('file open failure');
        end

        numChannels=36;

        try                          
            fid_2=fopen(fullfile(end_path, [end_name, '.baseDate']));
            base = fread(fid_2,'float');
            fclose(fid_2);
            a=mod(length(base),1000);
            data=zeros;
            if a==512
                data=base(513:length(base),:); 
            else
                data=base;
            end
        catch
            warning('file open failure');
        end 

        dataChannels=zeros;
        time = round((length(data))/numChannels/Fs);
        for t = 1:1:time
            for n = 1:1:numChannels
                for g = 1:1:Fs
                    dataChannels(n,(t-1)*Fs+g) = data((t-1)*numChannels*Fs+(n-1)*Fs+g);
                end
            end
        end

        valid_data = dataChannels(:, discard_points+1:end);
        [~, total_points] = size(valid_data);
        n_segments = floor(total_points / segment_length);
        baseDate_denoised = struct();
        valid_segment_count = 0; 
        for seg = 1:n_segments

            start_idx = (seg-1)*segment_length + 1;
            end_idx = seg*segment_length;
            current_segment = zeros(36, segment_length);
            for ch = 1:36
                raw_segment = valid_data(ch, start_idx:end_idx);
                if ismember(ch, channels_to_skip)
                    current_segment(ch, :) = raw_segment;
                else
                    current_segment(ch, :) = preprossing_mcg(raw_segment, Fs, 1, segment_length);
                end
            end
            
            exclusion_reasons = {}; 
            channels_to_check = [20, 21];
            
            for ch_idx = 1:length(channels_to_check)
                ch = channels_to_check(ch_idx);
                signal = current_segment(ch, :);
                
                [~, locs] = findpeaks(signal,... 
                    'MinPeakHeight', median(signal)*1.5,... 
                    'MinPeakDistance', round(0.4*Fs),  ...      
                    'MinPeakProminence', 0.3*range(signal(100:end-100)),... 
                    'WidthReference','halfprom' ...             
                );


                figure;
                plot(signal,'linewidth',1.5),hold on;
                plot(locs,signal( locs ),'r*');hold off;
                xlabel('Data points','fontsize',14)
                ylabel('Magnitude/pT','fontsize',14)
                title('R Peaks','fontsize',14)

                if length(locs) < 2
                    exclusion_reasons{end+1} = sprintf('Channal%d - Detected %d R waves', ch, length(locs));
                    continue; 
                end

                rr_intervals = diff(locs)/Fs; 
                

                abnormal_intervals = rr_intervals(rr_intervals < 0.3 | rr_intervals > 1.5);
                if ~isempty(abnormal_intervals)
                    exclusion_reasons{end+1} = sprintf('Channel %d - Abnormal RR interval [%.3fs]', ch, min(abnormal_intervals));
                end
                

                mean_rr = mean(rr_intervals);
                std_rr = std(rr_intervals);
                cv = std_rr/mean_rr;
                if cv > 0.3
                    exclusion_reasons{end+1} = sprintf('Channel %d - Coefficient of Variation %.1f%%', ch, cv*100);
                end
            end
            

            if ~isempty(exclusion_reasons)
                fprintf('Exclude the fragment %d：\n', seg);
                fprintf('→ %s\n', exclusion_reasons{:});
                continue;
            else
                fprintf('Inclusion segment %d：\n', seg);
                fprintf('→ The number of R wave peaks： %d\n', length(locs));
            end

            valid_segment_count = valid_segment_count + 1;
            field_name = sprintf('segment%d', valid_segment_count);
            baseDate_denoised.(field_name) = current_segment;

        end
    
        save_path = end_path; 
        file_name = 'processed_segments.mat';
        save(fullfile(save_path, file_name), 'baseDate_denoised', '-v7.3');
        disp(strcat( 'Saved in [' , end_path , '] of', file_name, '\n' ));
        disp('================================================================');

        clear data bfd
    end
    toc
    
end
