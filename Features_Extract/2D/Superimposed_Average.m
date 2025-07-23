function [baseDate_transform_bfd, P_info, R_info, T_info] = Superimposed_Average(baseDate_denoised, Fs, debug)

    segments = fieldnames(baseDate_denoised);
    numSegments = numel(segments);
    

    baseDate_transform_bfd = struct();
    P_info = struct();
    T_info = struct();
    R_info = struct();

    for segIdx = 1:numSegments
        segName = segments{segIdx};
        currentData = baseDate_denoised.(segName);
        
        try
         
            signal = currentData(21, :);
            [~, locs] = findpeaks(signal,...
                'MinPeakHeight', median(signal)*1.5,...
                'MinPeakDistance', round(0.4*Fs),...
                'MinPeakProminence', 0.3*range(signal(100:end-100)),...
                'WidthReference','halfprom');

            if length(locs) < 2
                warning('segment %s : The detected R wave is insufficient. Skip this segment.', segName);
                continue;
            end


            RR_intervals = diff(locs);
            avg_RR = round(mean(RR_intervals));
            pre_samples = round(avg_RR * 5/12);
            post_samples = round(avg_RR * 7/12);
            window_length = pre_samples + post_samples;
            R_info.(segName).Position = pre_samples;   
            R_info.(segName).avg_RR = avg_RR;         

            valid_indices = (locs >= pre_samples + 1) & (locs + post_samples - 1 <= size(currentData, 2));
            valid_locs = locs(valid_indices);
            
            num_valid_locs = length(valid_locs);
            num_channels = size(currentData, 1);
            data_segments = zeros(num_channels, num_valid_locs, window_length);
            for i = 1:num_valid_locs
                start_idx = valid_locs(i) - pre_samples;
                end_idx = valid_locs(i) + post_samples - 1;
                data_segments(:, i, :) = currentData(:, start_idx:end_idx);
            end
            baseDate_transform_bfd_seg = squeeze(mean(data_segments, 2));
            
            baseDate_transformed_interp = zeros(size(baseDate_transform_bfd_seg, 1),size(baseDate_transform_bfd_seg, 2));
            for t = 1:size(baseDate_transform_bfd_seg, 2)
    
                vec = baseDate_transform_bfd_seg(:, t);

       
                mat_col = reshape(vec, 6, 6);

               
                mat_row = mat_col.';

  
                mat_row(1, 1) = (mat_row(1, 2) + mat_row(2, 1) + mat_row(2, 2)) / 3;

   
                mat_row(1, 6) = (mat_row(1, 5) + mat_row(2, 6) + mat_row(2, 5)) / 3;


                mat_row(6, 1) = (mat_row(5, 1) + mat_row(6, 2) + mat_row(5, 2)) / 3;



                mat_row(6, 6) = (mat_row(5, 6) + mat_row(6, 5) + mat_row(5, 5)) / 3;

  
                mat_col_processed = mat_row.';
                vec_processed = mat_col_processed(:);

               
                baseDate_transformed_interp(:, t) = vec_processed;
            end

           
            avg_cycle_duration = window_length / Fs;
            P_search_start = round(window_length/12 * 5 - 0.35*avg_cycle_duration*Fs);
            P_search_end = round(window_length/12 * 5 - 0.1*avg_cycle_duration*Fs);
            T_search_start = round(window_length/12 * 5 + 0.15*avg_cycle_duration*Fs);
            T_search_end = round(window_length/12 * 5 + 0.5*avg_cycle_duration*Fs);
            
            ch21_avg = baseDate_transform_bfd_seg(21,:);
            [P_pos, P_is_peak, P_amp, P_start, P_end] = detect_wave_boundaries(ch21_avg, [P_search_start, P_search_end], 'P', Fs);
            [T_pos, T_is_peak, T_amp, T_start, T_end] = detect_wave_boundaries(ch21_avg, [T_search_start, T_search_end], 'T', Fs);
            
            
            backup_channels = [17, 16, 19, 22, 15, 14, 28];
            
           
            need_backup_P = false;
            if ~isempty(P_pos) && ~isempty(P_start) && ~isempty(P_end)
                p_dist_start = abs(P_start - P_pos);
                p_dist_end = abs(P_end - P_pos);
                if p_dist_start < 20 || p_dist_end < 20
                    need_backup_P = true;
                end
            end
            
           
            if isempty(P_pos) || isempty(P_start) || isempty(P_end) || need_backup_P
                
                backup_signal = baseDate_transform_bfd_seg(20,:);
                [temp_pos, temp_is_peak, temp_amp, temp_start, temp_end] = detect_wave_boundaries(backup_signal, [P_search_start, P_search_end], 'P', Fs);
                
                if ~isempty(temp_pos)
                    P_pos = temp_pos; P_is_peak = temp_is_peak; P_amp = temp_amp;
                    P_start = temp_start; P_end = temp_end;
                else
                   
                    for ch = backup_channels(1:end)
                        backup_signal = baseDate_transform_bfd_seg(ch,:);
                        [temp_pos, temp_is_peak, temp_amp, temp_start, temp_end] = detect_wave_boundaries(backup_signal, [P_search_start, P_search_end], 'P', Fs);
                        if ~isempty(temp_pos)
                            P_pos = temp_pos; P_is_peak = temp_is_peak; P_amp = temp_amp;
                            P_start = temp_start; P_end = temp_end;
                            break;
                        end
                    end
                end
            end
            
            
            need_backup_T = false;
            if ~isempty(T_pos) && ~isempty(T_start) && ~isempty(T_end)
                t_dist_start = abs(T_start - T_pos);
                t_dist_end = abs(T_end - T_pos);
                if t_dist_start < 20 || t_dist_end < 20
                    need_backup_T = true;
                end
            end
            
            
            if isempty(T_pos) || isempty(T_start) || isempty(T_end) || need_backup_T
                
                backup_signal = baseDate_transform_bfd_seg(20,:);
                [temp_pos, temp_is_peak, temp_amp, temp_start, temp_end] = detect_wave_boundaries(backup_signal, [T_search_start, T_search_end], 'T', Fs);
                
                if ~isempty(temp_pos)
                    T_pos = temp_pos; T_is_peak = temp_is_peak; T_amp = temp_amp;
                    T_start = temp_start; T_end = temp_end;
                else
                   
                    for ch = backup_channels(1:end)
                        backup_signal = baseDate_transform_bfd_seg(ch,:);
                        [temp_pos, temp_is_peak, temp_amp, temp_start, temp_end] = detect_wave_boundaries(backup_signal, [T_search_start, T_search_end], 'T', Fs);
                        if ~isempty(temp_pos)
                            T_pos = temp_pos; T_is_peak = temp_is_peak; T_amp = temp_amp;
                            T_start = temp_start; T_end = temp_end;
                            break;
                        end
                    end
                end
            end

           
            baseDate_transform_bfd.(segName) = baseDate_transformed_interp';
            P_info.(segName) = struct('Position',P_pos, 'IsPeak',P_is_peak, 'Amplitude',P_amp, 'Start',P_start, 'End',P_end);
            T_info.(segName) = struct('Position',T_pos, 'IsPeak',T_is_peak, 'Amplitude',T_amp, 'Start',T_start, 'End',T_end);
            
            if debug
              
                figure('Position', [100, 100, 1200, 900]);
                for ch = 1:num_channels
                    subplot(6,6,ch);
                    chan_data = baseDate_transform_bfd_seg(ch,:);
                    plot(chan_data, 'LineWidth',1.5, 'Color',[0 0 0]); hold on;
                    
                 
                    r_pos = pre_samples + 1;
                    scatter(r_pos, chan_data(r_pos), 20, 'o', 'MarkerEdgeColor', [1 0 0], 'MarkerFaceColor', [0.8 0 0], 'LineWidth',1.5);
                    
                   
                    if ~isempty(P_info.(segName).Position)
                        scatter(P_info.(segName).Position, chan_data(P_info.(segName).Position), 20, 'o', 'LineWidth',2, 'MarkerEdgeColor', [0 0.7 0]);
                        scatter([P_info.(segName).Start, P_info.(segName).End], chan_data([P_info.(segName).Start, P_info.(segName).End]), 20, 'v', 'LineWidth',1.5, 'MarkerEdgeColor', [0 1 0]);
                    end
                    
                   
                    if ~isempty(T_info.(segName).Position)
                        scatter(T_info.(segName).Position, chan_data(T_info.(segName).Position), 20, 'o', 'LineWidth',2, 'MarkerEdgeColor', [0 0 0.5]);
                        scatter([T_info.(segName).Start, T_info.(segName).End], chan_data([T_info.(segName).Start, T_info.(segName).End]), 20, 'v', 'LineWidth',1.5, 'MarkerEdgeColor', [0 0 1]);
                    end
                    
                    title(sprintf('Channel %d',ch)); grid on; axis tight;
                    if ch == 1
                        legend({'Average waveform','R peak','P peak','P Boundary','T peak','T Boundary'}, 'FontSize',8, 'Location','best');
                    end
                end
                sgtitle(sprintf('Segment %s | P:%d(%s) | T:%d(%s)', segName, ...
                    P_info.(segName).Position, ifelse(P_info.(segName).IsPeak, 'Peak', 'Trough'), ...
                    T_info.(segName).Position, ifelse(T_info.(segName).IsPeak, 'Peak', 'Trough')), 'FontSize',12, 'FontWeight','bold');
            end
        catch ME
            warning('Segment %s Failure: %s', segName, ME.message);
            continue;
        end
    end
end



function [pos, is_peak, amp, start, ending] = detect_wave_boundaries(signal, search_win, wave_type, Fs)

    order = 3; frame_len = 11;
    smoothed_signal = sgolayfilt(signal, order, frame_len);
    
  
    [pos, is_peak, amp] = detect_peak_valley(smoothed_signal, search_win, wave_type);
    
   
    if ~isempty(pos)
        [start, ending] = find_wave_edges(smoothed_signal, pos, is_peak, wave_type, Fs);
    else
        start = []; ending = [];
    end
end

function [pos, is_peak, amp] = detect_peak_valley(signal, search_win, wave_type)
    
    switch wave_type
        case 'P'
            min_prom = 0.03 * range(signal);
            peak_sort = 'descend';
        case 'T'
            min_prom = 0.03 * range(signal);
            peak_sort = 'descend';
    end
    
    [pos_pks, pos_locs] = findpeaks(signal(search_win(1):search_win(2)),...
        'MinPeakProminence', min_prom, 'SortStr', peak_sort);
    
    [neg_pks, neg_locs] = findpeaks(-signal(search_win(1):search_win(2)),...
        'MinPeakProminence', min_prom, 'SortStr', peak_sort);
    neg_pks = -neg_pks;

  
    all_amps = [pos_pks, neg_pks];
    all_locs = [pos_locs, neg_locs] + search_win(1) - 1;
    all_types = [true(length(pos_pks),1); false(length(neg_pks),1)];

    if ~isempty(all_amps)
        pos = all_locs(1);
        is_peak = all_types(1);
        amp = all_amps(1);
    else
        pos = []; is_peak = []; amp = [];
    end
end

function [start, ending] = find_wave_edges(signal, pos, is_peak, wave_type, Fs)
    
    switch wave_type
        case 'P'
            start = trace_slope(diff(signal), pos, -1, round(0.1*Fs), is_peak, wave_type, Fs);
            ending = trace_slope(diff(signal), pos, 1, round(0.1*Fs), is_peak, wave_type, Fs);
        case 'T'
            start = trace_slope(diff(signal), pos, -1, round(0.15*Fs), is_peak, wave_type, Fs);
            ending = trace_slope(diff(signal), pos, 1, round(0.15*Fs), is_peak, wave_type, Fs);
    end

    
  
    min_wave_width = round(0.04 * Fs); 
    if abs(ending - start) < min_wave_width
        if is_peak
            start = pos - round(0.04*Fs);
            ending = pos + round(0.04*Fs);
        else
            start = pos + round(0.04*Fs);
            ending = pos - round(0.04*Fs);
        end
    end
    

    start = min(max(start,1),length(signal));
    ending = min(max(ending,1),length(signal));
    if start > ending
        [start, ending] = deal(ending, start);
    end
end

function position = trace_slope(diff_signal, start_pos, direction, max_steps, is_peak, wave_type, Fs)
 
    window_size = round(0.08 * Fs); 
    window = diff_signal(max(1,start_pos-window_size):start_pos);
    baseline_noise = median(abs(window)) * 0.8; 
    if is_peak
    
        peak_value = max(abs(diff_signal(max(1,start_pos-10):min(end,start_pos+10))));
        dynamic_threshold = min(baseline_noise, peak_value*0.25); 
        
  
        steps = 0;
        current_pos = start_pos;
        
        
        while steps < max_steps
            switch wave_type
                case 'P'
                    step_size = min(10, max_steps - steps);
                case 'T'
                    step_size = min(10, max_steps - steps);      
            end
            
            current_pos = current_pos + direction*step_size;
            current_pos = max(1, min(length(diff_signal), current_pos));
            
            if abs(diff_signal(current_pos)) < dynamic_threshold
                break;
            end
            steps = steps + step_size;
        end
        
      
        refine_steps = 0;
        while refine_steps < step_size
            current_pos = current_pos - direction*1; 
            current_pos = max(1, min(length(diff_signal), current_pos));
            
            if abs(diff_signal(current_pos)) >= dynamic_threshold
                current_pos = current_pos + direction*1; 
                break;
            end
            refine_steps = refine_steps + 1;
        end
        position = current_pos;
    
    else

        peak_value = max(abs(diff_signal(max(1,start_pos-10):min(end,start_pos+10))));
        dynamic_threshold = -min(baseline_noise, peak_value*0.25); 
        

        steps = 0;
        current_pos = start_pos;

        while steps < max_steps
            switch wave_type
                case 'P'
                    step_size = min(10, max_steps - steps); 
                case 'T'
                    step_size = min(10, max_steps - steps); 
            end
            current_pos = current_pos + direction*step_size;
            current_pos = max(1, min(length(diff_signal), current_pos));
            
            if abs(diff_signal(current_pos)) > dynamic_threshold
                break;
            end
            steps = steps + step_size;
        end
        

        refine_steps = 0;
        while refine_steps < step_size
            current_pos = current_pos - direction*1; 
            current_pos = max(1, min(length(diff_signal), current_pos));
            
            if abs(diff_signal(current_pos)) <= dynamic_threshold
                current_pos = current_pos + direction*1; 
                break;
            end
            refine_steps = refine_steps + 1;
        end
        
        position = current_pos;
    end

end



function s = ifelse(condition, str1, str2)
    if condition
        s = str1;
    else
        s = str2;
    end
end