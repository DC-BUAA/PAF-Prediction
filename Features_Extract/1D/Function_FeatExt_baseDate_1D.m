function [Features_baseDate_1D] = Function_FeatExt_baseDate_1D(processed_segments,Fs)
    valid_channels = setdiff(1:36, [1 6 31 36]); 
    fields = fieldnames(processed_segments);
    
    for f = 1:numel(fields)
        data = processed_segments.(fields{f})(valid_channels, :);
        

        [amp_feat, wave_feat] = basic_features(data, Fs);
        freq_feat = frequency_features(data, Fs);
        HRV_feat = HRV_features(data, Fs);
        correlation_feat = channel_correlation(data);
        svd_feat = matrix_decomposition_features(data);
        lbp_feat = lbp_features(data);

        Features_baseDate_1D_segments = struct(...
            'Amplitude', amp_feat,...
            'Waveform', wave_feat,...
            'Frequency', freq_feat,...
            'HRV',HRV_feat,...
            'Correlation', correlation_feat,...
            'SVD', svd_feat,...
            'LBP', lbp_feat);
    
        [~,Features_baseDate_1D_segments_Vertical_Completely]  = Struct_to_Vertical(Features_baseDate_1D_segments,'Features_baseDate_1D_segments', '');
    
        if f==1
            Features_baseDate_1D(:,1) = Features_baseDate_1D_segments_Vertical_Completely(:,1);
            Features_baseDate_1D(:,2) = Features_baseDate_1D_segments_Vertical_Completely(:,2);
        else
            Features_baseDate_1D(:,f+1) = Features_baseDate_1D_segments_Vertical_Completely(:,2);
        end

    end


    numSegments = numel(fields);
    header = cell(1, numSegments + 1);
    header{1, 1} = 'Features';
    for seg = 1:numSegments
        header{1, seg + 1} = ['Segment' num2str(seg)];
    end
    Features_baseDate_1D = [header; Features_baseDate_1D];
    
end




function [amp_feat, wave_feat] = basic_features(data, Fs)

    [n_channels, n_samples] = size(data);
    

    amp_feat = struct();
    wave_feat = struct();
    
    for ch = 1:n_channels
        sig = data(ch,:);
        

        amp_feat.Max(ch) = max(sig);
        amp_feat.Mean(ch) = mean(sig);
        amp_feat.Var(ch) = var(sig);
        amp_feat.CV(ch) = std(sig)/mean(sig);
        amp_feat.Range(ch) = range(sig);
        amp_feat.Integral(ch) = sum(sig)/Fs;
        amp_feat.Area(ch) = sum(abs(sig))/Fs;
        

        wave_feat.Skewness(ch) = skewness(sig); 
        wave_feat.Kurtosis(ch) = kurtosis(sig);  
        rms_val = rms(sig);
        abs_mean_value = mean(abs(sig));
        peak_val = max(sig);
        wave_feat.Waveform(ch) = rms_val / abs_mean_value; 
        wave_feat.Peak(ch) = peak_val / rms_val;           
        wave_feat.Pulse(ch) = peak_val / abs_mean_value;  
        wave_feat.Margin(ch) = peak_val / std(sig);       
        wave_feat.Fluctuation(ch) = std(sig) / mean(sig);  
    end
end


function freq_feat = frequency_features(data, Fs)
    [n_channels, ~] = size(data);
    freq_feat = struct();
    
    for ch = 1:n_channels
        sig = data(ch,:);

        N = length(sig);
        Y = fft(sig);
        P2 = abs(Y/N); 
        P1 = P2(1:floor(N/2+1)); 
        P1(2:end-1) = 2*P1(2:end-1); 
        f = Fs*(0:(N/2))/N;
        [peakValue, peakIndex] = max(P1);
        freq_feat.MainFrequency(ch) = f(peakIndex);              
        freq_feat.MainAmplitude(ch) = peakValue;                 
        probabilities = P1 / sum(P1); 
        freq_feat.FrequencyEntropy(ch) = -sum(probabilities .* log(probabilities + eps)); 
        freq_feat.Energy(ch) = sum(P1.^2);                      
        freq_feat.CenterFrequency(ch) = sum(f .* probabilities); 
        PowerSpectrum = P1.^2;                        
        freq_feat.MedianPower(ch) = median(PowerSpectrum);      
        freq_feat.MaxPower(ch) = max(PowerSpectrum);           
    end

end

function HRV_feat = HRV_features(data, Fs)
    [n_channels, ~] = size(data);
    HRV_feat = struct();
    
    for ch = 1:n_channels
        sig = data(ch,:);

        [~, locs] = findpeaks(sig,... 
            'MinPeakHeight', median(sig)*1.5,... 
            'MinPeakDistance', round(0.4*Fs),  ...         
            'MinPeakProminence', 0.3*range(sig),... 
            'WidthReference','halfprom' ...                 
        );
      
        rr_intervals = diff(locs)/Fs; 
    
        HRV_feat.mean_rr(ch) = mean(rr_intervals);
        HRV_feat.SDNN(ch) = std(rr_intervals);
        HRV_feat.cv(ch) = std(rr_intervals)/mean(rr_intervals);
        HRV_feat.SDSD(ch) = std(diff(rr_intervals));


    end



end


function correlation_feat = channel_correlation(data)
    
    data = data';
    [N,C]=size(data);
    R=corr(data,data);
    
    R_Value=[];
    for i=1:C
       R_Value=[R_Value,R(i,i+1:C)];
       R(i,i:C)=0;
    end
    
    [count,~]=hist(R_Value,10);
    
    R_10Hist=count/sum(count);
    correlation_feat.R_Value=R_Value';

end


function [svd_feat] = matrix_decomposition_features(data)

    data = data';
    data=(data-mean(data))./(max(data)-min(data)); 

    [~,S,~]=svd(data');          
    wavesvdVar=diag(S(1:32,1:32)); 
    wavesvdVar=wavesvdVar/sum(wavesvdVar);
    
 
    SVD_Entropy=0;
    for i=1:32
       svd_feat.SVD_Entropy=SVD_Entropy-wavesvdVar(i)*log2(wavesvdVar(i)); 
    end
    q=2;
    svd_feat.SVD_Tsallis=(1/(q-1))*sum(1 - wavesvdVar.^q);
    svd_feat.SVD_Renyi=(1/(q-1))*log(sum(wavesvdVar.^q));
    

    svd_feat.SVD_S=wavesvdVar(1:10);
end


function [lbp_feat] = lbp_features(data)
    data = data';
    [R,C]=size(data);
    x=1:R;
    N=240;
    T = linspace(1, R, N)';
    for ch=1:C
    datas(:,ch) = interp1(x, data(:,ch), T.','spline');
    end
    
    inter=4; 
    R=N;
    LBP_10Hist=[];
    lbpval=[];
    
    for cl=1:C 
        count=1;
        for ri= inter+1:R-inter-1
 
            wavebuff=datas(ri-inter:ri+inter,cl);
            lbpv=zeros(2*inter+1,1);
            t=find(wavebuff>wavebuff(inter+1));
            lbpv(t)=1;
            lbpv(inter+1)=[];
            buff(count)=2^7*lbpv(8)+2^6*lbpv(7)+2^5*lbpv(6)+2^4*lbpv(5)+2^3*lbpv(4)+2^2*lbpv(3)+2^1*lbpv(2)+lbpv(1);
            count=count+1;
        end
        [N,X]=hist(buff,10); 
        LBP_10Hist=[LBP_10Hist;N];
    end
    lbp_feat.LBP_10Hist = reshape(LBP_10Hist', [], 1);

end