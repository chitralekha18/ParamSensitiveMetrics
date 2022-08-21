function Meas(dirname, outputdir)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Measure a certain audio's stats
%
% Inputs:   dirname: the dir name of all stat files
%           outputdir: the output stats files
% Outpus:   none, will be stored in _stats/sample_name.mat
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    cd('STSstep');
    dirname = ['../' dirname];
    outputdir = ['../' outputdir];

    disp(['Measuring statistics of ' dirname]);

    %% start the ltfat toolbox
    warning off
    addpath(genpath('_ltfat'))
    addpath(genpath('_minFunc_2012'))
    addpath(genpath('_sts'))
    
    mfb_mode = 'log20';
    if ~exist(['_system/AudSys_Setup_' mfb_mode '.mat'], "file")
        AudSys_Setup(mfb_mode);
    end
    load(['_system/AudSys_Setup_' mfb_mode '.mat'], 'fs', 'fs_d', 'compression', 'fcc', 'g', 'mfb', 'mfin');
    
    if ~exist(outputdir, "dir")
        mkdir(outputdir)
    end

    % If is a file
    if isfile(dirname)
        [~, filename, ~] = fileparts(dirname);
        MeasNSaveSingle(dirname, [outputdir '/' filename '.mat'], fs, fs_d, compression, fcc, g, mfb, mfin);
    elseif isfolder(dirname)
        dirlist = dir([dirname '/*.wav']);
        
        if 0==length(dirlist)
            fprintf("There is no wav file in this dir")
            return
        end

        for di = 1:length(dirlist)
            sample_name = dirlist(di);
            [~, filename, ~] = fileparts(sample_name.name);
            fprintf("%d/%d | %s\n", di, length(dirlist), sample_name.name);
            
            MeasNSaveSingle([dirname '/' filename '.wav'], [outputdir '/' filename '.mat'], fs, fs_d, compression, fcc, g, mfb, mfin);
        end
    else
        disp('No such file/folder') 
    end

    cd('..')
end


function MeasNSaveSingle(filename, outputfilename, fs, fs_d, compression, fcc, g, mfb, mfin)
    [x,fsx] = audioread(filename);
    
    x = x(1:floor(length(x)/fsx)*fsx);
    x = resample(x,fs,fsx);
    
    Px = 1/length(x) * sum(x.^2); % 1*1
    x = x * sqrt(1e-4/Px);        % 140000*1
    
    % measure input signal statistics 
    x_sub = ufilterbank(x,g,1)'; % 36*140000
    % dex_sub: 36*2800
    % exf_sub: 36*140000
    [dex_sub, ~] = Subband_Envelopes(x_sub,fs,fs_d,compression,'hilbert',fcc);
    
    % dexm_sub: 36*1 & 800*20
    dexm_sub = mfilterbank(dex_sub,mfb);
    
    % Px: 36*1
    % I:  36*1
    [Px,I] = Envelope_Power(dex_sub);
    
    % Mx: 36*4
    Mx = Envelope_Marginals(dex_sub);
    % Vx: 36*1
    Vx = Envelope_Variance(dex_sub);
    % Cx: 36*36
    Cx = Envelope_Correlation(dex_sub);
    % 36*20
    MPx = Modulation_Power(dexm_sub,dex_sub,fcc,mfin);
    % 36*20
    MVx = Modulation_Variance(dexm_sub,dex_sub);
    % 36*36*20
    MCx = Modulation_Correlation(dexm_sub);
    % I:  1*36
    I = I';

    save(outputfilename, 'Px','Mx','Vx','Cx','MPx','MVx','MCx','I');
end