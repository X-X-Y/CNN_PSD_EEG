function [ pxx,fpow,avgPsdFeatures ] = psdSort( inSignal,fs )
% ---------------------------------
%    求功率谱密度以及各个节律频带的信号功率
% ---------------------------------
% INPUT:
%   inSignal  输入信号
%   fs  采样频率
% OUTPUT:
%   pxx  功率谱密度
%   fpow  频率向量
%   psdFeatures  各节律频带的平均功率组成的数组

    [pxx, fpow] = pwelch(inSignal, [], [], [], fs);
    psd_alpha = bandpower(pxx, fpow, [8, 14], 'psd');
    psd_beta = bandpower(pxx, fpow, [14, 30], 'psd');
    avgPsdFeatures=[psd_alpha,psd_beta];
    
end

