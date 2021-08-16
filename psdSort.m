function [ pxx,fpow,avgPsdFeatures ] = psdSort( inSignal,fs )
% ---------------------------------
%    �������ܶ��Լ���������Ƶ�����źŹ���
% ---------------------------------
% INPUT:
%   inSignal  �����ź�
%   fs  ����Ƶ��
% OUTPUT:
%   pxx  �������ܶ�
%   fpow  Ƶ������
%   psdFeatures  ������Ƶ����ƽ��������ɵ�����

    [pxx, fpow] = pwelch(inSignal, [], [], [], fs);
    psd_alpha = bandpower(pxx, fpow, [8, 14], 'psd');
    psd_beta = bandpower(pxx, fpow, [14, 30], 'psd');
    avgPsdFeatures=[psd_alpha,psd_beta];
    
end

