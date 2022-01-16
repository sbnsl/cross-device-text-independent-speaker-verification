function [ outpuname ] = Activeaudio( inputname,outpuname, new_frequency)


[s,fs]=audioread(inputname);
[~,nc]=size(s);
if nc>1
    s=s(:,1);
end

if new_frequency~=fs
    s = resample(s,new_frequency,fs);
end

[vs,~]=vadsohn(s,new_frequency);

news=s(vs==1);
audiowrite(outpuname,news,new_frequency);
