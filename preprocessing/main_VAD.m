
SRC_FLDR='/home/Speaker/Voice_all_in_onefolder';
DST_FLDR='/home/Speaker/Voice_all_onefolder_active;


c = 1;
fs=48000;
filenames=dir(SRC_FLDR);
for f=3:length(filenames)
    filename=filenames(f).name;
    if strcmp(filename(end-2:end),'MP3')
    inptfile = strcat(SRC_FLDR,'/',filename);
    C = strsplit(filename,'.');
    C=C{1};
    outputfile=strcat(DST_FLDR,'/',C,'_active.wav');
    tic;Activeaudio(inptfile,outputfile,fs);toc
    
    c = c + 1
    end
end
