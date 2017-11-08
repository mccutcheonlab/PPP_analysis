%% Convert THPH1 files from TDT to Python via Matlab

clear all; close all;

% Folder locations
metafileFolder = 'R:\DA_and_Reward\es334\PPP1\';
metafile = strcat(metafileFolder,'PPP1_metafile.txt');
dataFolder = 'R:\DA_and_Reward\es334\PPP1\Tanks\';
saveFolder = 'R:\DA_and_Reward\es334\PPP1\Matlab Files\';

% Open metafile
fid = fopen(metafile);
C = textscan(fid, '%s %s %s %s %d %s %d %s %s %d %d %d %d %d %d %s %s %s %s %s %s %d %d','Delimiter','\t','HeaderLines',1);
fclose(fid);

% Loop through rows in metafile
for i = 1:size(C{1,1},1)
    tic
    rat = char(C{1,3}(i));
    session = char(C{1,4}(i));
    if C{1,23}(i) == 1 % checks to see if Row is to be included or not
        TDTfile = char(strcat(dataFolder,C{1,1}(i)));
        blue = char(C{1,16}(i));
        uv = char(C{1,17}(i));
        trialL = char(C{1,18}(i));
        trialR = char(C{1,19}(i));
        lickL = char(C{1,20}(i));
        lickR = char(C{1,21}(i));        
        try
            tdt2mat2py_PPP1(TDTfile,rat,session,blue,uv,trialL,trialR,lickL,lickR,saveFolder)
        catch ME
            disp(['Rat ' rat ', session = ' session ' has failed!'])
        end
    else
        disp(['Skipping Rat ' rat ', session = ' session '...'])    
    end
    toc
end



