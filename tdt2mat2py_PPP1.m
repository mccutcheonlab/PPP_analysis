function [output] = tdt2mat2py_PPP1(tank, rat, session, blueName, uvName, trialsL, trialsR, licksL, licksR, savefolder)

% clear all; close all 
% rat = 'PPP1.2';
% session = '5';
% tank = 'R:\DA_and_Reward\es334\PPP1\Tanks\Eelke-171020-105930';
% blueName = 'Dv1B';
% uvName = 'Dv2B';
% licksName = 'LiA_'
% savefolder = 'R:\DA_and_Reward\kp259\THPH2\TestSave\'

%% Extracts photometry data and fits signal FROM MULTIPLE BOXES (if there are multiple)
% Reads in TDT data into structured array using TDT function
fileinfo = strcat({'Rat '},rat,{': Session '}, session);
data = TDTbin2mat(tank);

%% Puts info into output file
output.info = data.info;

% Assigns processed data to new variables for easier referencing
output.blue = data.streams.(blueName).data';
output.uv = data.streams.(uvName).data';
output.fs = data.streams.(blueName).fs;
    
%% Gets TTLs
% This short code ensures that illeagal characters, such as underscores,
% aren't included

output.tick = data.epocs.Tick;
if strcmp(trialsL, 'none') == 0 
    output.trialsL = data.epocs.(trialsL);
    output.licksL = data.epocs.(licksL);
else
    output.trialsL = [];
    output.licksL = [];
end

if strcmp(trialsR, 'none') == 0 
    output.trialsR = data.epocs.(trialsR);
    output.licksR = data.epocs.(licksR);
else
    output.trialsR = [];
    output.licksR = [];
end

%% Save file with appropriate name

savefilename = strcat(savefolder,rat,session,'.mat');
save(savefilename, 'output');

    
    
