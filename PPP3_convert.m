% TDT test convert - mini-script for testing TDT conversion scripts
% In general, a similar script should be stored for each set of conversions
% somewhere near the data folder

[folder, name, ext] = fileparts(which(mfilename('fullpath')));

folder = 'R:\DA_and_Reward\gc214\PPP3\'

metafile = strcat(folder, 'PPP3_metafile.csv');
tankfolder = strcat(folder, 'tdtfile\');
savefolder = strcat(folder, 'matfiles\');

skipfiles = 0;
processfiles = 0;
nboxes = 2;

txtfileformat = '%s %s %s %s %d %s %d %s %s %d %d %d %d %d %d %s %s %s %s %s %s %d %d %d';

TDTmasterconvert(metafile, tankfolder, savefolder,...
    skipfiles, processfiles, nboxes, txtfileformat);


%%%
% for testing
% tic
% clear all; close all;
% tank = 'R:\DA_and_Reward\Shared\Scripts\THPH Tanks\Kate-170810-072909'
% data = TDTbin2mat(tank);
% toc
% 
% tic
% clear all; close all;
% tank = 'C:\Users\James Rig\Documents\Test data\Kate-170810-072909'
% data = TDTbin2mat(tank);
% toc