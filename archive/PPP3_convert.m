% TDT test convert - mini-script for testing TDT conversion scripts
% In general, a similar script should be stored for each set of conversions
% somewhere near the data folder

[folder, name, ext] = fileparts(which(mfilename('fullpath')));

folder = 'R:\DA_and_Reward\gc214\PPP3\'

tankfolder = strcat(folder, 'tdtfiles\');
savefolder = strcat(folder, 'matfiles\');

skipfiles = 1;
processfiles = 1;
nboxes = 2;

metafile = 'R:\DA_and_Reward\gc214\PPP3\PPP3.xlsx'
sheet = 'PPP3_metafile';
[~,~,a] = xlsread(metafile,sheet);

TDTmasterconvert(a, tankfolder, savefolder,...
     skipfiles, processfiles);

% This is a quick script that processes blue and UV signals using Vaibhav's
% procedure without having to reload all original 
if processfiles == 1
    TDTquickprocess(a, savefolder)
end


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

% 
% tank = 'R:\DA_and_Reward\Shared\Scripts\THPH Tanks\Kate-170810-072909';
% data = TDTbin2mat(tank);
% 
% tank = 'R:\DA_and_Reward\gc214\PPP3\tdtfiles\Giulia-180709-083142';
% data = TDTbin2mat(tank);
% 
% tank = 'R:\DA_and_Reward\gc214\PPP3\tdtfiles\Giulia-180709-100216';
% data = TDTbin2mat(tank);