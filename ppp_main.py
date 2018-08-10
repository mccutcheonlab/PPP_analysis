# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 16:53:38 2018

@author: James Rig
"""

import JM_general_functions as jmf
import JM_custom_figs as jmfig
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy.io as sio

## Colour scheme
col={}
col['np_cas'] = 'xkcd:silver'
col['np_malt'] = 'white'
col['lp_cas'] = 'xkcd:kelly green'
col['lp_malt'] = 'xkcd:light green'


class Session(object):
    
    def __init__(self, metafiledata, hrows, datafolder, outputfolder):
        self.medfile = metafiledata[hrows['medfile']]
        self.rat = metafiledata[hrows['rat']].replace('.', '-')
        self.session = metafiledata[hrows['session']]
        self.diet = metafiledata[hrows['dietgroup']]
        self.bottleL = metafiledata[hrows['bottleL']]
        self.bottleR = metafiledata[hrows['bottleR']]
        
        self.ttl_trialsL = metafiledata[hrows['ttl-trialL']]
        self.ttl_trialsR = metafiledata[hrows['ttl-trialR']]
        self.ttl_licksL = metafiledata[hrows['ttl-lickL']]
        self.ttl_licksR = metafiledata[hrows['ttl-lickR']]
        
        self.left = {}
        self.right = {}
        self.both = {}
        self.left['subs'] = metafiledata[hrows['bottleL']]
        self.right['subs'] = metafiledata[hrows['bottleR']]
        
        self.matlabfile = datafolder + self.rat + '_' + self.session + '.mat'
        
        self.outputfolder = outputfolder
        
    def loadmatfile(self):
        a = sio.loadmat(self.matlabfile, squeeze_me=True, struct_as_record=False) 
        self.output = a['output']
        self.fs = self.output.fs
        self.data = self.output.blue
        self.dataUV = self.output.uv
        
    def setticks(self):
        try:
            self.tick = self.output.tick.onset
        except AttributeError:
            self.tick = self.output.tick        
        
    def time2samples(self):
        maxsamples = len(self.tick)*int(self.fs)
        if (len(self.data) - maxsamples) > 2*int(self.fs):
            print('Something may be wrong with conversion from time to samples')
            print(str(len(self.data) - maxsamples) + ' samples left over. This is more than double fs.')
            self.t2sMap = np.linspace(min(self.tick), max(self.tick), maxsamples)
        else:
            self.t2sMap = np.linspace(min(self.tick), max(self.tick), maxsamples)

    def event2sample(self, EOI):
        idx = (np.abs(self.t2sMap - EOI)).argmin()   
        return idx

    def check4events(self):

        try:
            lt = getattr(self.output, self.ttl_trialsL)
            self.left['exist'] = True
            self.left['sipper'] = lt.onset
            self.left['sipper_off'] = lt.offset
            ll = getattr(self.output, self.ttl_licksL)
            self.left['licks'] = np.array([i for i in ll.onset if i<max(self.left['sipper_off'])])
            self.left['licks_off'] = ll.offset[:len(self.left['licks'])]
        except AttributeError:
            self.left['exist'] = False
            self.left['sipper'] = []
            self.left['sipper_off'] = []
            self.left['licks'] = []
            self.left['licks_off'] = []
           
        try:
            rt = getattr(self.output, self.ttl_trialsR)
            self.right['exist'] = True
            self.right['sipper'] = rt.onset
            self.right['sipper_off'] = rt.offset
            rl = getattr(self.output, self.ttl_licksR)
            self.right['licks'] = np.array([i for i in rl.onset if i<max(self.right['sipper_off'])])
            self.right['licks_off'] = rl.offset[:len(self.right['licks'])]
        except AttributeError:
            self.right['exist'] = False
            self.right['sipper'] = []
            self.right['sipper_off'] = []
            self.right['licks'] = []
            self.right['licks_off'] = []
            
        if self.left['exist'] == True and self.right['exist'] == True:
            try:
                first = findfreechoice(self.left['sipper'], self.right['sipper'])
                self.both['sipper'] = self.left['sipper'][first:]
                self.both['sipper_off'] = self.left['sipper_off'][first:]
                self.left['sipper'] = self.left['sipper'][:first-1]
                self.left['sipper_off'] = self.left['sipper_off'][:first-1]
                self.right['sipper'] = self.right['sipper'][:first-1]
                self.right['sipper_off'] = self.right['sipper_off'][:first-1]
                self.left['licks-forced'], self.left['licks-free'] = dividelicks(self.left['licks'], self.both['sipper'][0])
                self.right['licks-forced'], self.right['licks-free'] = dividelicks(self.right['licks'], self.both['sipper'][0])
                self.left['nlicks-forced'] = len(self.left['licks-forced'])
                self.right['nlicks-forced'] = len(self.right['licks-forced'])
                self.left['nlicks-free'] = len(self.left['licks-free'])
                self.right['nlicks-free'] = len(self.right['licks-free'])

            except IndexError:
                print('Problem separating out free choice trials')
                        
    def removephantomlicks(self):
        if self.left['exist'] == True:
            phlicks = jmf.findphantomlicks(self.licksL, self.trialsL, delay=3)
            self.licksL = np.delete(self.licksL, phlicks)
            self.licksL_off = np.delete(self.licksL_off, phlicks)
    
        if self.right['exist'] == True:
            phlicks = jmf.findphantomlicks(self.right['licks'], self.trialsR, delay=3)
            self.right['licks'] = np.delete(self.right['licks'], phlicks)
            self.right['licks_off'] = np.delete(self.right['licks_off'], phlicks)
            
    def sessionFig(self, ax):
        ax.plot(self.data, color='blue')
        try:
            ax.plot(self.dataUV, color='m')
        except:
            print('No UV data.')
        ax.set_xticks(np.multiply([0, 10, 20, 30, 40, 50, 60],60*self.fs))
        ax.set_xticklabels(['0', '10', '20', '30', '40', '50', '60'])
        ax.set_xlabel('Time (min)')
        ax.set_title('Rat ' + self.rat + ': Session ' + self.session)
        
    def makephotoTrials(self, bins, events, threshold=10):
        bgMAD = jmf.findnoise(self.data, self.randomevents,
                              t2sMap = self.t2sMap, fs = self.fs, bins=bins,
                              method='sum')          
        blueTrials, self.pps = jmf.snipper(self.data, events,
                                            t2sMap = self.t2sMap, fs = self.fs, bins=bins)        
        UVTrials, self.pps = jmf.snipper(self.dataUV, events,
                                            t2sMap = self.t2sMap, fs = self.fs, bins=bins)
        sigSum = [np.sum(abs(i)) for i in blueTrials]
        sigSD = [np.std(i) for i in blueTrials]
        noiseindex = [i > bgMAD*threshold for i in sigSum]

        return blueTrials, UVTrials, noiseindex
    
    def sessionlicksFig(self, ax):
        if x.left['exist'] == True:
            licks = self.left['lickdata']['licks']
            ax.hist(licks, range(0, 3600, 60), color=self.left['color'], alpha=0.4)          
            yraster = [ax.get_ylim()[1]] * len(licks)
            ax.scatter(licks, yraster, s=50, facecolors='none', edgecolors=self.left['color'])

        if x.right['exist'] == True:
            licks = self.right['lickdata']['licks']
            ax.hist(licks, range(0, 3600, 60), color=self.right['color'], alpha=0.4)          
            yraster = [ax.get_ylim()[1]] * len(licks)
            ax.scatter(licks, yraster, s=50, facecolors='none', edgecolors=self.right['color'])           
        
        ax.set_xticks(np.multiply([0, 10, 20, 30, 40, 50, 60],60))
        ax.set_xticklabels(['0', '10', '20', '30', '40', '50', '60'])
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Licks per min')
    #    ax.set_title('Rat ' + self.rat + ': Session ' + self.session)

    def setbottlecolors(self):
        casein_color = 'xkcd:pale purple'
        malt_color = 'xkcd:sky blue'
        
        # sets default colors, e.g. to be used on saccharin or water days
        self.left['color'] = 'xkcd:grey'
        self.right['color'] = 'xkcd:greyish blue'
           
        if 'cas' in self.bottleL:
            self.left['color'] = casein_color
        if 'malt' in self.bottleL:
            self.left['color'] = malt_color
        
        if 'cas' in self.bottleR:
            self.right['color'] = casein_color
        if 'malt' in self.bottleR:
            self.right['color'] = malt_color

    def side2subs(self):
        if 'cas' in self.left['subs']:
            self.cas = self.left
        if 'cas' in self.right['subs']:
            self.cas = self.right
        if 'malt' in self.left['subs']:
            self.malt = self.left
        if 'malt' in self.right['subs']:
            self.malt = self.right

def findfreechoice(left, right):
    first = [idx for idx, x in enumerate(left) if x in right][0]
    return first
        
def dividelicks(licks, time):
    before = [x for x in licks if x < time]
    after = [x for x in licks if x > time]
    
    return before, after    

def metafile2sessions(xlfile, metafile, datafolder, outputfolder, sheetname='metafile'):
    jmf.metafilemaker(xlfile, metafile, sheetname=sheetname, fileformat='txt')
    rows, header = jmf.metafilereader(metafile + '.txt')
    
    hrows = {}
    for idx, field in enumerate(header):
        hrows[field] = idx
               
    for row in rows:
        sessionID = row[hrows['rat']].replace('.','-') + '_' + row[hrows['session']]
        sessions[sessionID] = Session(row, hrows, datafolder, outputfolder)

# Extracts data from metafile

outputfolder = 'R:\\DA_and_Reward\\gc214\\PPP_combined\\output\\'

sessions = {}
metafile2sessions('R:\\DA_and_Reward\\es334\PPP1\\PPP1.xlsx',
                  'R:\\DA_and_Reward\\es334\PPP1\\PPP1_metafile',
                  'R:\\DA_and_Reward\\es334\\PPP1\\matfiles\\',
                  'R:\\DA_and_Reward\\es334\\PPP1\\output\\',
                  sheetname='metafile')
    

metafile2sessions('R:\\DA_and_Reward\\gc214\\PPP3\\PPP3.xlsx',
                  'R:\\DA_and_Reward\\gc214\\PPP3\\PPP3_metafile',
                  'R:\\DA_and_Reward\\gc214\\PPP3\\matfiles\\',
                  'R:\\DA_and_Reward\\gc214\\PPP3\\output\\',
                  sheetname='PPP3_metafile')

#for session in ['PPP3-1_s10', 'PPP3-2_s10', 'PPP3-3_s10', 'PPP3-4_s10', 'PPP3-5_s10', 'PPP3-8_s10']:
rats = []
for session in sessions:
    x = sessions[session]
    if x.rat not in rats:
        rats.append(x.rat)

rats_to_include = []
rats_to_exclude = ['PPP1-8', 'PPP3-1', 'PPP3-6', 'PPP3-7']
sessions_to_include = ['s16']

if len(rats_to_include) > 0:
    print('Overriding values in rats_to_exclude because of entry in rats_to_include.')
    rats_to_exclude = list(rats)
    for rat in rats_to_include:
        rats_to_exclude.remove(rat)

for session in sessions:
      
    x = sessions[session]
    
    if x.rat not in rats_to_exclude and x.session in sessions_to_include: 
        pdf_pages = PdfPages(x.outputfolder + session + '.pdf')
        
        x.loadmatfile()
        
        print('\nAnalysing rat ' + x.rat + ' in session ' + x.session)
        
        # Load in data from .mat file (convert from Tank first using Matlab script)
        x.loadmatfile()
        # Work out time to samples
        x.setticks()
        x.time2samples()       
        # Find out which bottles have TTLs/Licks associated with them     
        x.check4events()
        #        x.removephantomlicks()
        x.setbottlecolors()
        
        x.left['lickdata'] = jmf.lickCalc(x.left['licks'],
                          offset = x.left['licks_off'],
                          burstThreshold = 0.50)
        
        x.right['lickdata'] = jmf.lickCalc(x.right['licks'],
                  offset = x.right['licks_off'],
                  burstThreshold = 0.50)
        
        bins = 300
        
        x.randomevents = jmf.makerandomevents(120, max(x.tick)-120)
        x.bgTrials, x.pps = jmf.snipper(x.data, x.randomevents,
                                        t2sMap = x.t2sMap, fs = x.fs, bins=bins)
        
        if x.left['exist'] == True:
            x.left['snips_sipper'] = jmf.mastersnipper(x, x.left['sipper'])
            x.left['snips_licks'] = jmf.mastersnipper(x, x.left['lickdata']['rStart'])
            
            
            x.left['snips_licks_forced'] = jmf.mastersnipper(x, [licks for licks in x.left['lickdata']['rStart'] if licks < x.both['sipper'][0]])
            x.left['lats'] = jmf.latencyCalc(x.left['lickdata']['licks'], x.left['sipper'], cueoff=x.left['sipper_off'], lag=0)
         
        if x.right['exist'] == True:
            x.right['snips_sipper'] = jmf.mastersnipper(x, x.right['sipper'])
            x.right['snips_licks'] = jmf.mastersnipper(x, x.right['lickdata']['rStart'])
            x.right['snips_licks_forced'] = jmf.mastersnipper(x, [licks for licks in x.right['lickdata']['rStart'] if licks < x.both['sipper'][0]])
            x.right['lats'] = jmf.latencyCalc(x.right['lickdata']['licks'], x.right['sipper'], cueoff=x.right['sipper_off'], lag=0)
            
        makeBehavFigs(x)
        makePhotoFigs(x)
        
        x.side2subs()
          
        pdf_pages.close()
        plt.close('all')
        
try:
    pdf_pages.close()
    plt.close('all')
except:
    print('Everything already closed')
    
