# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:53:39 2017

@author: James Rig
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import JM_general_functions as jmf
import JM_custom_figs as jmfig

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

plt.style.use('seaborn-muted')

import os
import timeit

tic = timeit.default_timer()

datafolder = 'R:/DA_and_Reward/es334/PPP1/Matlab Files/'

class Rat(object):
    
    nRats = 0
    nSessions = 0
    
    def __init__(self, rat, diet):      
        self.rat = rat
        self.sessions = {}
        self.dietgroup = dietgroup
        
        Rat.nRats += 1
                
    def loadsession(self, data, header):
        self.session = str(data[3]) #should reference column of data with session number
        self.sessions[self.session] = Session(data, header, self.rat, self.session, self.dietgroup)
       
        Rat.nSessions += 1
        
class Session(object):
    
    def __init__(self, data, header, rat, session, dietgroup):
        self.hrow = {}
        for idx, col in enumerate(header):
            self.hrow[col] = data[idx]
        self.matlabfile = datafolder + self.hrow['rat'] + self.hrow['session'] + '.mat'
        self.medfile = datafolder + self.hrow['medfile']
        self.bottleL = self.hrow['bottleL']
        self.bottleR = self.hrow['bottleR']
        self.rat = str(rat)
        self.session = session
        self.dietgroup = str(dietgroup)
        
        self.bottles = {}

        self.left = {}
        self.right = {}
        self.both = {}
        self.left['subs'] = self.hrow['bottleL']
        self.right['subs'] = self.hrow['bottleR']

    def loadmatfile(self):
        a = sio.loadmat(self.matlabfile, squeeze_me=True, struct_as_record=False) 
        self.output = a['output']
        self.fs = self.output.fs
        self.data = self.output.blue
        self.dataUV = self.output.uv
        
    def time2samples(self):
        tick = self.output.tick.onset
        maxsamples = len(tick)*int(self.fs)
        if (len(self.data) - maxsamples) > 2*int(self.fs):
            print('Something may be wrong with conversion from time to samples')
            print(str(len(self.data) - maxsamples) + ' samples left over. This is more than double fs.')
            self.t2sMap = np.linspace(min(tick), max(tick), maxsamples)
        else:
            self.t2sMap = np.linspace(min(tick), max(tick), maxsamples)
            
    def event2sample(self, EOI):
        idx = (np.abs(self.t2sMap - EOI)).argmin()   
        return idx
    
    def check4events(self):        
        if hasattr(self.output.trialsL, 'onset'):
            self.left['exist'] = True
            self.left['sipper'] = self.output.trialsL.onset
            self.left['sipper_off'] = self.output.trialsL.offset
            self.left['licks'] = np.array([i for i in self.output.licksL.onset if i<max(self.left['sipper_off'])])
            self.left['licks_off'] = self.output.licksL.offset[:len(self.left['licks'])]
        else:
            self.left['exist'] = False
            self.left['sipper'] = []
            self.left['sipper_off'] = []
            self.left['licks'] = []
            self.left['licks_off'] = []
           
        if hasattr(self.output.trialsR, 'onset'):
            self.right['exist'] = True
            self.right['sipper'] = self.output.trialsR.onset
            self.right['sipper_off'] = self.output.trialsR.offset
            self.right['licks'] = np.array([i for i in self.output.licksR.onset if i<max(self.right['sipper_off'])])
            self.right['licks_off'] = self.output.licksR.offset[:len(self.right['licks'])]
        else:
            self.right['exist'] = False
            self.right['sipper'] = []
            self.right['sipper_off'] = []
            self.right['licks'] = []
            self.right['licks_off'] = []
            
        if self.left['exist'] == True and self.right['exist'] == True:
            first = [idx for idx, x in enumerate(self.left['sipper']) if x in self.right['sipper']][0]
            self.both['sipper'] = self.left['sipper'][first:]
            self.both['sipper_off'] = self.left['sipper_off'][first:]
            self.left['sipper'] = self.left['sipper'][:first-1]
            self.left['sipper_off'] = self.left['sipper'][:first-1]
            self.right['sipper'] = self.right['sipper'][:first-1]
            self.right['sipper_off'] = self.right['sipper_off'][:first-1]
                        
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
        
        self.Lcol = 'xkcd:grey'
        self.Rcol = 'xkcd:greyish blue'
           
        if 'cas' in self.bottleL:
            self.left['color'] = casein_color
        if 'malt' in self.bottleL:
            self.left['color'] = malt_color
        
        if 'cas' in self.bottleR:
            self.right['color'] = casein_color
        if 'malt' in self.bottleR:
            self.right['color'] = malt_color


metafile = 'R:/DA_and_Reward/es334/PPP1/PPP1_metafile.txt'
metafileData, metafileHeader = jmf.metafilereader(metafile)

exptsuffix = ''
includecol = 21

rats = {}

for i in metafileData:
    if int(i[includecol]) == 1:
        rowrat = str(i[2])
        dietgroup = str(i[5])
        if rowrat not in rats:
            rats[rowrat] = Rat(rowrat, dietgroup)
        rats[rowrat].loadsession(i, metafileHeader)
              
#for i in rats:
#    pdf_pages = PdfPages('R:/DA_and_Reward/es334/PPP1/output/' + i + exptsuffix + '.pdf')
#    for j in rats[i].sessions:        
for i in rats:
    pdf_pages = PdfPages('R:/DA_and_Reward/es334/PPP1/output/' + i + exptsuffix + '.pdf')
    for j in ['s10']:
        print('\nAnalysing rat ' + i + ' in session ' + j)
        
        # Load in data from .mat file (convert from Tank first using Matlab script)
        x = rats[i].sessions[j]
        x.loadmatfile()
        # Work out time to samples
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
        
        x.randomevents = jmf.makerandomevents(120, max(x.output.tick.onset)-120)
        x.bgTrials, x.pps = jmf.snipper(x.data, x.randomevents,
                                        t2sMap = x.t2sMap, fs = x.fs, bins=bins)
        
        if x.left['exist'] == True:
            x.left['snips_sipper'] = jmf.mastersnipper(x, x.left['sipper'])
            x.left['snips_licks'] = jmf.mastersnipper(x, x.left['lickdata']['rStart'])
            x.left['lats'] = jmf.latencyCalc(x.left['lickdata']['licks'], x.left['sipper'], cueoff=x.left['sipper_off'], lag=0)
            
        
        if x.right['exist'] == True:
            x.right['snips_sipper'] = jmf.mastersnipper(x, x.right['sipper'])
            x.right['snips_licks'] = jmf.mastersnipper(x, x.right['lickdata']['rStart'])
            x.right['lats'] = jmf.latencyCalc(x.right['lickdata']['licks'], x.right['sipper'], cueoff=x.right['sipper_off'], lag=0)
            
        makeBehavFigs(x)
        makePhotoFigs(x)
            
        
    pdf_pages.close()
    plt.close('all')