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

import dill

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
            self.leftTrials = True
            self.trialsL = self.output.trialsL.onset
            self.trialsL_off = self.output.trialsL.offset
            self.licksL = np.array([i for i in self.output.licksL.onset if i<max(self.trialsL_off)])
            self.licksL_off = self.output.licksL.offset[:len(self.licksL)]
        else:
            self.leftTrials = False
            self.trialsL = []
            self.trialsL_off = []
            self.licksL = []
            self.licksL_off = []
            
        if hasattr(self.output.trialsR, 'onset'):
            self.rightTrials = True
            self.trialsR = self.output.trialsR.onset
            self.trialsR_off = self.output.trialsR.offset
            self.licksR = np.array([i for i in self.output.licksR.onset if i<max(self.trialsR_off)])
            self.licksR_off = self.output.licksR.offset[:len(self.licksR)]
        else:
            self.rightTrials = False
            self.trialsR = []
            self.trialsR_off = []
            self.licksR = []
            self.licksR_off = []
            
        if self.leftTrials == True and self.rightTrials == True:
            print(len(self.trialsL))
            first = [idx for idx, x in enumerate(self.trialsL) if x in self.trialsR][0]
            print(first)
            self.trialsboth = self.trialsL[first:]
            self.trialsboth_off = self.trialsL_off[first:]
            self.trialsL = self.trialsL[:first-1]
            self.trialsL_off = self.trialsL_off[:first-1]
            self.trialsR = self.trialsR[:first-1]
            self.trialsR_off = self.trialsR_off[:first-1]

                        
    def removephantomlicks(self):
        if self.leftTrials == True:
            phlicks = jmf.findphantomlicks(self.licksL, self.trialsL, delay=3)
            self.licksL = np.delete(self.licksL, phlicks)
            self.licksL_off = np.delete(self.licksL_off, phlicks)
    
        if self.rightTrials == True:
            phlicks = jmf.findphantomlicks(self.licksR, self.trialsR, delay=3)
            self.licksR = np.delete(self.licksR, phlicks)
            self.licksR_off = np.delete(self.licksR_off, phlicks)
                        
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
        if x.leftTrials == True:
            ax.hist(self.lickDataL['licks'], range(0, 3600, 60), color=self.Lcol, alpha=0.4)          
            yraster = [ax.get_ylim()[1]] * len(self.lickDataL['licks'])
            ax.scatter(self.lickDataL['licks'], yraster, s=50, facecolors='none', edgecolors=self.Lcol)

        if x.rightTrials == True:
            ax.hist(self.lickDataR['licks'], range(0, 3600, 60), color=self.Rcol, alpha=0.4)          
            yraster = [ax.get_ylim()[1]] * len(self.lickDataR['licks'])
            ax.scatter(self.lickDataR['licks'], yraster, s=50, facecolors='none', edgecolors=self.Rcol)           
        
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
            self.Lcol = casein_color
        if 'malt' in self.bottleL:
            self.Lcol = malt_color
        
        if 'cas' in self.bottleR:
            self.Rcol = casein_color
        if 'malt' in self.bottleR:
            self.Rcol = malt_color

def makeBehavFigs(x):
    # Initialize figure
    behavFig = plt.figure(figsize=(8.27, 11.69), dpi=100)
    gs1 = gridspec.GridSpec(5, 2)
    gs1.update(left=0.10, right= 0.9, wspace=0.5, hspace = 0.7)
    plt.suptitle('Rat ' + x.rat + ': Session ' + x.session)
    
    ax = plt.subplot(gs1[0, :])
    x.sessionlicksFig(ax)

    if x.leftTrials == True:
        behavFigsCol(gs1, 0, x.lickDataL, x.trialsL, x.Lcol)
        
    if x.rightTrials == True:
        behavFigsCol(gs1, 1, x.lickDataR, x.trialsR, x.Rcol)
        
    ax = plt.subplot(gs1[4, 0])
    jmfig.latencyFig(ax, x)

    pdf_pages.savefig(behavFig)

def behavFigsCol(gs1, col, lickdata, cues, sidecol):
    ax = plt.subplot(gs1[1, col])
    jmfig.licklengthFig(ax, lickdata, color=sidecol)
    
    ax = plt.subplot(gs1[2, col])
    jmfig.iliFig(ax, lickdata, color=sidecol)
    
    ax = plt.subplot(gs1[3, col])
    jmfig.cuerasterFig(ax, cues, lickdata['licks'])
    
def makePhotoFigs(x):
    # Initialize photometry figure
    photoFig = plt.figure(figsize=(8.27, 11.69), dpi=100)
    gs1 = gridspec.GridSpec(6, 2)
    gs1.update(left=0.125, right= 0.9, wspace=0.4, hspace = 0.8)
    plt.suptitle('Rat ' + x.rat + ': Session ' + x.session)

    ax = plt.subplot(gs1[0, :])
    x.sessionFig(ax)

    if x.leftTrials == True:
        photoFigsCol(gs1, 0, x.pps,
                     x.trialsLSnips, x.trialsLSnipsUV, x.trialsLnoise,
                     x.licksLSnips, x.licksLSnipsUV, x.licksLnoise)

    if x.rightTrials == True:
        photoFigsCol(gs1, 1, x.pps,
                     x.trialsRSnips, x.trialsRSnipsUV, x.trialsRnoise,
                     x.licksRSnips, x.licksRSnipsUV, x.licksRnoise)
        
    if x.leftTrials == True and x.rightTrials == True:
        diffcueL = jmf.findphotodiff(x.trialsLSnips, x.trialsLSnipsUV, x.trialsLnoise)
        diffcueR = jmf.findphotodiff(x.trialsRSnips, x.trialsRSnipsUV, x.trialsRnoise)

        ax = plt.subplot(gs1[5, 0])
        jmfig.trialsMultShadedFig(ax, [diffcueL, diffcueR], x.pps,
                                  linecolor=[x.Lcol, x.Rcol], eventText = 'Sipper')

        difflickL = jmf.findphotodiff(x.licksLSnips, x.licksLSnipsUV, x.licksLnoise)
        difflickR = jmf.findphotodiff(x.licksRSnips, x.licksRSnipsUV, x.licksRnoise)

        ax = plt.subplot(gs1[5, 1])
        jmfig.trialsMultShadedFig(ax, [difflickL, difflickR], x.pps,
                                  linecolor=[x.Lcol, x.Rcol], eventText = 'Lick')
        
#    plt.savefig(userhome + '/Dropbox/Python/photometry/output-thph1-lp/' + x.rat + '.eps', format='eps', dpi=1000)
    pdf_pages.savefig(photoFig)
    
def photoFigsCol(gs1, col, pps, cues, cuesUV, cuesnoise, licks, licksUV, licksnoise):
    ax = plt.subplot(gs1[1, col])
    jmfig.trialsFig(ax, cues, pps, noiseindex = cuesnoise,
                    eventText = 'Sipper',
                    ylabel = 'Delta F / F0')
    
    ax = plt.subplot(gs1[2, col])
    jmfig.trialsMultShadedFig(ax, [cuesUV, cues], pps, noiseindex = cuesnoise,
                              eventText = 'Sipper')
    
    ax = plt.subplot(gs1[3, col])
    jmfig.trialsFig(ax, licks, pps, noiseindex = licksnoise,
                    eventText = 'First Lick',
                    ylabel = 'Delta F / F0')
    
    ax = plt.subplot(gs1[4, col])
    jmfig.trialsMultShadedFig(ax, [licksUV, licks], pps, noiseindex = licksnoise,
                              eventText = 'First Lick')
    
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
        x.removephantomlicks()
        x.setbottlecolors()
        
        x.lickDataL = jmf.lickCalc(x.licksL,
                          offset = x.licksL_off,
                          burstThreshold = 0.50)
        
        x.lickDataR = jmf.lickCalc(x.licksR,
                  offset = x.licksR_off,
                  burstThreshold = 0.50)
        
        bins = 300
        
        x.randomevents = jmf.makerandomevents(120, max(x.output.tick.onset)-120)
        x.bgTrials, x.pps = jmf.snipper(x.data, x.randomevents,
                                        t2sMap = x.t2sMap, fs = x.fs, bins=bins)
        
        if x.leftTrials == True:
            x.trialsLSnips, x.trialsLSnipsUV, x.trialsLnoise = x.makephotoTrials(bins, x.trialsL)
            x.licksLSnips, x.licksLSnipsUV, x.licksLnoise = x.makephotoTrials(bins, x.lickDataL['rStart'])
            x.latsL = jmf.latencyCalc(x.lickDataL['licks'], x.trialsL, cueoff=x.trialsL_off, lag=0)
            
        
        if x.rightTrials == True:
            x.trialsRSnips, x.trialsRSnipsUV, x.trialsRnoise = x.makephotoTrials(bins, x.trialsR)
            x.licksRSnips, x.licksRSnipsUV, x.licksRnoise = x.makephotoTrials(bins, x.lickDataR['rStart'])
            x.latsR = jmf.latencyCalc(x.lickDataR['licks'], x.trialsR, cueoff=x.trialsR_off, lag=0)
            
        makeBehavFigs(x)
        makePhotoFigs(x)
        
    pdf_pages.close()
    plt.close('all')
    
# For pickling data for opening later

#pickle_out = open('rats.pickle', 'wb')
#dill.dump(rats, pickle_out)
#pickle_out.close()