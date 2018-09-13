# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:00:38 2018

@author: jaimeHP
"""
import JM_general_functions as jmf
import JM_custom_figs as jmfig
import ppp_sessionfigs as sessionfigs
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy.io as sio
import JM_general_functions as jmf
import dill

class Session(object):
    
    def __init__(self, sessionID, metafiledata, hrows, datafolder, outputfolder):
        self.sessionID = sessionID
        self.sessiontype = metafiledata[hrows['stype']]
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
    
    sessions = {}
    
    for row in rows:
        sessionID = row[hrows['rat']].replace('.','-') + '_' + row[hrows['session']]
        sessions[sessionID] = Session(sessionID, row, hrows, datafolder, outputfolder)
    
    return sessions   

def assemble_sessions(sessions, rats,
                      rats_to_include=[],
                      rats_to_exclude=[],
                      sessions_to_include=[],
                      outputfile=[],
                      savefile=False,
                      makefigs=False):

    if len(rats_to_include) > 0:
        print('Overriding values in rats_to_exclude because of entry in rats_to_include.')
        rats_to_exclude = list(rats)
        for rat in rats_to_include:
            rats_to_exclude.remove(rat)
    
    for session in sessions:
          
        x = sessions[session]
        
        if x.rat not in rats_to_exclude and x.session in sessions_to_include: 
    
            try:
                x.loadmatfile()
        
                print('\nAnalysing rat ' + x.rat + ' in session ' + x.session)
                
                # Load in data from .mat file (convert from Tank first using Matlab script)
                x.loadmatfile()
                # Work out time to samples
                x.setticks()
                x.time2samples()       
                # Find out which bottles have TTLs/Licks associated with them     
                x.check4events()
    
                x.setbottlecolors()
                try:
                    x.left['lickdata'] = jmf.lickCalc(x.left['licks'],
                                      offset = x.left['licks_off'],
                                      burstThreshold = 0.50)
                except IndexError:
                    x.left['lickdata'] = 'none'
                    
                try:
                    x.right['lickdata'] = jmf.lickCalc(x.right['licks'],
                              offset = x.right['licks_off'],
                              burstThreshold = 0.50)
                except IndexError:
                    x.right['lickdata'] = 'none'
                
                bins = 300
                
                x.randomevents = jmf.makerandomevents(120, max(x.tick)-120)
                x.bgTrials, x.pps = jmf.snipper(x.data, x.randomevents,
                                                t2sMap = x.t2sMap, fs = x.fs, bins=bins)
                
                for side in [x.left, x.right]:   
                    if side['exist'] == True:
                        side['snips_sipper'] = jmf.mastersnipper(x, side['sipper'])
                        side['snips_licks'] = jmf.mastersnipper(x, side['lickdata']['rStart'])
                        try:
                            side['snips_licks_forced'] = jmf.mastersnipper(x, [licks for licks in side['lickdata']['rStart'] if licks < x.both['sipper'][0]])
                        except KeyError:
                            pass
                        
                        side['lats'] = jmf.latencyCalc(side['lickdata']['licks'], side['sipper'], cueoff=side['sipper_off'], lag=0)
                
                x.side2subs()
                 
                if makefigs == True:
                    pdf_pages = PdfPages(x.outputfolder + session + '.pdf')
                    sessionfigs.makeBehavFigs(x, pdf_pages)
                    sessionfigs.makePhotoFigs(x, pdf_pages)
            except:
                print('Could not extract data from ' + x.sessionID)

                
            try:
                pdf_pages.close()
                plt.close('all')
            except:
                print('Nothing to close')

    
    if savefile == True:
        pickle_out = open(savefile)
        dill.dump([sessions], pickle_out)
        pickle_out.close()



