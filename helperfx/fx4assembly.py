# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:49:34 2019

@author: jmc010
"""
import numpy as np
import timeit
import random
import matplotlib.pyplot as plt
import xlrd
import csv
import os
import tdt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import dill

import scipy.signal as sig
from scipy import stats

import sessionfigs as sessionfigs

import trompy as tp

class Session(object):
    
    def __init__(self, sessionID, metafiledata, hrows, datafolder, outputfolder):
        self.sessionID = sessionID
        self.sessiontype = metafiledata[hrows['stype']]
        self.medfile = metafiledata[hrows['medfile']]
        self.rat = metafiledata[hrows['rat']].replace('.', '-')
        self.session = metafiledata[hrows['session']]
        self.diet = metafiledata[hrows['dietgroup']]
        self.box = metafiledata[hrows['box']]
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
        
        self.left['metafilelicks'] = metafiledata[hrows['licksL']]
        self.right['metafilelicks'] = metafiledata[hrows['licksR']]
        
        self.tdtfile = datafolder + metafiledata[hrows['tdtfile']]
        self.SigBlue = metafiledata[hrows['sig-blue']]
        self.SigUV = metafiledata[hrows['sig-uv']]
        
        self.outputfolder = outputfolder
        
    def loaddata(self):
        try:
            tmp = tdt.read_block(self.tdtfile, evtype=['streams'], store=[self.SigBlue])
            self.data = getattr(tmp.streams, self.SigBlue)['data']
            self.fs = getattr(tmp.streams, self.SigBlue)['fs']

            tmp = tdt.read_block(self.tdtfile, evtype=['streams'], store=[self.SigUV])
            self.dataUV = getattr(tmp.streams, self.SigUV)['data']
            
            self.data_filt = tp.processdata(self.data, self.dataUV)

            self.ttls = tdt.read_block(self.tdtfile, evtype=['epocs']).epocs
            
            self.t2sMap = tp.time2samples(self.data, self.fs)
             
        except Exception as e:
            print('Unable to load data properly.', e)

    def check4events(self):
        try:
            lt = getattr(self.ttls, self.ttl_trialsL)
            self.left['exist'] = True
            self.left['sipper'] = lt.onset
            self.left['sipper_off'] = lt.offset
            ll = getattr(self.ttls, self.ttl_licksL)
            self.left['licks'] = np.array([i for i in ll.onset if i<max(self.left['sipper_off'])])
            self.left['licks_off'] = ll.offset[:len(self.left['licks'])]
        except AttributeError:
            self.left['exist'] = False
            self.left['sipper'] = []
            self.left['sipper_off'] = []
            self.left['licks'] = []
            self.left['licks_off'] = []
           
        try:
            rt = getattr(self.ttls, self.ttl_trialsR)
            self.right['exist'] = True
            self.right['sipper'] = rt.onset
            self.right['sipper_off'] = rt.offset
            rl = getattr(self.ttls, self.ttl_licksR)
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
                firstL, firstR = findfreechoice(self.left['sipper'], self.right['sipper'])
                self.both['sipper'] = self.left['sipper'][firstL:]
                self.both['sipper_off'] = self.left['sipper_off'][firstL:]
                self.left['sipper'] = self.left['sipper'][:firstL]
                self.left['sipper_off'] = self.left['sipper_off'][:firstL]
                self.right['sipper'] = self.right['sipper'][:firstR]
                self.right['sipper_off'] = self.right['sipper_off'][:firstR]
                self.left['licks-forced'], self.left['licks-free'] = dividelicks(self.left['licks'], self.both['sipper'][0])
                self.right['licks-forced'], self.right['licks-free'] = dividelicks(self.right['licks'], self.both['sipper'][0])
                self.left['nlicks-forced'] = len(self.left['licks-forced'])
                self.right['nlicks-forced'] = len(self.right['licks-forced'])
                self.left['nlicks-free'] = len(self.left['licks-free'])
                self.right['nlicks-free'] = len(self.right['licks-free'])

            except IndexError:
                print('Problem separating out free choice trials')
        else:
            self.left['licks-forced'] = self.left['licks']
            self.right['licks-forced'] = self.right['licks']
            
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

    def getlicksfrommetafile(self):

        if 'cas' in self.bottleL:
            self.cas = self.left['metafilelicks']
        if 'cas' in self.bottleR:
            self.cas = self.right['metafilelicks']
        if 'malt' in self.bottleL:
            self.malt = self.left['metafilelicks']
        if 'malt' in self.bottleR:
            self.malt = self.right['metafilelicks']

def findfreechoice(left, right):
    firstL = [idx for idx, x in enumerate(left) if x in right][0]
    firstR = [idx for idx, x in enumerate(right) if x in left][0]
    return firstL, firstR
        
def dividelicks(licks, time):
    before = [x for x in licks if x < time]
    after = [x for x in licks if x > time]
    
    return before, after  

def latencyCalc(licks, cueon, cueoff=10, nolat=np.nan, lag=3):
    if type(cueoff) == int:
        cueoff = [i+cueoff for i in cueon]
    lats=[]
    for on,off in zip(cueon, cueoff): 
        try:
            currentlat = [i-(on+lag) for i in licks if (i>on) and (i<off)][0]
        except IndexError:
            currentlat = nolat
        lats.append(currentlat)

    return(lats)

def metafile2sessions(xlfile, metafile, datafolder, outputfolder, sheetname='metafile'):
    tp.metafilemaker(xlfile, metafile, sheetname=sheetname, fileformat='txt')
    rows, header = tp.metafilereader(metafile + '.txt')
    
    hrows = {}
    for idx, field in enumerate(header):
        hrows[field] = idx
    
    sessions = {}
    
    for row in rows:
        sessionID = row[hrows['rat']].replace('.','-') + '_' + row[hrows['session']]
        sessions[sessionID] = Session(sessionID, row, hrows, datafolder, outputfolder)
    
    return sessions

def comparepeaks(cas, malt):
    result = stats.ttest_ind([peak for peak, noise in zip(cas['peak'], cas['noise']) if not noise],
                             [peak for peak, noise in zip(malt['peak'], malt['noise']) if not noise])
    return result

def variablesnipper(data_filt, fs, events, all_events_baseline,
                                      trialLength=30,
                                      snipfs=10,
                                      preTrial=10,
                                      threshold=8,
                                      peak_between_time=[0, 1],
                                      latency_events=[],
                                      latency_direction='pre',
                                      max_latency=30,
                                      verbose=True,
                                      removenoisefromaverage=True,
                                      **kwargs):
    
   # Parse arguments relating to trial length, bins etc

    if preTrial > trialLength:
        baseline = trialLength / 2
        print("preTrial is too long relative to total trialLength. Changing to half of trialLength.")
    else:
        baseline = preTrial
    
    if 'bins' in kwargs.keys():
        bins = kwargs['bins']
        print(f"bins given as kwarg. Fs for snip will be {trialLength/bins} Hz.")
    else:
        if snipfs > fs-1:
            print('Snip fs is too high, reducing to data fs')
            bins = 0
        else:
            bins = int(trialLength * snipfs)
    
    print('Number of bins is:', bins)
    
    if 'baselinebins' in kwargs.keys():
        baselinebins = kwargs['baselinebins']
        if baselinebins > bins:
            print('Too many baseline bins for length of trial. Changing to length of baseline.')
            baselinebins = int(baseline*snipfs)
        baselinebins = baselinebins
    else:
        baselinebins = int(baseline*snipfs)
    
    if len(events) < 1:
        print('Cannot find any events. All outputs will be empty.')
        return {}
    else:
        if verbose: print('{} events to analyze.'.format(len(events)))

    # find closest baseline event for each timelocked event
    events_baseline = []
    for event in events:
        events_baseline.append([e for e in all_events_baseline if e < event][-1])
        
    # calculate mean and SD of baseline by taking 10s window butting in 100 bins and working out
    baseline_snips,_ = tp.snipper(data_filt, events_baseline,
                           fs=fs,
                           bins=100,
                           preTrial=10,
                           trialLength=10,
                           adjustBaseline=False)
    
    baseline_mean = np.mean(baseline_snips, axis=1)
    baseline_sd = np.std(baseline_snips, axis=1)
    
    filt_snips,_ = tp.snipper(data_filt, events,
                                   fs=fs,
                                   bins=bins,
                                   preTrial=baseline,
                                   trialLength=trialLength,
                                   adjustBaseline=False)
    
    filt_snips_z = []
    for snip, mean, sd in zip(filt_snips, baseline_mean, baseline_sd):
        filt_snips_z.append([(x-mean)/sd for x in snip])
      
    return filt_snips_z

def assemble_sessions(sessions,
                      rats_to_include=[],
                      rats_to_exclude=[],
                      sessions_to_include=[],
                      outputfile=[],
                      savefile=False,
                      makefigs=False):
    
    # This section of code works out unique rats and makes a list of rats to
    # exclude from analysis
    rats = []
    for session in sessions:
        s = sessions[session]
        if s.rat not in rats:
            rats.append(s.rat)
    
    if len(rats_to_include) > 0:
        print('Overriding values in rats_to_exclude because of entry in rats_to_include.')
        rats_to_exclude = list(rats)
        for rat in rats_to_include:
            rats_to_exclude.remove(rat)
        
    # This section of code selects sessions for inclusion 
    sessions_to_remove = []
    
    for session in sessions:         
        s = sessions[session]
        
        if s.rat not in rats_to_exclude and s.session in sessions_to_include:
            try:
                process_rat(s)           
            except Exception as e:
                print('Could not extract data from ' + s.sessionID, e) 
            
            if makefigs == True:
#                try:                   
                pdf_pages = PdfPages(s.outputfolder + session + '.pdf')
                sessionfigs.makeBehavFigs(s, pdf_pages)
                sessionfigs.makePhotoFigs(s, pdf_pages)
#                except:
#                    print('Could not make figures from ' + s.sessionID)      
                try:
                    pdf_pages.close()
                    plt.close('all')
                except:
                    print('Nothing to close')
                    
        else:
            sessions_to_remove.append(session)
    
    for session in sessions_to_remove:
        sessions.pop(session)
        
    for rat in rats_to_exclude:
        idx = rats.index(rat)
        del rats[idx]
        

    
    if savefile == True:
        pickle_out = open(outputfile, 'wb')
        dill.dump([sessions, rats], pickle_out)
        pickle_out.close()
        
    return sessions

def savejustmetafiledata(sessions,
                         rats_to_exclude=[],
                         sessions_to_include=[],
                         outputfile=[]):
    
    rats = []
    sessions_to_remove = []
     
    for session in sessions:
        s = sessions[session]
        if s.session in sessions_to_include:
            if s.rat in rats_to_exclude:
                sessions_to_remove.append(session)
            else:
                if s.rat not in rats:
                    rats.append(s.rat)
                print('Including rat {}, session {}'.format(s.rat, s.session))
                s.getlicksfrommetafile()
        else:
            sessions_to_remove.append(session)
            
    for session in sessions_to_remove:
        sessions.pop(session)
        
    pickle_out = open(outputfile, 'wb')
    dill.dump([sessions, rats], pickle_out)
    pickle_out.close()
                
    return sessions

"""
process_rat is a function based on the notebook of the same name that was used
to develop and test the functions. It relies on functions in this script
(fx4assembly) as well as fx4behavior and fx4makingsnips.

Ideally, any changes should be tested in the notebook first before being transferred here.
"""

def process_rat(session):
    
    s = session
    print('\nAnalysing rat ' + s.rat + ' in session ' + s.session)
    s.loaddata()
    s.check4events()
    s.setbottlecolors()
    
    try:
        s.left['lickdata'] = tp.lickCalc(s.left['licks'],
                          offset = s.left['licks_off'],
                          burstThreshold = 0.50)
    except IndexError:
        s.left['lickdata'] = 'none'
        print('No left licks')
        
    try:
        s.right['lickdata'] = tp.lickCalc(s.right['licks'],
                  offset = s.right['licks_off'],
                  burstThreshold = 0.50)
    except IndexError:
        s.right['lickdata'] = 'none'
        print('No right licks')
        
    bins = 300
    
    for side in [s.left, s.right]:   
        if side['exist'] == True:
            side['snips_sipper'] = tp.mastersnipper(s.data, s.dataUV, s.data_filt, s.fs, side['sipper'],
                                                 peak_between_time=[0, 5],
                                                 latency_events=side['lickdata']['rStart'],
                                                 latency_direction='post')
            side['snips_licks'] = tp.mastersnipper(s.data, s.dataUV, s.data_filt, s.fs, side['lickdata']['rStart'],
                                                peak_between_time=[0, 5],
                                                latency_events=side['sipper'],
                                                latency_direction='pre')
                                               
            try:
                forced_licks = [licks for licks in side['lickdata']['rStart'] if licks in side['licks-forced']]
                side['snips_licks_forced'] = tp.mastersnipper(s.data, s.dataUV, s.data_filt, s.fs, forced_licks,
                                                           peak_between_time=[0, 5],
                                                           latency_events=side['sipper'],
                                                           latency_direction='pre')
                # side['snips_licks_forced']["filt_z_varBL"] = variablesnipper(s.data_filt, s.fs, forced_licks, side['sipper'])
                                                           
            except KeyError:
                pass
            try:
                side['lats'] = latencyCalc(side['lickdata']['licks'], side['sipper'], cueoff=side['sipper_off'], lag=0)
            except TypeError:
                print('Cannot work out latencies as there are lick and/or sipper values missing.')
                side['lats'] = []
    s.side2subs()
    try:
        s.peakdiff = comparepeaks(s.cas['snips_licks_forced'],
                                  s.malt['snips_licks_forced'])
    except:
        print("Could not compare peaks")

"""
This function and the line of code that follow it can be used to reduce the size of pickled files by removing
stream data and just leaving snips and other relevant info.

"""
def removestreamdata(picklefilein, picklefileout):
    try:
        pickle_in = open(picklefilein, 'rb')
    except FileNotFoundError:
        print('Cannot access pickled file')
        
    sessions, rats = dill.load(pickle_in)
    
    for session in sessions:
        s = sessions[session]
        delattr(s, 'data')
        delattr(s, 'dataUV')
        delattr(s, 'data_filt')
        
    pickle_out = open(picklefileout, 'wb')
    dill.dump([sessions, rats], pickle_out)
    pickle_out.close()
    
# removestreamdata("C:\\Github\\PPP_analysis\\data\\ppp_pref.pickle", "C:\\Github\\PPP_analysis\\data\\ppp_pref_nostreams.pickle")
        
    
        