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

import scipy.signal as sig

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

            self.ttls = tdt.read_block(self.tdtfile, evtype=['epocs']).epocs
        except:
            print('Unable to load data properly.')
            
    def setticks(self):
        try:
            self.tick = self.ttls['Tick'].onset
        except AttributeError:
            print('Problem setting ticks')        
        
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

def findfreechoice(left, right):
    first = [idx for idx, x in enumerate(left) if x in right][0]
    return first
        
def dividelicks(licks, time):
    before = [x for x in licks if x < time]
    after = [x for x in licks if x > time]
    
    return before, after  

def correctforbaseline(blue, uv):
    pt = len(blue)
    X = np.fft.rfft(uv, pt)
    Y = np.fft.rfft(blue, pt)
    Ynet = Y-X

    datafilt = np.fft.irfft(Ynet)

    datafilt = sig.detrend(datafilt)

    b, a = sig.butter(9, 0.012, 'low', analog=True)
    datafilt = sig.filtfilt(b, a, datafilt)
    
    return datafilt

def metafile2sessions(xlfile, metafile, datafolder, outputfolder, sheetname='metafile'):
    metafilemaker(xlfile, metafile, sheetname=sheetname, fileformat='txt')
    rows, header = metafilereader(metafile + '.txt')
    
    hrows = {}
    for idx, field in enumerate(header):
        hrows[field] = idx
    
    sessions = {}
    
    for row in rows:
        sessionID = row[hrows['rat']].replace('.','-') + '_' + row[hrows['session']]
        sessions[sessionID] = Session(sessionID, row, hrows, datafolder, outputfolder)
    
    return sessions

def metafilemaker(xlfile, metafilename, sheetname='metafile', fileformat='csv'):
    with xlrd.open_workbook(xlfile) as wb:
        sh = wb.sheet_by_name(sheetname)  # or wb.sheet_by_name('name_of_the_sheet_here')
        
        if fileformat == 'csv':
            with open(metafilename+'.csv', 'w', newline="") as f:
                c = csv.writer(f)
                for r in range(sh.nrows):
                    c.writerow(sh.row_values(r))
        if fileformat == 'txt':
            with open(metafilename+'.txt', 'w', newline="") as f:
                c = csv.writer(f, delimiter="\t")
                for r in range(sh.nrows):
                    c.writerow(sh.row_values(r))
    
def metafilereader(filename):
    
    f = open(filename, 'r')
    f.seek(0)
    header = f.readlines()[0]
    f.seek(0)
    filerows = f.readlines()[1:]
    
    tablerows = []
    
    for i in filerows:
        tablerows.append(i.split('\t'))
        
    header = header.split('\t')
    # need to find a way to strip end of line \n from last column - work-around is to add extra dummy column at end of metafile
    return tablerows, header
