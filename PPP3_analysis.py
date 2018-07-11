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
import numpy as np

## Colour scheme
col={}
col['np_cas'] = 'xkcd:silver'
col['np_malt'] = 'white'
col['lp_cas'] = 'xkcd:kelly green'
col['lp_malt'] = 'xkcd:light green'


class Session(object):
    
    def __init__(self, metafiledata):
        self.medfile = metafiledata[hrows['medfile']]
        self.rat = metafiledata[hrows['rat']]
        self.session = metafiledata[hrows['session']]
        self.diet = metafiledata[hrows['dietgroup']]
        self.bottleL = metafiledata[hrows['bottleL']]
        self.bottleR = metafiledata[hrows['bottleR']]
        

def sub2var(session, substance):
    
    if substance in session.bottleL:
        varsOut = ['b', 'c']        
    if substance in session.bottleR:
        varsOut = ['e', 'f']
    return varsOut


# Extracts data from metafile
metafile = 'R:\\DA_and_Reward\\gc214\\IPP1\\IPP1_metafile.txt'
medfolder = 'R:\\DA_and_Reward\\gc214\\IPP1\\MED-PC datafile\\'

rows, header = jmf.metafilereader(metafile)

hrows = {}
for idx, field in enumerate(header):
    hrows[field] = idx

# Sets up individual objects for each sessio and gets data from medfiles
sessions = {}
        
for row in rows:
    sessionID = row[hrows['rat']] + '-' + row[hrows['session']]
    sessions[sessionID] = Session(row)