# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 16:53:38 2018

@author: James Rig
"""

import JM_general_functions as jmf
import JM_custom_figs as jmfig
from ppp_assemble import *
from ppp_sessionfigs import *

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

# Extracts data from metafiles and sets up ppp_sessions dictionary

picklefolder = 'R:\\DA_and_Reward\\gc214\\PPP_combined\\output\\'

ppp1_sessions = metafile2sessions('R:\\DA_and_Reward\\es334\PPP1\\PPP1.xlsx',
                  'R:\\DA_and_Reward\\es334\PPP1\\PPP1_metafile',
                  'R:\\DA_and_Reward\\es334\\PPP1\\matfiles\\',
                  'R:\\DA_and_Reward\\es334\\PPP1\\output\\',
                  sheetname='metafile')
    

ppp3_sessions = metafile2sessions('R:\\DA_and_Reward\\gc214\\PPP3\\PPP3.xlsx',
                  'R:\\DA_and_Reward\\gc214\\PPP3\\PPP3_metafile',
                  'R:\\DA_and_Reward\\gc214\\PPP3\\matfiles\\',
                  'R:\\DA_and_Reward\\gc214\\PPP3\\output\\',
                  sheetname='PPP3_metafile')

ppp_sessions = {**ppp1_sessions, **ppp3_sessions}

# Code to indictae which files to assemble and whether to save and/or make figures
assemble_sacc = False
assemble_cond1 = False
assemble_cond2 = False
assemble_pref = True
assemble_single = False

savefile=True
makefigs=False

if assemble_sacc:
    sessions = assemble_sessions(ppp_sessions,
                  rats_to_include = [],
                  rats_to_exclude = ['PPP1-8', 'PPP3-1', 'PPP3-6', 'PPP3-7'],
                  sessions_to_include = ['s3', 's4', 's5'],
                  outputfile=picklefolder + 'ppp_sacc.pickle',
                  savefile=savefile,
                  makefigs=makefigs)

if assemble_cond1:
    sessions = assemble_sessions(ppp_sessions,
                  rats_to_include = [],
                  rats_to_exclude = ['PPP1-8', 'PPP3-1', 'PPP3-6', 'PPP3-7'],
                  sessions_to_include = ['s6', 's7', 's8', 's9'],
                  outputfile=picklefolder + 'ppp_cond1.pickle',
                  savefile=savefile,
                  makefigs=makefigs)

if assemble_cond2:
    assemble_sessions(ppp_sessions,
                  rats_to_include = [],
                  rats_to_exclude = ['PPP1-8', 'PPP3-1', 'PPP3-6', 'PPP3-7'],
                  sessions_to_include = ['s12', 's13', 's14', 's15'],
                  outputfile=picklefolder + 'ppp_cond2.pickle',
                  savefile=savefile,
                  makefigs=makefigs)

if assemble_pref:
    assemble_sessions(ppp_sessions,
                  rats_to_include = [],
                  rats_to_exclude = ['PPP1-8', 'PPP3-1', 'PPP3-6', 'PPP3-7'],
                  sessions_to_include = ['s10', 's11', 's16'],
                  outputfile='R:\\DA_and_Reward\\gc214\\PPP_combined\\output\\ppp_pref.pickle',
                  savefile=savefile,
                  makefigs=makefigs)

# Code to run for single rat
if assemble_single:
    sessions_to_add = assemble_sessions(ppp_sessions,
                  rats_to_include = ['PPP3-8'],
                  rats_to_exclude = ['PPP1-8', 'PPP3-1', 'PPP3-6', 'PPP3-7', 'PPP3-2', 'PPP3-8'],
                  sessions_to_include = ['s6'],
                  outputfile=picklefolder + 'ppp_test.pickle',
                  savefile=savefile,
                  makefigs=makefigs)

