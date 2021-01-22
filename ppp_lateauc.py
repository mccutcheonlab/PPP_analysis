# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 10:02:51 2021

@author: admin
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import matplotlib.lines as mlines
# import matplotlib.transforms as transforms

# import pandas as pd

from ppp_pub_figs_settings import *
# from ppp_pub_figs_fx import *
# from ppp_pub_figs_supp import *


# import dabest as db

import trompy as tp

# import scipy.io as sio
# import os
# import string
import numpy as np

import dill
import tdt


from scipy.stats import linregress

# Looks for existing data and if not there loads pickled file
try:
    type(sessions)
    print('Using existing data')
except NameError:
    print('Loading in data from pickled file')
    try:
        pickle_in = open('C:\\Github\\PPP_analysis\\data\\ppp_pref.pickle', 'rb')
    except FileNotFoundError:
        print('Cannot access pickled file')
    sessions, rats = dill.load(pickle_in)

pref_sessions = {}
for session in sessions:
    x = sessions[session]
    try:
        len(x.data)
        pref_sessions[x.sessionID] = x
    except AttributeError:
        pass
    
