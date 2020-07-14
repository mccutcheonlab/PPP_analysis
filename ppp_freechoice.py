# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 13:29:11 2020

@author: admin
"""

import trompy as tp
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ppp_pub_figs_settings import *
from ppp_pub_figs_fx import *
from ppp_pub_figs_supp import *


def get_free_signals(s, subs):
    
    dict2process = getattr(s, subs)
    ntotal = len(dict2process['snips_licks']['noise'])
    nforced = len(dict2process['snips_licks_forced']['noise'])
    nfree = ntotal - nforced
                  
    if nfree > 0:
        keys = dict2process['snips_licks'].keys()
        result = {}
        for key in keys:
            if len(dict2process['snips_licks'][key]) == ntotal:
                result[key] = keys = dict2process['snips_licks'][key][-nfree:]
        if nfree != len(result['noise']): print('Check get_free_signals!!')
    else:
        result = {}
    
    return result

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
    
keys = sessions.keys()

key = list(keys)[0]

for key in keys:
    s = sessions[key]

    s.cas['snips_licks_free'] = get_free_signals(s, 'cas')
    s.malt['snips_licks_free'] = get_free_signals(s, 'malt')
    


    
    

        

            
            
        

