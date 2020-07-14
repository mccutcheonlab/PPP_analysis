# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:01:10 2020

@author: admin
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

from ppp_pub_figs_settings import *
from ppp_pub_figs_fx import *
from ppp_pub_figs_supp import *

keys = ['pref1_cas_sip', 'pref1_malt_sip']

keys = ['pref1_cas_sip', 'pref1_malt_sip']
keys1 = ['pref1_cas_licks_free', 'pref1_malt_licks_free']
keys2 = ['pref1_cas_licks_free1st', 'pref1_malt_licks_free1st']

epoch = [100, 149]

diet = 'NR'
with plt.xkcd():
    f, ax = plt.subplots(ncols=2)
    averagetrace(ax[0], df_photo, diet, keys1, event='Licks')
    averagetrace(ax[1], df_photo, diet, keys2, event='Licks')
    
    ax[0].set_title("All free choices")
    ax[1].set_title("First free choice")
    # peakbargraph(ax[1], df_photo, diet, keys, epoch=epoch)
    
    for xval in epoch:
        ax[0].axvline(xval, linestyle='--', color='k', alpha=0.3)
    
    
    diet = 'PR'
    f, ax = plt.subplots(ncols=2)
    averagetrace(ax[0], df_photo, diet, keys1, event='Licks', colorgroup='exptl')
    averagetrace(ax[1], df_photo, diet, keys2, event='Licks', colorgroup='exptl')
    # peakbargraph(ax[1], df_photo, diet, keys, epoch=epoch, colorgroup='exptl')
    
    ax[0].set_title("All free choices")
    ax[1].set_title("First free choice")
    
    for xval in epoch:
        ax[0].axvline(xval, linestyle='--', color='k', alpha=0.3)
        


