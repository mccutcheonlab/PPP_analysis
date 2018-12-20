# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 15:20:48 2018

@author: James Rig
"""
import matplotlib.pyplot as plt
import numpy as np

import numpy.polynomial.polynomial as poly

import JM_custom_figs as jmfig
import JM_general_functions as jmf


from scipy import signal

import pandas as pd
    


snips = x.cas['snips_licks_forced']


z = snips['blue_z']
noise = snips['noise']

df = pd.DataFrame()

df['no_noise'] = [trial for trial, n in zip(z, noise) if not n]


def average_without_noise(snips, key='blue_z'):
    no_noise_snips = [trial for trial, noise in zip(snips[key], snips['noise') if not n]
    try:
        result = np.mean(no_noise_snips, axis=0)
        return result
    except:
        print('Problem averaging snips')
        return

df['no_noise'] = [average_without_noise(pref_sessions[x].cas['snips_licks_forced']) for x in pref_sessions if pref_sessions[x].session == j]

# [np.nanmean(pref_sessions[x].cas['snips_licks_forced']['blue_z'], axis=0) for x in pref_sessions if pref_sessions[x].session == j]



