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


def get_freelick_signals(s, subs):
    
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

def average_without_noise(snips, key='filt_z'):
    # Can change default key to switch been delatF (key='blue') and z-score (key='blue_z') 
    try:
        no_noise_snips = [trial for trial, noise in zip(snips[key], snips['noise']) if not noise]
        result = np.mean(no_noise_snips, axis=0)
        return result
    except:
        print('Problem averaging snips')
        return []
    
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
    
# keys = sessions.keys()

# key = list(keys)[0]

# for key in keys:
#     s = sessions[key]

#     s.cas['snips_licks_free'] = get_freelick_signals(s, 'cas')
#     s.malt['snips_licks_free'] = get_freelick_signals(s, 'malt')
    
#     s.both['snips_sipper'] = tp.mastersnipper(s.data, s.dataUV, s.data_filt, s.fs, s.both['sipper'],
#                              peak_between_time=[0, 5],
#                              latency_events=[s.cas['lickdata']['rStart'] + s.malt['lickdata']['rStart']],
#                              latency_direction='post')


all_sipper_NR = []
all_sipper_PR = []

all_cas_freelicks_NR = []
all_malt_freelicks_NR = []

all_cas_freelicks_PR = []
all_malt_freelicks_PR = []
    
s = sessions["PPP1-7_s10"]

def get_snips_withoutnoise(snips):
    
    try:
        no_noise_snips = [trial for trial, noise in zip(snips["filt_z"], snips['noise']) if not noise]
    except KeyError:
        print("Could not get no_noise snips, maybe no snips to analyze. Returning empty list.")
        no_noise_snips = []
    return no_noise_snips


for key in keys:
    s = sessions[key]
    if s.session == "s10":

        if s.diet == "NR":
            all_sipper_NR.append(average_without_noise(s.both["snips_sipper"]))
            
            snips = s.cas["snips_licks_free"]
            no_noise_snips = get_snips_withoutnoise(snips)
            all_cas_freelicks_NR.append(no_noise_snips)
            
            snips = s.malt["snips_licks_free"]
            no_noise_snips = get_snips_withoutnoise(snips)
            all_malt_freelicks_NR.append(no_noise_snips)
            
        elif s.diet == "PR":
            all_sipper_PR.append(average_without_noise(s.both["snips_sipper"]))
            
            snips = s.cas["snips_licks_free"]
            no_noise_snips = get_snips_withoutnoise(snips)
            all_cas_freelicks_PR.append(no_noise_snips)
            
            snips = s.malt["snips_licks_free"]
            no_noise_snips = get_snips_withoutnoise(snips)
            all_malt_freelicks_PR.append(no_noise_snips)



aucs_sipper_NR = [np.trapz(trace[100:150])/10 for trace in all_sipper_NR]
aucs_sipper_PR = [np.trapz(trace[100:150])/10 for trace in all_sipper_PR]

tp.barscatter([aucs_sipper_NR, aucs_sipper_PR])

f, ax = plt.subplots(gridspec_kw={"left": 0.15, "bottom": 0.2})

tp.shadedError(ax, all_sipper_NR)
tp.shadedError(ax, all_sipper_PR, linecolor="blue")

ax.set_ylabel("Z-Score")

ax.set_xticks([0, 100, 200, 300])
ax.set_xticklabels(['-10', '0', '10', '20'])
ax.set_xlabel('Time from sipper (s)')

f.savefig("C:\\Users\\jmc010\\Dropbox\\Publications in Progress\\PPP Paper\\04_JNS\\02_revision 1\\revision figs\\freechoice_sipper.jpg")


                
all_cas_freelicks_NR_flat = tp.flatten_list(all_cas_freelicks_NR)
all_malt_freelicks_NR_flat = tp.flatten_list(all_malt_freelicks_NR)
all_cas_freelicks_PR_flat = tp.flatten_list(all_cas_freelicks_PR)
all_malt_freelicks_PR_flat = tp.flatten_list(all_malt_freelicks_PR)

f, ax = plt.subplots(ncols=2, sharey=True,
                     gridspec_kw={"left": 0.15, "bottom": 0.2})


tp.shadedError(ax[0], all_cas_freelicks_NR_flat)
tp.shadedError(ax[0], all_malt_freelicks_NR_flat, linecolor="grey")

tp.shadedError(ax[1], all_cas_freelicks_PR_flat, linecolor="blue")
tp.shadedError(ax[1], all_malt_freelicks_PR_flat, linecolor="xkcd:light blue")

ax[0].set_ylabel("Z-Score")

for axis in [ax[0], ax[1]]:
    axis.set_xticks([0, 100, 200, 300])
    axis.set_xticklabels(['-10', '0', '10', '20'])
    axis.set_xlabel('Time from first lick (s)')

f.savefig("C:\\Users\\jmc010\\Dropbox\\Publications in Progress\\PPP Paper\\04_JNS\\02_revision 1\\revision figs\\freechoice_licks.jpg")


    
    

        

            
            
        

