# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 08:47:56 2017

@author: Jaime
"""

# Assembles data from PPP1 and PPP3 into pandas dataframes for plotting. Saves
# dataframes, df_behav and df_photo, as pickle object (ppp_dfs_pref)

# Choice data
import scipy.io as sio
import JM_general_functions as jmf
import JM_custom_figs as jmfig

import os
import string
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import dill

def choicetest(x):
    choices = []
    for trial, trial_off in zip(x.both['sipper'], x.both['sipper_off']):
        leftlick = [L for L in x.left['licks'] if (L > trial) and (L < trial_off)]
        rightlick = [L for L in x.right['licks'] if (L > trial) and (L < trial_off)]
        if len(leftlick) > 0:
            if len(rightlick) > 0:
                if leftlick < rightlick:
                    choices.append(x.bottleL[:3])
                else:
                    choices.append(x.bottleR[:3])
            else:
                choices.append(x.bottleL[:3])
        elif len(rightlick) > 0:
            choices.append(x.bottleR[:3])
        else:
            choices.append('missed')
    
    return choices

def prefcalc(x):
    cas = sum([1 for trial in x.choices if trial == 'cas'])
    malt = sum([1 for trial in x.choices if trial == 'mal'])
    pref = cas/(cas+malt)
    
    return pref

def excluderats(rats, ratstoexclude):  
    ratsX = [x for x in rats if x not in ratstoexclude]        
    return ratsX

def makemeansnips(snips, noiseindex):
    if len(noiseindex) > 0:
        trials = np.array([i for (i,v) in zip(snips, noiseindex) if not v])
    meansnip = np.mean(trials, axis=0)
        
    return meansnip

def removenoise(snipdata):
    # returns blue snips with noisey ones removed
    new_snips = [snip for (snip, noise) in zip(snipdata['blue'], snipdata['noise']) if not noise]
    return new_snips

def getsipper(snipdata):
    
    sipper = [lat for (lat, noise) in zip(snipdata['latency'], snipdata['noise']) if not noise]
    return sipper

def getfirstlick(side, event):
    sipper = side['sipper']
    licks = side['licks']
    firstlicks=[]
    for sip in sipper:
        firstlicks.append([l-sip for l in licks if l-sip>0][0])
        
    lats = [lat for (lat, noise) in zip(firstlicks, side[event]['noise']) if not noise]
    lats = [lat if (lat<20) else np.nan for lat in lats]
    return lats

def average_without_noise(snips, key='blue_z'):
    no_noise_snips = [trial for trial, noise in zip(snips[key], snips['noise']) if not noise]
    try:
        result = np.mean(no_noise_snips, axis=0)
        return result
    except:
        print('Problem averaging snips')
        return

# Looks for existing data and if not there loads pickled file
try:
    type(sessions)
    print('Using existing data')
except NameError:
    print('Loading in data from pickled file')
    try:
        pickle_in = open('R:\\DA_and_Reward\\gc214\\PPP_combined\\output\\ppp_pref.pickle', 'rb')
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

rats = {}
included_sessions = []
for session in pref_sessions:
    x = pref_sessions[session]
    if x.rat not in rats.keys():
        rats[x.rat] = x.diet
    if x.session not in included_sessions:
        included_sessions.append(x.session)
        
for session in pref_sessions:
    x = pref_sessions[session]          
    x.choices = choicetest(x)
    x.pref = prefcalc(x)

df_behav = pd.DataFrame([x for x in rats], columns=['rat'])
df_behav['diet'] = [rats.get(x) for x in rats]
df_behav.set_index(['rat', 'diet'], inplace=True)

for j, ch, pr, cas, malt in zip(included_sessions,
                                ['choices1', 'choices2', 'choices3'],
                                ['pref1', 'pref2', 'pref3'],
                                ['pref1_ncas', 'pref2_ncas', 'pref3_ncas'],
                                ['pref1_nmalt', 'pref2_nmalt', 'pref3_nmalt']):
    df_behav[ch] = [pref_sessions[x].choices for x in pref_sessions if pref_sessions[x].session == j]
    df_behav[pr] = [pref_sessions[x].pref for x in pref_sessions if pref_sessions[x].session == j]
    df_behav[cas] = [c.count('cas') for c in df_behav[ch]]
    df_behav[malt] = [m.count('mal') for m in df_behav[ch]]

for j, forc_cas, forc_malt, free_cas, free_malt in zip(included_sessions,
                        ['pref1_cas_forced', 'pref2_cas_forced', 'pref3_cas_forced'],
                        ['pref1_malt_forced', 'pref2_malt_forced', 'pref3_malt_forced'],
                        ['pref1_cas_free', 'pref2_cas_free', 'pref3_cas_free'],
                        ['pref1_malt_free', 'pref2_malt_free', 'pref3_malt_free']):
    df_behav[forc_cas] = [pref_sessions[x].cas['nlicks-forced'] for x in pref_sessions if pref_sessions[x].session == j]
    df_behav[forc_malt] = [pref_sessions[x].malt['nlicks-forced'] for x in pref_sessions if pref_sessions[x].session == j]
    df_behav[free_cas] = [pref_sessions[x].cas['nlicks-free'] for x in pref_sessions if pref_sessions[x].session == j]
    df_behav[free_malt] = [pref_sessions[x].malt['nlicks-free'] for x in pref_sessions if pref_sessions[x].session == j]
    
# Assembles dataframe with photometry data
df_photo = pd.DataFrame([x for x in rats], columns=['rat'])
df_photo['diet'] = [rats.get(x) for x in rats]
df_photo.set_index(['rat', 'diet'], inplace=True)

for j, c_sip_z, m_sip_z, c_licks_z, m_licks_z in zip(included_sessions,
                             ['pref1_cas_sip', 'pref2_cas_sip', 'pref3_cas_sip'],
                             ['pref1_malt_sip', 'pref2_malt_sip', 'pref3_malt_sip'],
                             ['pref1_cas_licks', 'pref2_cas_licks', 'pref3_cas_licks'],
                             ['pref1_malt_licks', 'pref2_malt_licks', 'pref3_malt_licks']):

    df_photo[c_sip_z] = [average_without_noise(pref_sessions[x].cas['snips_sipper']) for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[m_sip_z] = [average_without_noise(pref_sessions[x].malt['snips_sipper']) for x in pref_sessions if pref_sessions[x].session == j] 
    df_photo[c_licks_z] = [average_without_noise(pref_sessions[x].cas['snips_licks']) for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[m_licks_z] = [average_without_noise(pref_sessions[x].malt['snips_licks']) for x in pref_sessions if pref_sessions[x].session == j]

# adds licks and lanetcies - all values
#for j, c_licks_forc, m_licks_forc, c_lats_forc, m_lats_forc, c_lats_forc_fromsip, m_lats_forc_fromsip in zip(included_sessions,
#                           ['pref1_cas_licks_forced_all', 'pref2_cas_licks_forced_all', 'pref3_cas_licks_forced_all'],
#                           ['pref1_malt_licks_forced_all', 'pref2_malt_licks_forced_all', 'pref3_malt_licks_forced_all'],
#                           ['pref1_cas_lats_all', 'pref2_cas_lats_all', 'pref3_cas_lats_all'],
#                           ['pref1_malt_lats_all', 'pref2_malt_lats_all', 'pref3_malt_lats_all'],
#                           ['pref1_cas_lats_all_fromsip', 'pref2_cas_lats_all_fromsip', 'pref3_cas_lats_all_fromsip'],
#                           ['pref1_malt_lats_all_fromsip', 'pref2_malt_lats_all_fromsip', 'pref3_malt_lats_all_fromsip']):
#    df_photo[c_licks_forc] = [pref_sessions[x].cas['snips_licks_forced']['blue_z'] for x in pref_sessions if pref_sessions[x].session == j]
#    df_photo[m_licks_forc] = [pref_sessions[x].malt['snips_licks_forced']['blue_z'] for x in pref_sessions if pref_sessions[x].session == j]
#    df_photo[c_lats_forc] = [pref_sessions[x].cas['snips_licks_forced']['latency'] for x in pref_sessions if pref_sessions[x].session == j]
#    df_photo[m_lats_forc] = [pref_sessions[x].malt['snips_licks_forced']['latency'] for x in pref_sessions if pref_sessions[x].session == j]
#    df_photo[c_lats_forc_fromsip] = [pref_sessions[x].cas['lats'] for x in pref_sessions if pref_sessions[x].session == j]
#    df_photo[m_lats_forc_fromsip] = [pref_sessions[x].malt['lats'] for x in pref_sessions if pref_sessions[x].session == j]

# adds means of licks and latencies
for j, c_licks_forc, m_licks_forc, c_lats_forc, m_lats_forc, c_lats_forc_fromsip, m_lats_forc_fromsip, in zip(included_sessions,
                           ['pref1_cas_licks_forced', 'pref2_cas_licks_forced', 'pref3_cas_licks_forced'],
                           ['pref1_malt_licks_forced', 'pref2_malt_licks_forced', 'pref3_malt_licks_forced'],
                           ['pref1_cas_lats', 'pref2_cas_lats', 'pref3_cas_lats'],
                           ['pref1_malt_lats', 'pref2_malt_lats', 'pref3_malt_lats'],
                           ['pref1_cas_lats_fromsip', 'pref2_cas_lats_fromsip', 'pref3_cas_lats_fromsip'],
                           ['pref1_malt_lats_fromsip', 'pref2_malt_lats_fromsip', 'pref3_malt_lats_fromsip']):
    df_photo[c_licks_forc] = [average_without_noise(pref_sessions[x].cas['snips_licks_forced']) for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[m_licks_forc] = [average_without_noise(pref_sessions[x].malt['snips_licks_forced']) for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[c_lats_forc] = [np.nanmean(pref_sessions[x].cas['snips_licks_forced']['latency'], axis=0) for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[m_lats_forc] = [np.nanmean(pref_sessions[x].malt['snips_licks_forced']['latency'], axis=0) for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[c_lats_forc_fromsip] = [np.nanmean(pref_sessions[x].cas['lats'], axis=0) for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[m_lats_forc_fromsip] = [np.nanmean(pref_sessions[x].malt['lats'], axis=0) for x in pref_sessions if pref_sessions[x].session == j]

# Assembles dataframe for reptraces

groups = ['NR_cas', 'NR_malt', 'PR_cas', 'PR_malt']
rats = ['PPP1-7', 'PPP1-7', 'PPP1-4', 'PPP1-4']
pref_list = ['pref1', 'pref2', 'pref3']

traces_list = [[15, 18, 5, 3],
          [6, 3, 19, 14],
          [13, 13, 13, 9]]

event = 'snips_licks_forced'

df_reptraces = pd.DataFrame(groups, columns=['group'])
df_reptraces.set_index(['group'], inplace=True)

for s, pref, traces in zip(['s10', 's11', 's16'],
                           pref_list,
                           traces_list):

    df_reptraces[pref + '_photo_blue'] = ""
    df_reptraces[pref + '_photo_uv'] = ""
    df_reptraces[pref + '_licks'] = ""
    df_reptraces[pref + '_sipper'] = ""
    
    for group, rat, trace in zip(groups, rats, traces):
        
        x = pref_sessions[rat + '_' + s]
        
        if 'cas' in group:
            trial = x.cas[event]    
            run = x.cas['lickdata']['rStart'][trace]
            all_licks = x.cas['licks']
            all_sips = x.cas['sipper']
        elif 'malt' in group:
            trial = x.malt[event]    
            run = x.malt['lickdata']['rStart'][trace]
            all_licks = x.malt['licks']
            all_sips = x.malt['sipper']
        
        df_reptraces.at[group, pref + '_licks'] = [l-run for l in all_licks if (l>run-10) and (l<run+20)]
        df_reptraces.at[group, pref + '_sipper'] = [sip-run for sip in all_sips if (sip-run<0.01) and (sip-run>-10)]
        df_reptraces.at[group, pref + '_photo_blue'] = trial['blue'][trace]
        df_reptraces.at[group, pref + '_photo_uv'] = trial['uv'][trace]

rats = np.unique(rats)
df_heatmap = pd.DataFrame(rats, columns=['rat'])
df_heatmap.set_index(['rat'], inplace=True)

for s, pref in zip(['s10', 's11', 's16'],
                           pref_list):

    df_heatmap[pref + '_cas'] = ""
    df_heatmap[pref + '_malt'] = ""
    df_heatmap[pref + '_cas_event'] = ""
    df_heatmap[pref + '_malt_event'] = ""
    
    for rat in rats:
        x = pref_sessions[rat + '_' + s]
        
        df_heatmap.at[rat, pref + '_cas'] = removenoise(x.cas[event])
        df_heatmap.at[rat, pref + '_cas_event'] = getsipper(x.cas[event])        
        df_heatmap.at[rat, pref + '_malt'] = removenoise(x.malt[event])
        df_heatmap.at[rat, pref + '_malt_event'] = getsipper(x.malt[event])

# Assembles dataframe for reptraces     for SIPPER TRIALS    

groups = ['NR_cas', 'NR_malt', 'PR_cas', 'PR_malt']
rats = ['PPP1-7', 'PPP1-7', 'PPP1-4', 'PPP1-4']
pref_list = ['pref1', 'pref2', 'pref3']

traces_list = [[16, 14, 13, 5],
          [6, 3, 19, 14],
          [13, 13, 13, 9]]

event = 'snips_sipper'

df_reptraces_sip = pd.DataFrame(groups, columns=['group'])
df_reptraces_sip.set_index(['group'], inplace=True)

for s, pref, traces in zip(['s10', 's11', 's16'],
                           pref_list,
                           traces_list):

    df_reptraces_sip[pref + '_photo_blue'] = ""
    df_reptraces_sip[pref + '_photo_uv'] = ""
    df_reptraces_sip[pref + '_licks'] = ""
    df_reptraces_sip[pref + '_sipper'] = ""
    
    for group, rat, trace in zip(groups, rats, traces):
        
        x = pref_sessions[rat + '_' + s]
        
        if 'cas' in group:
            trial = x.cas[event]
            sip = x.cas['sipper'][trace]
            all_licks = x.cas['licks']
            all_sips = x.cas['sipper']
        elif 'malt' in group:
            trial = x.malt[event]
            sip = x.malt['sipper'][trace]
            all_licks = x.malt['licks']
            all_sips = x.malt['sipper']
        
        df_reptraces_sip.at[group, pref + '_licks'] = [l-sip for l in all_licks if (l>sip-10) and (l<sip+20)]
        df_reptraces_sip.at[group, pref + '_sipper'] = [s-sip for s in all_sips if (s-sip<0.01) and (s-sip>-10)]
        df_reptraces_sip.at[group, pref + '_photo_blue'] = trial['blue'][trace]
        df_reptraces_sip.at[group, pref + '_photo_uv'] = trial['uv'][trace]

rats = np.unique(rats)
df_heatmap_sip = pd.DataFrame(rats, columns=['rat'])
df_heatmap_sip.set_index(['rat'], inplace=True)



for s, pref in zip(['s10', 's11', 's16'],
                           pref_list):

    df_heatmap_sip[pref + '_cas'] = ""
    df_heatmap_sip[pref + '_malt'] = ""
    df_heatmap_sip[pref + '_cas_event'] = ""
    df_heatmap_sip[pref + '_malt_event'] = ""
    
    for rat in rats:
        x = pref_sessions[rat + '_' + s]
        
        df_heatmap_sip.at[rat, pref + '_cas'] = removenoise(x.cas[event])
        df_heatmap_sip.at[rat, pref + '_cas_event'] = getfirstlick(x.cas, event)
        df_heatmap_sip.at[rat, pref + '_malt'] = removenoise(x.malt[event])
        df_heatmap_sip.at[rat, pref + '_malt_event'] = getfirstlick(x.malt, event)

pickle_out = open('R:\\DA_and_Reward\\gc214\\PPP_combined\\output\\ppp_dfs_pref.pickle', 'wb')
dill.dump([df_behav, df_photo, df_reptraces, df_heatmap, df_reptraces_sip, df_heatmap_sip], pickle_out)
pickle_out.close()

## to find rep traces
#x = pref_sessions['PPP1-4_s10']
#trials = x.malt['snips_sipper']
#trialsb = trials['blue']
#
#
#
#i = 6
#fig, ax = plt.subplots()
#ax.plot(trialsb[i])
#ax.annotate(x.malt['lats'][i], xy=(100,0.1))
#

# possibles 'PPP1-4_s10' - 6, 13


### TO DO!!!
### remove noise trials from grouped data
### figure out a way of excluding certain rats (e.g. PPP1.8) maybe just a line that removes at beginning of this code
##
