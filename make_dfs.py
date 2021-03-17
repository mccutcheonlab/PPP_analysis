# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 08:47:56 2017

@author: Jaime
"""

# Assembles data from PPP1, PPP3, and PPP4 into pandas dataframes for plotting.
# Saves dataframes, df_behav and df_photo, as pickle object (ppp_dfs_pref)

# Choice data
import scipy.io as sio
import os
import string
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import trompy as tp
from scipy import stats
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

def removenoise(snipdata, key="filt_z"):
    # returns blue snips with noisey ones removed
    new_snips = [snip for (snip, noise) in zip(snipdata[key], snipdata['noise']) if not noise]
    return new_snips

def filt_as_delta(snipdata):
    # converts filtered signal into delta change in fluorescence and removes noise   
    new_snips=[]
    for (snip, noise) in zip(snipdata['filt'], snipdata['noise']): 
        if not noise:           
            new_snips.append(snip/np.abs(np.mean(snip[:99])))
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

def average_without_noise(snips, key='filt_z'):
    # Can change default key to switch been delatF (key='blue') and z-score (key='blue_z') 
    try:
        no_noise_snips = [trial for trial, noise in zip(snips[key], snips['noise']) if not noise]
        result = np.mean(no_noise_snips, axis=0)
        return result
    except:
        print('Problem averaging snips')
        return []
    
def average_peak_without_noise(snips):
    try:
        no_noise_peak = [peak for peak, noise in zip(snips["peak"], snips['noise']) if not noise]
        result = np.nanmean(no_noise_peak)
        return result
    except:
        print('Problem getting peaks')
        return []

def get_first_trial(snips, key='filt_z'):
    try:
        return snips[key][0]
    except:
        return []
    
def convert_events(events, t2sMap):
    events_convert=[]
    for x in events:
        events_convert.append(np.searchsorted(t2sMap, x, side="left"))
    
    return events_convert

def find_delta(df, keys_in, epoch=[100,149]):
    
    epochrange = range(epoch[0], epoch[1])
    
    keys_out = ['delta_1', 'delta_2', 'delta_3']
        
    for k_in, k_out in zip(keys_in, keys_out):
        cas_auc = [np.trapz(x[epochrange])/10 for x in df[k_in[0]]]
        malt_auc = [np.trapz(x[epochrange])/10 for x in df[k_in[1]]]
        df[k_out] = [c-m for c, m in zip(cas_auc, malt_auc)]
    
    return df

def shuffledcomp(cas, malt, nshuf=1000):
    ncas = len(cas)
    
    realdiff = np.mean(cas) - np.mean(malt)
    
    alltrials = cas+malt
    shufcomps = []
    for i in range(nshuf):
        shuftrials = np.random.permutation(alltrials)
        diff = np.mean(shuftrials[:ncas]) - np.mean(shuftrials[ncas:])
        shufcomps.append(diff)

    exceed = [1 for x in shufcomps if x > abs(realdiff)]
    
    return sum(exceed)/nshuf

def findpeak(snips):
    time_to_peak = []
    for snip in snips:
        time_to_peak.append(np.argmax(snip[100:])/10)
        
    return(time_to_peak)

def get_start_auc(key, makefig=False, figfile=""):

    print("Analysing", key)
    s = pref_sessions[key]
    
    sip0 = min(s.cas["sipper"][0], s.malt["sipper"][0])
    
    start_t = start_times[key] / s.fs
    
    print("Total time analysed is",  sip0-start_t)
    
    data = s.data_filt[int(start_t*s.fs):int(sip0*s.fs)]
    
    if makefig:
        data2plot = data+np.abs(np.min(data)*2)
        if s.diet == "NR":
            color=col["nr_cas"]
        elif s.diet == "PR":
            color = col["pr_cas"]
        f, ax = plt.subplots(figsize=(1.5,0.5))
        ax.plot(data2plot, color=color)
        # ax.text(0,0,key)
        tp.invisible_axes(ax)
        ax.plot([0, 0], [0, 0.1], color="k")
        ax.plot([0, s.fs*5], [0, 0], color="k")
        try:
            f.savefig(figfile)
        except:
            pass

    min_value = np.min(data)
    data = data + np.abs(min_value)
    
    return np.mean(data)

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

signal="filt_z"

for j, c_sip_z, m_sip_z, c_licks_z, m_licks_z in zip(included_sessions,
                             ['pref1_cas_sip', 'pref2_cas_sip', 'pref3_cas_sip'],
                             ['pref1_malt_sip', 'pref2_malt_sip', 'pref3_malt_sip'],
                             ['pref1_cas_licks', 'pref2_cas_licks', 'pref3_cas_licks'],
                             ['pref1_malt_licks', 'pref2_malt_licks', 'pref3_malt_licks']):

    df_photo[c_sip_z] = [average_without_noise(pref_sessions[x].cas['snips_sipper'], key=signal) for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[m_sip_z] = [average_without_noise(pref_sessions[x].malt['snips_sipper'], key=signal) for x in pref_sessions if pref_sessions[x].session == j] 
    df_photo[c_licks_z] = [average_without_noise(pref_sessions[x].cas['snips_licks'], key=signal) for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[m_licks_z] = [average_without_noise(pref_sessions[x].malt['snips_licks'], key=signal) for x in pref_sessions if pref_sessions[x].session == j]

# adds means of licks and latencies
for j, c_licks_forc, m_licks_forc, c_lats_forc, m_lats_forc, c_lats_forc_fromsip, m_lats_forc_fromsip, in zip(included_sessions,
                           ['pref1_cas_licks_forced', 'pref2_cas_licks_forced', 'pref3_cas_licks_forced'],
                           ['pref1_malt_licks_forced', 'pref2_malt_licks_forced', 'pref3_malt_licks_forced'],
                           ['pref1_cas_lats', 'pref2_cas_lats', 'pref3_cas_lats'],
                           ['pref1_malt_lats', 'pref2_malt_lats', 'pref3_malt_lats'],
                           ['pref1_cas_lats_fromsip', 'pref2_cas_lats_fromsip', 'pref3_cas_lats_fromsip'],
                           ['pref1_malt_lats_fromsip', 'pref2_malt_lats_fromsip', 'pref3_malt_lats_fromsip']):
    df_photo[c_licks_forc] = [average_without_noise(pref_sessions[x].cas['snips_licks_forced'], key=signal) for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[m_licks_forc] = [average_without_noise(pref_sessions[x].malt['snips_licks_forced'], key=signal) for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[c_lats_forc] = [np.nanmean(pref_sessions[x].cas['snips_licks_forced']['latency'], axis=0) for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[m_lats_forc] = [np.nanmean(pref_sessions[x].malt['snips_licks_forced']['latency'], axis=0) for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[c_lats_forc_fromsip] = [np.nanmean(pref_sessions[x].cas['lats'], axis=0) for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[m_lats_forc_fromsip] = [np.nanmean(pref_sessions[x].malt['lats'], axis=0) for x in pref_sessions if pref_sessions[x].session == j]

for j, c_lats_all, m_lats_all in zip(included_sessions,
                                     ['pref1_cas_lats_all', 'pref2_cas_lats_all', 'pref3_cas_lats_all'],
                                     ['pref1_malt_lats_all', 'pref2_malt_lats_all', 'pref3_malt_lats_all']):
    df_photo[c_lats_all] = [pref_sessions[x].cas['snips_licks_forced']['latency'] for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[m_lats_all] = [pref_sessions[x].malt['snips_licks_forced']['latency'] for x in pref_sessions if pref_sessions[x].session == j]
    
for j, col in zip(included_sessions, ['peakdiff_1', 'peakdiff_2', 'peakdiff_3']):
    df_photo[col] = [pref_sessions[x].peakdiff[0] for x in pref_sessions if pref_sessions[x].session == j]

# For calculating AUCs - immediately following first lick (5s)
for col_in, col_out in zip(['pref1_cas_licks_forced', 'pref2_cas_licks_forced', 'pref3_cas_licks_forced',
                            'pref1_malt_licks_forced', 'pref2_malt_licks_forced', 'pref3_malt_licks_forced'],
                            ['pref1_auc_cas', 'pref2_auc_cas', 'pref3_auc_cas',
                            'pref1_auc_malt', 'pref2_auc_malt', 'pref3_auc_malt']):
    df_photo[col_out] = [np.trapz(data[100:149])/10 for data in df_photo[col_in]]

# For calculating AUCs - immediately following first lick (5s)
for col_in, col_out in zip(['pref1_cas_licks_forced', 'pref2_cas_licks_forced', 'pref3_cas_licks_forced',
                            'pref1_malt_licks_forced', 'pref2_malt_licks_forced', 'pref3_malt_licks_forced'],
                            ['pref1_lateauc_cas', 'pref2_lateauc_cas', 'pref3_lateauc_cas',
                            'pref1_lateauc_malt', 'pref2_lateauc_malt', 'pref3_lateauc_malt']):
    df_photo[col_out] = [np.trapz(data[150:199])/10 for data in df_photo[col_in]]

for j, c_peak, m_peak in zip(included_sessions,
                                     ['pref1_cas_peak', 'pref2_cas_peak', 'pref3_cas_peak'],
                                     ['pref1_malt_peak', 'pref2_malt_peak', 'pref3_malt_peak']):
    df_photo[c_peak] = [average_peak_without_noise(pref_sessions[x].cas["snips_licks_forced"]) for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[m_peak] = [average_peak_without_noise(pref_sessions[x].malt["snips_licks_forced"]) for x in pref_sessions if pref_sessions[x].session == j]

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
    df_reptraces[pref + '_filt'] = ""
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
        df_reptraces.at[group, pref + '_filt'] = trial['filt'][trace]

rats = np.unique(rats)
df_heatmap = pd.DataFrame(rats, columns=['rat'])
df_heatmap.set_index(['rat'], inplace=True)

signal = "filt_z"

for s, pref in zip(['s10', 's11', 's16'],
                           pref_list):

    df_heatmap[pref + '_cas'] = ""
    df_heatmap[pref + '_malt'] = ""
    df_heatmap[pref + '_cas_event'] = ""
    df_heatmap[pref + '_malt_event'] = ""
    
    for rat in rats:
        x = pref_sessions[rat + '_' + s]
        
        df_heatmap.at[rat, pref + '_cas'] = removenoise(x.cas[event], key=signal)
        df_heatmap.at[rat, pref + '_cas_event'] = getsipper(x.cas[event])        
        df_heatmap.at[rat, pref + '_malt'] = removenoise(x.malt[event], key=signal)
        df_heatmap.at[rat, pref + '_malt_event'] = getsipper(x.malt[event])

### Makes dataframe with delta values for summary figure (fig. 6)
epoch = [100, 149]
photokeys = [['pref1_cas_licks_forced', 'pref1_malt_licks_forced'],
        ['pref2_cas_licks_forced', 'pref2_malt_licks_forced'],
        ['pref3_cas_licks_forced', 'pref3_malt_licks_forced']]

df_delta = find_delta(df_photo, photokeys, epoch=epoch)

### Makes dataframe for piechart data with significantly different rat-by-rat comparisons
cols = ["id", "rat", "session", "diet", "t-stat", "p-val"]

df_pies = pd.DataFrame(columns=cols)

for key in sessions.keys():
    s = sessions[key]
    tmp = s.cas['snips_licks_forced']
    cas_snips = [snip for snip, noise in zip(tmp['filt_z'], tmp['noise']) if noise == False]
    s.cas_auc_bytrial = [np.trapz(snip[100:149])/10 for snip in cas_snips]
    
    tmp = s.malt['snips_licks_forced']
    malt_snips = [snip for snip, noise in zip(tmp['filt_z'], tmp['noise']) if noise == False]
    s.malt_auc_bytrial = [np.trapz(snip[100:149])/10 for snip in malt_snips]
    
    print(s.rat, s.diet, s.session)
    shufttest = shuffledcomp(s.cas_auc_bytrial, s.malt_auc_bytrial)
    
    result = stats.ttest_ind(s.cas_auc_bytrial, s.malt_auc_bytrial)
    
    print(key, s.diet, result, s.peakdiff[1], shufttest)
    
    tmp = {"id": key,
            "rat": s.rat,
            "session": s.session,
            "diet": s.diet,
            "t-stat": result[0],
            "p-val": result[1],
            "shuf-p": shufttest}
    
    df_pies = df_pies.append(tmp, ignore_index=True)


pickle_out = open('C:\\Github\\PPP_analysis\\data\\ppp_dfs_pref.pickle', 'wb')
dill.dump([df_behav, df_photo, df_reptraces, df_heatmap, df_delta, df_pies], pickle_out)
pickle_out.close()


### For making dictionaries with latencies to lick and peak

NR_cas ={"time2peak": [], "latency": []}
NR_malt ={"time2peak": [], "latency": []}
PR_cas ={"time2peak": [], "latency": []}
PR_malt ={"time2peak": [], "latency": []}

for key in pref_sessions.keys():
    s = pref_sessions[key]
    if s.session == "s10":
        
        casdata = s.cas["snips_sipper"]
        maltdata = s.malt["snips_sipper"]
        
        if s.diet == "NR":
            NR_cas["time2peak"].append(findpeak(casdata["filt_z"]))
            NR_cas["latency"].append(casdata["latency"])
            
            NR_malt["time2peak"].append(findpeak(maltdata["filt_z"]))
            NR_malt["latency"].append(maltdata["latency"])
        elif s.diet == "PR":
            PR_cas["time2peak"].append(findpeak(casdata["filt_z"]))
            PR_cas["latency"].append(casdata["latency"])
            
            PR_malt["time2peak"].append(findpeak(maltdata["filt_z"]))
            PR_malt["latency"].append(maltdata["latency"])

df_nr_cas = pd.DataFrame(data={"latency": tp.flatten_list(NR_cas["latency"]), "time2peak": tp.flatten_list(NR_cas["time2peak"])})
df_nr_malt = pd.DataFrame(data={"latency": tp.flatten_list(NR_malt["latency"]), "time2peak": tp.flatten_list(NR_malt["time2peak"])})
df_pr_cas = pd.DataFrame(data={"latency": tp.flatten_list(PR_cas["latency"]), "time2peak": tp.flatten_list(PR_cas["time2peak"])})
df_pr_malt = pd.DataFrame(data={"latency": tp.flatten_list(PR_malt["latency"]), "time2peak": tp.flatten_list(PR_malt["time2peak"])})

pickle_out = open('C:\\Github\\PPP_analysis\\data\\ppp_dfs_latencies.pickle', 'wb')
dill.dump([df_nr_cas, df_nr_malt, df_pr_cas, df_pr_malt], pickle_out)
pickle_out.close()


### To get AUC from beginning of session
start_times = {"PPP1-1_s10": 62052,
               "PPP1-2_s10": 175600,
               "PPP1-3_s10": 99691,
               "PPP1-4_s10": 269571,
               "PPP1-5_s10": 105794,
               "PPP1-6_s10": 258890,
               "PPP1-7_s10": 110880,
               "PPP3-2_s10": 263458,
               "PPP3-3_s10": 78328,
               "PPP3-4_s10": 180053,
               "PPP3-5_s10": 119018,
               "PPP3-8_s10": 109863,
               "PPP4-1_s10": 1017,
               "PPP4-4_s10": 1017,
               "PPP4-6_s10": 92570}

NR_aucs = []
PR_aucs = []

for key in start_times.keys():
    auc = get_start_auc(key)
    diet = pref_sessions[key].diet
    if diet == "NR":
        NR_aucs.append(auc)
    elif diet == "PR":
        PR_aucs.append(auc)
    else:
        print("problem assigning AUC to group")
        
df_NR_startAUC = pd.DataFrame(data=NR_aucs)
df_PR_startAUC = pd.DataFrame(data=PR_aucs)

pickle_out = open('C:\\Github\\PPP_analysis\\data\\ppp_dfs_startAUC.pickle', 'wb')
dill.dump([df_NR_startAUC, df_PR_startAUC], pickle_out)
pickle_out.close()

### For making dataframes for conditioning data
try:
    pickle_in = open('C:\\Github\\PPP_analysis\\data\\ppp_cond.pickle', 'rb')
except FileNotFoundError:
    print('Cannot access pickled file')

cond_sessions, rats = dill.load(pickle_in)
    
rats = {}

for session in cond_sessions:
    x = cond_sessions[session]
    if x.rat not in rats.keys():
        rats[x.rat] = x.diet
    
cas_sessions = ['cond1-cas1', 'cond1-cas2']
malt_sessions = ['cond1-malt1', 'cond1-malt2']  

df_cond = pd.DataFrame([x for x in rats], columns=['rat'])
df_cond['diet'] = [rats.get(x) for x in rats]

for cas, malt in zip(cas_sessions, malt_sessions):
    df_cond[cas] = [np.float(cond_sessions[x].cas) for x in cond_sessions if cond_sessions[x].sessiontype == cas]
    df_cond[malt] = [np.float(cond_sessions[x].malt) for x in cond_sessions if cond_sessions[x].sessiontype == malt]

df_cond['cond1-cas-all'] = df_cond['cond1-cas1'] + df_cond['cond1-cas2']
df_cond['cond1-malt-all'] = df_cond['cond1-malt1'] + df_cond['cond1-malt2']

pickle_out = open('C:\\Github\\PPP_analysis\\data\\ppp_dfs_cond.pickle', 'wb')
dill.dump(df_cond, pickle_out)
pickle_out.close()