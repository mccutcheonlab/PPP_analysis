# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:06:13 2020

@author: admin
"""

import dill
import numpy as np
from scipy import stats

import pandas as pd

import matplotlib.pyplot as plt

try:
    pickle_folder = 'C:\\Github\\PPP_analysis\\data\\'
    pickle_in = open(pickle_folder + 'ppp_dfs_pref.pickle', 'rb')
    
    df_behav, df_photo, df_reptraces, df_heatmap, df_reptraces_sip, df_heatmap_sip, longtrace = pd.read_pickle(pickle_in)

    pickle_in = open(pickle_folder + 'ppp_dfs_cond1.pickle', 'rb')
    
    # df_cond1_behav = dill.load(pickle_in)
       
except FileNotFoundError:
    print('Cannot access pickled file(s)')

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




cols = ["id", "rat", "session", "diet", "t-stat", "p-val"]

df = pd.DataFrame(columns=cols)

for key in sessions.keys():
    s = sessions[key]
    # cas_snips = [snip['filt_z'] for snip in s.cas['snips_licks_forced'] if snip['noise'] == False]
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
    
    df = df.append(tmp, ignore_index=True)

def get_proportions(df, session, diet, p="p-val"):
    
    df = df[(df["session"] == session) & (df["diet"] == diet)]
    n = len(df)

    
    cas = len(df[(df[p] < 0.05) & (df["t-stat"] > 0)])
    malt = len(df[(df[p] < 0.05) & (df["t-stat"] < 0)])
    nonsig = len(df[df[p] > 0.05])
    
    if cas + malt + nonsig != n:
        print("Something wrong with calculations")
        return    
    
    return cas/n, malt/n, nonsig/n

def makepie(data, ax, labelsOn=False):
    
    labels = "Casein", "Malt", "Non-sig"
    colors = ["blue", "black", "xkcd:light grey"]
    
    if labelsOn == True:
        ax.pie(data, explode = explode, labels=labels, colors=colors, wedgeprops={"edgecolor":"k"})
    else:
        ax.pie(data, explode = explode, colors=colors, wedgeprops={"edgecolor":"k", 'linewidth': 1})
                                                                                    

savefolder = "C:\\Users\\jmc010\\Dropbox\\Publications in Progress\\PPP Paper\\04_JNS\\02_revision 1\\revision figs\\"

s10_NR = get_proportions(df, "s10", "NR")
s10_PR = get_proportions(df, "s10", "PR")

s11_NR = get_proportions(df, "s11", "NR")
s11_PR = get_proportions(df, "s11", "PR")

s16_NR = get_proportions(df, "s16", "NR")
s16_PR = get_proportions(df, "s16", "PR")


labels = "Casein", "Malt", "Non-sig"
labels = "", "", ""
colors = ["blue", "black", "xkcd:light grey"]
explode = [0.1, 0.1, 0.1]


f, ax = plt.subplots(ncols=3, nrows=2, figsize=(3, 2))

makepie(s10_NR, ax[0][0])
makepie(s10_PR, ax[1][0])

makepie(s11_NR, ax[0][1])
makepie(s11_PR, ax[1][1])

makepie(s16_NR, ax[0][2])
makepie(s16_PR, ax[1][2])

# ax[0][0].pie(s10_NR, explode = explode, labels=labels, colors=colors, wedgeprops={"edgecolor":"k")
# ax[1][0].pie(s10_PR, explode = explode, labels=labels, colors=colors)

# ax[0][1].pie(s11_NR, explode = explode, labels=labels, colors=colors)
# ax[1][1].pie(s11_PR, explode = explode, labels=labels, colors=colors)

# ax[0][2].pie(s16_NR, explode = explode, labels=labels, colors=colors)
# ax[1][2].pie(s16_PR, explode = explode, labels=labels, colors=colors)

f.savefig(savefolder + "piecharts.pdf")

    

    