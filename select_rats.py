# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:03:54 2020

@author: admin
"""

import dill

import matplotlib.pyplot as plt

import numpy as np

    
try:
    type(sessions_incl)
    print('Using existing data')
except NameError:
    print('Loading in data from pickled file')
    try:
        pickle_in = open('C:\\Github\\PPP_analysis\\data\\ppp_pref.pickle', 'rb')
    except FileNotFoundError:
        print('Cannot access pickled file')
    sessions_incl, rats_incl = dill.load(pickle_in)
    
try:
    type(sessions_excl)
    print('Using existing data')
except NameError:
    print('Loading in data from pickled file')
    try:
        pickle_in = open('C:\\Github\\PPP_analysis\\data\\ppp_pref_excl.pickle', 'rb')
    except FileNotFoundError:
        print('Cannot access pickled file')
    sessions_excl, rats_excl = dill.load(pickle_in)
    
    
def findsigtrials(snips, sigbins=3, threshold=1.64):
    
    sigvals_consecutive=[]
    for snip in snips:
        sigvals = [1 if t>threshold else 0 for t in snip[100:120]]
        
        sigvals.insert(0,0) # adds 0 at beginning to make looking for first increase reliable
        sigvals.insert(51,0) # adds 0 at beginning to make looking for first increase reliable

        d = np.diff(sigvals)
        
        ups = np.where(d==1)
        downs = np.where(d==-1)
        
        cons=[]
        for u, d in zip(ups, downs):
            cons.append(d-u)
        try:
            sigvals_consecutive.append(np.max(cons))
        except ValueError:
            sigvals_consecutive.append(0)

    #print(sigvals_consecutive)
    
    return [True if trial > sigbins else False for trial in sigvals_consecutive]

def calcsigtrialsbyevent(s, sigbins=5, threshold=3):
    
    cas_sip = findsigtrials(s.cas['snips_sipper']['filt_z'], sigbins=sigbins, threshold=threshold)
    malt_sip = findsigtrials(s.malt['snips_sipper']['filt_z'], sigbins=sigbins, threshold=threshold)
    cas_licks = findsigtrials(s.cas['snips_licks_forced']['filt_z'], sigbins=sigbins, threshold=threshold)
    malt_licks = findsigtrials(s.malt['snips_licks_forced']['filt_z'], sigbins=sigbins, threshold=threshold)
    
    sip = sum(cas_sip)+sum(malt_sip)
    licks = sum(cas_licks)+sum(malt_licks)
    
    print("For rat {} on day {}, there were {} significant responses to sipper extension and {} significant responses to licking.".format(
        s.rat, s.session, sip, licks))
    
    return sip, licks

def getsigtrials(sessions, days):
    totalsigtrials = []
    for session in sessions:
        s = sessions[session]
        for day in days:
            if s.session == day:
                try:
                    totalsigtrials.append(calcsigtrialsbyevent(s, sigbins=5, threshold=3))
                except:
                    pass
    return totalsigtrials

def getsigtrialsbyrat(sessions, days):
    rats={}
    for session in sessions:
        s=sessions[session]
        if s.rat not in rats.keys():
            rats[s.rat]=s.diet
    
    for rat in rats.keys():
        print(rat)
        eachday = []
        for day in days:          
            for session in sessions:
                s=sessions[session]
                if s.rat == rat and s.session == day:
                    eachday.append(sum(calcsigtrialsbyevent(s, sigbins=5, threshold=3)))
        rats[rat]=sum(eachday)
        print(sum(eachday))
        
    return rats

# print('\nIncluded rats')
# totalsigtrials_incl = getsigtrials(sessions_incl, days)


# print('\nExcluded rats')
# totalsigtrials_excl = getsigtrials(sessions_excl, days)


# f, ax = plt.subplots(ncols=2)
# f.subplots_adjust(wspace=0.3)
# for rat in totalsigtrials_incl:
#     ax[0].scatter(rat[0], rat[1], color='g')
#     #print(sum(rat[0]+rat[1]))
#     ax[1].plot(1, rat[0]+rat[1], color='g', marker='o')
    
# for rat in totalsigtrials_excl:
#     ax[0].scatter(rat[0], rat[1], color='r')
#     ax[1].plot(2, rat[0]+rat[1], color='r', marker='o')
    
# ax[0].set_xlabel('Significant sipper extensions')
# ax[0].set_ylabel('Significant lick extensions')

# ax[1].set_xlim([0.5, 2.5])
# ax[1].set_xticks([1,2])
# ax[1].set_xticklabels(['Incl.', 'Excl.'])
# ax[1].set_ylabel('Total significant trials, aligned to sipper or licking')

def plotsigtrialsbyevent(ax, data, color='red', sigbins=5, threshold=3):
    
    Larray = findsigtrials(data, sigbins=sigbins, threshold=threshold)
    
    for i, trial in enumerate(Larray):
        if trial == True:
            ax.plot(data[i], color=color)

def plotsigtrials(s, sigbins=5, threshold=3, plot='sig'):
    
    f, ax = plt.subplots(ncols=4, sharey=True)
    
    for axis, data in enumerate([s.cas['snips_sipper']['filt_z'],
                                  s.malt['snips_sipper']['filt_z'],
                                  s.cas['snips_licks_forced']['filt_z'],
                                  s.malt['snips_licks_forced']['filt_z']]):
        plotsigtrialsbyevent(ax[axis], data, sigbins=sigbins, threshold=threshold)

    
    ax[0].set_ylabel('Z-Score (filtered)')
    f.suptitle(s.rat)


### Works out total number of significant trials across all three test sessions

# days = ['s10', 's11', 's16']
            
# rats_incl = getsigtrialsbyrat(sessions_incl, days)
# rats_excl = getsigtrialsbyrat(sessions_excl, days)

# f, ax = plt.subplots()
# for rat in rats_incl:
#     total = rats_incl[rat]
#     ax.scatter(1, total, color='green')
    
# for rat in rats_excl:
#     total = rats_excl[rat]
#     ax.scatter(2, total, color='red')

# ax.set_xlim([0.5, 2.5])
# ax.set_xticks([1,2])
# ax.set_xticklabels(['Incl.', 'Excl.'])
# ax.set_ylabel('Total significant trials, aligned to sipper or licking')
    
    
s = sessions_incl['PPP4-3_s10']
plotsigtrials(s, sigbins=5, threshold=3)


    
# for row in df_photo:
#     print(row)
    
# df = df_photo['pref1_cas_sip']

# plt.plot(df[0])


# for session in sessions_incl:
#     s = sessions_incl[session]
#     print(s.rat, s.session, s.bgMAD)
    
# for session in sessions_excl:
#     s = sessions_excl[session]
#     print(s.rat, s.session, s.bgMAD)

    
    