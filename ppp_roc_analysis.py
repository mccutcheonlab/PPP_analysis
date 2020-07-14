# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:45:09 2020

@author: admin
"""

import dill
import time
import numpy as np

import trompy as tp

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
    
def get_snips(sessions, diet, prefsession, sol, event, datatype):
    
    eventkey = 'snips_' + event

    list_of_snips = []
    for key in sessions.keys():
        
        s = sessions[key]
        if s.session == prefsession and s.diet == diet:
            snip_dict = getattr(s, sol)[eventkey]
            try:
                noise = snip_dict['noise']
                snips = [snip for snip, noise in zip(snip_dict[datatype], noise) if not noise]
                snips = tp.resample_snips(snips)
                list_of_snips.append(snips)
            except:
                print(f'Could not extract snips for {key}, {event}')
    
    print(eventkey, diet, prefsession, np.shape(list_of_snips))

    return list_of_snips

def get_flat_snips(list_of_snips):
    
    return tp.flatten_list(list_of_snips)

roc_results = {}
for prefsession in ['s10', 's11', 's16']:
    roc_results[prefsession] = {}
    
    print("Analyzing pref session ", prefsession)
    
    pr_cas_sipper = get_snips(sessions, 'PR', prefsession, 'cas', 'sipper', 'filt_z')
    pr_malt_sipper = get_snips(sessions, 'PR', prefsession, 'malt', 'sipper', 'filt_z')
    nr_cas_sipper = get_snips(sessions, 'NR', prefsession, 'cas', 'sipper', 'filt_z')
    nr_malt_sipper = get_snips(sessions, 'NR', prefsession, 'malt', 'sipper', 'filt_z')
    
    pr_cas_licks = get_snips(sessions, 'PR', prefsession, 'cas', 'licks_forced', 'filt_z')
    pr_malt_licks = get_snips(sessions, 'PR', prefsession, 'malt', 'licks_forced', 'filt_z')
    nr_cas_licks = get_snips(sessions, 'NR', prefsession, 'cas', 'licks_forced', 'filt_z')
    nr_malt_licks = get_snips(sessions, 'NR', prefsession, 'malt', 'licks_forced', 'filt_z')
    
    pr_cas_licks_free = get_snips(sessions, 'PR', prefsession, 'cas', 'licks_free', 'filt_z')
    pr_malt_licks_free = get_snips(sessions, 'PR', prefsession, 'malt', 'licks_free', 'filt_z')
    nr_cas_licks_free = get_snips(sessions, 'NR', prefsession, 'cas', 'licks_free', 'filt_z')
    nr_malt_licks_free = get_snips(sessions, 'NR', prefsession, 'malt', 'licks_free', 'filt_z')
    
    # Uncomment to run lick comparisons
    n4shuf=1000
    ### Comparison of photo data aligned to SIPPER for PR rats
    data = [pr_cas_sipper, pr_malt_sipper]
    data_flat = [tp.flatten_list(d) for d in data]
    a, p = tp.run_roc_comparison(data_flat, n4shuf=n4shuf)
    roc_results[prefsession]['pr_sipper'] = {'a':a, 'p':p, 'data':data, 'data_flat':data_flat}
    
    ### Comparison of photo data aligned to SIPPER for NR rats
    data = [nr_cas_sipper, nr_malt_sipper]
    data_flat = [tp.flatten_list(d) for d in data]
    a, p = tp.run_roc_comparison(data_flat, n4shuf=n4shuf)
    roc_results[prefsession]['nr_sipper'] = {'a':a, 'p':p, 'data':data, 'data_flat':data_flat}
    
    ### Comparison of photo data aligned to LICK for PR rats
    data = [pr_cas_licks, pr_malt_licks]
    data_flat = [tp.flatten_list(d) for d in data]
    a, p = tp.run_roc_comparison(data_flat, n4shuf=n4shuf)
    roc_results[prefsession]['pr_licks'] = {'a':a, 'p':p, 'data':data, 'data_flat':data_flat}
    
    ## Comparison of photo data aligned to LICK for NR rats
    data = [nr_cas_licks, nr_malt_licks]
    data_flat = [tp.flatten_list(d) for d in data]
    a, p = tp.run_roc_comparison(data_flat, n4shuf=n4shuf)
    roc_results[prefsession]['nr_licks'] = {'a':a, 'p':p, 'data':data, 'data_flat':data_flat}
    
    ### Comparison of photo data aligned to FREE LICK for PR rats
    data = [pr_cas_licks_free, pr_malt_licks_free]
    data_flat = [tp.flatten_list(d) for d in data]
    a, p = tp.run_roc_comparison(data_flat, n4shuf=n4shuf)
    roc_results[prefsession]['pr_licks_free'] = {'a':a, 'p':p, 'data':data, 'data_flat':data_flat}
    
    ## Comparison of photo data aligned to FREE LICK for NR rats
    data = [nr_cas_licks_free, nr_malt_licks_free]
    data_flat = [tp.flatten_list(d) for d in data]
    a, p = tp.run_roc_comparison(data_flat, n4shuf=n4shuf)
    roc_results[prefsession]['nr_licks_free'] = {'a':a, 'p':p, 'data':data, 'data_flat':data_flat}

savefile=True
if savefile:
    pickle_out = open('C:\\Github\\PPP_analysis\\data\\ppp_roc_results.pickle', 'wb')
    dill.dump(roc_results, pickle_out)
    pickle_out.close()