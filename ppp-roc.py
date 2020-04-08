# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:45:09 2020

@author: admin
"""

import dill
import time

from fx4roc import *
from figs4roc import *

try:
    type(sessions)
    print('Using existing data')
except NameError:
    print('Loading in data from pickled file')
    try:
        pickle_in = open('C:\\Github\\PPP_analysis\\data\\ppp_pref_reduced.pickle', 'rb')
    except FileNotFoundError:
        print('Cannot access pickled file')
    sessions, rats = dill.load(pickle_in)

def flatten_list(listoflists):
    """ Flattens list of lists into a single list
    Args:
        listoflists: nested list
    Returns:
        flat_list: flattened list
    """
    try:
        flat_list = [item for sublist in listoflists for item in sublist]
        return flat_list
    except:
        print('Cannot flatten list. Maybe is in the wrong format. Returning empty list.')
        return []

def resample_snips(snips, factor=0.1):
    """ Resamples snips to collapse data into larger bins (e.g. for ROC analysis)
    Args
    snips: array of snips (list of lists)
    factor: constant to decide how to bin data (default=0.1)
    
    Returns
    snips: resamples snips
    
    """
    if len(snips)>0:
        n_bins = len(snips[0])
        out_bins = int(n_bins * factor)

        snips_out = []
        for snip in snips:
            snips_out.append(np.mean(np.reshape(snip, (out_bins, -1)), axis=1))
    
        return snips_out
    else:
        return []

def get_flat_snips(sessions, diet, prefsession, sol, event, datatype):
    
    eventkey = 'snips_' + event
    
    list_of_snips = []
    for key in sessions.keys():
        s = sessions[key]
        if s.session == prefsession and s.diet == diet:
            snips = getattr(s, sol)[eventkey][datatype]
            snips = resample_snips(snips)
            list_of_snips.append(snips)
            
    flat_snips = flatten_list(list_of_snips)
    
    return flat_snips

def run_roc_comparison(data, n4shuf=10, timer=True, savedata=""):
    """ Function to run ROC analysis with option for timing and saving resulting data
    Args
    data: list or array with two distributions to be compared. Normally should 
          be of the shape (2,x,y) where x is number of trials and can be different
          between each array and y is bins and should be identical.
          Example, data[0] can be 300x20 list of lists or array and data[1] can
          be 360x20.
    n4shuf: number of times to repeat roc with shuffled values to calculate ps
            default=10, so that it is fast to run, but for accurate p-vals should
            run 2000 times
    timer: Boolean, prints time taken if True
    savedata: insert complete filename here to save the results
    
    Returns
    a: list of ROC values (between 0 and 1) corresponding to bins provided
       (e.g. y in description above)
    p: list of p-vals that correspond to each ROC value in a
    
    """
    
    if timer: start_time = time.time()

    a, p = nanroc(data[0], data[1], n4shuf=n4shuf)
    
    if timer: print(f"--- Total ROC analysis took {(time.time() - start_time)} seconds ---")
    
    if len(savedata)>0:
        try:       
            pickle_out = open(savedata, 'wb')
            dill.dump([a, p, data], pickle_out)
            pickle_out.close()
        except:
            print("Cannot save. Check filename.")

    return a, p

roc_results = {}
for prefsession in ['s10', 's11', 's16']:
    roc_results[prefsession] = {}
    
    pr_cas_sipper = get_flat_snips(sessions, 'PR', prefsession, 'cas', 'sipper', 'filt_z')
    pr_malt_sipper = get_flat_snips(sessions, 'PR', prefsession, 'malt', 'sipper', 'filt_z')
    nr_cas_sipper = get_flat_snips(sessions, 'NR', prefsession, 'cas', 'sipper', 'filt_z')
    nr_malt_sipper = get_flat_snips(sessions, 'NR', prefsession, 'malt', 'sipper', 'filt_z')
    
    pr_cas_licks = get_flat_snips(sessions, 'PR', prefsession, 'cas', 'licks_forced', 'filt_z')
    pr_malt_licks = get_flat_snips(sessions, 'PR', prefsession, 'malt', 'licks_forced', 'filt_z')
    nr_cas_licks = get_flat_snips(sessions, 'NR', prefsession, 'cas', 'licks_forced', 'filt_z')
    nr_malt_licks = get_flat_snips(sessions, 'NR', prefsession, 'malt', 'licks_forced', 'filt_z')
    
    # Uncomment to run lick comparisons
    n4shuf=100
    ### Comparison of photo data aligned to SIPPER for PR rats on PREF DAY 1 
    a, p = run_roc_comparison([pr_cas_sipper, pr_malt_sipper], n4shuf=n4shuf)
    roc_results[prefsession]['pr_sipper'] = {'a':a, 'p':p}
    
    ### Comparison of photo data aligned to SIPPER for NR rats on PREF DAY 1 
    a, p = run_roc_comparison([nr_cas_sipper, nr_malt_sipper], n4shuf=n4shuf)
    roc_results[prefsession]['nr_sipper'] = {'a':a, 'p':p}
    
    ### Comparison of photo data aligned to LICK for PR rats on PREF DAY 1 
    a, p = run_roc_comparison([pr_cas_licks, pr_malt_licks], n4shuf=n4shuf)
    roc_results[prefsession]['pr_licks'] = {'a':a, 'p':p}
    
    ### Comparison of photo data aligned to LICK for NR rats on PREF DAY 1 
    a, p = run_roc_comparison([nr_cas_licks, nr_malt_licks], n4shuf=n4shuf)
    roc_results[prefsession]['nr_licks'] = {'a':a, 'p':p}
