# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 10:53:31 2018

@author: James Rig
"""

# ppp_publication_stats
import JM_general_functions as jmf
import dill
import pandas as pd

from scipy import stats

from subprocess import PIPE, run

Rscriptpath = 'C:\\Program Files\\R\\R-3.5.1\\bin\\Rscript'

# Looks for existing data and if not there loads pickled file
try:
    type(df_photo)
    print('Using existing data')
except NameError:
    print('Loading in data from pickled file')
    try:
        pickle_in = open('R:\\DA_and_Reward\\gc214\\PPP_combined\\output\\ppp_dfs_pref.pickle', 'rb')
    except FileNotFoundError:
        print('Cannot access pickled file')
    df_behav, df_photo, df_reptraces, df_heatmap, df_reptraces_sip, df_heatmap_sip = dill.load(pickle_in)

usr = jmf.getuserhome()

def extractandstack(df, cols_to_stack, new_cols=[]):
    new_df = df.loc[:,cols_to_stack]
    new_df = new_df.stack()
    new_df = new_df.to_frame()
    new_df.reset_index(inplace=True)

    if len(new_cols) > 1:
        try:
            new_df.columns = new_cols
        except ValueError:
            print('Wrong number of labels for new columns given as argument.')
            
    return new_df

def ppp_licksANOVA(df, cols, csvfile):
    
    df = extractandstack(df, cols, new_cols=['rat', 'diet', 'substance', 'licks'])
    df.to_csv(csvfile)
    result = run([Rscriptpath, "--vanilla", "ppp_licksANOVA.R", csvfile], stdout=PIPE, stderr=PIPE, universal_newlines=True)
    print(result.returncode, result.stderr, result.stdout)
    return result

def ppp_ttest_paired(df, subset, key1, key2):
    df = df.xs(subset, level=1)
    result = stats.ttest_rel(df[key1], df[key2])
    print(subset, result, '\n')
    return result

def ppp_ttest_unpaired(df, index1, index2, key):
    df1 = df.xs(index1, level=1)
    df2 = df.xs(index2, level=1)
    result = stats.ttest_ind(df1[key], df2[key])
    print(key, result, '\n')
    return result

def ppp_full_ttests(df, keys, verbose=True):
    
    if verbose: print('t-test for NON-RESTRICTED, casein vs. malt')
    ppp_ttest_paired(df, 'NR', keys[0], keys[1])
    
    if verbose: print('t-test for PROTEIN RESTRICTED, casein vs. malt')
    ppp_ttest_paired(df, 'PR', keys[0], keys[1])
    
    if verbose: print('t-test for CASEIN, NR vs. PR')
    ppp_ttest_unpaired(df, 'NR', 'PR', keys[0])
    
    if verbose: print('t-test for MALTODEXTRIN, NR vs. PR')
    ppp_ttest_unpaired(df, 'NR', 'PR', keys[1])
    
def stats_pref_behav(prefsession='1', verbose=True):
    if verbose: print('\nAnalysis of preference session ' + prefsession)
        
    forcedkeys = ['pref' + prefsession + '_cas_forced',
                  'pref' + prefsession + '_malt_forced']
    
    latkeys = ['pref' + str(prefsession) + '_cas_lats_fromsip',
               'pref' + str(prefsession) + '_malt_lats_fromsip']
    
    freekeys = ['pref' + prefsession + '_cas_free',
                'pref' + prefsession + '_malt_free']
    
    choicekeys = ['pref' + str(prefsession) + '_ncas',
                  'pref' + str(prefsession) + '_nmalt']

    if verbose: print('\nANOVA on FORCED LICK trials\n')
    ppp_licksANOVA(df_behav,
                   forcedkeys,
                   usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref' + prefsession + '_forc_licks.csv')
   
    if verbose: print('\ANOVA on LATENCIES on forced lick trials')
    ppp_licksANOVA(df_photo,
                   latkeys,
                   usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref' + prefsession + '_forc_licks.csv')

    ppp_full_ttests(df_photo, latkeys)
    
    if verbose: print('\nANOVA on FREE LICK trials\n')
    ppp_licksANOVA(df_behav,
                   freekeys,
                   usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref' + prefsession + '_free_licks.csv')
    
    ppp_full_ttests(df_behav, freekeys)

    if verbose: print('\nANOVA of CHOICE data\n')
    ppp_licksANOVA(df_behav,
                   choicekeys,
                   usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref' + prefsession + '_choice.csv')
    
    ppp_full_ttests(df_behav, choicekeys)

stats_pref_behav()
stats_pref_behav(prefsession='2')
stats_pref_behav(prefsession='3')


def stats_pref_photo(prefsession='1', verbose=True):
    
    if verbose: print('\nAnalysis of preference session ' + prefsession)
    
    forcedkeys = ['pref' + prefsession + '_cas_licks_peak',
                  'pref' + prefsession + '_malt_licks_peak']

    if verbose: print('\nANOVA of photometry data, casein vs. maltodextrin\n')
    ppp_licksANOVA(df_photo,
                   forcedkeys,
                   usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref'+ prefsession+ '_forc_licks.csv')
    
    ppp_full_ttests(df_photo, forcedkeys)
    

stats_pref_photo()
stats_pref_photo(prefsession='2')
stats_pref_photo(prefsession='3')