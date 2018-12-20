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
import numpy as np

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

def ppp_2wayANOVA(df, cols, csvfile):
    
    df = extractandstack(df, cols, new_cols=['rat', 'diet', 'substance', 'licks'])
    df.to_csv(csvfile)
    result = run([Rscriptpath, "--vanilla", "ppp_licksANOVA.R", csvfile], stdout=PIPE, stderr=PIPE, universal_newlines=True)
    print(result.returncode, result.stderr, result.stdout)
    return result

def ppp_summaryANOVA_2way(df, cols, csvfile):
    df = extractandstack(df, cols, new_cols=['rat', 'diet', 'prefsession', 'value'])
    df.to_csv(csvfile)
    result = run([Rscriptpath, "--vanilla", "ppp_summaryANOVA_2way.R", csvfile], stdout=PIPE, stderr=PIPE, universal_newlines=True)
    print(result.returncode, result.stderr, result.stdout)
    return result

def ppp_summaryANOVA_1way(df, cols, csvfile, dietgroup):
    
    df = df.xs(dietgroup, level=1)    
    df = extractandstack(df, cols, new_cols=['rat', 'prefsession', 'value'])
    df.to_csv(csvfile)
    result = run([Rscriptpath, "--vanilla", "ppp_summaryANOVA_1way.R", csvfile], stdout=PIPE, stderr=PIPE, universal_newlines=True)
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

def ppp_summary_ttests(df, keys, dietgroup, verbose=True):
    if verbose: print('t-test for session 1 vs session 2')
    result = ppp_ttest_paired(df, dietgroup, keys[0], keys[1])
    print('Bonferroni corrected p-value = ', result[1]*3)
    
    if verbose: print('t-test for session 1 vs session 3')
    result = ppp_ttest_paired(df, dietgroup, keys[0], keys[2])
    print('Bonferroni corrected p-value = ', result[1]*3)
    
    if verbose: print('t-test for session 2 vs session 3')
    result = ppp_ttest_paired(df, dietgroup, keys[1], keys[2])
    print('Bonferroni corrected p-value = ', result[1]*3)
    
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
    ppp_2wayANOVA(df_behav,
                   forcedkeys,
                   usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref' + prefsession + '_forc_licks.csv')
   
    if verbose: print('\ANOVA on LATENCIES on forced lick trials')
    ppp_2wayANOVA(df_photo,
                   latkeys,
                   usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref' + prefsession + '_forc_licks.csv')

    ppp_full_ttests(df_photo, latkeys)
    
    if verbose: print('\nANOVA on FREE LICK trials\n')
    ppp_2wayANOVA(df_behav,
                   freekeys,
                   usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref' + prefsession + '_free_licks.csv')
    
    ppp_full_ttests(df_behav, freekeys)

    if verbose: print('\nANOVA of CHOICE data\n')
    ppp_2wayANOVA(df_behav,
                   choicekeys,
                   usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref' + prefsession + '_choice.csv')
    
    ppp_full_ttests(df_behav, choicekeys)


def stats_pref_photo(df, prefsession='1', verbose=True):
        
    keys = ['pref' + prefsession + '_auc_cas',
            'pref' + prefsession + '_auc_malt']

    if verbose: print('\nAnalysis of preference session ' + prefsession)

    if verbose: print('\nANOVA of photometry data, casein vs. maltodextrin\n')
    ppp_2wayANOVA(df_photo,
                   keys,
                   usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref'+ prefsession+ '_forc_licks.csv')
    
    ppp_full_ttests(df_photo, keys)

def stats_summary_behav(verbose=True):
    if verbose: print('\nAnalysis of summary data - BEHAVIOUR')
    
    choicekeys = ['pref1', 'pref2', 'pref3']
    
    ppp_summaryANOVA_2way(df_behav,
                   choicekeys,
                   usr + '\\Documents\\GitHub\\PPP_analysis\\df_summary_behav.csv')
    
    if verbose: print('\nOne-way ANOVA on NR-PR rats')
    ppp_summaryANOVA_1way(df_behav,
               choicekeys,
               usr + '\\Documents\\GitHub\\PPP_analysis\\df_summary_behav_NR.csv',
               'NR')
    
    ppp_summary_ttests(df_behav, choicekeys, 'NR')
    
    if verbose: print('\nOne-way ANOVA on PR-NR rats')
    ppp_summaryANOVA_1way(df_behav,
               choicekeys,
               usr + '\\Documents\\GitHub\\PPP_analysis\\df_summary_behav_PR.csv',
               'PR')

def stats_summary_photo(verbose=True):
    if verbose: print('\nAnalysis of summary data - PHOTOMETRY')
    
    photokeys = ['pref1_delta', 'pref2_delta', 'pref3_delta']
    
    ppp_summaryANOVA_2way(df_photo,
                   photokeys,
                   usr + '\\Documents\\GitHub\\PPP_analysis\\df_summary_photo.csv')
    
    if verbose: print('\nOne-way ANOVA on NR-PR rats')
    ppp_summaryANOVA_1way(df_photo,
               photokeys,
               usr + '\\Documents\\GitHub\\PPP_analysis\\df_summary_photo_NR.csv',
               'NR')
    
    if verbose: print('\nOne-way ANOVA on PR-NR rats')
    ppp_summaryANOVA_1way(df_photo,
               photokeys,
               usr + '\\Documents\\GitHub\\PPP_analysis\\df_summary_photo_PR.csv',
               'PR')

def make_stats_df(df, key_suffixes, prefsession='1', epoch=[100, 119]):
    epochrange = range(epoch[0], epoch[1])
    
    keys_in, keys_out = [], []
    for suffix, short_suffix in zip(key_suffixes, ['cas', 'malt']):
        keys_in.append('pref' + prefsession + suffix)
        keys_out.append('pref' + prefsession + '_auc_' + short_suffix)

    for key_in, key_out in zip(keys_in, keys_out):
        df_photo[key_out] = [np.trapz(rat[epochrange]) for rat in df[key_in]]

    df['pref' + prefsession + '_delta'] = [cas-malt for cas, malt in zip(df[keys_out[0]], df[keys_out[1]])]
    
    return df

epoch = [100,119]
keys = ['_cas_licks_forced', '_malt_licks_forced']

for session in [1, 2, 3]:
    df_photo = make_stats_df(df_photo, keys, prefsession=str(session), epoch=epoch)

#
#stats_pref_behav()
#stats_pref_behav(prefsession='2')
#stats_pref_behav(prefsession='3')

#stats_pref_photo(df_photo)
#stats_pref_photo(df_photo, prefsession='2')
#stats_pref_photo(df_photo, prefsession='3')

stats_summary_behav()
stats_summary_photo()
