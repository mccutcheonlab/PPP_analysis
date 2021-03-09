# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 10:53:31 2018

@author: James Rig
"""

# ppp_publication_stats
import dill
import pandas as pd

from scipy import stats
import numpy as np

import trompy as tp

from subprocess import PIPE, run

Rscriptpath = 'C:\\Program Files\\R\\R-4.0.3\\bin\\Rscript'
statsfolder = 'C:\\Github\\PPP_analysis\\stats\\'

# Attempts to load pickled file

try:
    pickle_folder = 'C:\\Github\\PPP_analysis\\data\\'
    pickle_in = open(pickle_folder + 'ppp_dfs_pref.pickle', 'rb')
    
    pickle_in = open(pickle_folder + 'ppp_dfs_pref.pickle', 'rb')
    df_behav, df_photo, df_reptraces, df_heatmap, df_reptraces_sip, df_heatmap_sip, longtrace = pd.read_pickle(pickle_in)

    pickle_in = open(pickle_folder + 'ppp_dfs_cond1.pickle', 'rb')
    
    # df_cond1_behav = dill.load(pickle_in)
       
except FileNotFoundError:
    print('Cannot access pickled file(s)')

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

def extractandstack_multi(df, cols1_to_stack, cols2_to_stack, new_cols=[]):
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

def sidakcorr(pval, ncomps=3):
    corr_p = 1-((1-pval)**ncomps)   
    return corr_p

def sidakcorr_R(robj, ncomps=3):
    pval = (list(robj.rx('p.value'))[0])[0]
    corr_p = 1-((1-pval)**ncomps)   
    return corr_p

def ppp_2wayANOVA(df, cols, csvfile):
    
    df = extractandstack(df, cols, new_cols=['rat', 'diet', 'substance', 'licks'])
    df.to_csv(csvfile)
    result = run([Rscriptpath, "--vanilla", "ppp_licksANOVA.R", csvfile], stdout=PIPE, stderr=PIPE, universal_newlines=True)
    print(result.returncode, result.stderr, result.stdout)
    return result

def ppp_3wayANOVA(df, cols1, cols2, csvfile):
    
    df = extractandstack_multi(df, cols1, cols2, new_cols=['rat', 'diet', 'col1', 'col2', 'licks'])
    df.to_csv(csvfile)
#    result = run([Rscriptpath, "--vanilla", "ppp_licksANOVA.R", csvfile], stdout=PIPE, stderr=PIPE, universal_newlines=True)
#    print(result.returncode, result.stderr, result.stdout)
#    return result

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

def ppp_ttest_paired(df, subset, key1, key2, ncomps=3):
    df = df.xs(subset, level=1)
    result = stats.ttest_rel(df[key1], df[key2])
    print(subset, result)
    print('With Sidak correction: ', tp.sidakcorr(result[1], ncomps=ncomps), '\n')
    return result

def ppp_ttest_unpaired(df, index1, index2, key, ncomps=3):
    df1 = df.xs(index1, level=1)
    df2 = df.xs(index2, level=1)
    result = stats.ttest_ind(df1[key], df2[key])
    print(key, result)
    print('With Sidak correction: ', tp.sidakcorr(result[1], ncomps=ncomps), '\n')
    return result

def ppp_ttest_onesample(df, index, key):
    df_new = df.xs(index, level=1)
    result = stats.ttest_1samp(df_new[key], 0.5)
    print(index, key, result, '\n')
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
    result = ppp_ttest_paired(df, dietgroup, keys[0], keys[1], ncomps=2)
    
    if verbose: print('t-test for session 1 vs session 3')
    result = ppp_ttest_paired(df, dietgroup, keys[0], keys[2], ncomps=2)

def stats_conditioning(condsession='1', verbose=True):
    if verbose: print('\nAnalysis of conditioning sessions ' + condsession)
    df = df_cond1_behav
    
    keys1 = ['cond' + condsession + '-cas1-licks',
            'cond' + condsession + '-cas2-licks']
    keys2 = ['cond' + condsession + '-malt1-licks',
            'cond' + condsession + '-malt2-licks']
    
    keys = ['cond' + condsession + '-cas-all',
            'cond' + condsession + '-malt-all']

    if verbose: print('\nANOVA on CONDITIONING trials\n')
    ppp_3wayANOVA(df_cond1_behav,
                   keys1, keys2,
                   statsfolder + 'df_cond' + condsession + '_licks.csv')    


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
    
    prefkey = ['pref' + str(prefsession)]

    if verbose: print('\nANOVA on FORCED LICK trials\n')
    ppp_2wayANOVA(df_behav,
                   forcedkeys,
                   statsfolder + 'df_pref' + prefsession + '_forc_licks.csv')
   
    if verbose: print('\ANOVA on LATENCIES on forced lick trials')
    ppp_2wayANOVA(df_photo,
                   latkeys,
                   statsfolder + 'df_pref' + prefsession + '_forc_licks.csv')

    ppp_full_ttests(df_photo, latkeys)
    
    if verbose: print('\nANOVA on FREE LICK trials\n')
    ppp_2wayANOVA(df_behav,
                   freekeys,
                   statsfolder + 'df_pref' + prefsession + '_free_licks.csv')
    
    ppp_full_ttests(df_behav, freekeys)

    if verbose: print('\nANOVA of CHOICE data\n')
    ppp_2wayANOVA(df_behav,
                   choicekeys,
                   statsfolder + 'df_pref' + prefsession + '_choice.csv')
    
    ppp_full_ttests(df_behav, choicekeys)
    
    ppp_ttest_unpaired(df_behav, 'NR', 'PR', prefkey)
    ppp_ttest_onesample(df_behav, 'NR', prefkey)
    ppp_ttest_onesample(df_behav, 'PR', prefkey)

def stats_pref_photo(df, prefsession='1', verbose=True):
        
    keys = ['pref' + prefsession + '_auc_cas',
            'pref' + prefsession + '_auc_malt']

    if verbose: print('\nAnalysis of preference session ' + prefsession)

    if verbose: print('\nANOVA of photometry data, casein vs. maltodextrin\n')
    ppp_2wayANOVA(df_photo,
                   keys,
                   statsfolder + 'df_pref' + prefsession+ '_forc_licks_auc.csv')
    
    ppp_full_ttests(df_photo, keys)
    
    keys = ['pref' + prefsession + '_lateauc_cas',
            'pref' + prefsession + '_lateauc_malt']

    if verbose: print('\nAnalysis of preference session ' + prefsession)

    if verbose: print('\nANOVA of photometry data (late AUC), casein vs. maltodextrin\n')
    ppp_2wayANOVA(df_photo,
                   keys,
                   statsfolder + 'df_pref' + prefsession+ '_forc_licks_lateauc.csv')
    
    ppp_full_ttests(df_photo, keys)

def stats_summary_behav(verbose=True):
    if verbose: print('\nAnalysis of summary data - BEHAVIOUR')
    
    choicekeys = ['pref1', 'pref2', 'pref3']
    
    ppp_summaryANOVA_2way(df_behav,
                   choicekeys,
                   statsfolder + 'df_summary_behav.csv')
    
    if verbose: print('\nOne-way ANOVA on NR-PR rats')
    ppp_summaryANOVA_1way(df_behav,
               choicekeys,
               statsfolder + 'df_summary_behav_NR.csv',
               'NR')
    
    ppp_summary_ttests(df_behav, choicekeys, 'NR')
    
    if verbose: print('\nOne-way ANOVA on PR-NR rats')
    ppp_summaryANOVA_1way(df_behav,
               choicekeys,
               statsfolder + 'df_summary_behav_PR.csv',
               'PR')
    
    ppp_summary_ttests(df_behav, choicekeys, 'PR')

def stats_summary_photo(verbose=True, use_tvals=False):
    if verbose: print('\nAnalysis of summary data - PHOTOMETRY')
    
    if use_tvals:
        photokeys = ['peakdiff_1', 'peakdiff_2', 'peakdiff_3']
    else:
        photokeys = ['pref1_delta', 'pref2_delta', 'pref3_delta']

    ppp_summaryANOVA_2way(df_photo,
                   photokeys,
                   statsfolder + 'df_summary_photo.csv')
    
    if verbose: print('\nOne-way ANOVA on NR-PR rats')
    ppp_summaryANOVA_1way(df_photo,
               photokeys,
               statsfolder + 'df_summary_photo_NR.csv',
               'NR')
    
    if verbose: print('\nOne-way ANOVA on PR-NR rats')
    ppp_summaryANOVA_1way(df_photo,
               photokeys,
               statsfolder + 'df_summary_photo_PR.csv',
               'PR')

def make_stats_df(df, key_suffixes, prefsession='1', epoch=[100, 149]):
    epochrange = range(epoch[0], epoch[1])
    
    keys_in, keys_out = [], []
    for suffix, short_suffix in zip(key_suffixes, ['cas', 'malt']):
        keys_in.append('pref' + prefsession + suffix)
        keys_out.append('pref' + prefsession + '_auc_' + short_suffix)

    for key_in, key_out in zip(keys_in, keys_out):
        df_photo[key_out] = [np.trapz(rat[epochrange])/10 for rat in df[key_in]]

    df['pref' + prefsession + '_delta'] = [cas-malt for cas, malt in zip(df[keys_out[0]], df[keys_out[1]])]
    
    return df

# for analysing licks
epoch =[100,149]
keys = ['_cas_licks_forced', '_malt_licks_forced']

# for analysing sipper
# epoch = [100,149]
# keys = ['_cas_sip', '_malt_sip']

for session in [1, 2, 3]:
    df_photo = make_stats_df(df_photo, keys, prefsession=str(session), epoch=epoch)

def stats_summary_photo_casvmalt():
    print('Boo yeah')

def stats_pref_ind(prefsession=1):
    
    if prefsession == 1:
        day = 's10'
    elif prefsession == 2:
        day = 's11'
    elif prefsession == 3:
        day = 's16'
    
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
        
    for session in sessions:
        s=sessions[session]
        if s.session == day:
            print(s.rat, s.session, '\n', s.peakdiff)
            
            

def stats_pref_postpref1(df_behav, df_photo, diet, prefsession='2', verbose=True):
    if verbose: print(f'\nAnalysis of preference session {prefsession} for {diet} rats.')
        
    forcedkeys = ['pref' + prefsession + '_cas_forced',
                  'pref' + prefsession + '_malt_forced']
    
    latkeys = ['pref' + str(prefsession) + '_cas_lats_fromsip',
               'pref' + str(prefsession) + '_malt_lats_fromsip']
    
    freekeys = ['pref' + prefsession + '_cas_free',
                'pref' + prefsession + '_malt_free']
    
    choicekeys = ['pref' + str(prefsession) + '_ncas',
                  'pref' + str(prefsession) + '_nmalt']
    
    prefkey = ['pref' + str(prefsession)]
    
    photokeys = ['pref' + prefsession + '_auc_cas',
            'pref' + prefsession + '_auc_malt']
    
    if verbose: print('\nt-test on FORCED LICK trials\n')
    ppp_ttest_paired(df_behav, diet, forcedkeys[0], forcedkeys[1])
    
    if verbose: print('\nt-test on LATENCIES on forced lick trials\n')
    ppp_ttest_paired(df_photo, diet, latkeys[0], latkeys[1])
    
    if verbose: print('\nt-test on FREE LICK trials\n')
    ppp_ttest_paired(df_behav, diet, freekeys[0], freekeys[1])
    
    if verbose: print('\nt-test on CHOICE data\n')
    ppp_ttest_paired(df_behav, diet, choicekeys[0], choicekeys[1])
    
    if verbose: print('\nt-test on PREF RATIO data\n')
    ppp_ttest_onesample(df_behav, diet, prefkey)
    
    if verbose: print('\nt-test on PHOTOMETRY data\n')
    ppp_ttest_paired(df_photo, diet, photokeys[0], photokeys[1])


# Using SPSS for statistical analysis of conditioning data because 3-way ANOVA
# csvfile=statsfolder+'cond1_behav.csv'
# df_cond1_behav.to_csv(csvfile)
# stats_conditioning()
#stats_conditioning(condsession='2')

# stats_pref_behav()
# stats_pref_behav(prefsession='2')
# stats_pref_behav(prefsession='3')

stats_pref_photo(df_photo)
# stats_pref_photo(df_photo, prefsession='2')
# stats_pref_photo(df_photo, prefsession='3')

# stats_pref_ind(prefsession=1)

# stats_summary_behav()
# stats_summary_photo(use_tvals=False)
# stats_summary_photo_casvmalt()


csvfile = statsfolder + "df_summary_photo2wayNR.csv"
diet = 'NR'
cols_to_stack= ['pref1_auc_cas', 'pref2_auc_cas', 'pref3_auc_cas', 'pref1_auc_malt', 'pref2_auc_malt', 'pref3_auc_malt']


df = df_photo.xs(diet, level=1)

df = extractandstack(df, cols_to_stack, new_cols=['rat', 'datalabel', 'value'])

df['prefsession'] = [label[:5] for label in df['datalabel']]

sol =[]
for label in df['datalabel']:
    if 'cas' in label:
        sol.append('cas')
    else:
        sol.append('malt')

df['sol'] = sol

df.to_csv(csvfile)
result = run([Rscriptpath, "--vanilla", "ppp_summaryANOVA_2way_within_within.R", csvfile], stdout=PIPE, stderr=PIPE, universal_newlines=True)
print(result.returncode, result.stderr, result.stdout)


### New stats with rejigged figures
# stats_pref_postpref1(df_behav, df_photo, diet='NR', prefsession='2')

# stats_pref_postpref1(df_behav, df_photo, diet='NR', prefsession='3')

# stats_pref_postpref1(df_behav, df_photo, diet='PR', prefsession='2')

# stats_pref_postpref1(df_behav, df_photo, diet='PR', prefsession='3')

#prefsession='1'
#
#latkeys = ['pref' + str(prefsession) + '_cas_lats_fromsip',
#           'pref' + str(prefsession) + '_malt_lats_fromsip']
#
#df = extractandstack(df_photo, latkeys, new_cols=['rat', 'diet', 'substance', 'licks'])
#
#df.to_csv('C:\\Users\\James Rig\\Dropbox\\Publications in Progress\\PPP Paper\\Stats\\pref1_lats.csv')
