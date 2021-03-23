# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 09:24:45 2021

@author: admin
"""

from subprocess import PIPE, run
from scipy import stats   
import pandas as pd
import numpy as np
import trompy as tp
import xlrd

Rscriptpath = 'C:\\Program Files\\R\\R-4.0.4\\bin\\Rscript'
statsfolder = 'C:\\Github\\PPP_analysis\\stats\\'

book = xlrd.open_workbook("C:\\Github\\PPP_analysis\\stats\\estimation_stats.xlsx")

run([Rscriptpath, "--vanilla", statsfolder+"installez.R"], stdout=PIPE, stderr=PIPE, universal_newlines=True)

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
    Rfile=statsfolder+"ppp_licksANOVA.R"
    result = run([Rscriptpath, "--vanilla", Rfile, csvfile], stdout=PIPE, stderr=PIPE, universal_newlines=True)
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
    Rfile=statsfolder+"ppp_summaryANOVA_2way.R"
    result = run([Rscriptpath, "--vanilla", Rfile, csvfile], stdout=PIPE, stderr=PIPE, universal_newlines=True)
    print(result.returncode, result.stderr, result.stdout)
    return result

def ppp_summaryANOVA_2way_within_diet(df, cols, csvfile, dietgroup):
        
    df = df.xs(dietgroup, level=1)  
    df = extractandstack(df, cols, new_cols=['rat', 'prefsession', 'substance', 'value'])
    df.to_csv(csvfile)
    Rfile=statsfolder+"ppp_summaryANOVA_2way.R"
    result = run([Rscriptpath, "--vanilla", Rfile, csvfile], stdout=PIPE, stderr=PIPE, universal_newlines=True)
    print(result.returncode, result.stderr, result.stdout)
    return result

def ppp_summaryANOVA_1way(df, cols, csvfile, dietgroup):
    
    df = df.xs(dietgroup, level=1)    
    df = extractandstack(df, cols, new_cols=['rat', 'prefsession', 'value'])
    df.to_csv(csvfile)
    Rfile=statsfolder+"ppp_summaryANOVA_1way.R"
    result = run([Rscriptpath, "--vanilla", Rfile, csvfile], stdout=PIPE, stderr=PIPE, universal_newlines=True)
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


def stats_pref_behav(df_behav, df_photo, prefsession='1', verbose=True):
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

    # ppp_full_ttests(df_photo, latkeys)
    
    if verbose: print('\nANOVA on FREE LICK trials\n')
    ppp_2wayANOVA(df_behav,
                   freekeys,
                   statsfolder + 'df_pref' + prefsession + '_free_licks.csv')
    
    # ppp_full_ttests(df_behav, freekeys)

    if verbose: print('\nANOVA of CHOICE data\n')
    ppp_2wayANOVA(df_behav,
                   choicekeys,
                   statsfolder + 'df_pref' + prefsession + '_choice.csv')
    
    # ppp_full_ttests(df_behav, choicekeys)
    
    # ppp_ttest_unpaired(df_behav, 'NR', 'PR', prefkey)
    # ppp_ttest_onesample(df_behav, 'NR', prefkey)
    # ppp_ttest_onesample(df_behav, 'PR', prefkey)

def stats_pref_photo(df_photo, prefsession='1', verbose=True):
        
    keys = ['pref' + prefsession + '_auc_cas',
            'pref' + prefsession + '_auc_malt']

    if verbose: print('\nAnalysis of preference session ' + prefsession)

    if verbose: print('\nANOVA of photometry data, casein vs. maltodextrin\n')
    ppp_2wayANOVA(df_photo,
                   keys,
                   statsfolder + 'df_pref' + prefsession+ '_forc_licks_auc.csv')
    
    # ppp_full_ttests(df_photo, keys)
    
    keys = ['pref' + prefsession + '_lateauc_cas',
            'pref' + prefsession + '_lateauc_malt']

    if verbose: print('\nAnalysis of preference session ' + prefsession)

    if verbose: print('\nANOVA of photometry data (late AUC), casein vs. maltodextrin\n')
    ppp_2wayANOVA(df_photo,
                   keys,
                   statsfolder + 'df_pref' + prefsession+ '_forc_licks_lateauc.csv')
    
    # ppp_full_ttests(df_photo, keys)

def stats_summary_behav(df_behav, tests = ["2way", "NR2PR", "PR2NR"], verbose=True):
    if verbose: print('\nAnalysis of summary data - BEHAVIOUR')
    
    choicekeys = ['pref1', 'pref2', 'pref3']
    
    if "2way" in tests:
        if verbose: print('\nTwo-way ANOVA')
        ppp_summaryANOVA_2way(df_behav,
                        choicekeys,
                        statsfolder + 'df_summary_behav.csv')
    
    if "NR2PR" in tests:
        if verbose: print('\nOne-way ANOVA on NR-PR rats')
        ppp_summaryANOVA_1way(df_behav,
                   choicekeys,
                   statsfolder + 'df_summary_behav_NR.csv',
                   'NR')
    
    # ppp_summary_ttests(df_behav, choicekeys, 'NR')
    
    if "PR2NR" in tests:
        if verbose: print('\nOne-way ANOVA on PR-NR rats')
        ppp_summaryANOVA_1way(df_behav,
                   choicekeys,
                   statsfolder + 'df_summary_behav_PR.csv',
                   'PR')
    
    # ppp_summary_ttests(df_behav, choicekeys, 'PR')

def stats_summary_photo_difference(df_photo, tests = ["2way", "NR2PR", "PR2NR"], verbose=True):
    if verbose: print('\nAnalysis of summary data - PHOTOMETRY')
    
    photokeys = ['delta_1', 'delta_2', 'delta_3']
    
    if "2way" in tests:
        ppp_summaryANOVA_2way(df_photo,
                       photokeys,
                       statsfolder + 'df_summary_photo.csv')
    
    if "NR2PR" in tests:
        if verbose: print('\nOne-way ANOVA on NR-PR rats')
        ppp_summaryANOVA_1way(df_photo,
                   photokeys,
                   statsfolder + 'df_summary_photo_NR.csv',
                   'NR')
    
    if "PR2NR" in tests:
        if verbose: print('\nOne-way ANOVA on PR-NR rats')
        ppp_summaryANOVA_1way(df_photo,
                   photokeys,
                   statsfolder + 'df_summary_photo_PR.csv',
                   'PR')

def stats_summary_photo_both_solutions(df_photo, diet):
    csvfile = statsfolder + "df_summary_photo2way.csv"
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
    Rfile = statsfolder+"ppp_summaryANOVA_2way_within_within.R"
    result = run([Rscriptpath, "--vanilla", Rfile, csvfile], stdout=PIPE, stderr=PIPE, universal_newlines=True)
    print(result.returncode, result.stderr, result.stdout)

def output_estimation_stats(book, sheet, rowx, string="", unit=""):
    sh = book.sheet_by_name(sheet)
    print("{} = {:.2f} {} [95%CI {:.2f}, {:.2f}], p={:.3f}".format(
        string,
        sh.cell_value(rowx=rowx, colx=7),
        unit,
        sh.cell_value(rowx=rowx, colx=9),
        sh.cell_value(rowx=rowx, colx=10),
        sh.cell_value(rowx=rowx, colx=18)))
    


# def make_stats_df(df_photo, key_suffixes, prefsession='1', epoch=[100, 149]):
#     epochrange = range(epoch[0], epoch[1])
    
#     keys_in, keys_out = [], []
#     for suffix, short_suffix in zip(key_suffixes, ['cas', 'malt']):
#         keys_in.append('pref' + prefsession + suffix)
#         keys_out.append('pref' + prefsession + '_auc_' + short_suffix)

#     for key_in, key_out in zip(keys_in, keys_out):
#         df_photo[key_out] = [np.trapz(rat[epochrange])/10 for rat in df[key_in]]

#     df_out['pref' + prefsession + '_delta'] = [cas-malt for cas, malt in zip(df_out[keys_out[0]], df_out[keys_out[1]])]
    
    # return df
