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
    df_behav, df_photo = dill.load(pickle_in)

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
    
# Prepare data for stats on preference day FORCED licks

# Stats on preference day 1 - behaviour
ppp_licksANOVA(df_behav,
               ['forced1-cas', 'forced1-malt'],
               usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref1_forc_licks.csv')

ppp_licksANOVA(df_behav,
               ['free1-cas', 'free1-malt'],
               usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref1_free_licks.csv')


ppp_ttest_paired(df_behav, 'NR', 'free1-cas', 'free1-malt')
ppp_ttest_paired(df_behav, 'PR', 'free1-cas', 'free1-malt')

ppp_ttest_unpaired(df_behav, 'NR', 'PR', 'free1-cas')
ppp_ttest_unpaired(df_behav, 'NR', 'PR', 'free1-malt')

ppp_licksANOVA(df_behav,
               ['ncas1', 'nmalt1'],
               usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref1_choice.csv')

ppp_ttest_paired(df_behav, 'NR', 'ncas1', 'nmalt1')
ppp_ttest_paired(df_behav, 'PR', 'ncas1', 'nmalt1')

ppp_ttest_unpaired(df_behav, 'NR', 'PR', 'ncas1')
ppp_ttest_unpaired(df_behav, 'NR', 'PR', 'nmalt1')


# Stats on pref 1- photometry

ppp_licksANOVA(df_photo,
               ['cas1_licks_peak', 'malt1_licks_peak'],
               usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref1_forc_licks.csv')

ppp_ttest_paired(df_photo, 'NR', 'cas1_licks_peak', 'malt1_licks_peak')
ppp_ttest_paired(df_photo, 'PR', 'cas1_licks_peak', 'malt1_licks_peak')

ppp_ttest_unpaired(df_photo, 'NR', 'PR', 'cas1_licks_peak')
ppp_ttest_unpaired(df_photo, 'NR', 'PR', 'malt1_licks_peak')

#ppp_licksANOVA(df_behav,
#               ['forced2-cas', 'forced2-malt'],
#               usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref2_forc_licks.csv')
#
#ppp_licksANOVA(df_behav,
#               ['free2-cas', 'free2-malt'],
#               usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref2_free_licks.csv')
#
#ppp_licksANOVA(df_behav,
#               ['forced3-cas', 'forced3-malt'],
#               usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref3_forc_licks.csv')
#
#ppp_licksANOVA(df_behav,
#               ['free3-cas', 'free3-malt'],
#               usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref3_free_licks.csv')


