# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 10:53:31 2018

@author: James Rig
"""

# ppp_publication_stats
import JM_general_functions as jmf
import dill
import pandas as pd

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

# Prepare data for stats on preference day FORCED licks
    
ppp_licksANOVA(df_behav,
               ['forced1-cas', 'forced1-malt'],
               usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref1_forc_licks.csv')

ppp_licksANOVA(df_behav,
               ['free1-cas', 'free1-malt'],
               usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref1_free_licks.csv')

ppp_licksANOVA(df_behav,
               ['forced2-cas', 'forced2-malt'],
               usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref2_forc_licks.csv')

ppp_licksANOVA(df_behav,
               ['free2-cas', 'free2-malt'],
               usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref2_free_licks.csv')

ppp_licksANOVA(df_behav,
               ['forced3-cas', 'forced3-malt'],
               usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref3_forc_licks.csv')

ppp_licksANOVA(df_behav,
               ['free3-cas', 'free3-malt'],
               usr + '\\Documents\\GitHub\\PPP_analysis\\df_pref3_free_licks.csv')


