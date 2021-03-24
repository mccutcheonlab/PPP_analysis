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

from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression

# Change this line to point to Rscript installation
Rscriptpath = 'C:\\Program Files\\R\\R-4.0.4\\bin\\Rscript'
statsfolder = '..\\stats\\'

book = xlrd.open_workbook(statsfolder+"estimation_stats.xlsx")

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

def ppp_2wayANOVA(df, cols, csvfile):
    
    df = extractandstack(df, cols, new_cols=['rat', 'diet', 'substance', 'licks'])
    df.to_csv(csvfile)
    Rfile=statsfolder+"ppp_licksANOVA.R"
    result = run([Rscriptpath, "--vanilla", Rfile, csvfile], stdout=PIPE, stderr=PIPE, universal_newlines=True)
    print(result.returncode, result.stderr, result.stdout)
    return result

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
    
    if verbose: print('\nANOVA on FREE LICK trials\n')
    ppp_2wayANOVA(df_behav,
                   freekeys,
                   statsfolder + 'df_pref' + prefsession + '_free_licks.csv')

    if verbose: print('\nANOVA of CHOICE data\n')
    ppp_2wayANOVA(df_behav,
                   choicekeys,
                   statsfolder + 'df_pref' + prefsession + '_choice.csv')

def stats_pref_photo(df_photo, prefsession='1', verbose=True):
        
    keys = ['pref' + prefsession + '_auc_cas',
            'pref' + prefsession + '_auc_malt']

    if verbose: print('\nAnalysis of preference session ' + prefsession)

    if verbose: print('\nANOVA of photometry data, casein vs. maltodextrin\n')
    ppp_2wayANOVA(df_photo,
                   keys,
                   statsfolder + 'df_pref' + prefsession+ '_forc_licks_auc.csv')
    
    keys = ['pref' + prefsession + '_lateauc_cas',
            'pref' + prefsession + '_lateauc_malt']

    if verbose: print('\nAnalysis of preference session ' + prefsession)

    if verbose: print('\nANOVA of photometry data (late AUC), casein vs. maltodextrin\n')
    ppp_2wayANOVA(df_photo,
                   keys,
                   statsfolder + 'df_pref' + prefsession+ '_forc_licks_lateauc.csv')

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
    
    if "PR2NR" in tests:
        if verbose: print('\nOne-way ANOVA on PR-NR rats')
        ppp_summaryANOVA_1way(df_behav,
                   choicekeys,
                   statsfolder + 'df_summary_behav_PR.csv',
                   'PR')

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
    
def calcmodel(df_behav, df_delta, diet, n=5000):

    if diet != "both":
        dfx1 = df_behav.xs(diet, level=1)
        dfx2 = df_delta.xs(diet, level=1)
    else:
        dfx1 = df_behav
        dfx2 = df_delta
        
    x1vals = np.array([dfx1['pref1'].to_numpy(), dfx1['pref2'].to_numpy(), dfx1['pref3'].to_numpy()]).reshape((1,-1))
    x2vals = np.array([dfx2[photo_keys[0]].to_numpy(), dfx2[photo_keys[1]].to_numpy(), dfx2[photo_keys[2]].to_numpy()]).reshape((1,-1))
    
    pr=pearsonr(x1vals.squeeze(), x2vals.squeeze())
    print("Pearson R for {}: r={:.2f}, p={:.3f}".format(diet, pr[0], pr[1]))
    
    X = np.vstack((x1vals, x2vals)).T
    
    nrats = int(np.shape(X)[0] / 3)
    
    y = np.array([1]*nrats + [2]*nrats + [3]*nrats)
    
    model = LinearRegression(normalize=True).fit(X, y)
    
    print("Betas for {}: behavior, {:.2f}; photometry, {:.2f}".format(
        diet, model.coef_[0], model.coef_[1]))
    
    coefs = []
    for i in range(0, n):
        sample_index = np.random.choice(range(0, len(y)), len(y))
        
        X_samples = X[sample_index]
        y_samples = y[sample_index]
        
        lr = LinearRegression()
        lr.fit(X_samples, y_samples)
        coefs.append(lr.coef_)
        
    return model, np.array(coefs)

def calcmodel_state(df_behav, df_delta, diet, n=5000):
    
    if diet != "both":
        dfx1 = df_behav.xs(diet, level=1)
        dfx2 = df_delta.xs(diet, level=1)
    else:
        dfx1 = df_behav
        dfx2 = df_delta
        
    x1vals = np.array([dfx1['pref1'].to_numpy(), dfx1['pref2'].to_numpy(), dfx1['pref3'].to_numpy()]).reshape((1,-1))
    x2vals = np.array([dfx2['delta_1'].to_numpy(), dfx2['delta_2'].to_numpy(), dfx2['delta_3'].to_numpy()]).reshape((1,-1))

    pr=pearsonr(x1vals.squeeze(), x2vals.squeeze())
    # print("Pearson R for {}: r={:.2f}, p={:.3f}".format(diet, pr[0], pr[1]))
    
    X = np.vstack((x1vals, x2vals)).T
    
    nrats = int(np.shape(X)[0] / 3)
    
    if diet == "NR":
        y = np.array([1]*nrats + [0]*nrats + [0]*nrats)
    elif diet == "PR":
        y = np.array([0]*nrats + [1]*nrats + [1]*nrats)
    elif diet == "both":
        print("difficult")

    model = LinearRegression(normalize=True).fit(X, y)
    print("Betas for {}: behavior, {:.2f}; photometry, {:.2f}".format(
        diet, model.coef_[0], model.coef_[1]))
    
    coefs = []
    for i in range(0, n):
        sample_index = np.random.choice(range(0, len(y)), len(y))
        
        X_samples = X[sample_index]
        y_samples = y[sample_index]
        
        lr = LinearRegression()
        lr.fit(X_samples, y_samples)
        coefs.append(lr.coef_)
        
    return model, np.array(coefs)

photo_keys = ["delta_1", "delta_2", "delta_3"]

