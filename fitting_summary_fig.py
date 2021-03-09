# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 16:41:09 2021

@author: admin
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
from scipy.stats import gaussian_kde
from scipy.stats.stats import pearsonr

import matplotlib.pyplot as plt


from ppp_pub_figs_settings import *
import trompy as tp

def find_delta(df, keys_in, epoch=[100,149]):
    
    epochrange = range(epoch[0], epoch[1])
    
    keys_out = ['delta_1', 'delta_2', 'delta_3']
        
    for k_in, k_out in zip(keys_in, keys_out):
        cas_auc = [np.trapz(x[epochrange])/10 for x in df[k_in[0]]]
        malt_auc = [np.trapz(x[epochrange])/10 for x in df[k_in[1]]]
        df[k_out] = [c-m for c, m in zip(cas_auc, malt_auc)]
    
    return df

summary_photo_keys = [['pref1_cas_licks_forced', 'pref1_malt_licks_forced'],
                  ['pref2_cas_licks_forced', 'pref2_malt_licks_forced'],
                  ['pref3_cas_licks_forced', 'pref3_malt_licks_forced']]

df_delta = find_delta(df_photo, summary_photo_keys)

photo_keys = ["delta_1", "delta_2", "delta_3"]
# photo_keys = ["peakdiff_1", "peakdiff_2", "peakdiff_3"] # for t-values


def calcmodel(df_behav, df_delta, diet, n=5000):

    if diet != "both":
        dfx1 = df_behav.xs(diet, level=1)
        dfx2 = df_delta.xs(diet, level=1)
    else:
        dfx1 = df_behav
        dfx2 = df_delta
        
    x1vals = np.array([dfx1['pref1'].to_numpy(), dfx1['pref2'].to_numpy(), dfx1['pref3'].to_numpy()]).reshape((1,-1))
    x2vals = np.array([dfx2[photo_keys[0]].to_numpy(), dfx2[photo_keys[1]].to_numpy(), dfx2[photo_keys[2]].to_numpy()]).reshape((1,-1))

        
    print(len(x1vals.squeeze()))
    print("Pearson R for ", diet, pearsonr(x1vals.squeeze(), x2vals.squeeze()))
    
    X = np.vstack((x1vals, x2vals)).T
    
    nrats = int(np.shape(X)[0] / 3)
    
    y = np.array([1]*nrats + [2]*nrats + [3]*nrats)
    print(y)
    
    model = LinearRegression(normalize=True).fit(X, y)
    
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
        
    print(len(x1vals.squeeze()))
    print("Pearson R for ", diet, pearsonr(x1vals.squeeze(), x2vals.squeeze()))
    
    X = np.vstack((x1vals, x2vals)).T
    
    nrats = int(np.shape(X)[0] / 3)
    
    if diet == "NR":
        y = np.array([1]*nrats + [0]*nrats + [0]*nrats)
    elif diet == "PR":
        y = np.array([0]*nrats + [1]*nrats + [1]*nrats)
    elif diet == "both":
        print("difficult")

    model = LinearRegression(normalize=True).fit(X, y)
    
    coefs = []
    for i in range(0, n):
        sample_index = np.random.choice(range(0, len(y)), len(y))
        
        X_samples = X[sample_index]
        y_samples = y[sample_index]
        
        lr = LinearRegression()
        lr.fit(X_samples, y_samples)
        coefs.append(lr.coef_)
        
    return model, np.array(coefs)

# def pearson4diets(df_behav, df_photo, diet, n=5000):
    

def compare2normal(y):
    mu = 0
    sigma = np.std(y)
    normal = np.random.normal(mu, sigma, 5000)
    
    test = ttest_ind(y, normal)
    print(test)

modelNR, coefsNR = calcmodel(df_behav, df_delta, "NR")
modelPR, coefsPR = calcmodel(df_behav, df_delta, "PR")
modelboth, coefsboth = calcmodel(df_behav, df_delta, "both")

modelNR, coefsNR = calcmodel_state(df_behav, df_delta, "NR")
modelPR, coefsPR = calcmodel_state(df_behav, df_delta, "PR")


# takes absolute value to take into account that direction of change is opposite
coefsNR[:,0] = np.abs(coefsNR[:,0])
coefsPR[:,0] = np.abs(coefsPR[:,0])

coefsNR[:,1] = np.abs(coefsNR[:,1])
coefsPR[:,1] = np.abs(coefsPR[:,1])

pp_coef_test = ttest_ind(coefsNR[:,0], coefsPR[:,0])
delta_coef_test = ttest_ind(coefsNR[:,1], coefsPR[:,1])

densityNR = gaussian_kde(coefsNR[:,0])
densityPR = gaussian_kde(coefsPR[:,0])

xs = np.linspace(0, 4)

f, ax = plt.subplots()
ax.plot(xs, densityNR(xs))
ax.plot(xs, densityPR(xs))

densityNR = gaussian_kde(coefsNR[:,1])
densityPR = gaussian_kde(coefsPR[:,1])

xs = np.linspace(0, 0.3)

f, ax = plt.subplots()
ax.plot(xs, densityNR(xs))
ax.plot(xs, densityPR(xs))


# To correlate protein preference and photometry

# pp = X[:,0].reshape(-1,1)
# delta = X[:,1].reshape(-1,1)

# pearsonr(pp.squeeze(), delta.squeeze())
# np.corrcoef(pp.squeeze(), delta.squeeze())

# newmodel = LinearRegression(normalize=True).fit(pp, delta)

# coefs = []

# for i in range(0, 5000):
#     sample_index = np.random.choice(range(0, len(pp)), len(pp))
    
#     X_samples = delta[sample_index]
#     y_samples = pp[sample_index]
    
#     lr = LinearRegression()
#     lr.fit(X_samples, y_samples)
#     coefs.append(lr.coef_)
    
# density = gaussian_kde(tp.flatten_list(coefs))

# xs = np.linspace(0, 10)

# f, ax = plt.subplots()
# ax.plot(xs, density(xs))


