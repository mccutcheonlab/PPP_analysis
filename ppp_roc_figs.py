# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:21:24 2020

@author: admin
"""
import dill
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import trompy as tp

import scipy.stats as stats

from ppp_pub_figs_settings import *

try:
    pickle_in = open('C:\\Github\\PPP_analysis\\data\\ppp_roc_results.pickle', 'rb')
except FileNotFoundError:
        print('Cannot access pickled file')
        
roc_results = dill.load(pickle_in)

def get_data_from_dict(roc_dict, prefsession, subset):
    data_to_plot = roc_dict[prefsession][subset]
    
    a = data_to_plot['a']
    p = data_to_plot['p']
    data = data_to_plot['data']
    data_flat = data_to_plot['data_flat']
    
    return [a, p, data, data_flat]

def ppp_plot_roc_and_peak(roc_results, session, key, colors, peakbetween=[10, 13]):
    

    # get appropraite data from roc_results file
    [a, p, data, data_flat] = get_data_from_dict(roc_results, session, key)
    
    #initialize figure
    f = plt.figure(figsize=(6, 2))
    
    outer_gs = f.add_gridspec(1,2, wspace=0.7, width_ratios=[1, 0.5])
    gsdict = {'gsx': 0, 'gsy': 0, 'gs_spec': outer_gs}
    
    # plot ROC figure
    f, ax1 = tp.plot_ROC_and_line(f, a, p, data_flat[0], data_flat[1],
                      cdict=[colors[0], 'white', colors[1]],
                      colors = colors,
                      labels=['Casein', 'Maltodextrin'],
                      ylabel='Z-score',
                      xlabel='Time from first lick',
                      gridspec_dict=gsdict)
    
    # get data for peak plots
    caspeak, maltpeak = [], []
    
    for cas, malt in zip(data[0], data[1]):
        caspeak.append(np.mean(sum_of_epoch_from_snips(cas, peakbetween)))
        maltpeak.append(np.mean(sum_of_epoch_from_snips(malt, peakbetween)))
        
    ax2 = f.add_subplot(outer_gs[0,1])
    tp.barscatter([caspeak, maltpeak], paired=True,
                  scattersize=50,
                  barfacecoloroption = 'individual',
                  barfacecolor = colors,
                  scatteredgecolor = ['xkcd:charcoal'],
                  scatterlinecolor = 'xkcd:charcoal',
                  barlabels=['Cas', 'Malt'],
                  ylabel='AUC',
                  ax=ax2)
    
    print(stats.ttest_rel(caspeak, maltpeak))
    
    return {'f':f, 'a':a, 'p':p, 'caspeak':caspeak, 'maltpeak': maltpeak}
    
def sum_of_epoch_from_snips(snips, peakbetween):
    start = peakbetween[0]
    stop = peakbetween[1]
    return [np.sum(trial[start:stop]) for trial in snips]

colors_pr = [col['pr_cas'], col['pr_malt']]
colors_nr = [col['nr_cas'], col['nr_malt']]
figs_dict = {}

peakbetween=[10, 14]

figs_dict['s10_pr_licks'] = ppp_plot_roc_and_peak(roc_results, 's10', 'pr_licks', colors_pr, peakbetween=peakbetween)

figs_dict['s11_pr_licks'] = ppp_plot_roc_and_peak(roc_results, 's11', 'pr_licks', colors_pr, peakbetween=peakbetween)

figs_dict['s16_pr_licks'] = ppp_plot_roc_and_peak(roc_results, 's16', 'pr_licks', colors_pr, peakbetween=peakbetween)

figs_dict['s10_nr_licks'] = ppp_plot_roc_and_peak(roc_results, 's10', 'nr_licks', colors_nr, peakbetween=peakbetween)

figs_dict['s11_nr_licks'] = ppp_plot_roc_and_peak(roc_results, 's11', 'nr_licks', colors_nr, peakbetween=peakbetween)

figs_dict['s16_nr_licks'] = ppp_plot_roc_and_peak(roc_results, 's16', 'nr_licks', colors_nr, peakbetween=peakbetween)


figs_dict['s10_pr_sipper'] = ppp_plot_roc_and_peak(roc_results, 's10', 'pr_sipper', colors_pr, peakbetween=peakbetween)

figs_dict['s11_pr_sipper'] = ppp_plot_roc_and_peak(roc_results, 's11', 'pr_sipper', colors_pr, peakbetween=peakbetween)

figs_dict['s16_pr_sipper'] = ppp_plot_roc_and_peak(roc_results, 's16', 'pr_sipper', colors_pr, peakbetween=peakbetween)

figs_dict['s10_nr_sipper'] = ppp_plot_roc_and_peak(roc_results, 's10', 'nr_sipper', colors_nr, peakbetween=peakbetween)

figs_dict['s11_nr_sipper'] = ppp_plot_roc_and_peak(roc_results, 's11', 'nr_sipper', colors_nr, peakbetween=peakbetween)

figs_dict['s16_nr_sipper'] = ppp_plot_roc_and_peak(roc_results, 's16', 'nr_sipper', colors_nr, peakbetween=peakbetween)

pdf_pages = PdfPages('C:/Github/PPP_analysis/figs/roc_figs.pdf')
for key in figs_dict.keys():
    fig = figs_dict[key]['f']
    pdf_pages.savefig(fig)

pdf_pages.close()


[a, p, data, data_flat] = get_data_from_dict(roc_results, 's10', 'pr_sipper')


    