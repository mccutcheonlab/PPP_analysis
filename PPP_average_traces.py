# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:42:34 2020

@author: admin
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.transforms as transforms

import pandas as pd

from ppp_pub_figs_settings import *
from ppp_pub_figs_fx import *
from ppp_pub_figs_supp import *

import pandas as pd

import trompy as tp

def photogroup_panel(df_photo, keys, dietgroup, colorgroup="control"):
    
    epoch=[100,149]
    
    event='Licks'
    
    f = plt.figure(figsize=(1.5, 2))
    
    
    gs = gridspec.GridSpec(2,1,
                                         height_ratios=[0.15,1],
                                         hspace=0.0,
                                         left=0.3, right=0.8, top=0.9, bottom=0.2)
    
    
    ax1 = f.add_subplot(gs[1,0])
    averagetrace(ax1, df_photo, dietgroup, keys, event=event, fullaxis=True, colorgroup=colorgroup)
    ax1.set_ylim([-1.5, 3.2])
    for xval in epoch:
        ax1.axvline(xval, linestyle='--', color='k', alpha=0.3)

        
    ax2 = f.add_subplot(gs[0,0], sharex=ax1)
    

    ax2.axis('off')
    if event == 'Sipper':
        ax2.plot(100,0, 'v', color='xkcd:silver')
        ax2.annotate(event, xy=(100, 0), xytext=(0,5), textcoords='offset points',
            ha='center', va='bottom')
    elif event == 'Licks':
        ax2.plot([100,150], [0,0], color='xkcd:silver', linewidth=3)
        ax2.annotate(event, xy=(125, 0), xytext=(0,5), textcoords='offset points',
            ha='center', va='bottom')
    
    return f

keys = ['pref1_cas_licks_forced', 'pref1_malt_licks_forced']

fig2_p6 = photogroup_panel(df_photo, keys, "NR", colorgroup="control")
fig2_p6.savefig(savefolder + "fig2_p6_average_NR.pdf")

fig2_p7 = photogroup_panel(df_photo, keys, "PR", colorgroup="expt")
fig2_p7.savefig(savefolder + "fig2_p7_average_PR.pdf")


keys = ['pref2_cas_licks_forced', 'pref2_malt_licks_forced']
fig4_p5 = photogroup_panel(df_photo, keys, "NR", colorgroup="expt")
fig4_p5.savefig(savefolder + "fig4_p5_average_NR.pdf")

keys = ['pref3_cas_licks_forced', 'pref3_malt_licks_forced']
fig4_p11 = photogroup_panel(df_photo, keys, "NR", colorgroup="expt")
fig4_p11.savefig(savefolder + "fig4_p11_average_NR.pdf")





keys = ['pref2_cas_licks_forced', 'pref2_malt_licks_forced']
fig5_p5 = photogroup_panel(df_photo, keys, "PR", colorgroup="control")
fig5_p5.savefig(savefolder + "fig5_p5_average_PR.pdf")

keys = ['pref3_cas_licks_forced', 'pref3_malt_licks_forced']
fig5_p11 = photogroup_panel(df_photo, keys, "PR", colorgroup="control")
fig5_p11.savefig(savefolder + "fig5_p11_average_PR.pdf")




