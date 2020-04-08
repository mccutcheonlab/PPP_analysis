# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 11:00:35 2019

@author: James Rig
"""

all_lats = []
for cas, malt in zip(df_photo['pref1_cas_lats_all'], df_photo['pref1_malt_lats_all']):
    all_lats.append(cas)
    all_lats.append(malt)
    
all_lats_flat = jmf.flatten_list(all_lats)

all_lats_flat = [x for x in all_lats_flat if np.isnan(x) == False]

plt.hist(all_lats_flat, 100, normed=1,  histtype='step', cumulative=True)


