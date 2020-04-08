# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 16:25:03 2018

@author: James Rig
"""
rat = 'PPP1-7'
s = 's10'
x = pref_sessions[rat + '_' + s]

event = 'snips_licks_forced'

len(x.cas[event]['noise'])
len(x.cas[event]['latency'])

(x.cas[event]['noise']).count(False)

a = removenoise(x.cas[event])
b = getsipper(x.cas[event])

len(a)
len(b)

c = [2,4,5,8,12,3,4,14]

#d = np.asarray(c)
#
#d[d>10]=100
#c.replace()
#
#c = [n if (n<10) else None for n in c]


c = np.asarray(c, dtype=float)

c[c>10] = np.nan
