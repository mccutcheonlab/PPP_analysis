# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:57:17 2020

@author: admin
"""

sessions_to_remove =['PPP4-2_s10', 'PPP4-3_s10', 'PPP4-5_s10', 'PPP4.7_s10', 'PPP4-8_s10',
                     'PPP4-2_s11', 'PPP4-3_s11', 'PPP4-5_s11', 'PPP4.7_s11', 'PPP4-8_s11',
                     'PPP4-2_s16', 'PPP4-3_s16', 'PPP4-5_s16', 'PPP4.7_s16', 'PPP4-8_s16']

for session in sessions_to_remove:
    sessions.pop(session)


rats = []
for session in sessions:
    s = sessions[session]
    if s.rat not in rats:
        rats.append(s.rat)  