# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:39:12 2018

@author: jaimeHP
"""

xlfile = 'R:\\DA_and_Reward\\gc214\\PPP_combined\\PPP_body weight and food intake.xlsx'

# Body weight data
df = pd.read_excel(xlfile, sheet_name='PPP_bodyweight')
df.set_index(['rat', 'diet'], inplace=True)

df.iloc[:, df.rows.get_level_values('diet') == 'NR']

df.xs('PR', level=1)