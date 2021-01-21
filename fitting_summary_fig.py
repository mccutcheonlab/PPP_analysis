# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 16:41:09 2021

@author: admin
"""
import numpy as np
from sklearn.linear_model import LinearRegression

pp1 = [0.9, 0.8, 0.85]
pp2 = [0.6, 0.7, 0.8]
pp3 = [0.4, 0.2, 0.3]

pp3 = [0.9, 0.8, 0.85]
pp2 = [0.6, 0.7, 0.8]
pp1 = [0.4, 0.2, 0.3]

df1 = [10, 15, 12]
df2 = [6, 9, 10]
df3 = [4, 5, 2]

df1 = [10, 15, 12]
df2 = [11, 19, 10]
df3 = [12, 13, 9]


ppall = pp1+pp2+pp3
dfall = df1+df2+df3

merged_list = [[p, d] for p, d in zip(ppall, dfall)]

X = np.array(merged_list)
y = [1,1,1,2,2,2,3,3,3]


# a = [1,2,3]
# b = [4,5]
# c = [2,2,2]

# X = np.array([a,c])
# y = b


reg = LinearRegression(normalize=True).fit(X, y)

print(reg.coef_)


