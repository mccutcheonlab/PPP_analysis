#! C:\\Program Files\\R\\R-3.5.1\\bin\\Rscript

df <- read.csv("C:\\Users\\James Rig\\Documents\\GitHub\\PPP_analysis\\df_days_stacked.csv")

library(ez)

bw_analysis <- ezANOVA(data=df,dv=bw,wid=rat,within=day,between=diet,type=3)

print(bw_analysis)