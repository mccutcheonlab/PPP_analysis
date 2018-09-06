filename <- commandArgs(trailingOnly=TRUE)

df <- read.csv(filename)

library(ez)

analysis <- ezANOVA(data=df,dv=licks,wid=rat,within=substance,between=diet,type=3)

print(analysis)