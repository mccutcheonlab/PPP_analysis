filename <- commandArgs(trailingOnly=TRUE)

df <- read.csv(filename)

require(ez)

library(ez)

analysis <- ezANOVA(data=df,dv=value,wid=rat,within=.(prefsession, sol), type=3)

print(analysis)