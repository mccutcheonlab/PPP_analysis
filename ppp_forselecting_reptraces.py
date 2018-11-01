## to find rep traces
x = pref_sessions['PPP1-4_s10']
trials = x.cas['snips_sipper']
trialsb = trials['blue']

i = 13
fig, ax = plt.subplots()
ax.plot(trialsb[i])
ax.annotate(x.cas['lats'][i], xy=(100,0.1))

# PR - PPP1-4
# cas - 2, 3, 4, 6*, 7, 13, 18, 19
# malt - 5, 11, 15


# NR - PPP1.7
# cas - 0, 9, 10, 14, 16*
# malt - 1big, 7double, 8, 14, 17, 19


