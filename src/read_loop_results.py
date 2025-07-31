

# Wrapper to generate and run many different ox scenarios

# TO DO: Read in the rmse summary and the ox params summary and merge
# show distribution of rmses for each value/bin for each var, to show where min is

# Plot the different params for the top 50 runs (double rankings) to see if there is a pattern

# basic packages 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from datetime import datetime as dt
import os
import sys
from pathlib import Path
from datetime import datetime
import shutil

parentdir = os.getcwd()

# Loop data
loop_params = pd.read_csv("output/loops/loopparams_testconfigs_looptest_250730_142913")
loop_res = pd.read_csv(parentdir+"/output/output_summary_loops.csv")
loop_fullres = pd.merge(loop_params, loop_res, on='runname')

# Select only models capturing the D33S dip
loop_fullres = loop_fullres[(loop_fullres['Laki6_D33S']<0)] # & (loop_fullres['Laki7_D33S']<0.0)]
# No values of Laki 7 are less than 0!

# Alt method, using weight where heighest weight is best
rmse_d34_weight = 1-(loop_fullres['rmse_d34S']-np.nanmin(loop_fullres['rmse_d34S']))/(np.nanmax(loop_fullres['rmse_d34S'])-np.nanmin(loop_fullres['rmse_d34S']))
rmse_D33_weight = 1-(loop_fullres['rmse_D33S']-np.nanmin(loop_fullres['rmse_D33S']))/(np.nanmax(loop_fullres['rmse_D33S'])-np.nanmin(loop_fullres['rmse_D33S']))
weights = [1,1] # weight for d34S and D33S rankings
total_weight = (rmse_d34_weight*weights[0]+rmse_D33_weight*weights[1])/sum(weights)
loop_fullres["weights"] = np.array(total_weight)
total_weight_rankings = np.argsort(-total_weight)
loop_fullres_ord_2 = loop_fullres.iloc[total_weight_rankings]
# Rank fits
rmse_d34S_argsort = np.argsort(loop_fullres['rmse_d34S'])
rmse_d34S_rankings = np.array([ np.where(rmse_d34S_argsort==n)[0][0] for n in np.arange(0,len(rmse_d34S_argsort)) ])
rmse_D33S_argsort = np.argsort(loop_fullres['rmse_D33S'])
rmse_D33S_rankings = np.array([ np.where(rmse_D33S_argsort==n)[0][0] for n in np.arange(0,len(rmse_D33S_argsort)) ])
weights = [1,1] # weight for d34S and D33S rankings
overall_rankings = np.argsort(rmse_d34S_rankings*weights[0]+rmse_D33S_rankings*weights[1])
loop_fullres_ord = loop_fullres.iloc[overall_rankings]

fig, ax = plt.subplots(1,1,figsize=(8,8))
plt.plot(loop_fullres['rmse_d34S'],loop_fullres['rmse_D33S'],"o",alpha=0.2)
fig.tight_layout()
plt.show()


# Plot the top 50 values
n_show = 50
bestdata = loop_fullres_ord.iloc[0:n_show]
# Alt method, just taking the overall top ones
#bestdata = loop_fullres[(loop_fullres['rmse_d34S']<3.95) & (loop_fullres['rmse_D33S']<0.089)]
#n_show = len(bestdata)
cols = ["oh_start","oh_end","tmi_start","tmi_end","shift_start","shift_end"]
fig, ax = plt.subplots(7,1,figsize=(8,8))
best_values = []
for n,c in enumerate(cols):
    best_value = (np.nansum(bestdata[c]*bestdata["weights"]))/(np.nansum(bestdata["weights"]))
    ax[n].plot(np.arange(0,n_show),bestdata[c],"o")
    ax[n].plot(np.arange(0,n_show),[best_value]*n_show,"-")
    if n == len(c):
        ax[n].set_xlabel("Ranking")
    ax[n].set_ylabel(c)
    best_values = best_values+[best_value]
ax[6].plot(np.arange(0,n_show),bestdata["weights"],"o")
fig.tight_layout()
plt.show()

# Notes: Using the different methods, different groups are identified as the best ones
# Using loop_fullres[(loop_fullres['rmse_d34S']<3.95) & (loop_fullres['rmse_D33S']<0.089)] gives the most consistent
# results for the best values, identifying the cluster with okay d34S and good D33S.
# The ranking results are scattered. The reason is that optimizing d34S and D33S does not co-occur.
# The results are also "worse" than the manually selected values because of the low statistical weight of the small
# dip in D33S that is of high scientific importance.