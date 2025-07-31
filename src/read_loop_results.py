

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

# Rank fits
rmse_d34S_argsort = np.argsort(loop_fullres['rmse_d34S'])
rmse_d34S_rankings = np.array([ np.where(rmse_d34S_argsort==n)[0][0] for n in np.arange(0,len(rmse_d34S_argsort)) ])
rmse_D33S_argsort = np.argsort(loop_fullres['rmse_D33S'])
rmse_D33S_rankings = np.array([ np.where(rmse_D33S_argsort==n)[0][0] for n in np.arange(0,len(rmse_D33S_argsort)) ])
weights = [1,1] # weight for d34S and D33S rankings
overall_rankings = np.argsort(rmse_d34S_rankings*weights[0]+rmse_D33S_rankings*weights[1])
loop_fullres_ord = loop_fullres.iloc[overall_rankings]
# Alt method
rmse_d34_weight = (loop_fullres['rmse_d34S']-np.nanmin(loop_fullres['rmse_d34S']))/(np.nanmax(loop_fullres['rmse_d34S'])-np.nanmin(loop_fullres['rmse_d34S']))
rmse_D33_weight = (loop_fullres['rmse_D33S']-np.nanmin(loop_fullres['rmse_D33S']))/(np.nanmax(loop_fullres['rmse_D33S'])-np.nanmin(loop_fullres['rmse_D33S']))
total_weight = (rmse_d34_weight+rmse_D33_weight)/2
total_weight_rankings = np.argsort(total_weight)
loop_fullres_ord_2 = loop_fullres.iloc[total_weight_rankings]

# Plot the top 50 values

# Plot oxidation pathways
n_show = 50
bestdata = loop_fullres_ord_2.iloc[0:n_show]
rmse_d34_weight = (bestdata['rmse_d34S']-np.nanmin(bestdata['rmse_d34S']))/(np.nanmax(bestdata['rmse_d34S'])-np.nanmin(bestdata['rmse_d34S']))
rmse_D33_weight = (bestdata['rmse_D33S']-np.nanmin(bestdata['rmse_D33S']))/(np.nanmax(bestdata['rmse_D33S'])-np.nanmin(bestdata['rmse_D33S']))

cols = ["oh_start","oh_end","tmi_start","tmi_end","shift_start","shift_end"]
fig, ax = plt.subplots(6,1,figsize=(8,8))
best_values = []
for n,c in enumerate(cols):
    best_value = ( bestdata[c]*(1-bestdata['rmse_d34S'])+bestdata[c]*(1-bestdata['rmse_D33S']) ) / (np.nansum((1-bestdata['rmse_d34S']))+(1-bestdata['rmse_D33S'])) 
    ax[n].plot(np.arange(0,n_show),loop_fullres_ord[c].iloc[0:n_show ],"o")
    ax[n].plot(np.arange(0,n_show),[np.nanmean(loop_fullres_ord[c].iloc[0:n_show])]*n_show,"-")
    if n == len(c):
        ax[n].set_xlabel("Ranking")
    ax[n].set_ylabel(c)
fig.tight_layout()
plt.show()


plt.savefig("figs/loops/"+runname+"/ox_pathways_"+runname+".pdf") 