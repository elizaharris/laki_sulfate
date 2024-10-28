#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Simulate S isotopic composition during the Laki eruption

# Author: Eliza Harris
# Created on Mon Oct 28

# Tasks/questions:
# Over how long (in depth terms) do we think Laki is releasing SO2? I have set it up as a "point" emission in time; is that reasonable?

#%% Preamble 

# basic packages 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from datetime import datetime as dt
import os

parentdir = os.getcwd()

def get_alpha_with_temp(temp,alpha): # Provide temperature and temperature dependence and get alpha and uncertainty at that T
    value = alpha[0] + temp*alpha[2]
    unc_pt2 = abs(alpha[3]/alpha[2])*temp*alpha[2]
    unc = ( alpha[1]**2 + unc_pt2**2 )**0.5
    return(value,unc)

#%% Input parameters: (mean,sd) with Gaussian distribution unless otherwise specified

# General conditions
temp = 0 # 0 degC for atmospheric oxidation: Is this okay?
r_34S = 1/22.7 # These are the ratios for IAEA-S-1 but this won't make any difference to the calcs; from https://www.sciencedirect.com/science/article/abs/pii/S0016703701006111
r_33S = 1/126.9
frac_oxidised_bcg_tot = 0.5 # Fraction of background SO2 oxidised: Is this right? Better guesses? If it's too low we can't converge with the background d34S easily

# Emitted SO2 isotopic composition
d34S_bcgSO2 = (-1,1) # Currently just mirroring volc: Do we have a better value here?
d33S_bcgSO2 = (0,0.01) # Guess
d34S_volcSO2 = (-1,1) # From Will, email
d33S_volcSO2 = (0,0.01) # Guess

# 34S fractionation (T dependent): (a,b,c,d) where alpha-1 = (a+b) - (c+d)*T, where T is in deg C
# T dependences from Harris et al. 2013
alpha34_OH_T = (10.60,0.73,0.004,0.015)
alpha34_OH = get_alpha_with_temp(temp,alpha34_OH_T)
alpha34_H2O2_T = (16.51,0.15,-0.085,0.004)
alpha34_H2O2 = get_alpha_with_temp(temp,alpha34_H2O2_T)
alpha34_TMI_T = (-5.039,0.044,-0.237,0.004)
alpha34_TMI = get_alpha_with_temp(temp,alpha34_TMI_T)

# 33S fractionation 
theta33_OH = (0.503,0.007)
theta33_H2O2 = (0.511,0.003)
theta33_TMIcold = (0.498,0.003)
theta33_TMIwarm = (0.537,0.004)

# Get the Laki data
laki_data = pd.read_csv(parentdir+'/data/laki_s_isotope_data.csv', sep=',', header=0)

#%% Solve the background scenario

# Find mean bcg data
bcg_samples = ["laki_bottom_bcg","laki_top_bcg"]
laki_data["bcg"] = 0
for s in bcg_samples:
    laki_data.loc[laki_data["Name"]==s,"bcg"] = 1
bcg = laki_data[laki_data["Name"]==s].copy() # Space for bcg mean data
for c in laki_data.columns:
    if isinstance(laki_data.loc[laki_data["Name"]==s,c].values[0],str):
        bcg[c] = "background" # deal with string vars
    elif c.count("_unc")==0:
        bcg[c] = np.nanmean(laki_data.loc[laki_data["bcg"]==1,c])
    else:
        bcg[c] = np.nanmean(laki_data.loc[laki_data["bcg"]==1,c])
        # For now just take the mean of the two stds, but we could use a more correct treatment
        # eg. https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups

# Find fraction of oxidation caused by each pathway
frac_OH_H2O2 = pd.DataFrame(np.zeros((101,6)))
frac_OH_H2O2.columns = ["f_OH","f_H2O2","d_SO4","d_SO4_unc","resid","resid_unc"]
frac_OH_H2O2["f_OH"] = np.arange(0,1.01,0.01)
frac_OH_H2O2["f_H2O2"] = 1 - frac_OH_H2O2["f_OH"]
for n in np.arange(frac_OH_H2O2.shape[0]):
    r_0 = (d34S_bcgSO2[0]/1000 + 1)*r_34S # R for emitted SO2
    # Use RP / R0 = (1-f^alpha)/(1-f) to find RP for each reaction
    # OH pathway
    f = 1 - frac_OH_H2O2["f_OH"].iloc[n]*frac_oxidised_bcg_tot  # Fraction remaining respective to this ox pathway
    a = alpha34_OH[0]/1000+1
    r_P_OH = (1-f**a)/(1-f)*r_0
    d_OH = (r_P_OH/r_34S - 1)*1000
    a_low = (alpha34_OH[0]-alpha34_OH[1])/1000+1 # low unc limit
    r_P_OH_low = (1-f**a_low)/(1-f)*r_0
    d_OH_low = (r_P_OH_low/r_34S - 1)*1000
    a_high = (alpha34_OH[0]+alpha34_OH[1])/1000+1 # high unc limit
    r_P_OH_high = (1-f**a_high)/(1-f)*r_0
    d_OH_high = (r_P_OH_high/r_34S - 1)*1000
    d_OH = (d_OH,np.nanmean([abs(d_OH_low-d_OH),abs(d_OH_high-d_OH)])) # mean dOH, uncertainty
    # H2O2 pathway
    f = 1 - frac_OH_H2O2["f_H2O2"].iloc[n]*frac_oxidised_bcg_tot  # Fraction remaining respective to this ox pathway
    a = alpha34_H2O2[0]/1000+1
    r_P_H2O2 = (1-f**a)/(1-f)*r_0
    d_H2O2 = (r_P_H2O2/r_34S - 1)*1000
    a_low = (alpha34_H2O2[0]-alpha34_H2O2[1])/1000+1
    r_P_H2O2_low = (1-f**a_low)/(1-f)*r_0
    d_H2O2_low = (r_P_H2O2_low/r_34S - 1)*1000
    a_high = (alpha34_H2O2[0]+alpha34_H2O2[1])/1000+1
    r_P_H2O2_high = (1-f**a_high)/(1-f)*r_0
    d_H2O2_high = (r_P_H2O2_high/r_34S - 1)*1000
    d_H2O2 = (d_H2O2,np.nanmean([abs(d_H2O2_low-d_H2O2),abs(d_H2O2_high-d_H2O2)])) # mean dOH, uncertainty
    # Mix the products
    frac_OH_H2O2["d_SO4"].iloc[n] = d_OH[0]*frac_OH_H2O2["f_OH"].iloc[n] + d_H2O2[0]*frac_OH_H2O2["f_H2O2"].iloc[n]
    frac_OH_H2O2["d_SO4_unc"].iloc[n] = ( (d_OH[1]*frac_OH_H2O2["f_OH"].iloc[n])**2 + (d_H2O2[1]*frac_OH_H2O2["f_H2O2"].iloc[n])**2 )**0.5
    # Compare to the measured background
    frac_OH_H2O2["resid"].iloc[n] = frac_OH_H2O2["d_SO4"].iloc[n] - bcg["d34S_permil"]
    frac_OH_H2O2["resid_unc"].iloc[n] = ( frac_OH_H2O2["d_SO4_unc"].iloc[n]**2 + bcg["d34S_unc_permil"]**2 )**0.5
resid_norm = abs(frac_OH_H2O2["resid"])/frac_OH_H2O2["resid_unc"]
r = np.where(resid_norm == np.nanmin(resid_norm))[0][0]
f_OH = np.round(frac_OH_H2O2["f_OH"].iloc[r],2)
f_H2O2 = np.round(1-f_OH,2)

# Plot data and background
fig = plt.figure(figsize=(12,6))
plt.plot(laki_data["depth_m"],laki_data["depth_m"]*0+frac_OH_H2O2["d_SO4"].iloc[r],"r-",label="d34S bcg, modelled")
plt.plot(laki_data["depth_m"],laki_data["depth_m"]*0+frac_OH_H2O2["d_SO4"].iloc[r]+frac_OH_H2O2["d_SO4_unc"].iloc[r],"r:")
plt.plot(laki_data["depth_m"],laki_data["depth_m"]*0+frac_OH_H2O2["d_SO4"].iloc[r]-frac_OH_H2O2["d_SO4_unc"].iloc[r],"r:")
plt.plot(laki_data["depth_m"],laki_data["depth_m"]*0+bcg["d34S_permil"].iloc[0],"b-",label="d34S bcg, measured")
plt.plot(laki_data["depth_m"],laki_data["depth_m"]*0+bcg["d34S_permil"].iloc[0]+bcg["d34S_unc_permil"].iloc[0],"b:")
plt.plot(laki_data["depth_m"],laki_data["depth_m"]*0+bcg["d34S_permil"].iloc[0]-bcg["d34S_unc_permil"].iloc[0],"b:")
plt.errorbar(laki_data.loc[laki_data["bcg"]==1,"depth_m"],laki_data.loc[laki_data["bcg"]==1,"d34S_permil"],laki_data.loc[laki_data["bcg"]==1,"d34S_unc_permil"],marker="o",ls="",label="background")
plt.errorbar(laki_data.loc[laki_data["bcg"]==0,"depth_m"],laki_data.loc[laki_data["bcg"]==0,"d34S_permil"],laki_data.loc[laki_data["bcg"]==0,"d34S_unc_permil"],marker="o",ls="",label="volc")
plt.xlabel("depth")
plt.ylabel("d34S_sulfate")
plt.legend()
fig.tight_layout()
fig.show() 
plt.savefig("figs/background.pdf") 
plt.show()

#%% Set up oxidation pathways during the volcanic period
timesteps = 10000
laki_modelled = pd.DataFrame(np.zeros((timesteps,6)))
laki_modelled.columns = ["t","depth","f_remaining","SO2_volc","SO4_volc","d34S_volc","d34S_tot","d33S_volc","d33S_tot"] 
# T is arbitrary time since eruption, according to timesteps
# depth is the approximate depth equivalent (mainly used for plotting)
# f_remaining is the fraction of unoxidised SO2 remaining
# SO2_volc is the amount of volcanic SO2 remaining
# SO4_volc is the amount of volcanic SO2 oxidised in that timestep, to allow mixing with background (assuming short lifetime of volcanic SO4 - is this valid?)
# d34S_volc and d34S_tot are the isotopic compositions of deposited volcanic sulfate and total sulfate (mixed with background)
# d33S, analogously as above
