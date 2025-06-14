#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Simulate S isotopic composition during the Laki eruption

# Author: Eliza Harris
# Created on Mon Oct 28

# Tasks/questions:
# Scenarios: All volcanic oxidation TMI vs. all H2O2/OH vs. half:half

#%% Preamble 

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

#%% Get run name (from command line input)

if (len(sys.argv)>1) & (len(sys.argv[1])>0):
    runname = sys.argv[1]
    print("\nName of run:", runname)
else:
    print("No run name given, exiting.")
    sys.exit()
if 0:
    runname = "testconfigs_WillOptimal"
    
configfile = parentdir+'/configs/configs_'+runname+'.csv'
if not os.path.isfile(configfile):
    print("No config file for this run, exiting.")
    sys.exit()
oxfile = parentdir+'/configs/oxpathways_'+runname+'.csv'
if not os.path.isfile(oxfile):
    print("No oxidation pathways file for this run, exiting.")
    sys.exit()
    
# Create the directories (removes and overwrites if exists already)
dir = parentdir+'/figs/'+runname
if os.path.exists(dir):
    shutil.rmtree(dir)
os.mkdir(parentdir+'/figs/'+runname)
dir = parentdir+'/output/'+runname
if os.path.exists(dir):
    shutil.rmtree(dir)
os.mkdir(parentdir+'/output/'+runname)
os.mkdir(parentdir+'/output/'+runname+"/iterations_individual/")

#%% Some functions to set up

def get_alpha_with_temp(temp,alpha): # Provide temperature and temperature dependence and get alpha and uncertainty at that T
    value = alpha[0] + temp*alpha[2]
    unc_pt2 = abs(alpha[3]/alpha[2])*temp*alpha[2]
    unc = ( alpha[1]**2 + unc_pt2**2 )**0.5
    return(value,unc)

# Add low/hi uncertainty to this and later use it in the bcg and other sections!
def rayleigh_product(e,d_0,f,r_std):
    a = e/1000+1
    r_0 = (d_0/1000 + 1)*r_std
    r_P = (1-f**a)/(1-f)*r_0
    d_P = (r_P/r_std - 1)*1000
    return(d_P)

def rayleigh_rem_substrate(e,d_0,f,r_std):
    a = e/1000+1
    r_0 = (d_0/1000 + 1)*r_std
    r_S = (f**(a-1))*r_0
    d_S = (r_S/r_std - 1)*1000
    return(d_S)

def rayleigh_product_33S(e,theta,d_0_34,d_0_33,f,r_std_33,r_std_34):
    #e_abs = np.abs(e)
    #a_34 = e_abs/1000+1
    a_34 = e/1000+1
    r_0_34 = (d_0_34/1000 + 1)*r_std_34
    r_P_34 = (1-f**a_34)/(1-f)*r_0_34
    d_P_34 = (r_P_34/r_std_34 - 1)*1000
    #a_33 = np.exp(theta*np.log(e_abs/1000+1))
    a_33 = np.exp(theta*np.log(e/1000+1))
    r_0_33 = (d_0_33/1000 + 1)*r_std_33
    r_P_33 = (1-f**a_33)/(1-f)*r_0_33
    d_P_33 = (r_P_33/r_std_33 - 1)*1000 
    D_P_33 = 1000*( (d_P_33/1000+1) - (d_P_34/1000+1)**0.515 )
    #return(d_P_33*e/e_abs,D_P_33)
    return(d_P_33,D_P_33)

#%% Input parameters: (mean,sd) with Gaussian distribution unless otherwise specified
# In this version, mostly from config file

configs = pd.read_csv(parentdir+'/configs/configs_'+runname+'.csv', sep=',', header=0, usecols=['param','value1','value2','value3'])

# General conditions
temp_min = configs[configs["param"]=="temp"]["value1"].iloc[0]
temp_max = configs[configs["param"]=="temp"]["value2"].iloc[0]
if np.isnan(temp_max):
    temp_max = temp_min # If only one T is given, T is treated as having no uncertainty
r_34S = 1/22.7 # These are the ratios for IAEA-S-1 but this won't make any difference to the calcs; from https://www.sciencedirect.com/science/article/abs/pii/S0016703701006111
r_33S = 1/126.9
frac_oxidised_bcg_tot = configs[configs["param"]=="frac_oxidised_bcg_tot"]["value1"].iloc[0]

# Emitted SO2 isotopic composition
d34S_bcgSO2 = (configs[configs["param"]=="d34S_bcgSO2"]["value1"].iloc[0], configs[configs["param"]=="d34S_bcgSO2"]["value2"].iloc[0])
d33S_bcgSO2 = (((d34S_bcgSO2[0]/1000+1)**0.515 - 1)*1000,0.01) # D33S = 0
d34S_volcSO2 = (configs[configs["param"]=="d34S_volcSO2"]["value1"].iloc[0], configs[configs["param"]=="d34S_volcSO2"]["value2"].iloc[0])
d33S_volcSO2 = (((d34S_volcSO2[0]/1000+1)**0.515 - 1)*1000,0.01) # D33S = 0; is this right?

# Details of the eruption
full_volc_length = configs[configs["param"]=="full_volc_length"]["value1"].iloc[0]
decay_rate = configs[configs["param"]=="decay_rate"]["value1"].iloc[0]
so2_emission_length = configs[configs["param"]=="so2_emission_length"]["value1"].iloc[0]
so2_emission_tail_length = configs[configs["param"]=="so2_emission_tail_length"]["value1"].iloc[0]
total_so2 = configs[configs["param"]=="total_so2"]["value1"].iloc[0]
mean_res_time_so2 = configs[configs["param"]=="mean_res_time_so2"]["value1"].iloc[0]
mean_res_time_so4 = configs[configs["param"]=="mean_res_time_so4"]["value1"].iloc[0]

# 34S fractionation (T dependent): (a,b,c,d) where alpha-1 = (a+b) - (c+d)*T, where T is in deg C
# T dependences from Harris et al. 2013
alpha34_OH_T = (10.60,0.73,0.004,0.015)
alpha34_H2O2_T = (16.51,0.15,-0.085,0.004)
alpha34_TMI_T = (-5.039,0.44,-0.237,0.004)
background_T = 0
alpha34_H2O2_bcg = get_alpha_with_temp(background_T,alpha34_H2O2_T)
alpha34_OH_bcg = get_alpha_with_temp(background_T,alpha34_OH_T)
alpha34_TMI_bcg = get_alpha_with_temp(background_T,alpha34_TMI_T)

# 33S fractionation 
theta33_OH = (0.503,0.007)
theta33_H2O2 = (0.511,0.003)
theta33_TMIcold = (0.498,0.003) # below 20 C
theta33_TMIwarm = (0.537,0.004)

# Get the Laki data
fname = configs[configs["param"]=="laki_data_file"]["value3"].iloc[0]
laki_data = pd.read_csv(parentdir+'/data/'+fname, sep=',', header=0)

# Get the ox pathways
ox_pathways = pd.read_csv(parentdir+'/configs/oxpathways_'+runname+'.csv', sep=',', header=0)
ox_sum = ox_pathways["OH"]+ox_pathways["H2O2"]+ox_pathways["TMI"]
if (np.nanmax(ox_sum)>1) | (np.nanmax(ox_sum)<1):
    print("Some ox pathways do not add up to 1; normalising")
    ox_pathways["OH"] = ox_pathways["OH"]/ox_sum
    ox_pathways["H2O2"] = ox_pathways["H2O2"]/ox_sum
    ox_pathways["TMI"] = ox_pathways["TMI"]/ox_sum
    
#%% Solve the background scenario

# Find mean bcg data
bcg_samples = ["laki_bottom_bcg","laki_top_bcg","Laki1","Laki2-3","Laki20-21"]
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
# New bcg method: find one fractionation factor and then fractionation (from 28.11.2024)
for n in np.arange(frac_OH_H2O2.shape[0]):
    r_0 = (d34S_bcgSO2[0]/1000 + 1)*r_34S # R for emitted SO2
    # Use RP / R0 = (1-f^alpha)/(1-f) to find RP for total reaction
    e_comb = frac_OH_H2O2["f_OH"].iloc[n]*alpha34_OH_bcg[0] + frac_OH_H2O2["f_H2O2"].iloc[n]*alpha34_H2O2_bcg[0]
    e_comb_unc = ( (frac_OH_H2O2["f_OH"].iloc[n]*alpha34_OH_bcg[1])**2 + (frac_OH_H2O2["f_H2O2"].iloc[n]*alpha34_H2O2_bcg[1])**2 )**0.5
    # Comb pathway
    f = 1 - frac_oxidised_bcg_tot  # Fraction remaining 
    d_SO4 = rayleigh_product(e=e_comb,d_0=d34S_bcgSO2[0],f=f,r_std=r_34S)
    d_SO4_low = rayleigh_product(e=e_comb-e_comb_unc,d_0=d34S_bcgSO2[0],f=f,r_std=r_34S)
    d_SO4_high = rayleigh_product(e=e_comb+e_comb_unc,d_0=d34S_bcgSO2[0],f=f,r_std=r_34S)
    d_SO4 = (d_SO4,np.nanmean([abs(d_SO4_low-d_SO4),abs(d_SO4_high-d_SO4)])) # mean dOH, uncertainty
    # Save the results
    frac_OH_H2O2["d_SO4"].iloc[n] = d_SO4[0]
    frac_OH_H2O2["d_SO4_unc"].iloc[n] = d_SO4[1]
    # Compare to the measured background
    frac_OH_H2O2["resid"].iloc[n] = frac_OH_H2O2["d_SO4"].iloc[n] - bcg["d34S_permil"]
    frac_OH_H2O2["resid_unc"].iloc[n] = ( frac_OH_H2O2["d_SO4_unc"].iloc[n]**2 + bcg["d34S_unc_permil"]**2 )**0.5
resid_norm = abs(frac_OH_H2O2["resid"])/frac_OH_H2O2["resid_unc"]
r = np.where(resid_norm == np.nanmin(resid_norm))[0][0]
f_OH = np.round(frac_OH_H2O2["f_OH"].iloc[r],2)
f_H2O2 = np.round(1-f_OH,2)
d34S_bcg = frac_OH_H2O2["d_SO4"].iloc[r]

# Calc 33S for optimised bcg scenario
f = 1 - f_OH*frac_oxidised_bcg_tot
bcg_33S_OH = rayleigh_product_33S(e=alpha34_OH_bcg[0],theta=theta33_OH[0],d_0_34=d34S_bcgSO2[0],d_0_33=d33S_bcgSO2[0],f=f,r_std_34=r_34S,r_std_33=r_33S)
f = 1 - f_H2O2*frac_oxidised_bcg_tot
bcg_33S_H2O2 = rayleigh_product_33S(e=alpha34_H2O2_bcg[0],theta=theta33_H2O2[0],d_0_34=d34S_bcgSO2[0],d_0_33=d33S_bcgSO2[0],f=f,r_std_34=r_34S,r_std_33=r_33S)
d33S_bcg = bcg_33S_OH[0]*f_OH + bcg_33S_H2O2[0]*f_H2O2
D33S_bcg = bcg_33S_OH[1]*f_OH + bcg_33S_H2O2[1]*f_H2O2

# Plot data and background
fig, ax = plt.subplots(2,1,figsize=(12,6))
# 34S
ax[0].plot(laki_data["depth_m"],laki_data["depth_m"]*0+d34S_bcg,"r-",label="d34S bcg, modelled")
ax[0].plot(laki_data["depth_m"],laki_data["depth_m"]*0+d34S_bcg+frac_OH_H2O2["d_SO4_unc"].iloc[r],"r:")
ax[0].plot(laki_data["depth_m"],laki_data["depth_m"]*0+d34S_bcg-frac_OH_H2O2["d_SO4_unc"].iloc[r],"r:")
ax[0].plot(laki_data["depth_m"],laki_data["depth_m"]*0+bcg["d34S_permil"].iloc[0],"b-",label="d34S bcg, measured")
ax[0].plot(laki_data["depth_m"],laki_data["depth_m"]*0+bcg["d34S_permil"].iloc[0]+bcg["d34S_unc_permil"].iloc[0],"b:")
ax[0].plot(laki_data["depth_m"],laki_data["depth_m"]*0+bcg["d34S_permil"].iloc[0]-bcg["d34S_unc_permil"].iloc[0],"b:")
ax[0].errorbar(laki_data.loc[laki_data["bcg"]==1,"depth_m"],laki_data.loc[laki_data["bcg"]==1,"d34S_permil"],laki_data.loc[laki_data["bcg"]==1,"d34S_unc_permil"],marker="o",ls="",label="background")
ax[0].errorbar(laki_data.loc[laki_data["bcg"]==0,"depth_m"],laki_data.loc[laki_data["bcg"]==0,"d34S_permil"],laki_data.loc[laki_data["bcg"]==0,"d34S_unc_permil"],marker="o",ls="",label="volc")
ax[0].set_xlabel("depth")
ax[0].set_ylabel("d34S_sulfate")
ax[0].legend()
# 33S
ax[1].plot(laki_data["depth_m"],laki_data["depth_m"]*0+D33S_bcg,"r-",label="D34S bcg, modelled")
ax[1].plot(laki_data["depth_m"],laki_data["depth_m"]*0+bcg["D33S_permil"].iloc[0],"b-",label="D33S bcg, measured")
ax[1].plot(laki_data["depth_m"],laki_data["depth_m"]*0+bcg["D33S_permil"].iloc[0]+bcg["D33S_unc_permil"].iloc[0],"b:")
ax[1].plot(laki_data["depth_m"],laki_data["depth_m"]*0+bcg["D33S_permil"].iloc[0]-bcg["D33S_unc_permil"].iloc[0],"b:")
ax[1].errorbar(laki_data.loc[laki_data["bcg"]==1,"depth_m"],laki_data.loc[laki_data["bcg"]==1,"D33S_permil"],laki_data.loc[laki_data["bcg"]==1,"D33S_unc_permil"],marker="o",ls="",label="background")
ax[1].errorbar(laki_data.loc[laki_data["bcg"]==0,"depth_m"],laki_data.loc[laki_data["bcg"]==0,"D33S_permil"],laki_data.loc[laki_data["bcg"]==0,"D33S_unc_permil"],marker="o",ls="",label="volc")
ax[1].set_xlabel("depth")
ax[1].set_ylabel("D33S_sulfate")
fig.tight_layout()
plt.savefig("figs/"+runname+"/background_d34S_D33S_"+runname+".pdf") 
plt.show()


#%% Look at oxidation pathways during the volcanic period

timesteps = int(full_volc_length*365) # Daily time steps for the ~1.2 years affected (defined at the start)
laki_modelled = pd.DataFrame(np.zeros((timesteps,16)))
laki_modelled.columns = ["t","temp","depth","so2_emitted","so2_pool","so2_oxidised","so4_pool","so4_removed","d34S_so2_pool","d34S_so4_pool","d33S_so4_pool","D33S_so4_pool","f_volc","d34S_so4_tot","d33S_so4_tot","D33S_so4_tot"] 
# T is arbitrary time since eruption, according to timesteps
# depth is the approximate depth equivalent (mainly used for plotting)
# so2 emitted is emissions in Tg per day
# so2 pool is the so2 pool at the END of the day (eg. after daily addition and removal)
# so2 oxidised is the amount of so2 removed by oxidation
# so4 pool and removed are analogous (although so4 removed by deposition)
# d34S_so2_pool and so4 are the pool isotopic compositions at the END of the day
# f_volc is the fraction of deposited sulfate from volcanic sources (for mixing to find deposited isotopic composition)

# Some set up...
laki_modelled["t"] = np.arange(0,timesteps)
so2_per_day = total_so2/(so2_emission_length + so2_emission_tail_length/2) # Max SO2 emissions (for first XX days)

# Set up volc pathways on model timesteps
pathways_volc = pd.DataFrame(np.zeros((timesteps,3)))
pathways_volc.columns = ["OH","TMI","H2O2"] 
pathways_volc["OH"] = np.interp(laki_modelled["t"],ox_pathways["fraction_of_volc"]*int(full_volc_length*365),ox_pathways["OH"])
pathways_volc["H2O2"] = np.interp(laki_modelled["t"],ox_pathways["fraction_of_volc"]*int(full_volc_length*365),ox_pathways["H2O2"])
pathways_volc["TMI"] = np.interp(laki_modelled["t"],ox_pathways["fraction_of_volc"]*int(full_volc_length*365),ox_pathways["TMI"])

laki_its = dict() # Space for results of iterations
n_its = 100
for i in np.arange(n_its):
    print(i)
    this_laki_modelled = laki_modelled.copy()
    # Start with T uncertainty
    temp = np.random.uniform(temp_min,temp_max)
    alpha34_H2O2 = get_alpha_with_temp(temp,alpha34_H2O2_T)
    alpha34_OH = get_alpha_with_temp(temp,alpha34_OH_T)
    alpha34_TMI = get_alpha_with_temp(temp,alpha34_TMI_T)
    this_laki_modelled["temp"] = temp
    # Set up isotopic frac uncertainty
    alpha34S_i = {'OH': alpha34_OH[0] + np.random.normal(0,1,1)*alpha34_OH[1],
                  'H2O2': alpha34_H2O2[0] + np.random.normal(0,1,1)*alpha34_H2O2[1],
                'TMI': alpha34_TMI[0] + np.random.normal(0,1,1)*alpha34_TMI[1] }
    alpha34_volc_i = pathways_volc["OH"]*alpha34S_i["OH"] + pathways_volc["H2O2"]*alpha34S_i["H2O2"] + pathways_volc["TMI"]*alpha34S_i["TMI"]
    theta_33_i_TMI = theta33_TMIcold[0] + np.random.normal(0,1,1)*theta33_TMIcold[1] if temp<=20 else theta33_TMIwarm[0] + np.random.normal(0,1,1)*theta33_TMIwarm[1] 
    theta33_i = {'OH': theta33_OH[0] + np.random.normal(0,1,1)*theta33_OH[1] ,
                  'H2O2': theta33_H2O2[0] + np.random.normal(0,1,1)*theta33_H2O2[1],
                'TMI': theta_33_i_TMI }
    # Loop through the days...
    for n,t in enumerate(this_laki_modelled["t"]):
        # Get the SO2 pool to work on
        so2_pool_start = this_laki_modelled.loc[n-1,"so2_pool"] if n>0 else 0
        # Emit SO2 if t < SO2_emission_length
        if t <= so2_emission_length:
            this_laki_modelled.loc[n,"so2_emitted"] = so2_per_day 
        elif t <= so2_emission_length + so2_emission_tail_length:
            proportion = 1 - (t - so2_emission_length)/so2_emission_tail_length
            this_laki_modelled.loc[n,"so2_emitted"] = so2_per_day*proportion
        else:
            this_laki_modelled.loc[n,"so2_emitted"] = 0
        so2_pool = so2_pool_start + this_laki_modelled.loc[n,"so2_emitted"]
        # Remove SO2 according to lifetime
        this_laki_modelled.loc[n,"so2_oxidised"] = so2_pool * (1 - np.exp(-1/mean_res_time_so2) )
        this_laki_modelled.loc[n,"so2_pool"] = so2_pool - this_laki_modelled.loc[n,"so2_oxidised"]
        # Get the SO4 pool
        so4_pool_start = this_laki_modelled.loc[n-1,"so4_pool"] if n>0 else 0
        so4_pool = so4_pool_start + this_laki_modelled.loc[n,"so2_oxidised"] # Oxidised SO2 -> SO4
        this_laki_modelled.loc[n,"so4_removed"] = so4_pool * (1 - np.exp(-1/mean_res_time_so4) )
        this_laki_modelled.loc[n,"so4_pool"] = so4_pool - this_laki_modelled.loc[n,"so4_removed"]
        # SO2/SO4 d34S isotopic composition
        d34S_so2_pool_start = this_laki_modelled.loc[n-1,"d34S_so2_pool"] if n>0 else 0
        d34S_so2_pool = (d34S_so2_pool_start*so2_pool_start + d34S_volcSO2[0]*this_laki_modelled.loc[n,"so2_emitted"])/(so2_pool)
        f_remaining = 1 - this_laki_modelled.loc[n,"so2_oxidised"]/so2_pool
        this_laki_modelled.loc[n,"d34S_so2_pool"] = rayleigh_rem_substrate(e=alpha34_volc_i.iloc[n],d_0=d34S_so2_pool,f=f_remaining,r_std=r_34S)
        this_laki_modelled.loc[n,"d34S_so4_pool"] = rayleigh_product(e=alpha34_volc_i.iloc[n],d_0=d34S_so2_pool,f=f_remaining,r_std=r_34S)
        # Print a summary
        # print("d34S_SO2_pool = ",str(d34S_so2_pool),"; d34S_SO4_pool = ",str(this_laki_modelled.loc[n,"d34S_so4_pool"]),"; e = ",str(alpha34_volc_i,))
        # D33S
        d_0_33 = ((d34S_so2_pool/1000+1)**0.515 - 1)*1000
        dD33_TMI = rayleigh_product_33S(e=alpha34S_i["TMI"],theta=theta33_i["TMI"],d_0_34=d34S_so2_pool,d_0_33=d_0_33,f=f_remaining,r_std_34=r_34S,r_std_33=r_33S)
        dD33_OH = rayleigh_product_33S(e=alpha34S_i["OH"],theta=theta33_i["OH"],d_0_34=d34S_so2_pool,d_0_33=d_0_33,f=f_remaining,r_std_34=r_34S,r_std_33=r_33S)
        dD33_H2O2 = rayleigh_product_33S(e=alpha34S_i["H2O2"],theta=theta33_i["H2O2"],d_0_34=d34S_so2_pool,d_0_33=d_0_33,f=f_remaining,r_std_34=r_34S,r_std_33=r_33S)
        d33S_so4 = pathways_volc["TMI"].iloc[n]*dD33_TMI[0] + pathways_volc["OH"].iloc[n]*dD33_OH[0] + pathways_volc["H2O2"].iloc[n]*dD33_H2O2[0]
        D33S_so4 = pathways_volc["TMI"].iloc[n]*dD33_TMI[1] + pathways_volc["OH"].iloc[n]*dD33_OH[1] + pathways_volc["H2O2"].iloc[n]*dD33_H2O2[1]
        this_laki_modelled.loc[n,"d33S_so4_pool"] = d33S_so4
        this_laki_modelled.loc[n,"D33S_so4_pool"] = D33S_so4

    # Mix volcanic S with background S
    # Based on max 96% contribution of volcanic sulfate to total deposited sulfate 
    # (this is the max in Will's excel, maybe there is a better way to estimate?)
    this_laki_modelled["f_volc"] = this_laki_modelled["so4_removed"]/np.nanmax(this_laki_modelled["so4_removed"])*0.96
    this_laki_modelled["d34S_so4_tot"] = this_laki_modelled["d34S_so4_pool"]*this_laki_modelled["f_volc"] + d34S_bcg*(1-this_laki_modelled["f_volc"])
    this_laki_modelled["d33S_so4_tot"] = this_laki_modelled["d33S_so4_pool"]*this_laki_modelled["f_volc"] + d33S_bcg*(1-this_laki_modelled["f_volc"])
    this_laki_modelled["D33S_so4_tot"] = this_laki_modelled["D33S_so4_pool"]*this_laki_modelled["f_volc"] + D33S_bcg*(1-this_laki_modelled["f_volc"])

    # Save as csv and to dict
    this_laki_modelled.to_csv("output/"+runname+"/iterations_individual/"+runname+"_"+str(i)+"_model_output.csv")
    laki_its[str(i)] = this_laki_modelled
    
# Find average and stdev of all the runs
laki_mod_mean = laki_modelled.copy()
laki_mod_sd = laki_modelled.copy()
for c in laki_modelled.columns:
    tmp = np.zeros((laki_modelled.shape[0],n_its))
    for i in np.arange(n_its):
        tmp[:,i] = laki_its[str(i)][c]
    laki_mod_mean[c] = np.nanmean(tmp,axis=1)
    laki_mod_sd[c] = np.nanstd(tmp,axis=1)

# Match depth as best as possible... (0.05 m is roughly a month)
laki_start_depth = [ 60.10, 60.40 ] # Range for the start of the eruption
laki_end_depth = [ 59.35, 59.65 ] # Range for the end of the eruption
depth_steps = 0.01 # Move depth by this much per iteration
laki_start_depth_block = np.arange(laki_start_depth[0],laki_start_depth[1]+depth_steps,depth_steps)
laki_end_depth_block = np.arange(laki_end_depth[0],laki_end_depth[1]+depth_steps,depth_steps)
laki_start_depth_full = np.tile(laki_start_depth_block,len(laki_end_depth_block))
laki_end_depth_full = np.repeat(laki_end_depth_block,len(laki_start_depth_block))
laki_depth_res = np.zeros((len(laki_end_depth_full),4))+np.nan
for n in np.arange(0,len(laki_end_depth_full)):
    thisdepth = np.linspace(laki_start_depth_full[n],laki_end_depth_full[n],timesteps)
    cols = ['d34S_so4_tot', 'd33S_so4_tot','D33S_so4_tot']
    laki_mod_interp = pd.DataFrame(np.zeros((laki_data.shape[0],len(cols)*2)))
    laki_mod_interp.columns = cols + [ c + "_sd" for c in cols]
    for c in cols:
        laki_mod_interp[c] = np.interp(-laki_data["depth_m"],-thisdepth,laki_mod_mean[c])
        laki_mod_interp[c+"_sd"] = np.interp(-laki_data["depth_m"],-thisdepth,laki_mod_sd[c])
    laki_mod_resid = laki_mod_interp.copy()
    laki_mod_resid['d34S_so4_tot'] = laki_mod_interp['d34S_so4_tot'] - laki_data["d34S_permil"]
    laki_mod_resid['d34S_so4_tot_sd'] = ( laki_mod_interp['d34S_so4_tot_sd']**2 + laki_data["d34S_unc_permil"]**2 )**0.5
    laki_mod_resid['d33S_so4_tot'] = laki_mod_interp['d33S_so4_tot'] - laki_data["d33S_permil"]
    laki_mod_resid['d33S_so4_tot_sd'] = ( laki_mod_interp['d33S_so4_tot_sd']**2 + laki_data["d33S_unc_permil"]**2 )**0.5
    laki_mod_resid['D33S_so4_tot'] = laki_mod_interp['D33S_so4_tot'] - laki_data["D33S_permil"]
    laki_mod_resid['D33S_so4_tot_sd'] = ( laki_mod_interp['D33S_so4_tot_sd']**2 + laki_data["D33S_unc_permil"]**2 )**0.5
    rmse_d34S = ( np.nansum((laki_mod_resid.loc[laki_data["bcg"]==0,"d34S_so4_tot"])**2)/(sum(laki_data["bcg"]==0)) )**0.5
    rmse_D33S = ( np.nansum((laki_mod_resid.loc[laki_data["bcg"]==0,"D33S_so4_tot"])**2)/(sum(laki_data["bcg"]==0)) )**0.5
    laki_depth_res[n,0] = rmse_d34S
    laki_depth_res[n,1] = rmse_D33S
    laki_depth_res[n,2] = laki_start_depth_full[n]
    laki_depth_res[n,3] = laki_end_depth_full[n]
rmse_d34S_argsort = np.argsort(laki_depth_res[:,0])
rmse_d34S_rankings = np.array([ np.where(rmse_d34S_argsort==n)[0][0] for n in np.arange(0,len(laki_end_depth_full)) ])
rmse_D33S_argsort = np.argsort(laki_depth_res[:,1])
rmse_D33S_rankings = np.array([ np.where(rmse_D33S_argsort==n)[0][0] for n in np.arange(0,len(laki_end_depth_full)) ])
overall_rankings = np.argsort(rmse_d34S_rankings+rmse_D33S_rankings)
tmp = laki_depth_res[overall_rankings,:]
laki_start_depth_best = tmp[0,2] # Best start depth for matching
laki_end_depth_best = tmp[0,3] # Best start depth for matching
laki_modelled["depth"] = np.linspace(laki_start_depth_best,laki_end_depth_best,timesteps)
laki_mod_mean["depth"] = laki_modelled["depth"]
laki_mod_sd["depth"] = laki_modelled["depth"]

# Save
laki_mod_mean.to_csv("output/"+runname+"/"+runname+"_"+str(i)+"_model_output.csv")
laki_mod_sd.to_csv("output/"+runname+"/"+runname+"_"+str(i)+"_model_output_sd.csv")

# Find model - data residuals and stats and save/print
cols = ['d34S_so4_tot', 'd33S_so4_tot','D33S_so4_tot']
laki_mod_interp = pd.DataFrame(np.zeros((laki_data.shape[0],len(cols)*2)))
laki_mod_interp.columns = cols + [ c + "_sd" for c in cols]
for c in cols:
    laki_mod_interp[c] = np.interp(-laki_data["depth_m"],-laki_mod_mean["depth"],laki_mod_mean[c])
    laki_mod_interp[c+"_sd"] = np.interp(-laki_data["depth_m"],-laki_mod_mean["depth"],laki_mod_sd[c])
laki_mod_resid = laki_mod_interp.copy()
laki_mod_resid['d34S_so4_tot'] = laki_mod_interp['d34S_so4_tot'] - laki_data["d34S_permil"]
laki_mod_resid['d34S_so4_tot_sd'] = ( laki_mod_interp['d34S_so4_tot_sd']**2 + laki_data["d34S_unc_permil"]**2 )**0.5
laki_mod_resid['d33S_so4_tot'] = laki_mod_interp['d33S_so4_tot'] - laki_data["d33S_permil"]
laki_mod_resid['d33S_so4_tot_sd'] = ( laki_mod_interp['d33S_so4_tot_sd']**2 + laki_data["d33S_unc_permil"]**2 )**0.5
laki_mod_resid['D33S_so4_tot'] = laki_mod_interp['D33S_so4_tot'] - laki_data["D33S_permil"]
laki_mod_resid['D33S_so4_tot_sd'] = ( laki_mod_interp['D33S_so4_tot_sd']**2 + laki_data["D33S_unc_permil"]**2 )**0.5
rmse_d34S = ( np.nansum((laki_mod_resid.loc[laki_data["bcg"]==0,"d34S_so4_tot"])**2)/(sum(laki_data["bcg"]==0)) )**0.5
rmse_D33S = ( np.nansum((laki_mod_resid.loc[laki_data["bcg"]==0,"D33S_so4_tot"])**2)/(sum(laki_data["bcg"]==0)) )**0.5

# Save the residuals RMSE to a master file
this_resid = pd.DataFrame(np.zeros((1,4+len(laki_data["Name"])*2)))
this_resid.columns = ["runname","datetime","rmse_d34S","rmse_D33S"] + [ s+"_d34S" for s in laki_data["Name"] ] + [ s+"_D33S" for s in laki_data["Name"] ]
this_resid["runname"] = runname
this_resid["datetime"] = datetime.now().strftime("%Y%m%d_%H%M%S")
this_resid["rmse_d34S"] = rmse_d34S
this_resid["rmse_D33S"] = rmse_D33S
this_resid[[ s+"_d34S" for s in laki_data["Name"] ]] = laki_mod_resid['d34S_so4_tot']
this_resid[[ s+"_D33S" for s in laki_data["Name"] ]] = laki_mod_resid['D33S_so4_tot']
my_file = Path("output/output_summary.csv")
if my_file.is_file():
    tmp = pd.read_csv(my_file, sep=',', header=0).drop("Unnamed: 0",axis=1)
    tmp = pd.concat([tmp,this_resid])
    tmp.to_csv("output/output_summary.csv")
else:
    this_resid.to_csv("output/output_summary.csv")

# Plot data and background
fig, ax = plt.subplots(4,1,figsize=(8,8))
ax[0].plot(-laki_data["depth_m"],laki_data["depth_m"]*0+d34S_bcg,"g:",label="d34S bcg, modelled") # plot background
ax[0].plot(-laki_mod_mean["depth"],laki_mod_mean["d34S_so4_tot"],"r-",label="d34S volc, modelled") # model
ax[0].plot(-laki_mod_mean["depth"],laki_mod_mean["d34S_so4_tot"]-laki_mod_sd["d34S_so4_tot"],"r:")
ax[0].plot(-laki_mod_mean["depth"],laki_mod_mean["d34S_so4_tot"]+laki_mod_sd["d34S_so4_tot"],"r:")
ax[0].errorbar(-laki_data.loc[laki_data["bcg"]==1,"depth_m"],laki_data.loc[laki_data["bcg"]==1,"d34S_permil"],laki_data.loc[laki_data["bcg"]==1,"d34S_unc_permil"],marker="o",ls="",label="background")
ax[0].errorbar(-laki_data.loc[laki_data["bcg"]==0,"depth_m"],laki_data.loc[laki_data["bcg"]==0,"d34S_permil"],laki_data.loc[laki_data["bcg"]==0,"d34S_unc_permil"],marker="o",ls="",label="volc")
ax[0].set_xlabel("depth")
ax[0].set_xlim([-60.45,-59.7])
ax[0].set_ylabel("d34S_sulfate")
ax[0].legend()

ax[1].plot(-laki_data["depth_m"],laki_data["depth_m"]*0,"g:") # plot 0
ax[1].errorbar(-laki_data.loc[laki_data["bcg"]==1,"depth_m"],laki_mod_resid.loc[laki_data["bcg"]==1,"d34S_so4_tot"],laki_mod_resid.loc[laki_data["bcg"]==1,"d34S_so4_tot_sd"],marker="o",ls="",label="background")
ax[1].errorbar(-laki_data.loc[laki_data["bcg"]==0,"depth_m"],laki_mod_resid.loc[laki_data["bcg"]==0,"d34S_so4_tot"],laki_mod_resid.loc[laki_data["bcg"]==0,"d34S_so4_tot_sd"],marker="o",ls="",label="volc; rmse="+str(np.round(rmse_d34S,3)) )
ax[1].set_xlabel("depth")
ax[1].set_xlim([-60.45,-59.7])
ax[1].set_ylabel("d34S_sulfate, residual")
ax[1].legend()

ax[2].plot(-laki_data["depth_m"],laki_data["depth_m"]*0+D33S_bcg,"g:",label="d34S bcg, modelled")
ax[2].plot(-laki_mod_mean["depth"],laki_mod_mean["D33S_so4_tot"],"r-",label="d34S volc, modelled")
ax[2].plot(-laki_mod_mean["depth"],laki_mod_mean["D33S_so4_tot"]-laki_mod_sd["D33S_so4_tot"],"r:")
ax[2].plot(-laki_mod_mean["depth"],laki_mod_mean["D33S_so4_tot"]+laki_mod_sd["D33S_so4_tot"],"r:")
ax[2].errorbar(-laki_data.loc[laki_data["bcg"]==1,"depth_m"],laki_data.loc[laki_data["bcg"]==1,'D33S_permil'],laki_data.loc[laki_data["bcg"]==1,"D33S_unc_permil"],marker="o",ls="",label="background")
ax[2].errorbar(-laki_data.loc[laki_data["bcg"]==0,"depth_m"],laki_data.loc[laki_data["bcg"]==0,'D33S_permil'],laki_data.loc[laki_data["bcg"]==0,"D33S_unc_permil"],marker="o",ls="",label="volc")
ax[2].set_xlabel("depth")
ax[2].set_xlim([-60.45,-59.7])
ax[2].set_ylabel("D33S_sulfate")

ax[3].plot(-laki_data["depth_m"],laki_data["depth_m"]*0,"g:") # plot 0
ax[3].errorbar(-laki_data.loc[laki_data["bcg"]==1,"depth_m"],laki_mod_resid.loc[laki_data["bcg"]==1,"D33S_so4_tot"],laki_mod_resid.loc[laki_data["bcg"]==1,"D33S_so4_tot_sd"],marker="o",ls="",label="background")
ax[3].errorbar(-laki_data.loc[laki_data["bcg"]==0,"depth_m"],laki_mod_resid.loc[laki_data["bcg"]==0,"D33S_so4_tot"],laki_mod_resid.loc[laki_data["bcg"]==0,"D33S_so4_tot_sd"],marker="o",ls="",label="volc; rmse="+str(np.round(rmse_D33S,3)) )
ax[3].set_xlabel("depth")
ax[3].set_xlim([-60.45,-59.7])
ax[3].set_ylabel("D33S_sulfate, residual")
ax[3].legend()
fig.tight_layout()
plt.savefig("figs/"+runname+"/model-volc_d34S_d33S_"+runname+".pdf") 
plt.show()

# Plot other model params
fig, ax = plt.subplots(5,1,figsize=(6,8))
ax[0].plot(laki_mod_mean["t"],laki_mod_mean["so2_emitted"],"b-",label="so2_emitted")
ax[0].plot(laki_mod_mean["t"],laki_mod_mean["so2_emitted"]-laki_mod_sd["so2_emitted"],"b:")
ax[0].plot(laki_mod_mean["t"],laki_mod_mean["so2_emitted"]+laki_mod_sd["so2_emitted"],"b:")
ax[0].plot(laki_mod_mean["t"],laki_mod_mean['so2_oxidised'],"r-",label='so2_oxidised')
ax[0].plot(laki_mod_mean["t"],laki_mod_mean["so2_oxidised"]-laki_mod_sd["so2_oxidised"],"r:")
ax[0].plot(laki_mod_mean["t"],laki_mod_mean["so2_oxidised"]+laki_mod_sd["so2_oxidised"],"r:")
ax[0].set_ylabel("so2 (~megatons)")
ax[0].legend()
ax[1].plot(laki_mod_mean["t"],laki_mod_mean['so2_pool'],"r-",label='so2_pool')
ax[1].plot(laki_mod_mean["t"],laki_mod_mean["so2_pool"]-laki_mod_sd["so2_pool"],"r:")
ax[1].plot(laki_mod_mean["t"],laki_mod_mean["so2_pool"]+laki_mod_sd["so2_pool"],"r:")
ax[1].plot(laki_mod_mean["t"],laki_mod_mean['so4_pool'],"b-",label='so4_pool')
ax[1].plot(laki_mod_mean["t"],laki_mod_mean["so4_pool"]-laki_mod_sd["so4_pool"],"b:")
ax[1].plot(laki_mod_mean["t"],laki_mod_mean["so4_pool"]+laki_mod_sd["so4_pool"],"b:")
ax[1].plot(laki_mod_mean["t"],laki_mod_mean['so4_removed'],'c',label='so4_removed')
ax[1].plot(laki_mod_mean["t"],laki_mod_mean["so4_removed"]-laki_mod_sd["so4_removed"],"c:")
ax[1].plot(laki_mod_mean["t"],laki_mod_mean["so4_removed"]+laki_mod_sd["so4_removed"],"c:")
ax[1].set_ylabel("so2 (~megatons)")
ax[1].legend()
ax[2].plot(laki_mod_mean["t"],np.cumsum(laki_mod_mean['so2_emitted']),"r-",label='cumulative so2 emissions')
ax[2].plot(laki_mod_mean["t"],np.cumsum(laki_mod_mean['so4_removed']),"b-",label='cumulative so4 removal')
ax[2].set_ylabel("so4 (~megatons)")
ax[2].legend()
ax[3].plot(laki_mod_mean["t"],laki_mod_mean['f_volc'],"b-",label='f_volc')
ax[3].plot(laki_mod_mean["t"],laki_mod_mean["f_volc"]-laki_mod_sd["f_volc"],"b:")
ax[3].plot(laki_mod_mean["t"],laki_mod_mean["f_volc"]+laki_mod_sd["f_volc"],"b:")
ax[3].set_ylabel("fraction of SO4 from volc")
ax[3].legend()
ax[4].plot(laki_mod_mean["t"],laki_mod_mean['d34S_so2_pool']*0+d34S_volcSO2[0],"g--",label='so2_volc_emit')
ax[4].plot(laki_mod_mean["t"],laki_mod_mean['d34S_so2_pool']*0+d34S_bcgSO2[0],"k:",label='so2_bcg_emit')
ax[4].plot(laki_mod_mean["t"],laki_mod_mean['d34S_so2_pool'],"b-",label='so2_pool')
ax[4].plot(laki_mod_mean["t"],laki_mod_mean['d34S_so4_pool'],"m-",label='so4_pool')
ax[4].plot(laki_mod_mean["t"],laki_mod_mean["d34S_so4_tot"],"r-",label='so4_tot')
ax[4].set_ylabel("d34S")
ax[4].set_ylim([-30,20])
ax[4].legend()
ax[4].set_xlabel("days since eruption")
fig.tight_layout()
plt.savefig("figs/"+runname+"/model-params_"+runname+".pdf") 
plt.show()

# Plot oxidation pathways
fig, ax = plt.subplots(2,1,figsize=(12,6))
ax[0].plot(ox_pathways["fraction_of_volc"]*int(full_volc_length*365),ox_pathways["OH"],"rx",label="OH-input")
ax[0].plot(ox_pathways["fraction_of_volc"]*int(full_volc_length*365),ox_pathways["H2O2"],"gx",label="H2O2-input")
ax[0].plot(ox_pathways["fraction_of_volc"]*int(full_volc_length*365),ox_pathways["TMI"],"bx",label="TMI-input")
ax[0].plot(laki_mod_mean["t"],pathways_volc["OH"],"r-",label="OH")
ax[0].plot(laki_mod_mean["t"],pathways_volc["H2O2"],"g-",label="H2O2")
ax[0].plot(laki_mod_mean["t"],pathways_volc["TMI"],"b-",label="TMI")
ax[0].set_xlabel("Progression of eruption")
ax[0].set_ylabel("Fraction of SO2 oxidation")
ax[0].legend()
ax[1].plot(laki_mod_mean["t"],laki_mod_mean['so2_oxidised']*pathways_volc["OH"],"r-",label="OH")
ax[1].plot(laki_mod_mean["t"],laki_mod_mean['so2_oxidised']*pathways_volc["H2O2"],"g-",label="H2O2")
ax[1].plot(laki_mod_mean["t"],laki_mod_mean['so2_oxidised']*pathways_volc["TMI"],"b-",label="TMI")
ax[1].set_xlabel("Progression of eruption")
ax[1].set_ylabel("so2 oxidised (~megatons)")
ax[1].legend()
fig.tight_layout()
plt.savefig("figs/"+runname+"/ox_pathways_"+runname+".pdf") 
plt.show()