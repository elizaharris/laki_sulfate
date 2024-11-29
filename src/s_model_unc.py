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

parentdir = os.getcwd()

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

runname = "test_28112024"

# General conditions
temp = 0 # 0 degC for atmospheric oxidation: Is this okay?
r_34S = 1/22.7 # These are the ratios for IAEA-S-1 but this won't make any difference to the calcs; from https://www.sciencedirect.com/science/article/abs/pii/S0016703701006111
r_33S = 1/126.9
frac_oxidised_bcg_tot = 0.9 # Fraction of background SO2 oxidised: Is this right? Better guesses? If it's too low we can't converge with the background d34S easily

# Emitted SO2 isotopic composition
d34S_bcgSO2 = (5,1) # Much marine sulfur so higher; mixed with volcanic - still needs refinement
d33S_bcgSO2 = (((d34S_bcgSO2[0]/1000+1)**0.515 - 1)*1000,0.01) # D33S = 0
d34S_volcSO2 = (-1,1) # From Will, email
d33S_volcSO2 = (((d34S_volcSO2[0]/1000+1)**0.515 - 1)*1000,0.01) # D33S = 0; is this right?

# 34S fractionation (T dependent): (a,b,c,d) where alpha-1 = (a+b) - (c+d)*T, where T is in deg C
# T dependences from Harris et al. 2013
alpha34_OH_T = (10.60,0.73,0.004,0.015)
alpha34_OH = get_alpha_with_temp(temp,alpha34_OH_T)
alpha34_H2O2_T = (16.51,0.15,-0.085,0.004)
alpha34_H2O2 = get_alpha_with_temp(temp,alpha34_H2O2_T)
alpha34_TMI_T = (-5.039,0.44,-0.237,0.004)
alpha34_TMI = get_alpha_with_temp(temp,alpha34_TMI_T)

# 33S fractionation 
theta33_OH = (0.503,0.007)
theta33_H2O2 = (0.511,0.003)
theta33_TMIcold = (0.498,0.003)
theta33_TMIwarm = (0.537,0.004)

# Details of the eruption
full_volc_length = 1.2 # Years over which the volcanic influence is evident in the ice core; length of simulation
decay_rate = 0.001 # Decay rate for volcanic SO2: This defines how fast the SO2 is removed (see later section for details)
so2_emission_length = 2*30 # days between start of eruption and emissions decreasing strongly (based roughly on Schmidt et al. www.atmos-chem-phys.net/10/6025/2010/ who state that Laki was most vigourous for 1.5 months, and on Fig 2 from THORDARSON AND SELF 2003)
so2_emission_tail_length = 3*30 # days over which SO2 emissions decline linearly to 0
total_so2 = 120 # Tg of SO2 emitted, total (will be evenly spread across the period defined above)
mean_res_time_so2 = 20 # Mean residence time in days (mid of values in Schmidt et al)
mean_res_time_so4 = 8 # Mean residence time in days (mid of values in Schmidt et al)
pathways_volc = [1,0,0] # Fraction of oxidation in the volcanic plume from TMIs, OH, H2O2 

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
    d_OH = rayleigh_product(e=alpha34_OH[0],d_0=d34S_bcgSO2[0],f=f,r_std=r_34S)
    d_OH_low = rayleigh_product(e=alpha34_OH[0]-alpha34_OH[1],d_0=d34S_bcgSO2[0],f=f,r_std=r_34S)
    d_OH_high = rayleigh_product(e=alpha34_OH[0]+alpha34_OH[1],d_0=d34S_bcgSO2[0],f=f,r_std=r_34S)
    d_OH = (d_OH,np.nanmean([abs(d_OH_low-d_OH),abs(d_OH_high-d_OH)])) # mean dOH, uncertainty
    # H2O2 pathway
    f = 1 - frac_OH_H2O2["f_H2O2"].iloc[n]*frac_oxidised_bcg_tot  # Fraction remaining respective to this ox pathway
    d_H2O2 = rayleigh_product(e=alpha34_H2O2[0],d_0=d34S_bcgSO2[0],f=f,r_std=r_34S)
    d_H2O2_low = rayleigh_product(e=alpha34_H2O2[0]-alpha34_H2O2[1],d_0=d34S_bcgSO2[0],f=f,r_std=r_34S)
    d_H2O2_high = rayleigh_product(e=alpha34_H2O2[0]+alpha34_H2O2[1],d_0=d34S_bcgSO2[0],f=f,r_std=r_34S)
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
d34S_bcg = frac_OH_H2O2["d_SO4"].iloc[r]

# Calc 33S for optimised bcg scenario
f = 1 - f_OH*frac_oxidised_bcg_tot
bcg_33S_OH = rayleigh_product_33S(e=alpha34_OH[0],theta=theta33_OH[0],d_0_34=d34S_bcgSO2[0],d_0_33=d33S_bcgSO2[0],f=f,r_std_34=r_34S,r_std_33=r_33S)
f = 1 - f_H2O2*frac_oxidised_bcg_tot
bcg_33S_H2O2 = rayleigh_product_33S(e=alpha34_H2O2[0],theta=theta33_H2O2[0],d_0_34=d34S_bcgSO2[0],d_0_33=d33S_bcgSO2[0],f=f,r_std_34=r_34S,r_std_33=r_33S)
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
plt.savefig("figs/background_d34S_D33S_"+runname+".pdf") 
plt.show()

#%% Look at oxidation pathways during the volcanic period

timesteps = int(full_volc_length*365) # Daily time steps for the ~1.2 years affected (defined at the start)
laki_modelled = pd.DataFrame(np.zeros((timesteps,15)))
laki_modelled.columns = ["t","depth","so2_emitted","so2_pool","so2_oxidised","so4_pool","so4_removed","d34S_so2_pool","d34S_so4_pool","d33S_so4_pool","D33S_so4_pool","f_volc","d34S_so4_tot","d33S_so4_tot","D33S_so4_tot"] 
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
laki_start_depth = 60.25 # Roughly corresponding to the 1.2 years (if you have better data we could change these estimates I made from your figure)
laki_end_depth = 59.5
laki_modelled["depth"] = np.linspace(laki_start_depth,laki_end_depth,timesteps)
so2_per_day = total_so2/(so2_emission_length + so2_emission_tail_length/2) # Max SO2 emissions (for first XX days)

laki_its = dict() # Space for results of iterations
n_its = 50
for i in np.arange(n_its):
    print(i)
    this_laki_modelled = laki_modelled.copy()
    alpha34_volc = pathways_volc[0]*alpha34_TMI[0] + pathways_volc[1]*alpha34_OH[0] + pathways_volc[2]*alpha34_H2O2[0] # First scenario, all volc ox from TMI - Add others later!
    alpha34_volc_i = alpha34_volc + np.random.normal(0,1,1)*(pathways_volc[0]*alpha34_TMI[1] + pathways_volc[1]*alpha34_OH[1] + pathways_volc[2]*alpha34_H2O2[1]) # First scenario, all volc ox from TMI - Add others later!
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
        this_laki_modelled.loc[n,"d34S_so2_pool"] = rayleigh_rem_substrate(e=alpha34_volc_i,d_0=d34S_so2_pool,f=f_remaining,r_std=r_34S)
        this_laki_modelled.loc[n,"d34S_so4_pool"] = rayleigh_product(e=alpha34_volc_i,d_0=d34S_so2_pool,f=f_remaining,r_std=r_34S)
        # D33S
        d_0_33 = ((d34S_so2_pool/1000+1)**0.515 - 1)*1000
        alpha34_TMI_i = alpha34_TMI[0] + np.random.normal(0,1,1)*alpha34_TMI[1]
        theta33_TMIcold_i = theta33_TMIcold[0] + np.random.normal(0,1,1)*theta33_TMIcold[1]     
        alpha34_OH_i = alpha34_OH[0] + np.random.normal(0,1,1)*alpha34_OH[1]
        theta33_OH_i = theta33_OH[0] + np.random.normal(0,1,1)*theta33_OH[1] 
        alpha34_H2O2_i = alpha34_H2O2[0] + np.random.normal(0,1,1)*alpha34_H2O2[1]
        theta33_H2O2_i = theta33_H2O2[0] + np.random.normal(0,1,1)*theta33_H2O2[1] 
        dD33_TMI = rayleigh_product_33S(e=alpha34_TMI_i,theta=theta33_TMIcold_i,d_0_34=d34S_so2_pool,d_0_33=d_0_33,f=f_remaining,r_std_34=r_34S,r_std_33=r_33S)
        dD33_OH = rayleigh_product_33S(e=alpha34_OH_i,theta=theta33_OH_i,d_0_34=d34S_so2_pool,d_0_33=d_0_33,f=f_remaining,r_std_34=r_34S,r_std_33=r_33S)
        dD33_H2O2 = rayleigh_product_33S(e=alpha34_H2O2_i,theta=theta33_H2O2_i,d_0_34=d34S_so2_pool,d_0_33=d_0_33,f=f_remaining,r_std_34=r_34S,r_std_33=r_33S)
        d33S_so4 = pathways_volc[0]*dD33_TMI[0] + pathways_volc[1]*dD33_OH[0] + pathways_volc[2]*dD33_H2O2[0]
        D33S_so4 = pathways_volc[0]*dD33_TMI[1] + pathways_volc[1]*dD33_OH[1] + pathways_volc[2]*dD33_H2O2[1]
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
    this_laki_modelled.to_csv("output/iterations_individual/"+runname+"_"+str(i)+"_model_output.csv")
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
laki_mod_mean.to_csv("output/"+runname+"_"+str(i)+"_model_output.csv")
laki_mod_sd.to_csv("output/"+runname+"_"+str(i)+"_model_output_sd.csv")

# Plot data and background
fig, ax = plt.subplots(2,1,figsize=(12,6))
ax[0].plot(-laki_data["depth_m"],laki_data["depth_m"]*0+d34S_bcg,"g:",label="d34S bcg, modelled") # plot background
ax[0].plot(-laki_mod_mean["depth"],laki_mod_mean["d34S_so4_tot"],"r-",label="d34S volc, modelled") # model
ax[0].plot(-laki_mod_mean["depth"],laki_mod_mean["d34S_so4_tot"]-laki_mod_sd["d34S_so4_tot"],"r:")
ax[0].plot(-laki_mod_mean["depth"],laki_mod_mean["d34S_so4_tot"]+laki_mod_sd["d34S_so4_tot"],"r:")
ax[0].errorbar(-laki_data.loc[laki_data["bcg"]==1,"depth_m"],laki_data.loc[laki_data["bcg"]==1,"d34S_permil"],laki_data.loc[laki_data["bcg"]==1,"d34S_unc_permil"],marker="o",ls="",label="background")
ax[0].errorbar(-laki_data.loc[laki_data["bcg"]==0,"depth_m"],laki_data.loc[laki_data["bcg"]==0,"d34S_permil"],laki_data.loc[laki_data["bcg"]==0,"d34S_unc_permil"],marker="o",ls="",label="volc")
ax[0].set_xlabel("depth")
ax[0].set_ylabel("d34S_sulfate")
ax[0].legend()
ax[1].plot(-laki_data["depth_m"],laki_data["depth_m"]*0+D33S_bcg,"g:",label="d34S bcg, modelled")
ax[1].plot(-laki_mod_mean["depth"],laki_mod_mean["D33S_so4_tot"],"r-",label="d34S volc, modelled")
ax[1].plot(-laki_mod_mean["depth"],laki_mod_mean["D33S_so4_tot"]-laki_mod_sd["D33S_so4_tot"],"r:")
ax[1].plot(-laki_mod_mean["depth"],laki_mod_mean["D33S_so4_tot"]+laki_mod_sd["D33S_so4_tot"],"r:")
ax[1].errorbar(-laki_data.loc[laki_data["bcg"]==1,"depth_m"],laki_data.loc[laki_data["bcg"]==1,'D33S_permil'],laki_data.loc[laki_data["bcg"]==1,"D33S_unc_permil"],marker="o",ls="",label="background")
ax[1].errorbar(-laki_data.loc[laki_data["bcg"]==0,"depth_m"],laki_data.loc[laki_data["bcg"]==0,'D33S_permil'],laki_data.loc[laki_data["bcg"]==0,"D33S_unc_permil"],marker="o",ls="",label="volc")
ax[1].set_xlabel("depth")
ax[1].set_ylabel("D33S_sulfate")
fig.tight_layout()
plt.savefig("figs/model-volc_d34S_d33S_"+runname+".pdf") 
plt.show()

# Plot other model params
fig, ax = plt.subplots(4,1,figsize=(6,8))
ax[0].plot(laki_mod_mean["t"],laki_mod_mean["so2_emitted"],"b-",label="so2_emitted")
ax[0].plot(laki_mod_mean["t"],laki_mod_mean["so2_emitted"]-laki_mod_sd["so2_emitted"],"b:")
ax[0].plot(laki_mod_mean["t"],laki_mod_mean["so2_emitted"]+laki_mod_sd["so2_emitted"],"b:")
ax[0].plot(laki_mod_mean["t"],laki_mod_mean['so2_oxidised'],"r-",label='so2_oxidised')
ax[0].plot(laki_mod_mean["t"],laki_mod_mean["so2_oxidised"]-laki_mod_sd["so2_oxidised"],"r:")
ax[0].plot(laki_mod_mean["t"],laki_mod_mean["so2_oxidised"]+laki_mod_sd["so2_oxidised"],"r:")
ax[0].set_ylabel("so2 (~megatons)")
ax[0].legend()
ax[1].plot(laki_mod_mean["t"],laki_mod_mean['so2_pool'],"b-",label='so2_pool')
ax[1].plot(laki_mod_mean["t"],laki_mod_mean["so2_pool"]-laki_mod_sd["so2_pool"],"b:")
ax[1].plot(laki_mod_mean["t"],laki_mod_mean["so2_pool"]+laki_mod_sd["so2_pool"],"b:")
ax[1].plot(laki_mod_mean["t"],np.cumsum(laki_mod_mean['so2_emitted']),"r-",label='cumulative so2 emissions')
ax[1].set_ylabel("so2 (~megatons)")
ax[1].legend()
ax[2].plot(laki_mod_mean["t"],laki_mod_mean['so4_pool'],"b-",label='so4_pool')
ax[2].plot(laki_mod_mean["t"],laki_mod_mean["so4_pool"]-laki_mod_sd["so4_pool"],"b:")
ax[2].plot(laki_mod_mean["t"],laki_mod_mean["so4_pool"]+laki_mod_sd["so4_pool"],"b:")
ax[2].plot(laki_mod_mean["t"],laki_mod_mean['so4_removed'],'r',label='so4_removed')
ax[2].plot(laki_mod_mean["t"],laki_mod_mean["so4_removed"]-laki_mod_sd["so4_removed"],"r:")
ax[2].plot(laki_mod_mean["t"],laki_mod_mean["so4_removed"]+laki_mod_sd["so4_removed"],"r:")
ax[2].set_ylabel("so4 (~megatons)")
ax[2].legend()
ax[3].plot(laki_mod_mean["t"],laki_mod_mean['f_volc'],"b-",label='f_volc')
ax[3].plot(laki_mod_mean["t"],laki_mod_mean["f_volc"]-laki_mod_sd["f_volc"],"b:")
ax[3].plot(laki_mod_mean["t"],laki_mod_mean["f_volc"]+laki_mod_sd["f_volc"],"b:")
ax[3].set_ylabel("fraction of SO4 from volc")
ax[3].legend()
ax[3].set_xlabel("days since eruption")
fig.tight_layout()
plt.savefig("figs/model-params_"+runname+".pdf") 
plt.show()
