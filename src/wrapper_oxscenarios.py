
# Wrapper to generate and run many different ox scenarios

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

#%% Get base run name (from command line input)
if (len(sys.argv)>1) & (len(sys.argv[1])>0):
    baserunname = sys.argv[1]
    print("\nName of run:", baserunname)
else:
    print("No run name given, exiting.")
    sys.exit()
if 0:
    baserunname = "testconfigs_looptest"
    
# Check con file exists
configfile = parentdir+'/configs/configs_'+baserunname+'.csv'
if not os.path.isfile(configfile):
    print("No config file for this run, exiting.")
    sys.exit()
    
# Remove previous ouput summary for loops
os.remove("output/output_summary_loops.csv")
    
#%% Start loop here
n_iterations = 2000
params = np.zeros((n_iterations,9))
allrunnames = []
for n in np.arange(n_iterations):
    print("Running iteration "+str(n)+" of "+str(n_iterations))
    start_ox = np.random.dirichlet([1, 1, 1]) # A dirichlet samples within the simplex to give uniform dist for all 3 numbers
    oh_start = start_ox[0]
    h2o2_start = start_ox[1]
    tmi_start = start_ox[2]
    end_ox = np.random.dirichlet([1, 1, 1]) 
    oh_end = end_ox[0]
    h2o2_end = end_ox[1]
    tmi_end = end_ox[2]
    shift_start = np.random.uniform(0, 1)
    shift_end = np.random.uniform(shift_start, 1)
    tmp = [n,oh_start,h2o2_start,tmi_start,oh_end,h2o2_end,tmi_end,shift_start,shift_end]
    tmp = [ round(v,3) for v in tmp ]
    params[n,:] = tmp
        
    # Generate specific run name 
    timestamp = dt.now().strftime("%y%m%d_%H%M%S")
    runname = baserunname + "_" + timestamp
    allrunnames = allrunnames + [runname]

    # Create con file for this run
    shutil.copy2(parentdir+'/configs/configs_'+baserunname+'.csv', parentdir+'/configs/configs_'+runname+'.csv')
        
    # Create ox scenarios 
    n_values = 51
    ox = pd.DataFrame(np.zeros((n_values,4)))
    ox.columns = ["fraction_of_volc","OH","H2O2","TMI" ]
    ox["fraction_of_volc"] = np.linspace(0,1,n_values)
    r = np.where( (ox["fraction_of_volc"]>=shift_start) & (ox["fraction_of_volc"]<=shift_end) )[0] # where the shift in ox pathways happens
    oh = np.full(n_values, oh_start, dtype=float)
    oh[r] = np.linspace(oh_start,oh_end,len(r))
    oh[ox["fraction_of_volc"]>shift_end] = oh_end
    ox["OH"] = oh
    h2o2 = np.full(n_values, h2o2_start, dtype=float)
    h2o2[r] = np.linspace(h2o2_start,h2o2_end,len(r))
    h2o2[ox["fraction_of_volc"]>shift_end] = h2o2_end
    ox["H2O2"] = h2o2
    tmi = np.full(n_values, tmi_start, dtype=float)
    tmi[r] = np.linspace(tmi_start,tmi_end,len(r))
    tmi[ox["fraction_of_volc"]>shift_end] = tmi_end
    ox["TMI"] = tmi

    ox.to_csv(parentdir+'/configs/oxpathways_'+runname+'.csv', index=False)

    #%% Run the model
    runcode = "python ./src/s_model_unc_newbcg_config_opt_looprun.py " + runname
    os.system(runcode)
        
    #%% Move the config and ox pathways to the appropriate loop folder
    configdir = parentdir+'/configs/loopconfigs'
    shutil.move(parentdir+'/configs/configs_'+runname+'.csv', configdir+'/configs_'+runname+'.csv')
    shutil.move(parentdir+'/configs/oxpathways_'+runname+'.csv', configdir+'/oxpathways_'+runname+'.csv')

params_df = pd.DataFrame(params)
params_df.columns = ["runname","oh_start","h2o2_start","tmi_start","oh_end","h2o2_end","tmi_end","shift_start","shift_end"]
params_df["runname"] = allrunnames
params_df.to_csv("output/loops/loopparams_"+allrunnames[0],index = False)