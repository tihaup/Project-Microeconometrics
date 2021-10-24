import pandas as pd
import os
import numpy as np
import statsmodels.formula.api as smf

###############################################################################

def get_data():
    df = pd.read_stata("data/ReplicationDataset_ThePriceofForcedAttendance.dta")
    df["grade"] = df["grade"].astype(float)

    # treatment variable
    df["treat"] = 0      
    df.loc[df["firstyeargpa"] < 7, ["treat"]] = 1
    
    # centered running variable
    df["firstyeargpa_centered"] = -1*(df["firstyeargpa"] - 7)
    
    # pass course variable
    df["passcourse"] = 0
    df.loc[df["grade"] >= 5.5, ["passcourse"]] = 1 
    
    return df

########################################################################

def get_truncated_data(df,bandwidth,cohort,coursetype):
    
    if cohort==1:
        df_temp = df.loc[df["cohort"] < 6]
    elif cohort==6:
        df_temp = df.loc[df["cohort"] == 6]
    elif cohort== "all cohorts":
        pass
    
    
    if bandwidth == "total range":
        pass
    else:
        df_temp = df_temp.loc[df_temp["firstyeargpa"]<=7 + bandwidth] 
        df_temp = df_temp.loc[df_temp["firstyeargpa"]>=7 - bandwidth] 
    
    
    if coursetype == "all courses":
        pass
    elif coursetype in ["voluntary","encouraged","forced"]:
        df_temp = df_temp.loc[df_temp["coursepolicy"]== coursetype] 
    
    
    df_temp.reset_index(inplace=True)
    
    return df_temp
    

#################################################################################

def collect_each_student(df):
    sing_id = [df["studentid"][0]]      # studendid of each student
    sing_gpa = [df["firstyeargpa"][0]]   # firstyeargpa of each student

    for i in range(len(df)-1):
    
        if df["studentid"][i] != df["studentid"][i+1]:
            sing_id.append(df["studentid"][i+1])
            sing_gpa.append(df["firstyeargpa"][i+1])
    
    df_temp = pd.DataFrame(sing_id, columns=["studentid"])
    df_temp["firstyeargpa"] = sing_gpa
    return df_temp    
    

##################################################################################


def get_bins_func(df,variable,cohort,coursetype):     
    mean_loc = np.zeros((20,1))
    numobs_loc = np.zeros((20,1))
    pos_loc = np.zeros((20,1))
    df_temp = get_truncated_data(df,"total range",cohort,coursetype)
    #df.temp = df.loc[df["coursepolicy"] == coursetype]           

    for i, xlow in enumerate(np.arange(6.5,7.5,0.05)):
    
        df_temp1 = df_temp
    
        df_temp1 = df_temp1.loc[df_temp1["firstyeargpa"]>=xlow]
        df_temp1 = df_temp1.loc[df_temp1["firstyeargpa"]< xlow+0.05]
        
        #df.temp1 = df.temp1.loc[df.temp1["attendance"]!=0]
        
        mean_loc[i] = df_temp1[variable].mean()
        numobs_loc[i] = len(df_temp1)
        
    return(mean_loc, numobs_loc)

####################################################################################


def get_interactionterms(df_input):
    
    df = df_input
    
    ### treatment interaction term:
    df["pol1"] = df["firstyeargpa"] - 7
    df["pol1t"] = df["pol1"]*df["treat"]
    
    ### Coursetype indicator:
    df["volcourse"] = 0 
    df.loc[df["coursepolicy"] == "voluntary", ["volcourse"]] = 1 

    df["forcourse"] = 0 
    df.loc[df["coursepolicy"] == "forced", ["forcourse"]] = 1 

    ### Interaction terms: treatment x coursetype
    df["treatmentvol"] = df["treat"]*df["volcourse"]
    df["treatmentfor"] = df["treat"]*df["forcourse"]

    df["pol1vol"] = df["pol1"]*df["volcourse"]
    df["pol1for"] = df["pol1"]*df["forcourse"]

    df["pol1tvol"] = df["pol1t"]*df["volcourse"]
    df["pol1tfor"] = df["pol1t"]*df["forcourse"]
    
    return df



########################################################################################

def get_kweights(df_input,bandwidth):
    
    df = df_input
    df["kwgt"] = (1-abs((df["firstyeargpa"]-7)/bandwidth))
    
    return df

########################################################################################

def get_fakecutoff(df,coursetype,y_var):
    
    rslt_temp = np.zeros((4,3))
    
    for i,c in enumerate([6,8,8.25,9]):
    
        ### data
        df_reg = get_truncated_data(df,"total range",1,coursetype)
    
        ### create running variable centered at fake cutoff and fake treatment variables
        df_reg["X_fake"] = -1*(df_reg["firstyeargpa"] - c)
        df_reg["treat_fake"] = 0 
        df_reg.loc[df_reg["firstyeargpa"] < c, ["treat_fake"]] = 1
        df_reg["treat_X_fake"] = df_reg["treat_fake"] * df_reg["X_fake"]
        df_reg["kwgt_fake"] = 0
        df_reg.loc[abs(df_reg["X_fake"]) <= 0.365, ["kwgt_fake"]] = (1-abs((df["firstyeargpa"]-c)/0.365))
        
        df_reg = df_reg.loc[df_reg["firstyeargpa"]<= c + 0.365] 
        df_reg = df_reg.loc[df_reg["firstyeargpa"]>= c - 0.365]
        
        
        
        
        ### locally linear regression 
        formula = y_var + " ~ treat_fake + X_fake + treat_X_fake"
        rslt = smf.ols(formula=formula, data=df_reg, weights=df_reg["kwgt_fake"]).fit(cov_type='cluster',cov_kwds={'groups': df_reg["studentid"]})
        
        ### save results
        rslt_temp[i,0] = c
        rslt_temp[i,1] = rslt.params[1]
        rslt_temp[i,2] = rslt.pvalues[1]
        #rslt_temp = np.round(rslt_temp,3)
    return rslt_temp    
    
    

###########################################################################################################################


def get_results_abolition(df,coursetype):
    
    rslt_temp = np.zeros((4,1))
    df_temp = get_truncated_data(df,0.365,6,coursetype)
    df_temp1 = get_interactionterms(df_temp)
    df_reg = get_kweights(df_temp1,0.365)
    
    rslt = smf.ols(formula="stdgradeabolition ~ treat + firstyeargpa_centered + pol1t", data=df_reg, weights=df_reg["kwgt"]).fit(cov_type='cluster',cov_kwds={'groups': df_reg["studentid"]})
    rslt_temp[0,0] = rslt.params[1]
    rslt_temp[1,0] = rslt.bse[1]
    rslt_temp[2,0] = rslt.pvalues[1]
    rslt_temp[3,0] = len(df_reg)
    rslt_temp = np.round(rslt_temp,3)
    
    return rslt_temp



######################################################################################################################

def get_bandwidth_results(df,coursetype,y_var):
    ### empty results canvas:
    rslt_temp = np.zeros((6,3))
    
    for i,h in enumerate([0.5, 0.4, 0.365, 0.3, 0.2, 0.1]):
    
        ### data within bandwidth:
        df_temp = get_truncated_data(df,h,1,coursetype)
        df_temp1 = get_interactionterms(df_temp)
        df_reg = get_kweights(df_temp1, h)
        

        ### locally linear regression 
        formula = y_var + " ~ treat + firstyeargpa_centered + pol1t"
        rslt = smf.ols(formula=formula, data=df_reg, weights=df_reg["kwgt"]).fit(cov_type='cluster',cov_kwds={'groups': df_reg["studentid"]})
        
        ### save regression results
        rslt_temp[i,0] = h
        rslt_temp[i,1] = rslt.params[1]
        rslt_temp[i,2] = rslt.pvalues[1]
        
    return rslt_temp 

###############################################################################################################


def get_bandwidth_results2(df,coursetype,y_var):
    ### empty results canvas:
    rslt_temp = np.zeros((6,3))
    
    for i,h in enumerate([0.5, 0.4, 0.365, 0.3, 0.2, 0.1]):
    
        ### data within bandwidth:
        df_temp = get_truncated_data(df,h,1,coursetype)
        df_temp1 = get_interactionterms(df_temp)
        df_reg = get_kweights(df_temp1, h)
        df_reg["X2"] = df_reg["firstyeargpa_centered"]**2
        df_reg["pol1t2"] = df_reg["X2"]*df_reg["treat"]

        ### locally quadratic regression 
        
        formula = y_var + " ~ treat + firstyeargpa_centered + X2 + pol1t + pol1t2"
        #formula = y_var + " ~ treat + firstyeargpa_centered + pol1t"
        rslt = smf.ols(formula=formula, data=df_reg, weights=df_reg["kwgt"]).fit(cov_type='cluster',cov_kwds={'groups': df_reg["studentid"]})
        
        ### save regression results
        rslt_temp[i,0] = h
        rslt_temp[i,1] = rslt.params[1]
        rslt_temp[i,2] = rslt.pvalues[1]
        
    return rslt_temp 


##########################################################################################################


def get_bandwidth_results3(df,coursetype,y_var):
    ### empty results canvas:
    rslt_temp = np.zeros((6,3))
    
    for i,h in enumerate([0.5, 0.4, 0.365, 0.3, 0.2, 0.1]):
    
        ### data within bandwidth:
        df_temp = get_truncated_data(df,h,1,coursetype)
        df_temp1 = get_interactionterms(df_temp)
        df_reg = get_kweights(df_temp1, h)
        df_reg["X2"] = df_reg["firstyeargpa_centered"]**2
        df_reg["X3"] = df_reg["firstyeargpa_centered"]**3
        df_reg["pol1t2"] = df_reg["X2"]*df_reg["treat"]
        df_reg["pol1t3"] = df_reg["X3"]*df_reg["treat"]

        ### locally cubic regression 
        
        formula= y_var + " ~ treat + firstyeargpa_centered + X2 + X3 + pol1t + pol1t2 + pol1t3"
        #formula = y_var + " ~ treat + firstyeargpa_centered + pol1t"
        rslt = smf.ols(formula=formula, data=df_reg, weights=df_reg["kwgt"]).fit(cov_type='cluster',cov_kwds={'groups': df_reg["studentid"]})
        
        ### save regression results
        rslt_temp[i,0] = h
        rslt_temp[i,1] = rslt.params[1]
        rslt_temp[i,2] = rslt.pvalues[1]
        
    return rslt_temp 

######################################################################################################################