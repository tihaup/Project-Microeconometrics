import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf



def get_data_Rmagic():

    path_data = "../data/ReplicationDataset_ThePriceofForcedAttendance.dta"
    df = pd.read_stata(path_data)
    df["grade"] = df["grade"].astype(float)

    ### Treatment variable and centered running variable
    df["treat"] = 0      
    df.loc[df["firstyeargpa"] < 7, ["treat"]] = 1
    df["firstyeargpa_centered"] = -1*(df["firstyeargpa"] - 7)
    
    ### pass course variable
    df["passcourse"] = 0
    df.loc[df["grade"] >= 5.5, ["passcourse"]] = 1 
    
    return df


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



def get_fakecutoff_val(df,coursetype,c,y_var):
    
    ### data
    df_reg = get_truncated_data(df,"total range",1,coursetype)
    
    ### create running variable centered at fake cutoff and fake treatment variables
    df_reg["X_fake"] = -1*(df_reg["firstyeargpa"] - c)
    df_reg["treat_fake"] = 0 
    df_reg.loc[df_reg["firstyeargpa"] < c, ["treat_fake"]] = 1
    df_reg["treat_X_fake"] = df_reg["treat_fake"] * df_reg["X_fake"]
    df_reg["kwgt_fake"] = 0
    df_reg.loc[abs(df_reg["X_fake"]) <= 0.365, ["kwgt_fake"]] = (1-abs((df["firstyeargpa"]-c)/0.365))

    
    return df_reg
