import pandas as pd
import os
import numpy as np
from stargazer.stargazer import Stargazer
from scipy.stats import ttest_ind
from tabulate import tabulate
import statsmodels.formula.api as smf

from auxiliary.auxiliary_attendance_datasets import *

############################################################################################################

def get_variable_description(df):
    df_var = {"Variable": df.columns.values, "Type": df.dtypes, 
              "Description":["unique student identifier","cohorts ranging from 2008 to 2014",
                             "attendance policy for above-7 students",
                             "second-year grades",
                             "second-year grades standardized","stdgrade of the abolition cohort",
                             "attendance rate of tutorials", "first-year average grade",
                             "indicator if first-year gpa was below 7",
                             "first-year average grade centered around 7",
                             "indicator if a second-year course was passed"]}
    df_var = pd.DataFrame(df_var, columns=["Variable","Type","Description"])
    return df_var.style.hide_index()


############################################################################################################

def get_table1(df,bandwidth):
    
    ### mean and differences:
    att_mean0,att_mean1 = round(df.groupby("treat")["attendance"].mean(),3)
    att_diff = round(att_mean0 - att_mean1,3)
    grade_mean0,grade_mean1 = round(df.groupby("treat")["grade"].mean(),3)
    grade_diff = round(grade_mean0 - grade_mean1,3)
    
    ### standard deviations:
    att_sd0, att_sd1 = round(df.groupby("treat")["attendance"].std(),3)
    grade_sd0, grade_sd1 = round(df.groupby("treat")["grade"].std(),3)
    
    ### number of observations:
    num_ind = df["treat"].value_counts()
    num_ind[2] = num_ind[1] + num_ind[0] 
    
    ### t-test of different means:
    diff_stat = np.empty(2)
    var = ["attendance","grade"]
    for i,column in enumerate(var):
        treated = df.query("treat == 1")[column]
        control = df.query("treat == 0")[column]
        diff_stat[i] = round(ttest_ind(treated, control)[1],3)
    
    ### table:
    info = [["Variable   |   First-year GPA:",f" [{7-bandwidth},7.0]",f"[7,{7+bandwidth}]","Difference","p-value"] 
            ,["Course level (second year)","","","",""]
            ,["Grade","","","",""]
            ,["mean",grade_mean1,grade_mean0,grade_diff,diff_stat[1]]
            ,["standard deviation",grade_sd1,grade_sd0,"",""]
            ,["","","","","",""]
            ,["Tutorial attendance","","","",""]
            ,["mean",round(att_mean1,3),round(att_mean0,3),att_diff,diff_stat[0]]
            ,["standard deviation",round(att_sd1,3),round(att_sd0,3),"",""]
            ,["","","","","",""]
            ,["Observations",num_ind[1],num_ind[0],num_ind[2],""]]
    
    print(tabulate(info ,headers='firstrow',tablefmt='fancy_grid'))    

    
    
    
#############################################################################################################

def get_table3(df):
    
    ### regressions:
    rslt = smf.ols(formula="stdgrade ~ treat + pol1+ pol1t", data=df, weights=df["kwgt"]).fit(cov_type='cluster',cov_kwds={'groups': df["studentid"]})
    rslt1 = rslt
    
    formula2 = "stdgrade ~ treat + treatmentvol + treatmentfor + volcourse + forcourse + pol1 + pol1t + pol1vol + pol1tvol + pol1for + pol1tfor"
    rslt = smf.ols(formula=formula2, data=df,weights=df["kwgt"] ).fit(cov_type='cluster',cov_kwds={'groups': df["studentid"]})
    rslt2 = rslt
    
    ### Table stargazer:
    stargazer = Stargazer([rslt1,rslt2])
    stargazer.custom_columns(["column 1","column 4" ], [1,1])
    stargazer.title("Table 3 - Effects on standardized grades")
    stargazer.show_model_numbers(False)
    stargazer.significant_digits(2)
    stargazer.covariate_order([ "treat","treatmentvol","treatmentfor"])
    stargazer.rename_covariates({"treat": "1st-year GPA is below 7",
                                 "treatmentvol":"Attendance is voluntary x treatment",
                                 "treatmentfor":"Absence is penalized x treatment"})
    stargazer.show_degrees_of_freedom(False)
    stargazer.add_line('Fixed Effects', ['No', 'No'])

    return stargazer


###############################################################################################


def get_table4(rslt,coursetype):

    rslt = np.round(rslt,3)

    info = [["Variable","Treatment Effect","Stand. Error","p-value","Observations"] 
            ,["Attendance rate","","","",""]
            ,["1st-year GPA below 7",rslt[0,0],rslt[0,1],rslt[0,2],rslt[0,3]]
            ,["","","","",""]
            ,["Grade (standardized)","","","",""]
            ,["1st-year GPA below 7",rslt[1,0],rslt[1,1],rslt[1,2],rslt[1,3]]
            ,["","","","",""]
            ,["Passes course","","","",""]
            ,["1st-year GPA below 7",rslt[2,0],rslt[2,1],rslt[2,2],rslt[2,3]]]
    print(f"Table 4 - {coursetype} course type")    
    print(tabulate(info ,headers='firstrow',tablefmt='fancy_grid'))    
    

    
    
#######################################################################################################

def get_table5(df_c15, df_c6):
    
    av11 = round(df_c15["grade"].loc[df_c15["firstyeargpa"].between(6.9,6.9999)].mean(),3)
    av12 = round(df_c15["grade"].loc[df_c15["firstyeargpa"].between(7,7.1)].mean(),3)

    n11 = len(df_c15["grade"].loc[df_c15["firstyeargpa"].between(6.9,6.9999)])
    n12 = len(df_c15["grade"].loc[df_c15["firstyeargpa"].between(7,7.1)])

    av21 = round(df_c6["grade"].loc[df_c6["firstyeargpa"].between(6.9,6.9999)].mean(),3)
    av22 = round(df_c6["grade"].loc[df_c6["firstyeargpa"].between(7,7.1)].mean(),3)

    n21 = len(df_c6["grade"].loc[df_c6["firstyeargpa"].between(6.9,6.9999)])
    n22 = len(df_c6["grade"].loc[df_c6["firstyeargpa"].between(7,7.1)])
    
    # (1.) difference within cohort 1-5:
    treated = df_c15["grade"].loc[df_c15["firstyeargpa"].between(6.9,6.9999)]
    control = df_c15["grade"].loc[df_c15["firstyeargpa"].between(7,7.1)]
    diff_stat1 = round(ttest_ind(treated,control)[1],3)
    
    # (2.) difference within cohort 6:
    treated = df_c6["grade"].loc[df_c6["firstyeargpa"].between(6.9,6.9999)]
    control = df_c6["grade"].loc[df_c6["firstyeargpa"].between(7,7.1)]
    diff_stat2 = round(ttest_ind(treated,control)[1],3)
    
    # (3.) difference between cohorts below 7:
    treated = df_c15["grade"].loc[df_c15["firstyeargpa"].between(6.9,6.9999)]
    control = df_c6["grade"].loc[df_c6["firstyeargpa"].between(6.9,6.9999)]
    diff_stat3 = round(ttest_ind(treated,control)[1],3)
    
    # (4.) difference between cohorts above 7:
    treated = df_c15["grade"].loc[df_c15["firstyeargpa"].between(7,7.1)]
    control = df_c6["grade"].loc[df_c6["firstyeargpa"].between(7,7.1)]
    diff_stat4 = round(ttest_ind(treated,control)[1],3)
    
    # Create a table:
    info = [["Cohort   |   First-year GPA:","[6.9,7.0]","[7.0,7.1]","Difference","p-value"], ["2009-2013","","","",""] 
            ,["Second-year Grade Average",av11,av12,round(av12-av11,3),diff_stat1]
            ,["Observations",n11,n12,"",""]
            ,["","","","","",""]
            ,["2014","","","",""]
            ,["Second-year Grade Average",av21,av22,round(av22-av21,3),diff_stat2]
            ,["Observations",n21,n22,"",""]
            ,["","","","","",""]
            ,["Difference between cohorts",round(av21-av11,3),round(av22-av12,3),"" ,"" ]
           , ["p-value",diff_stat3,diff_stat4,"",""]]

    print(tabulate(info ,headers='firstrow',tablefmt='fancy_grid'))
    
    
    
#########################################################################################################################


def get_table_A6(df):
    
    ### get results
    r_vol = get_results_abolition(df,"voluntary")
    r_enc = get_results_abolition(df,"encouraged")
    r_for = get_results_abolition(df,"forced")
    
    ### table
    info = [["Course Type","Treatment Effect","Standard Error","p-value","Observations"]
            ,["Voluntary", r_vol[0], r_vol[1], r_vol[2], r_vol[3]]
            ,["Encouraged", r_enc[0], r_enc[1], r_enc[2], r_enc[3]]
            ,["Forced", r_for[0], r_for[1], r_for[2], r_for[3]]
            ]

    print(tabulate(info ,headers='firstrow',tablefmt='fancy_grid'))
    
    
#############################################################################################################    
    
    
    
def get_table_spec1(df,coursetype,rslt_rdrobust):
    df1 = get_truncated_data(df,0.365,1,coursetype)
    df11 = get_interactionterms(df1)
    df_reg = get_kweights(df11,0.365)
    
    rslt = smf.ols(formula="attendance ~ treat + firstyeargpa_centered + pol1t", data=df_reg, weights=df_reg["kwgt"]).fit(cov_type='cluster',cov_kwds={'groups': df_reg["studentid"]})
    rslt_att_b = rslt.params[1]
    rslt_att_p = rslt.pvalues[1]
    
    rslt = smf.ols(formula="stdgrade ~ treat + firstyeargpa_centered + pol1t", data=df_reg, weights=df_reg["kwgt"]).fit(cov_type='cluster',cov_kwds={'groups': df_reg["studentid"]})
    rslt_grade_b = rslt.params[1]
    rslt_grade_p = rslt.pvalues[1]
    
    rslt = smf.ols(formula="passcourse ~ treat + firstyeargpa_centered + pol1t", data=df_reg, weights=df_reg["kwgt"]).fit(cov_type='cluster',cov_kwds={'groups': df_reg["studentid"]})
    rslt_pass_b = rslt.params[1]
    rslt_pass_p = rslt.pvalues[1]
    
    rslt_att_b = np.round(rslt_att_b,3)
    rslt_att_p = np.round(rslt_att_p,3)
    rslt_grade_b = np.round(rslt_grade_b,3)
    rslt_grade_p = np.round(rslt_grade_p,3)
    rslt_pass_b = np.round(rslt_pass_b,3)
    rslt_pass_p = np.round(rslt_pass_p,3)
    rslt_r = np.round(rslt_rdrobust,3)
    
    info = [["Variable","Treatment Effect","p-value","Treatment Effect (rdrobust)","p-value"]
        ,["Attendance Rate",rslt_att_b,rslt_att_p,rslt_r[0,0],rslt_r[0,2]] 
        ,["Standardized Grades",rslt_grade_b,rslt_grade_p,rslt_r[1,0],rslt_r[1,2]]
        ,["Passes Course",rslt_pass_b,rslt_pass_p,rslt_r[2,0],rslt_r[2,2]]
        ]

    print(tabulate(info ,headers='firstrow',tablefmt='fancy_grid'))
    
    
    
    
##################################################################################################    
    
def get_table_spec2(df,coursetype):
    
    ### get data
    df_temp = get_truncated_data(df,0.365,1,coursetype)
    df_temp1 = get_interactionterms(df_temp)
    df_reg = get_kweights(df_temp1,0.365)
    
    ### create polynomials
    df_reg["X2"] = df_reg["firstyeargpa_centered"]**2
    df_reg["X3"] = df_reg["firstyeargpa_centered"]**3
    df_reg["pol1t2"] = df_reg["X2"]*df_reg["treat"]
    df_reg["pol1t3"] = df_reg["X3"]*df_reg["treat"]
    
    ### linear regression
    rslt = smf.ols(formula="stdgrade ~ treat + firstyeargpa_centered + pol1t", data=df_reg, weights=df_reg["kwgt"]).fit(cov_type='cluster',cov_kwds={'groups': df_reg["studentid"]})
    rslt_b = rslt.params[1]
    rslt_p = rslt.pvalues[1]
    
    ### quadratic regression
    rslt2 = smf.ols(formula="stdgrade ~ treat + firstyeargpa_centered + X2 + pol1t + pol1t2", data=df_reg, weights=df_reg["kwgt"]).fit(cov_type='cluster',cov_kwds={'groups': df_reg["studentid"]})
    rslt2_b = rslt2.params[1]
    rslt2_p = rslt2.pvalues[1]
    
    ### cubic regression
    rslt3 = smf.ols(formula="stdgrade ~ treat + firstyeargpa_centered + X2 + X3 + pol1t + pol1t2 + pol1t3", data=df_reg, weights=df_reg["kwgt"]).fit(cov_type='cluster',cov_kwds={'groups': df_reg["studentid"]})
    rslt3_b = rslt3.params[1]
    rslt3_p = rslt3.pvalues[1]
    
    ### Table
    info = [["Order of Polynomial","Treatment Effect","p-value"]
            ,["Locally Linear",rslt_b,rslt_p] 
            ,["Locally Quadratic",rslt2_b,rslt2_p]
            ,["Locally Cubic",rslt3_b,rslt3_p]
            ]

    print(tabulate(info ,headers='firstrow',tablefmt='fancy_grid'))    
    
    
#################################################################################################################



def get_table_fakecutoff(rslt_input):

    rslt_bw = np.round(rslt_input,3)
    
    info = [["Fake Cutoff at","Treatment Effect","p-value"]
            ,[rslt_bw[0,0],rslt_bw[0,1],rslt_bw[0,2]]
            ,[rslt_bw[1,0],rslt_bw[1,1],rslt_bw[1,2]]
            ,[rslt_bw[2,0],rslt_bw[2,1],rslt_bw[2,2]]
            ,[rslt_bw[3,0],rslt_bw[3,1],rslt_bw[3,2]]
            ]

    print(tabulate(info ,headers='firstrow',tablefmt='fancy_grid'))     
    
    
####################################################################################################

def get_table_spec3(rslt_input):
    
    rslt_bw = np.round(rslt_input,3)
    
    info = [["Bandwidth","Treatment Effect","p-value"]
            ,[rslt_bw[0,0],rslt_bw[0,1],rslt_bw[0,2]]
            ,[rslt_bw[1,0],rslt_bw[1,1],rslt_bw[1,2]]
            ,[rslt_bw[2,0],rslt_bw[2,1],rslt_bw[2,2]]
            ,[rslt_bw[3,0],rslt_bw[3,1],rslt_bw[3,2]]
            ,[rslt_bw[4,0],rslt_bw[4,1],rslt_bw[4,2]]
            ,[rslt_bw[5,0],rslt_bw[5,1],rslt_bw[5,2]]
            ]

    print(tabulate(info ,headers='firstrow',tablefmt='fancy_grid'))
    
    
    
#####################################################################################################


def get_table_spec3_comparison(rslt_lin, rslt_qua, rslt_cub):
    
    rslt_lin = np.round(rslt_lin,3)
    rslt_qua = np.round(rslt_qua,3)
    rslt_cub = np.round(rslt_cub,3)
    
    info = [["Bandwidth","Effect (linear)","p-value","Effect (quadratic)","p-value","Effect (cubic)","p-value"]
            ,[rslt_lin[0,0],rslt_lin[0,1],rslt_lin[0,2],rslt_qua[0,1],rslt_qua[0,2],rslt_cub[0,1],rslt_cub[0,2]]             
            ,[rslt_lin[1,0],rslt_lin[1,1],rslt_lin[1,2],rslt_qua[1,1],rslt_qua[1,2],rslt_cub[1,1],rslt_cub[1,2]]
            ,[rslt_lin[2,0],rslt_lin[2,1],rslt_lin[2,2],rslt_qua[2,1],rslt_qua[2,2],rslt_cub[2,1],rslt_cub[2,2]]
            ,[rslt_lin[3,0],rslt_lin[3,1],rslt_lin[3,2],rslt_qua[3,1],rslt_qua[3,2],rslt_cub[3,1],rslt_cub[3,2]]
            ,[rslt_lin[4,0],rslt_lin[4,1],rslt_lin[4,2],rslt_qua[4,1],rslt_qua[4,2],rslt_cub[4,1],rslt_cub[4,2]]
            ,[rslt_lin[5,0],rslt_lin[5,1],rslt_lin[5,2],rslt_qua[5,1],rslt_qua[5,2],rslt_cub[5,1],rslt_cub[5,2]]
            ]
    print(tabulate(info ,headers='firstrow',tablefmt='fancy_grid'))
    