import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from stargazer.stargazer import Stargazer
from scipy.stats import ttest_ind
from tabulate import tabulate
import statsmodels.formula.api as smf



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


def get_variable_description(df):
    df_var = {"Variable": df.columns.values, "Type": df.dtypes, 
              "Description":["unique student identifier","cohorts ranging from 2008 to 2014",
                             "attendance policy for above-7 students (voluntary, encouraged or forced)",
                             "second-year grades",
                             "second-year grades standardized","stdgrade of the abolition cohort",
                             "attendance rate of tutorials", "first-year average grade",
                             "indicator if first-year gpa was below 7",
                             "first-year average grade centered around 7",
                             "indicator if a second-year course was passed"]}
    df_var = pd.DataFrame(df_var, columns=["Variable","Type","Description"])
    return df_var.style.hide_index()



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
    
    
    
def plot_gpahisto(df,bins):
    plt.figure(figsize=(10, 6))
    plt.hist(df["firstyeargpa"],bins=bins)#,edgecolor = "black")
    plt.axvline(x=7, color='r')
    #plt.axvline(x=6.635, color="orange") 
    #plt.axvline(x=7.335, color="orange")

    #plt.fill_between(x1, 0, 50, color = 'k', alpha = 0.5)

    plt.title("Histogram of First Year GPA")
    plt.xlabel("First-year GPA")
    plt.ylabel("Number of observations")
    
    
    
def get_subsetvol(cohort,df):
    if cohort == "1to5":
        df_temp = df.loc[df["cohort"] < 6]
    if cohort == "6":
        df_temp = df.loc[df["cohort"] == 6]
    
    df_temp = df_temp.loc[df_temp["coursepolicy"]=="voluntary"]
    df_temp.reset_index(inplace=True)
    return df_temp



def get_bins_func(df,variable,coursetype):     
    mean_loc = np.zeros((20,1))
    numobs_loc = np.zeros((20,1))
    pos_loc = np.zeros((20,1))
    df_temp = get_truncated_data(df,"total range",1,coursetype)
    #df.temp = df.loc[df["coursepolicy"] == coursetype]           

    for i, xlow in enumerate(np.arange(6.5,7.5,0.05)):
    
        df_temp1 = df_temp
    
        df_temp1 = df_temp1.loc[df_temp1["firstyeargpa"]>=xlow]
        df_temp1 = df_temp1.loc[df_temp1["firstyeargpa"]< xlow+0.05]
        
        #df.temp1 = df.temp1.loc[df.temp1["attendance"]!=0]
        
        mean_loc[i] = df_temp1[variable].mean()
        numobs_loc[i] = len(df_temp1)
        
    return(mean_loc, numobs_loc)



def get_figure1_1(df,coursetype):
    
    ### get positions of local averages
    pos_loc = np.zeros((20,1))
    for i, xlow in enumerate(np.arange(6.5,7.5,0.05)):
        pos_loc[i,0] = xlow+0.05/2
    
    ### attendance results
    att_mean_loc, att_numobs_loc = get_bins_func(df,"attendance",coursetype)
    m_left, b_left = np.polyfit(pos_loc[3:10,0], att_mean_loc[3:10,0], 1)
    m_right, b_right = np.polyfit(pos_loc[10:18,0], att_mean_loc[10:18,0], 1)
    m3_left2, m2_left2, m1_left2, b_left2 = np.polyfit(pos_loc[:10,0], att_mean_loc[:10,0], 3)
    m3_right2, m2_right2, m1_right2, b_right2 = np.polyfit(pos_loc[10:,0], att_mean_loc[10:,0], 3)
    
    ### stdgrade results
    std_mean_loc, std_numobs_loc = get_bins_func(df,"stdgrade","voluntary")
    m_left1, b_left1 = np.polyfit(pos_loc[3:10,0], std_mean_loc[3:10,0], 1)
    m_right1, b_right1 = np.polyfit(pos_loc[10:18,0], std_mean_loc[10:18,0], 1)
    m3_left22,m2_left22, m1_left22, b_left22 = np.polyfit(pos_loc[:10,0], std_mean_loc[:10,0], 3)
    m3_right22,m2_right22, m1_right22, b_right22 = np.polyfit(pos_loc[10:,0], std_mean_loc[10:,0], 3)
    
    ### plot both graphs
    fig, ax = plt.subplots(1,2,figsize=(14, 6))
    plt.subplots_adjust(wspace=0.5)

    plt.subplot(121)
    plt.title(f"Attendance in {coursetype} courses")
    ax = plt.scatter(pos_loc, att_mean_loc[:,0],s = att_numobs_loc[:,0],  facecolors='black', edgecolors='black')
    plt.ylim(0.4,1)
    plt.axvline(x=7, color='r')
    # plot the locally fitted linear regression line:
    plt.plot(pos_loc[3:10,0], m_left*pos_loc[3:10,0]+b_left,color="b")     
    plt.plot(pos_loc[10:18,0], m_right*pos_loc[10:18,0]+b_right, color="b")
    # plot the locally fitted cubic regression line:
    plt.plot(pos_loc[:10,0], m1_left2*pos_loc[:10,0] + m2_left2*((pos_loc)**2)[:10,0]+ m3_left2*((pos_loc)**3)[:10,0] +b_left2, color="grey")
    plt.plot(pos_loc[10:,0], m1_right2*pos_loc[10:,0] + m2_right2*((pos_loc)**2)[10:,0]+m3_right2*((pos_loc)**3)[10:,0] +b_right2, color="grey")
    plt.xlabel('1st-year GPA')
    plt.ylabel('Attendance rate')

    plt.subplot(122)
    plt.title(f"Grades in {coursetype} courses")
    ax = plt.scatter(pos_loc, std_mean_loc[:,0],s = std_numobs_loc[:,0],  facecolors='black', edgecolors='black')
    plt.ylim(-0.5,0.5)
    plt.axvline(x=7, color='r')
    # plot the locally fitted linear regression line:
    plt.plot(pos_loc[3:10,0], m_left1*pos_loc[3:10,0]+b_left1,color="b")     
    plt.plot(pos_loc[10:18,0], m_right1*pos_loc[10:18,0]+b_right1, color="b") 
    # plot the locally fitted cubic regression line:
    plt.plot(pos_loc[:10,0], m1_left22*pos_loc[:10,0] + m2_left22*((pos_loc)**2)[:10,0]+ m3_left22*((pos_loc)**3)[:10,0] +b_left22, color="grey")
    plt.plot(pos_loc[10:,0], m1_right22*pos_loc[10:,0] + m2_right22*((pos_loc)**2)[10:,0]+ m3_right22*((pos_loc)**3)[10:,0] +b_right22, color="grey")
    plt.xlabel('1st-year GPA')
    plt.ylabel('Grades (standardized)')

    plt.plot()



def get_figure1_2(df,coursetype):
    
    ### get positions of local averages
    pos_loc = np.zeros((20,1))
    for i, xlow in enumerate(np.arange(6.5,7.5,0.05)):
        pos_loc[i,0] = xlow+0.05/2
    
    df_reg = get_truncated_data(df,0.3,1,coursetype)
    X_L = df_reg.loc[df_reg["treat"]==1]
    X_R = df_reg.loc[df_reg["treat"]==0]
    
    df_reg3 = get_truncated_data(df,0.5,1,coursetype)
    X_L3 = df_reg.loc[df_reg["treat"]==1]
    X_R3 = df_reg.loc[df_reg["treat"]==0]
    
    ### attendance results
    att_mean_loc, att_numobs_loc = get_bins_func(df,"attendance",coursetype)
    m_left, b_left = np.polyfit(X_L["firstyeargpa"],X_L["attendance"],1)
    m_right, b_right = np.polyfit(X_R["firstyeargpa"],X_R["attendance"],1)
    m3_left2, m2_left2, m1_left2, b_left2 = np.polyfit(X_L3["firstyeargpa"],X_L3["attendance"], 3)
    m3_right2, m2_right2, m1_right2, b_right2 = np.polyfit(X_R3["firstyeargpa"],X_R3["attendance"], 3)
    
    ### stdgrade results
    std_mean_loc, std_numobs_loc = get_bins_func(df,"stdgrade","voluntary")
    m_left1, b_left1 = np.polyfit(X_L["firstyeargpa"],X_L["stdgrade"],1)
    m_right1, b_right1 = np.polyfit(X_R["firstyeargpa"],X_R["stdgrade"],1)
    m3_left22,m2_left22, m1_left22, b_left22 = np.polyfit(X_L3["firstyeargpa"],X_L3["stdgrade"], 3)
    m3_right22,m2_right22, m1_right22, b_right22 = np.polyfit(X_R3["firstyeargpa"],X_R3["stdgrade"], 3)
    
    ### plot both graphs
    fig, ax = plt.subplots(1,2,figsize=(14, 6))
    plt.subplots_adjust(wspace=0.5)

    plt.subplot(121)
    plt.title(f"Attendance in {coursetype} courses")
    ax = plt.scatter(pos_loc, att_mean_loc[:,0],s = att_numobs_loc[:,0],  facecolors='none', edgecolors='black')
    plt.ylim(0.4,1)
    plt.axvline(x=7, color='r')
    # plot the locally fitted linear regression line:
    plt.plot(pos_loc[3:10,0], m_left*pos_loc[3:10,0]+b_left,color="b")     
    plt.plot(pos_loc[10:18,0], m_right*pos_loc[10:18,0]+b_right, color="b")
    # plot the locally fitted cubic regression line:
    plt.plot(pos_loc[:10,0], m1_left2*pos_loc[:10,0] + m2_left2*((pos_loc)**2)[:10,0]+ m3_left2*((pos_loc)**3)[:10,0] +b_left2, color="grey")
    plt.plot(pos_loc[10:,0], m1_right2*pos_loc[10:,0] + m2_right2*((pos_loc)**2)[10:,0]+m3_right2*((pos_loc)**3)[10:,0] +b_right2, color="grey")
    plt.xlabel('1st-year GPA')
    plt.ylabel('Attendance rate')

    plt.subplot(122)
    plt.title(f"Grades in {coursetype} courses")
    ax = plt.scatter(pos_loc, std_mean_loc[:,0],s = std_numobs_loc[:,0],  facecolors='none', edgecolors='black')
    plt.ylim(-0.5,0.5)
    plt.axvline(x=7, color='r')
    # plot the locally fitted linear regression line:
    plt.plot(pos_loc[3:10,0], m_left1*pos_loc[3:10,0]+b_left1,color="b")     
    plt.plot(pos_loc[10:18,0], m_right1*pos_loc[10:18,0]+b_right1, color="b") 
    # plot the locally fitted cubic regression line:
    plt.plot(pos_loc[:10,0], m1_left22*pos_loc[:10,0] + m2_left22*((pos_loc)**2)[:10,0]+ m3_left22*((pos_loc)**3)[:10,0] +b_left22, color="grey")
    plt.plot(pos_loc[10:,0], m1_right22*pos_loc[10:,0] + m2_right22*((pos_loc)**2)[10:,0]+ m3_right22*((pos_loc)**3)[10:,0] +b_right22, color="grey")
    plt.xlabel('1st-year GPA')
    plt.ylabel('Grades (standardized)')

    plt.plot()


    
    
    
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

def get_kweights(df_input,bandwidth):
    
    df = df_input
    df["kwgt"] = (1-abs((df["firstyeargpa"]-7)/bandwidth))
    
    return df


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
        #print(df_reg[["X_fake","kwgt_fake"]])
        #df_reg["kwgt_fake"] = (1-abs((df["firstyeargpa"]-c)/0.365))
    
        ### locally linear regression 
        formula = y_var + " ~ treat_fake + X_fake + treat_X_fake"
        rslt = smf.ols(formula=formula, data=df_reg, weights=df_reg["kwgt_fake"]).fit(cov_type='cluster',cov_kwds={'groups': df_reg["studentid"]})
        
        ### save results
        rslt_temp[i,0] = c
        rslt_temp[i,1] = rslt.params[1]
        rslt_temp[i,2] = rslt.pvalues[1]
        #rslt_temp = np.round(rslt_temp,3)
    return rslt_temp    
    
    
def get_table_fakecutoff(rslt_input):

    rslt_bw = np.round(rslt_input,3)
    
    info = [["Fake Cutoff at","Treatment Effect","p-value"]
            ,[rslt_bw[0,0],rslt_bw[0,1],rslt_bw[0,2]]
            ,[rslt_bw[1,0],rslt_bw[1,1],rslt_bw[1,2]]
            ,[rslt_bw[2,0],rslt_bw[2,1],rslt_bw[2,2]]
            ,[rslt_bw[3,0],rslt_bw[3,1],rslt_bw[3,2]]
            ]

    print(tabulate(info ,headers='firstrow',tablefmt='fancy_grid'))     
    
    
    
    
    
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



def get_table_spec4(rslt_input):
    
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