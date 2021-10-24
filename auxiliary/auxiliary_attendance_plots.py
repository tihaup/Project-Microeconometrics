import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

from auxiliary.auxiliary_attendance_datasets import *



########################################################################

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
    
    
    
########################################################################

def get_figure1(df,coursetype):
    
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
    att_mean_loc, att_numobs_loc = get_bins_func(df,"attendance",1,coursetype)
    m_left, b_left = np.polyfit(X_L["firstyeargpa"],X_L["attendance"],1)
    m_right, b_right = np.polyfit(X_R["firstyeargpa"],X_R["attendance"],1)
    m3_left2, m2_left2, m1_left2, b_left2 = np.polyfit(X_L3["firstyeargpa"],X_L3["attendance"], 3)
    m3_right2, m2_right2, m1_right2, b_right2 = np.polyfit(X_R3["firstyeargpa"],X_R3["attendance"], 3)
    
    ### stdgrade results
    std_mean_loc, std_numobs_loc = get_bins_func(df,"stdgrade",1,coursetype)
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


    
    
##########################################################################################################################
    
    
    
    
    
def get_figure1_abolition(df,coursetype):
    
    ### get positions of local averages
    pos_loc = np.zeros((20,1))
    for i, xlow in enumerate(np.arange(6.5,7.5,0.05)):
        pos_loc[i,0] = xlow+0.05/2
    
    
    mean1_loc, numobs1_loc = get_bins_func(df,"stdgradeabolition",6,"voluntary")
    mean2_loc, numobs2_loc = get_bins_func(df,"stdgradeabolition",6,"encouraged")
    mean3_loc, numobs3_loc = get_bins_func(df,"stdgradeabolition",6,"forced")
    
    df_reg = get_truncated_data(df,0.365,6,"voluntary")
    X_L = df_reg.loc[df_reg["treat"]==1]
    X_R = df_reg.loc[df_reg["treat"]==0]
    m_left1, b_left1 = np.polyfit(X_L["firstyeargpa"],X_L["stdgradeabolition"],1)
    m_right1, b_right1 = np.polyfit(X_R["firstyeargpa"],X_R["stdgradeabolition"],1)
    
    df_reg = get_truncated_data(df,0.365,6,"encouraged")
    X_L = df_reg.loc[df_reg["treat"]==1]
    X_R = df_reg.loc[df_reg["treat"]==0]
    m_left2, b_left2 = np.polyfit(X_L["firstyeargpa"],X_L["stdgradeabolition"],1)
    m_right2, b_right2 = np.polyfit(X_R["firstyeargpa"],X_R["stdgradeabolition"],1)
    
    df_reg = get_truncated_data(df,0.365,6,"forced")
    X_L = df_reg.loc[df_reg["treat"]==1]
    X_R = df_reg.loc[df_reg["treat"]==0]
    m_left3, b_left3 = np.polyfit(X_L["firstyeargpa"],X_L["stdgradeabolition"],1)
    m_right3, b_right3 = np.polyfit(X_R["firstyeargpa"],X_R["stdgradeabolition"],1)
    
    fig, ax = plt.subplots(1,3,figsize=(20, 7))
    
    plt.subplot(131)
    plt.title("Voluntary Courses")
    ax = plt.scatter(pos_loc, mean1_loc[:,0],s = numobs1_loc[:,0],  facecolors='none', edgecolors='black')
    plt.ylim(-0.5,0.5)
    plt.axvline(x=7, color='r')
    plt.plot(pos_loc[3:10,0], m_left1*pos_loc[3:10,0]+b_left1,color="b")     
    plt.plot(pos_loc[10:18,0], m_right1*pos_loc[10:18,0]+b_right1, color="b")
    plt.xlabel('1st-year GPA')
    plt.ylabel('Grades (standardized)')
    
    plt.subplot(132) 
    plt.title("Encouraged Courses")
    ax = plt.scatter(pos_loc, mean2_loc[:,0],s = numobs2_loc[:,0],  facecolors='none', edgecolors='black')
    plt.ylim(-0.5,0.5)
    plt.axvline(x=7, color='r')
    plt.plot(pos_loc[3:10,0], m_left2*pos_loc[3:10,0]+b_left2,color="b")     
    plt.plot(pos_loc[10:18,0], m_right2*pos_loc[10:18,0]+b_right2, color="b")
    plt.xlabel('1st-year GPA')
    plt.ylabel('Grades (standardized)')
    
    
    plt.subplot(133)
    plt.title("Forced Courses")
    ax = plt.scatter(pos_loc, mean3_loc[:,0],s = numobs3_loc[:,0],  facecolors='none', edgecolors='black')
    plt.ylim(-0.5,0.5)
    plt.axvline(x=7, color='r')
    plt.plot(pos_loc[3:10,0], m_left3*pos_loc[3:10,0]+b_left3,color="b")     
    plt.plot(pos_loc[10:18,0], m_right3*pos_loc[10:18,0]+b_right3, color="b")
    plt.xlabel('1st-year GPA')
    plt.ylabel('Grades (standardized)')
    
#################################################################################################################