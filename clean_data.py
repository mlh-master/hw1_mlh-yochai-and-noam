# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    CTG_features_no_DR=CTG_features.drop(columns=['DR'])
    columns = list(CTG_features_no_DR)
    c_ctg={}
    for i in columns:
        c_ctg[i]=pd.to_numeric(CTG_features_no_DR[i], errors='coerce').dropna().values
        # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    c_cdf = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    CTG_features_no_DR = CTG_features.drop(columns=['DR'])
    columns = list(CTG_features_no_DR)
    for i in columns:
        # printing the third element of the column
        c_cdf[i] = pd.to_numeric(CTG_features_no_DR[i], errors='coerce').fillna(np.random.choice(CTG_features_no_DR[i].values)).values
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary={}
    columns = list(c_feat)
    for col in columns:
        #d_summay[col]={c_feat[col].describe().iloc[1:5].to_dict()}
        d_summary[col]={"min":c_feat[col].min(),
                        "Q1":c_feat[col].quantile(0.25),
                        "median": c_feat[col].median(),
                        "Q3":c_feat[col].quantile(0.75),
                        "max":c_feat[col].max()}

    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    columns = list(c_feat)
    for col in columns:
        IQR=d_summary[col]["Q3"]-d_summary[col]["Q1"]
        Q_MIN=d_summary[col]["Q1"]-1.5*IQR
        Q_MAX = d_summary[col]["Q3"] + 1.5 * IQR
        c_no_outlier[col]=c_feat[col][(c_feat[col]>=Q_MIN) & (c_feat[col]<=Q_MAX)].values
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)


def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    filt_feature= c_cdf[feature][ c_cdf[feature] <= thresh].values
    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    nsd_res={}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    columns = list(CTG_features)
        # nsd_res[x]=(CTG_features[x].values-CTG_features[x].min())/(CTG_features[x].max()-CTG_features[x].min())
        # nsd_res[y] = (CTG_features[y].values - CTG_features[y].min()) / (CTG_features[y].max() - CTG_features[y].min())
    for col in columns:
        if mode == "none":
            nsd_res[col]=CTG_features[col].values
        if mode == "MinMax":
            nsd_res[col] =(CTG_features[col].values-CTG_features[col].min())/(CTG_features[col].max()-CTG_features[col].min())
        if mode == "mean":
            nsd_res[col] = (CTG_features[col].values - CTG_features[col].mean()) / (CTG_features[col].max() - CTG_features[col].min())
        if mode=='standard':
            nsd_res[col]=(CTG_features[col].values-CTG_features[col].mean())/CTG_features[col].std()
    if flag==True:
        nsd_res=pd.DataFrame(nsd_res)
        nsd_res[x].hist(bins=50)
        nsd_res[y].hist(bins=50)
        plt.xlabel('Histogram Width')
        plt.ylabel('Count')
        plt.show()
    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)

