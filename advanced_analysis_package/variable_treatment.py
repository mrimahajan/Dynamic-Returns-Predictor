import pandas as pd
import numpy as np
from scipy.stats import skew,kurtosis
from scipy.special import cbrt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chisquare
from matplotlib import pyplot as plt
from pandas.tools.plotting import table
import os
import pickle
from sklearn.linear_model import LinearRegression

## s is column name on which exponential smoothning is to be applied
## high = True when smoothning required above 99 percentile and low when smoothning required below 1 percentile
def exponential_smoothning(data,s,alpha=.5,high=True,low=True):
    p1,p5,p95,p99 = data[s].quantile(.01),data[s].quantile(.05),data[s].quantile(.95),data[s].quantile(.99)
    if alpha<=0 or alpha>=1:
        raise ValueError('alpha should be between 0 and 1')
    data.sort_values(s,ascending=True,inplace=True)
    if high==True:
        def get_st_increasing(array,alpha):
            st = array[0]
            i=1
            while i < len(array):
                st = alpha*array[i]+(1-alpha)*st
                i+=1
            return st
        s0 = get_st_increasing(list(np.array(data.loc[(data[s]>=p95) & (data[s]<=p99) & (data[s].notnull()),s])),alpha)
        
        for i in data[(data[s]>p99) & (data[s].notnull())].index.tolist():
            data.loc[i,s] = alpha*data.loc[i,s] + (1-alpha)*s0
            s0 = data.loc[i,s]
            
        if low==True:
            def get_st_decreasing(array,alpha):
                st = array[-1]
                i=len(array)-2
                while i > 0:
                    st = alpha*array[i+1]+(1-alpha)*st
                    i-=1
                return st
            s0 = get_st_decreasing(list(np.array(data.loc[(data[s]>=p1) & (data[s]<=p5) & (data[s].notnull()),s])),alpha)

            for i in data[(data[s]<p1) & (data[s].notnull())].index.tolist()[::-1]:
                data.loc[i,s] = alpha*data[s].loc[i,s] + (1-alpha)*s0
                s0 = data.loc[i,s]

## s is column on which capping and flooring is to be applied
## a and b are higher and lower limits given by analyst
## high and low are true as per conditional capping and flooring at higher and lower ends
def capping_and_flooring(data,s,a,b,high=True,low=True):
    if high==True:
        data.loc[(data[s]>a) & (data[s].notnull()),s] =a
    if low==True:
        data.loc[(data[s]<b) & (data[s].notnull()),s] = b

# s is categorical column
def make_dummies(data,s):
    conversion_list = []
    df_value = pd.DataFrame()
    df_value['counts'] = data[s].value_counts(dropna=False)
    df_value.sort_values(by='counts',ascending=False,inplace=True)
    for category in df_value.index.tolist()[:-1]:
        conversion_list.append(category)
        data[s+'_dum_'+str(category)] = 0
        data.loc[data[s]==category,s+'_dum_'+str(category)]=1
    return conversion_list

# s is categorical column
def make_dummies_binary(data,s):
    df_value=pd.DataFrame()
    df_value['counts']=data[s].value_counts(dropna=False)
    categories = df_value.index.tolist()
    total_categories = len(categories)
    dummies = len(str(int(bin(total_categories)[2:],10)))
    bin_conv=[]
    for i in range(total_categories):
        bin_conv.append(int(bin(i)[2:],10))
    conv_list=[]
    for i,cat in enumerate(categories):
        conv_list.append((cat,bin_conv[i]))
    for j in range(dummies):
        data[s+'_dum_'+str(j)]=0
        for i,cat in enumerate(categories):
            data.loc[data[s]==cat,s+'_dum_'+str(j)]=bin_conv[i]%10
            bin_conv[i]=bin_conv[i]//10
    return conv_list
        