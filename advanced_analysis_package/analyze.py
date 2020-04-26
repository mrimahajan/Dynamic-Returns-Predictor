import pandas as pd
import numpy as np
from scipy.stats import skew,kurtosis
from scipy.special import cbrt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chisquare
from matplotlib import pyplot as plt
from pandas.tools.plotting import table

def numerical_categorical_division(df):
    	numerical,categorical = [],[]
    	for col in df.columns:
    	    if 'int' in str(df[col].dtypes) or 'float' in str(df[col].dtypes):
    	        numerical.append(col)
    	    else:
    	        categorical.append(col)
    	return numerical,categorical

def edd(data,dv=None,regression=True,percentile=[.01,.05,.1,.5,.9,.95,.99],cv=[2,3]):
    numerical,categorical = numerical_categorical_division(data)
    df_desc = data.describe().transpose()
    df_desc['Var'] = df_desc.index
    df_desc.reset_index(inplace=True)
    df_desc.drop('count',axis=1,inplace=True)
    df_desc['skewness'] = df_desc['Var'].apply(lambda x: skew(np.array(data.loc[data[x].notnull(),x])))
    df_desc['kurtosis'] = df_desc['Var'].apply(lambda x: kurtosis(np.array(data.loc[data[x].notnull(),x]),fisher=False))
    for pct in percentile:
        df_desc['p'+str(int(pct*100))] = df_desc['Var'].apply(lambda x: data[x].quantile(pct))
    
    for dev in cv:
        df_desc['mean-'+str(int(dev))+'sigma'] = df_desc['mean'] - dev*df_desc['std']
        df_desc['mean+'+str(int(dev))+'sigma'] = df_desc['mean'] + dev*df_desc['std']
    
    df_desc['type']='numeric'

    df_categorical = pd.DataFrame()
    df_categorical['Var']=np.array(categorical)
    df_categorical['type']='categorical'
    for col in [c for c in df_desc.columns if c not in ['Var','type']]:
        df_categorical[col]=np.nan
    for col in categorical:
        df_var = data[col].value_counts(ascending=True,dropna=False).cumsum()/data.shape[0]
        df_cat = pd.DataFrame(df_var)
        df_cat.reset_index(inplace=True)
        df_cat.columns = ['categories','cum_pct']
        df_categorical.loc[df_categorical['Var']==col,'min'] = list(df_cat['categories'])[0]
        df_categorical.loc[df_categorical['Var']==col,'max'] = list(df_cat['categories'])[-1]
        for pct in percentile:
            df_categorical.loc[df_categorical['Var']==col,'p'+str(int(pct*100))] = list(df_cat.loc[df_cat['cum_pct']>= pct,'categories'])[0]

  
        del df_var
        del df_cat

    df_categorical = df_categorical[df_desc.columns]
    edd = pd.concat([df_desc,df_categorical])
    del df_desc
    del df_categorical
    edd['count'] = edd['Var'].apply(lambda x: data[data[x].notnull()].shape[0])
    edd['nmiss'] = data.shape[0]-edd['count']
    edd['missing_rate'] = np.array(edd['nmiss']).astype('float')/data.shape[0] * 100
    edd['unique'] = edd['Var'].apply(lambda x: len(data[x].value_counts().index.tolist()))
    col_list = ['Var','type','count','nmiss','missing_rate','unique','std','skewness','kurtosis','mean','min'] + \
    ['mean-'+str(int(dev))+'sigma' for dev in cv] + ['p'+str(int(pct*100)) for pct in percentile] + \
    ['mean+'+str(int(dev))+'sigma' for dev in cv] + ['max']

    edd = edd[col_list]

    if dv:
        edd['correlation/p_value'] = np.nan
        if regression==True:
            corr_matrix = data.corr()
            for col in numerical:
                edd.loc[edd['Var']==col,'correlation/p_value'] = corr_matrix.loc[col,dv]
            for col in categorical:
                try:
                    mod = ols(dv+' ~ '+ col,data=data).fit()
                    aov_table = sm.stats.anova_lm(mod,type=2)
                    edd.loc[edd['Var']==col,'correlation/p_value'] = aov_table.loc[col,'PR(>F)']
                except Exception as e:
                    print(col)
                    print(e)

        else:
            for col in numerical:
                try:
                    mod = ols(col+' ~ '+ dv,data=data).fit()
                    aov_table = sm.stats.anova_lm(mod,type=2)
                    edd.loc[edd['Var']==col,'correlation/p_value'] = aov_table.loc[dv,'PR(>F)']
                except Exception as e:
                    print(col)
                    print(e)

            for col in categorical:
                f =pd.crosstab(data[dv],data[col],dropna=False)
                edd.loc[edd['Var']==col,'correlation/p_value'] = chisquare(np.reshape(np.array(f),np.product(f.shape))).pvalue
    edd.reset_index(inplace=True)
    edd.drop('index',axis=1,inplace=True)
    return edd


##path = folder where you want to save your plots and tables
def graphical_analysis(data,dv,path='',regression=True):
    numerical,categorical = numerical_categorical_division(data)
    if regression:
        for col in numerical:
            if col != dv:
                ax = data.plot(col,dv)
                fig = ax.get_figure()
                fig.savefig(path+col+'.png',dpi=1000)
        for col in categorical:
            if col != dv:
                ax = data.boxplot(dv,by=col)
                fig = ax.get_figure()
                fig.savefig(path+col+'.png',dpi=1000)
    else:
        for col in numerical:
            if col != dv:
                ax = data.boxplot(col,by=dv)
                fig = ax.get_figure()
                fig.savefig(path+col+'.png',dpi=1000)
        for col in categorical:
            if col != dv:
                f =pd.crosstab(data[dv],data[col],dropna=False)
                for cat in f.columns:
                    f[cat] = f[cat].apply(lambda x: x/f[cat].sum()*100)
                ax = plt.subplot(111, frame_on=False) # no visible frame
                ax.xaxis.set_visible(False)  # hide the x axis
                ax.yaxis.set_visible(False)  # hide the y axis

                table(ax, f)  

                plt.savefig(path+col+'.png',dpi=1000)


