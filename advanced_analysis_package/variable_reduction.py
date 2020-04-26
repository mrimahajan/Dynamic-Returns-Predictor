import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.discrete.discrete_model as sm
import statsmodels.api as lm
from sklearn.preprocessing import StandardScaler
from patsy import dmatrices
import statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor


## cutoff is the minimum corelation forr dependency in 2 variables
def inter_correlation_clusters(data,cutoff=.7):
    correlations=data.corr()
    graph={}
    columns=data.columns
    for i in range(len(columns)):
        graph[i]=[]
        for j in range(len(columns)):
            if i!=j and np.abs(correlations.iloc[i,j])>cutoff:
                graph[i].append(j)
    
    tree_set={}
    component = 0
    visited = [0 for i in range(len(columns))]
    def dfs(i):
        visited[i]=1
        try:
            tree_set[component].append(i)
        except KeyError:
            tree_set[component] = [i]
            
        for j in graph[i]:
            if visited[j]==0:
                dfs(j)
    for i in range(len(columns)):
        if visited[i]==0:
            dfs(i)
            component+=1
        else:
            continue
    
    tree_cluster={}
    for key in list(tree_set.keys()):
        tree_cluster[key] = [columns[i] for i in tree_set[key]]
        
    return tree_cluster


## return columns which are not correlated above cutoff
def varclus(data,cutoff,maxkeep=1,maxdrop=None):
    columns = []
    correlations=data.corr()
    clusters= inter_correlation_clusters(data,cutoff=cutoff)
    print(clusters)
    
    cols = list(data.columns)
    
    def distance(c1,c2):
        return np.max([[np.abs(correlations.loc[i,j]) for i in clusters[c1]] for j in clusters[c2]])
    
    def next_closest(c):
        minima=0
        point=c
        for c1 in [i for i in list(clusters.keys()) if i!=c]:
            dist = distance(c,c1)
            if dist>minima:
                minima=dist
                point=c1
        return point
    
    def get_squared_ratio(col,own_cluster,next_cluster):
        y = np.array(data[col])
        x = np.array(data[own_cluster].drop(col,axis=1))
        model=LinearRegression()
        model.fit(x,y)
        y_pred = list(model.predict(x))
        del x
        del model
        r2_own = r2_score(y,y_pred)
        del y_pred
        x = np.array(data[next_cluster])
        model=LinearRegression()
        model.fit(x,y)
        y_pred = list(model.predict(x))
        del x
        del model
        r2_next = r2_score(y,y_pred)
        del y
        del y_pred
        
        return float(1-r2_own)/(1-r2_next)
    
    for c1 in list(clusters.keys()):
        clus_len = len(clusters[c1])
        if clus_len>1:
            own_cluster = clusters[c1]
            next_cluster = clusters[next_closest(c1)]
            ratio_list = []
            for col in clusters[c1]:
                col_ratio = get_squared_ratio(col,own_cluster,next_cluster)
                ratio_list.append((col,col_ratio))
            print(ratio_list)
            ratio_list = sorted(ratio_list,key = lambda x: x[1])
            if maxdrop is not None:
            	columns += [col[0] for col in ratio_list[:-min(maxdrop,clus_len)]]
            else:
            	columns += [col[0] for col in ratio_list[:min(maxkeep,clus_len)]]
        else:
            columns.append(clusters[c1][0])
            
    return columns

## return columns to be dropped by vif reduction method
def vif_reduction(data,limit=2.5):
	vif_drop_cols = []
	def variance_inflation(fn_data):
	    vif = pd.DataFrame()
	    vif['features'] = fn_data.columns
	    vif['vif factor'] = [variance_inflation_factor(fn_data.values, i) for i in range(fn_data.shape[1])]
	    vif.sort_values(by='vif factor',ascending=False,inplace=True)
	    vif.reset_index(inplace=True)
	    vif.drop(['index'],axis=1,inplace=True)
	    print(vif)
	    return tuple(vif.loc[0,:].values)

	def reduction(fn_data,cutoff=limit):
	    vif = variance_inflation(fn_data)
	    if vif[1]<=cutoff:
	        return
	    else:
	        fn_data.drop(vif[0],axis=1,inplace=True)
	        vif_drop_cols.append(vif[0])
	        print(vif[0]+' dropped')
	        del vif
	        reduction(fn_data,cutoff=limit)
	reduction(data)
	return vif_drop_cols

## return columns to be dropped by backward selection method
def backward_selection(df,dv,regression=True,alpha=.05):
    flag=0
    cols_dropped=[dv]
    if regression:
        while flag==0:
            model = lm.OLS(endog=np.array(df[dv]),exog=np.array(df.drop(cols_dropped,axis=1)))
            results = model.fit()
            pvalues=list(results.pvalues)
            drop_index = pvalues.index(max(pvalues))
            col_drop = df.drop(cols_dropped,axis=1).columns[drop_index]
            print(col_drop+'-'+str(pvalues[drop_index]))
            if pvalues[drop_index]> alpha:
                cols_dropped.append(col_drop)
            else:
                flag=1
    else:
        while flag==0:
            model = sm.Logit(endog=np.array(df[dv]),exog=np.array(df.drop(cols_dropped,axis=1)))
            results = model.fit()
            pvalues=list(results.pvalues)
            drop_index = pvalues.index(max(pvalues))
            col_drop = df.drop(cols_dropped,axis=1).columns[drop_index]
            print(col_drop+'-'+str(pvalues[drop_index]))
            if pvalues[drop_index]> alpha:
                cols_dropped.append(col_drop)
            else:
                flag=1

    cols_dropped.remove(dv)
    return cols_dropped


