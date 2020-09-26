#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 09:06:09 2020

@author: bxp151
"""

import numpy as np
import pandas as pd
import plotnine as pn
from scipy.stats import skew
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from minepy import MINE
from scipy import stats
from scipy.spatial import distance

def checkNested(obj):
    
    '''
    ----------   
    
    Parameters
    ----------
    obj:    a list, series, or dataframe
    
    Returns
    -------
    True if any nested objects reside in passed object
    
    Example 
    -------
    a = [1,2,3]
    b = [a,a]
    c = (1,2,3)
    d = pd.DataFrame({'first' : [1, 2, 3, (1,2,3), 4, 5, 6],
                  'second': [2, 4, 5, [1,3,4], 6, 7, 8]}
                 , columns = ['first', 'second'])

    checkNested(a)
    Out[59]: False
    
    checkNested(b)
    Out[60]: True
    
    checkNested(c)
    Out[61]: False
    
    checkNested(d)
    Out[62]: True


    ---------- 
    '''  
    
    # object types
    otypes = (list, pd.core.series.Series, pd.core.frame.DataFrame)
    
    if isinstance(obj, otypes): pass
    else: return print("\nFunction only accepts:\n" +
                       "List, Series, or Dataframe\n")
    
    
    # nested types    
    ntypes = (list, tuple, set, np.ndarray, 
              pd.core.indexes.base.Index,
              pd.core.series.Series, 
              pd.core.frame.DataFrame)

        

    # Interates through obj and returns true if type in nestdtypes
    
    # dataframes
    if isinstance(obj, (pd.core.frame.DataFrame)):
        for row in range(len(obj)):
            for col in range(len(obj.columns)):
                if isinstance(obj.iloc[row,col], ntypes):
                    return True
    else: pass
    
    # non dataframes
    for item in obj:
        if isinstance(item, otypes): 
            return True
       
    
    return False


def returnType(d, r):    
    
    
    if r == 'list': return list(d)
    if r == 'set': return d
    if r == 'array': return np.array(d)
    if r == 'frame': return pd.DataFrame(d)
 

def calcDrop(res):
    
    '''
    ----------   
    
    Parameters
    ----------
    res:    results table from correlation functions

    Returns
    -------
    List of columns to drop

    Example 
    -------
    No example - Function is called by correlation functions

    ---------- 
    '''    
    # All variables with correlation > cutoff
    all_corr_vars = list(set(res['v1'].tolist() + res['v2'].tolist()))
    
    # All unique variables in drop column
    poss_drop = list(set(res['drop'].tolist()))

    # Keep any variable not in drop column
    keep = list(set(all_corr_vars).difference(set(poss_drop)))
     
    # Drop any variables in same row as a keep variable
    p = res[ res['v1'].isin(keep)  | res['v2'].isin(keep) ][['v1', 'v2']]
    q = list(set(p['v1'].tolist() + p['v2'].tolist()))
    drop = (list(set(q).difference(set(keep))))

    # Remove drop variables from possible drop 
    poss_drop = list(set(poss_drop).difference(set(drop)))
    
    # subset res dataframe to include possible drop pairs
    m = res[ res['v1'].isin(poss_drop)  | res['v2'].isin(poss_drop) ][['v1', 'v2','drop']]
        
    # remove rows that are decided (drop), take set and add to drops
    more_drop = set(list(m[~m['v1'].isin(drop) & ~m['v2'].isin(drop)]['drop']))
    for item in more_drop:
        drop.append(item)
      
        
    return drop


def loadBoston(t = 'all'):
    '''
    ----------   
    
    Parameters
    ----------
    t: option for return set 
        all (default)  all columns with dtypes fixed and columns dropped
        num'           only numeric dtypes
    
    Returns
    -------
    Boston Dataset, 2 objects
        x   predictors  dataframe
        y   target      series      
    
    -------
    '''
    source = 'http://lib.stat.cmu.edu/datasets/boston_corrected.txt'
    df = pd.read_table(source, skiprows= 9)
    df.columns = map(str.lower, df.columns)   
    
    # t = 'all'
    
    df = df.drop( ['obs.', 'town#', 'medv', 'tract'],axis = 1)
    
    if t == 'all':
        
        df['chas'] = df['chas'].astype('category')

        # Modify rad as ordinal
        r = pd.Series(range(df['rad'].min(), df['rad'].max() + 1))
        rad_cat = CategoricalDtype(categories=list(r), ordered=True)
        df['rad'] = df['rad'].astype(rad_cat)
    
    else:
        df = df.drop( ['rad', 'town', 'chas'],axis = 1)

    
    x = df.drop('cmedv', axis = 1)
    y = df['cmedv']  
    
    return x, y
    

def describe(df):
    

    '''
    ----------   
    
    Parameters
    ----------
    df : dataframe to analyze

    Returns
    -------
    Dataframe with statistics for each variable:
    
    q_zer:   number of zeros
    p_zer:   percent zeros
    q_na:    number of missing 
    p_na:    percent missing  
    q_inf:   quanitity of infinity  
    p_inf:   percent infinity
    dtype:   python dtype  
    uniqe:    unique levels


    Example 
    -------
    source = 'http://lib.stat.cmu.edu/datasets/boston_corrected.txt'
    BostonHousing2 = pd.read_table(source, skiprows= 9)
    describe(BostonHousing2.iloc[:,0:5])
    
          variable  q_zer  p_zer  q_na  p_na  q_inf  p_inf    dtype  unique
    0     OBS.        0      0.0     0   0.0      0    0.0    int64     506
    1     TOWN        0      0.0     0   0.0      0    0.0   object      92
    2    TOWN#        1      0.2     0   0.0      0    0.0    int64      92
    3    TRACT        0      0.0     0   0.0      0    0.0    int64     506
    4      LON        0      0.0     0   0.0      0    0.0  float64     375
    

    ---------- 
    '''        
    
    # Input Checks
    if isinstance(df, (pd.core.frame.DataFrame)):
        if checkNested(df): 
            print("\nPlease collapse any nested values in your dataframe\n")
            return
        else: 
            pass
    else:
        print("\nFunction only accetps dataframes\n")
        return

    # Get number of rows in dataframe
    obscnt = len(df)
    
    # get info() function into buffer to manipulate in to dataframe
    import io
    buffer = io.StringIO()
    df.info(null_counts = True, verbose = True, buf=buffer)
    s = buffer.getvalue()
    
    # Slice only important part of info()
    left = ' 0   '
    right = 'dtypes: '
    slice = s[s.index(left) + 1: s.index(right)]

    # convert into dataframe
    eda = pd.read_fwf(io.StringIO(slice), names = ['#', 'variable', 'Non-Null Count', 'dtype'])
    
    # drop the # column
    eda = eda.drop(['#'], axis = 1)
    
    
    # calcuate q_zer column
    def cntzero(series):
        
        "Returns # of zeros in a numeric series or zero for non-numeric series"
        
        if series.dtype not in ['int64', 'float64']:
            return 0
        else:
            return sum(series == 0)   
    eda.insert(1, 'q_zer', df.apply(cntzero, axis = 0).values )
    
    # add p_zer column
    eda.insert(2, 'p_zer', round(eda['q_zer'] / obscnt * 100, 2))
    
    # add q_na column
    # Get number as text from non-null count
    eda['Non-Null Count'] = [ item[0] for item in eda['Non-Null Count'].str.split() ]
    
    # populate missing count into 'non-null count' column 
    eda['Non-Null Count'] = obscnt - eda['Non-Null Count'].astype(int)
    
    # rename column to q_na
    eda = eda.rename(columns = {'Non-Null Count': 'q_na'})
    
    
    # add percent na column p_na
    eda.insert(4, 'p_na', eda['q_na'] / obscnt)
    
    # add q_inf column
    def cntinf(series):
        "Returns # of inf in a numeric series or zero for non-numeric"
        if series.dtype not in ['int64', 'float64']:
            return 0
        else:
            return sum(np.isinf(series))
        
    eda.insert(5, 'q_inf', df.apply(cntinf, axis = 0).values )
    
    # add p_inf column
    eda.insert(6, 'p_inf', eda['q_inf'] / obscnt )
      
    # add unique values column
    eda['unique'] = df.nunique().values  
    
    return eda


def glimpse(df):
    
    
    '''
    ----------   
    
    Parameters
    ----------
    df: dataframe to glimpse into

    Returns
    -------
    All variables, dtypes, and first five observations

    Example 
    -------
    source = 'http://lib.stat.cmu.edu/datasets/boston_corrected.txt'
    BostonHousing2 = pd.read_table(source, skiprows= 9)
    glimpse(BostonHousing2.iloc[:,0:4])
    
    Rows: 506
    Columns: 4

      variable   dtype                            first_five_observations
    0     OBS.   int64                                    [1, 2, 3, 4, 5]
    1     TOWN  object  [Nahant, Swampscott, Swampscott, Marblehead, M...
    2    TOWN#   int64                                    [0, 1, 1, 2, 2]
    3    TRACT   int64                     [2011, 2021, 2022, 2031, 2032]

    ---------- 
    '''        
    # Input Checks
    if isinstance(df, (pd.core.frame.DataFrame)): pass
    else: return print("\nFunction only accetps dataframes\n")
    if checkNested(df): return print("\nPlease collapse any nested values in your dataframe\n")


    
    # grab first columns from describe dataframe
    g = describe(df)[['variable', 'dtype']]
    # create new column to store observations
    g['first_five_observations'] = ''
    
    # get the first 5 items for each variable
    # transpose the data frame and store the values
    x = df.apply((pd.DataFrame.head), axis = 0 ).T.values
    
    # populate values into new dataframe column
    for i in range(0, len(x)):
        g['first_five_observations'][i] = x[i]
    
    r = '\nRows: ' + str(len(df))
    c = 'Columns: ' + str(len(df.columns))
    g = print(r + '\n' + c + '\n\n' + str(g))
    
    return g


def plotfreq(df):
    '''
    ----------   
    
    Parameters
    ----------
    df: dataframe generated by freq()

    Returns
    -------
    Bar chart with frequencies & percentages in descending order
        
    Example 
    -------
    plotfreq(df)

    Warning 
    -------
    This function will likely not plot more than 100 unique levels properly.
    
    ---------- 
    '''    
    
    # input checks
    if isinstance(df, (pd.core.frame.DataFrame)): pass
    else: return print("\nFunction only accetps dataframes\n") 
    
    if len(df.columns) == 4: pass
    else: return print("\nInput must be a dataframe generated by freq()\n")
    
    if len(same(df.columns[1:4], ['freq', 'perc', 'cump'])) == 3: pass
    else: return print("\nInput must be a dataframe generated by freq()\n")
    
    # label for plot
    lbl =  df['freq'].astype(str).str.cat('[ ' + df['perc'].astype(str) + '%' + ' ]'
                                          , sep = '   ') 
    # create variable to be used in aes
    aesx = 'reorder(' + df.columns[0] + ', freq)'
       
        # build plot
    plot = ( 
      pn.ggplot(df) +
      pn.aes(x = aesx, 
             y = 'freq', 
             fill = 'freq', 
             label = lbl) +
      pn.geom_bar(stat = 'identity') +
      pn.coord_flip() +
      pn.theme(axis_text_y = pn.element_text(size=6, weight = 'bold'), 
            legend_position = 'none') + 
      pn.labs(x=df.columns[0], y="Freq") +
      pn.scale_fill_gradient2(mid='bisque', high='blue') +
      pn.geom_text(size = 6,
                   nudge_y = .7)
    )
    
    return plot



def freq(srs):
    '''
    ----------   
    
    Parameters
    ----------
    srs:    series to analyze
    plot:   flag inicating whether to plot table

    Returns
    -------
    If plot is false: Dataframe with the following columns:
        
    col:    Unique levels in the column
    freq:   Count of each level
    perc:   Percent each level contributes
    cump:   Cumulative percent 
    
    If plot is true: prints above Dataframe and displays plot
    
    Example 
    -------
    source = 'http://lib.stat.cmu.edu/datasets/boston_corrected.txt'
    BostonHousing2 = pd.read_table(source, skiprows= 9)
    freq(BostonHousing2['TOWN'])
    
                     TOWN  freq  perc    cump
    0           Cambridge    30  5.93    5.93
    1   Boston Savin Hill    23  4.55   10.47
    2                Lynn    22  4.35   14.82
    3      Boston Roxbury    19  3.75   18.58
    4              Newton    18  3.56   22.13
    ..                ...   ...   ...     ...
    87           Medfield     1  0.20   99.21
    88             Millis     1  0.20   99.41
    89          Topsfield     1  0.20   99.60
    90              Dover     1  0.20   99.80
    91            Norwell     1  0.20  100.00

    ---------- 
    '''    
    
    # input checks
    if isinstance(srs, (pd.core.series.Series)): pass
    else: return print("\nFunction only accetps series\n") 
            
    # Create frequency dataframe
    cnts = srs.value_counts()
    perc = round(cnts / sum(cnts.values) * 100, 2)
    cump = round(100 * (cnts.cumsum() / cnts.sum()), 2)
    
    df = pd.DataFrame(data = dict(var = cnts.keys(),
                                   freq = cnts,
                                   perc = perc,
                                   cump = cump))
    df.rename(columns={'var': srs.name}, inplace=True)
    df = df.reset_index(drop = True)
    
    return df


def diff(a,b, r = 'list'):
    
    
    '''
    ----------   
    
    Parameters
    ----------
    a : First object to compare
    b : Second object to compare
    r : Return object type, optional (default: list)
    
    a,b can only be list, set, array, series, index, tuple.  The types
    don't need to match.
    
    Returns
    -------
    The unique values of a not in b, or b not in a

    Example 
    -------
    a = [1,1,2,3,4,5]
    b = [2,3,4,5,6,6]
    
    diff(a,b)
    [1]
    
    diff(b,a)
    [6]
    
    ---------- 
    '''        
    
 
    
    t = (list, tuple, set, np.ndarray, pd.core.series.Series, 
     pd.core.indexes.base.Index)
   
    if isinstance(a, t) & isinstance(b, t):
        return returnType(set(a).difference(set(b)), r)
    else:
        return print("\nFunction only accepts:\n" +
                     "List, Tuple, Set, ndarray, Series, or Index")
              
    
def same(a,b, r = 'list'):
    
    '''
    ----------
    
    Parameters
    ----------
    a : First object to compare
    b : Second object to compare
    r : Return object type, optional (default: list)
    
    a,b can be any of index, list, set, array, series, index, tuple.  The types
    don't need to match.
    
    Returns
    -------
    The unique values shared by a and b

    Example 
    -------
    x = [1,1,2,3,4,5]
    y = [2,3,4,5,6,6]
    
    same(x,y)
    [2, 3, 4, 5]
    
    same(y,x)
    [2, 3, 4, 5]
    
    ---------- 
    '''        
        
    # check a,b are in t
    t = (list, tuple, set, np.ndarray, pd.core.series.Series, 
         pd.core.indexes.base.Index)
    
    # if checkTypes([a,b], t):return type_return(set(a).intersection(set(b)), r)
    
    if isinstance(a, t) & isinstance(b, t):
        return returnType(set(a).intersection(set(b)), r)
    else:
        return print("\nFunction only accepts:\n" +
                     "List, Tuple, Set, ndarray, Series, or Index")


def skew_df(df):
    '''
    ----------   
    
    Parameters
    ----------
    df:     dataframe to analyze

    Returns
    -------
    Dataframe with the following columns:
        
    index:      Variable name
    dtype:      Python dtype
    skewness:   Skewness statistic calculated by skew function
    
    magnitude:   
    2-high              Skewness less than -1 or greater than 1     
    1-medium            Skewness between -1 and -0.5 or 0.5 and 1   
    0-approx_symmetric  Skewness between -0.5 and 0.5             
    
    Example 
    -------
    source = 'http://lib.stat.cmu.edu/datasets/boston_corrected.txt'
    BostonHousing2 = pd.read_table(source, skiprows= 9)
    skew_df(BostonHousing2)
    
               dtype  skewness           magnitude
    CHAS       int64  3.395799              2-high
    B        float64 -2.881798              2-high
    MEDV     float64  1.104811              2-high
    CMEDV    float64  1.107616              2-high
    CRIM     float64  5.207652              2-high
    ZN       float64  2.219063              2-high
    RAD        int64  1.001833              2-high
    DIS      float64  1.008779              2-high
    AGE      float64 -0.597186            1-medium
    PTRATIO  float64 -0.799945            1-medium
    TAX        int64  0.667968            1-medium
    LSTAT    float64  0.903771            1-medium
    NOX      float64  0.727144            1-medium
    RM       float64  0.402415  0-approx_symmetric
    TOWN#      int64  0.039088  0-approx_symmetric
    INDUS    float64  0.294146  0-approx_symmetric
    LAT      float64 -0.086421  0-approx_symmetric
    LON      float64 -0.204775  0-approx_symmetric
    TRACT      int64 -0.434515  0-approx_symmetric
    OBS.       int64  0.000000  0-approx_symmetric

    ---------- 
    '''    
    
    
    # input checks
    if isinstance(df, (pd.core.frame.DataFrame)): pass
    else: return print("\nFunction only accetps dataframes\n") 
    
    d = df.select_dtypes('number')
    if len(d.columns) == 0: return print("\nDataframe has no numeric columns\n") 

    

    def skew_series(srs):
        s = skew(srs.array)
        a = 'placeholder'
        return ([srs.dtype,float(s), a])
        
    result = d.apply(skew_series, axis = 0).transpose()
    result.columns = ['dtype', 'skewness', 'magnitude']
    
    result['skewness'] = result['skewness'].astype(float)
    
   
    def magnitude_skew(x):
        # caculate magnitude of skewness
        w = abs(x)
        if w > 1:
            return '2-high'
        if w <= 1 and w > 0.5:
            return '1-medium'
        else:
            return '0-approx_symmetric'
            
    # run apply on analysis
    result['magnitude'] = result['skewness'].apply(magnitude_skew)
        
    # sort values
    result.sort_values('magnitude', ascending = False, inplace=True)
    
    return result



def associationMeasures(X, y):
    

    r = pd.DataFrame(data = 0.0, index=X.columns, 
                     columns = ['mic', 'pearson', 'spearman', 'cosine'])
    
    mine = MINE(alpha=0.6, c=15) 
    
    for col in X.columns:
        mine.compute_score(X[col], y)
        r.loc[col, 'mic'] = mine.mic()
        r.loc[col, 'pearson'] = abs(stats.pearsonr(X[col], y)[0])
        r.loc[col, 'spearman'] = abs(stats.spearmanr(X[col], y)[0])
        r.loc[col, 'cosine'] = abs(1-distance.cosine(X[col], y))
    return r


    
def corr(X, y = None, cut = 0.9, ret = 'name') :
       
    '''
    ----------   
    
    Parameters
    ----------
    X:      predictors dataframe
    y:      target
    cut:    correlation cutoff
    ret:    'name'  returns names of columns
            'ind'   returns indexes of columns
            'X'    returns datafrme with columns dropped

    Returns
    -------
    This function analyzes the correlation matrix for the dataframe and 
    returns a list of columns to drop based on the correlation cutoff.  It
    uses the average correlation for the row and column in the matrix and 
    compares it with the cell value to decide on potential drop candidates.
    
    It uses the calcDrop() function in order to calculate which columns 
    should be dropped.  For more information please visit:
        
    https://towardsdatascience.com/are-you-dropping-too-many-correlated-features-d1c96654abe6
        
    Example 
    -------

    source = 'http://lib.stat.cmu.edu/datasets/boston_corrected.txt'
    X = pd.read_table(source, skiprows= 9)
    X.columns = map(str.lower, X.columns)
    # Keeping only numeric columns (non-target)
    X = X.drop( ['obs.', 'town#', 'medv', 'cmedv', 'tract', 
                    'rad', 'town', 'chas'],axis = 1)
    
    corrX(X, cut = 0.6)
    ['nox', 'indus', 'lstat', 'dis']
    
    corrX(X, cut = 0.6, ret = 'ind')
    [5, 4, 12, 8]
    
    ---------- 
    '''    

    
    # Input Checks
    if isinstance(X, (pd.core.frame.DataFrame)): pass
    else: return print("\nFunction only accetps dataframes\n")
    if checkNested(X): return print("\nPlease collapse any nested values"  + 
                                     "in your dataframe\n")

    # check for at least 2 numeric columns
    if len(X.select_dtypes('number').columns) >= 2: pass
    else: return print("Dataframe must have 2 or more numeric columns")
    

    # Get correlation matrix and upper triagle
    corr_mtx = X.corr().abs()
    avg_corr = corr_mtx.mean(axis = 1)
    up = corr_mtx.where(np.triu(np.ones(corr_mtx.shape), k=1).astype(np.bool))
    
    dropcols = list()
    
    res = pd.DataFrame(columns=(['v1', 'v2', 'v1.target', 
                                 'v2.target','corr', 'drop' ]))
    
    for row in range(len(up)-1):
        col_idx = row + 1
        for col in range (col_idx, len(up)):
            if(corr_mtx.iloc[row, col] > cut):
                if(avg_corr.iloc[row] > avg_corr.iloc[col]): 
                    dropcols.append(row)
                    drop = corr_mtx.columns[row]
                else: 
                    dropcols.append(col)
                    drop = corr_mtx.columns[col]
                
                s = pd.Series([ corr_mtx.index[row],
                up.columns[col],
                avg_corr[row],
                avg_corr[col],
                up.iloc[row,col],
                drop],
                index = res.columns)
        
                res = res.append(s, ignore_index = True)
    
    dropcols_names = calcDrop(res)
    dropcols_idx = list(X.columns.get_indexer(dropcols_names))
    

        
    if ret == 'ind':    return(dropcols_idx)
    if ret == 'name':  return(dropcols_names)
    if ret == 'X':     return(X.drop(dropcols_names, axis = 1))
    


class CategoricalOtherLevel( BaseEstimator, TransformerMixin ):
    
    def __init__(self, colname, threshold):
        self.colname = colname
        self.threshold = threshold
        self.notothers = pd.Series()
        
    
    def fit( self, X, y = None):

        #self.notothers = catOtherLevel(X[self.colname], self.threshold)
        
        # get frequency table
        f = freq(X[self.colname])[[self.colname, 'perc']]
        
        # get  (not "others") to create lookup table
        self.notothers = f[ f['perc'] > (self.threshold * 100) ][self.colname]
        
        return self
    
    def transform( self, X, y = None):
        # if srs in o then replace with "other"
        # for i in X.index:
        #     if sum(self.others.str.contains(X[self.colname][i])) > 0 :
        #         X.at[i, self.colname] = 'other'
        # return X
        for i in X.index:
            if sum(self.notothers.str.contains(X[self.colname][i])):
                pass
            else:
                X.at[i, self.colname] = 'other'
        return X

        

class CorrelationFilter( BaseEstimator, TransformerMixin ):
           
    def __init__(self, cut = 0.9, corrType = 'X'):
        self.corrType = corrType
        self.cut = cut
        # self.target = target
        self.names = []
  

        
    def fit( self, X, y = None ):
        if self.corrType == 'X': 
            self.names = corr(X, cut = self.cut)
            
        
        
        # if self.corrType == 'XY': 
        #     # self.names = corrY(X, self.target ,cut = self.cut)
        #     self.names = corrY(X, y=y, cut = self.cut)
            
        return self
    
    def transform( self, X, y = None ):
        return X.drop(self.names, axis = 1)
    
  
class ColumnSelect( BaseEstimator, TransformerMixin ):

    def __init__( self, feature_names):
        self.feature_names = feature_names
       
    def fit( self, X, y = None ):
        return self 
    
    def transform( self, X,  y = None ):
        return X[self.feature_names]

