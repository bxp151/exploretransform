import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import plotnine as pn
from scipy.stats import skew
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from minepy import MINE
from scipy import stats
from dcor import distance_correlation

        
def nested(obj, retloc = False):
    
    '''
    ----------   
    
    Parameters
    ----------
    obj:        a list, series, or dataframe
    retloc:     True or False
    
    Returns
    -------
    
    retloc = True       
        Returns locations of nested objects: For dataframes, it returns tuples
        For other objects it returns a list of indicies
    retloc = False      
        Returns True if any nested objects reside in passed object, False otherwise
    
    Example 
    -------
    a = pd.DataFrame({'first' : [1, 2, 3, (1,2,3), 4, 5, 6],
              'second': [2, 4, 5, [1,3,4], 6, 7, 8]}
              , columns = ['first', 'second'])
    
    nested(a, locs = True)
    [(3, 0), (3, 1)]

    nested(a)
    Out[59]: False
    
    ---------- 
    '''  

    # object types
    otypes = (list, pd.core.series.Series, pd.core.frame.DataFrame)
    
    # store locations of nested items
    locs = list() 
    
    if isinstance(obj, otypes): pass
    else: return "Function only accepts: List, Series, or Dataframe"
        
    # nested types    
    ntypes = (list, tuple, set, np.ndarray, 
              pd.core.indexes.base.Index,
              pd.core.series.Series, 
              pd.core.frame.DataFrame)

    
    # dataframes
    if isinstance(obj, (pd.core.frame.DataFrame)):
        for row in range(len(obj)):
            for col in range(len(obj.columns)):
                if isinstance(obj.iloc[row,col], ntypes):
                    locs.append((row,col))
    
    
    else: #other types
        
        for i in range(len(obj)):
            if isinstance(obj[i], ntypes): 
                locs.append(i)
            
    
    if retloc: return locs
    else: return len(locs) > 0


def loadboston():
    '''
    ----------   
    
    Parameters
    ----------
    None
    
    Returns
    -------
    Boston corrected data objects:
        1. df  X and y     dataframe    
        2. X   predictors  dataframe
        3. y   target      series
    
    -------
    '''
    source = 'https://raw.githubusercontent.com/bxp151/exploretransform/master/data/boston_corrected.txt'
    df = pd.read_table(source, skiprows= 9)
    df.columns = map(str.lower, df.columns)   
    df = df.drop( ['obs.', 'town#', 'medv', 'tract'],axis = 1)
    df['chas'] = df['chas'].astype('category')

    # Modify rad as ordinal
    r = pd.Series(range(df['rad'].min(), df['rad'].max() + 1))
    rad_cat = CategoricalDtype(categories=list(r), ordered=True)
    df['rad'] = df['rad'].astype(rad_cat)
    
    x = df.drop('cmedv', axis = 1)
    y = df['cmedv']  
    
    return df, x, y
  
  
def explore(X):
    

    '''
    ----------   
    
    Parameters
    ----------
    X:       dataframe to analyze

    Returns
    -------
    Dataframe with statistics for each variable:
    
    variable   name of column
    obs        number of observations
    q_zer      number of zeros
    p_zer      percent zeros
    q_na       number of missing 
    p_na       percent missing  
    q_inf      quanitity of infinity  
    p_inf      percent infinity
    dtype      Python dtype  

    Example 
    -------
    import exploretransform as et
    df, X, y = et.loadboston()
    et.explore(df.iloc[:,0:5])
    
       variable  obs  q_zer  p_zer  q_na  p_na  q_inf  p_inf     dtype
    0      town  506      0   0.00     0   0.0      0    0.0    object
    1       lon  506      0   0.00     0   0.0      0    0.0   float64
    2       lat  506      0   0.00     0   0.0      0    0.0   float64
    3      crim  506      0   0.00     0   0.0      0    0.0   float64
    4        zn  506    372  73.52     0   0.0      0    0.0   float64
    

    ---------- 
    '''        
   
    # Input Checks
    if isinstance(X, (pd.core.frame.DataFrame)):
        if nested(X): 
            return "Please collapse any nested values in your dataframe"     
        else: 
            pass
    else:
        return "Function only accetps dataframes"
    
    
    # counts zeros for numeric dtype and returns zero for others
    def cntzero(series):
        if is_numeric_dtype(series): return sum(series == 0)
        else:return 0 
        
    # counts inf values for numeric dtype and returns zero for others
    def cntinf(series):
        if is_numeric_dtype(series): return sum(np.isinf(series))
        else: return 0
        
    df = pd.DataFrame({'variable': X.columns})
    df['obs'] = len(X)
    df['q_zer'] = X.apply(cntzero, axis = 0).values
    df['p_zer'] = round(df['q_zer'] / len(X) * 100, 2)
    df['q_na'] = X.isna().sum().values
    df['p_na'] = round(df['q_na'] / len(X) * 100, 2)
    df['q_inf'] = X.apply(cntinf, axis = 0).values
    df['p_inf'] = round(df['q_inf'] / len(X) * 100, 2)
    df['dtype'] = X.dtypes.to_frame('dtypes').reset_index()['dtypes']

    
    return df


def peek(X):
    
    
    '''
    ----------   
    
    Parameters
    ----------
    X: dataframe to peek into

    Returns
    -------
    Columns based on passed dataframe:
        
    variable    name of variable
    dtype	    Python dtype
    lvls	    unique values of variable
    obs	        number of observations
    head	    first five observations

    Example 
    -------
    import exploretransform as et
    df, X, y = et.loadboston()
    et.peek(df.iloc[:,0:5])
    
       variable     dtype  ...  obs                                               head
    0      town    object  ...  506  [Nahant, Swampscott, Swampscott, Marblehead, M...
    1       lon   float64  ...  506       [-70.955, -70.95, -70.936, -70.928, -70.922]
    2       lat   float64  ...  506          [42.255, 42.2875, 42.283, 42.293, 42.298]
    3      crim   float64  ...  506  [0.00632, 0.02731, 0.02729, 0.0323699999999999...
    4        zn   float64  ...  506                         [18.0, 0.0, 0.0, 0.0, 0.0]

    ---------- 
    '''        
    # Input Checks
    if isinstance(X, (pd.core.frame.DataFrame)): pass
    else: return "Function only accetps dataframes"
    if nested(X): return "Please collapse any nested values in your dataframe"

    g = pd.DataFrame({'variable': X.columns, 
                      'dtype': X.dtypes.to_frame('dtypes').reset_index()['dtypes']}, 
                     index=(range(0,len(X.columns))))
    g['lvls'] = X.nunique().values 
    g['obs'] = len(X)
    
    g['head'] = ''
    
    # get the first 5 items for each variable
    # transpose the data frame and store the values
    x = X.apply((pd.DataFrame.head), axis = 0 ).T.values
    for i in range(0, len(x)):
        g.at[i,'head'] = x[i]
    
    
    return g


def plotfreq(freqdf):
    '''
    ----------   
    
    Parameters
    ----------
    freqdf  dataframe generated by freq()

    Returns
    -------
    Bar chart with frequencies & percentages in descending order
        
    Example 
    -------
    import exploretransform as et
    df, X, y = et.loadboston()
    et.plotfreq(et.freq(X['town']))

    Warning 
    -------
    This function will likely not plot more than 100 unique levels properly.
    
    ---------- 
    '''    

    
    # input checks
    if isinstance(freqdf, (pd.core.frame.DataFrame)): pass
    else: return print("\nFunction only accetps dataframes\n") 
    
    if len(freqdf.columns) == 4: pass
    else: return print("\nInput must be a dataframe generated by freq()\n")
    
    if sum(freqdf.columns[1:4] == ['freq', 'perc', 'cump']) == 3: pass
    else: return print("\nInput must be a dataframe generated by freq()\n")
    
    if len(freqdf) < 101: pass
    else: return print("\nUnable to plot more than 100 items")
    
    # label for plot
    lbl =  freqdf['freq'].astype(str).str.cat('[ ' + freqdf['perc'].astype(str) + '%' + ' ]'
                                          , sep = '   ') 
    # create variable to be used in aes
    aesx = 'reorder(' + freqdf.columns[0] + ', freq)'
       
        # build plot
    plot = ( 
      pn.ggplot(freqdf) +
      pn.aes(x = aesx, 
             y = 'freq', 
             fill = 'freq', 
             label = lbl) +
      pn.geom_bar(stat = 'identity') +
      pn.coord_flip() +
      pn.theme(axis_text_y = pn.element_text(size=6, weight = 'bold'), 
            legend_position = 'none') + 
      pn.labs(x=freqdf.columns[0], y="Freq") +
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

    Returns
    -------
    Dataframe with the following columns:
        
    <name>  The unique values of the series
    freq    Count of each level
    perc    Percent each level contributes
    cump    Cumulative percent 
    
    
    Example 
    -------
    import exploretransform as et
    df, X, y = et.loadboston()
    et.freq(X['town'])
    
                     town  freq  perc    cump
    0           Cambridge    30  5.93    5.93
    1   Boston Savin Hill    23  4.55   10.47
    2                Lynn    22  4.35   14.82
    3      Boston Roxbury    19  3.75   18.58
    4              Newton    18  3.56   22.13
    ..                ...   ...   ...     ...
    87          Topsfield     1  0.20   99.21
    88         Manchester     1  0.20   99.41
    89              Dover     1  0.20   99.60
    90            Hanover     1  0.20   99.80
    91            Lincoln     1  0.20  100.00

    ---------- 
    '''    
    
    # input checks
    if isinstance(srs, (pd.core.series.Series)): pass
    else: return "Function only accetps series"
            
    # Create frequency dataframe
    cnts = srs.value_counts()
    perc = round(cnts / sum(cnts.values) * 100, 2)
    cump = round(100 * (cnts.cumsum() / cnts.sum()), 2)
    
    freqdf = pd.DataFrame(data = dict(var = cnts.keys(),
                                   freq = cnts,
                                   perc = perc,
                                   cump = cump))
    freqdf.rename(columns={'var': srs.name}, inplace=True)
    freqdf = freqdf.reset_index(drop = True)
    
    return freqdf


def corrtable(X, y = None, cut = 0.9, methodx = 'spearman', methody = None, full = False):
    
    
    '''
    ----------   
    
    Parameters
    ----------
    X           predictors dataframe
    y           target (unused in exploretransform v 1.0.0)
    cut         correlation threshold
    full       
        True    Returns the full corrtable with drop column
        False   (default) Returns without the drop column
    
    methodx     used to calculate correlations amount predictors
    methody*    used to calculate correlations between predictors & target
                *(unused in exploretransform v 1.0.0) 
    
    pearson     standard correlation coefficient
    kendall     Kendall Tau correlation coefficient
    spearman    Spearman rank correlation
    callable    callable with input two 1d ndarrays and returning a float. Note 
                that the returned matrix from corr will have 1 along the 
                diagonals and will be symmetric regardless of the callable's 
                behavior.

    Returns
    -------
    This function analyzes the correlation matrix for the dataframe. It
    uses the average correlation for the row and column in the matrix and 
    compares it with the cell value to decide on potential drop candidates.
    
    Columns
    
    v1          varaible 1
    v2          variable 2
    v1.target   metric used to compare v1 and v2 for drop
    v2.target   metric used to compare v1 and v2 for drop
    corr        pairwise correlation based on method
    drop        if the correlation > threshold, the drop decision 
    
    For more information please visit
        
    https://towardsdatascience.com/are-you-dropping-too-many-correlated-features-d1c96654abe6
        
    Example 
    -------
    import exploretransform as et
    df, X, y = et.loadboston()
    X = X.select_dtypes('number')  
    et.corrtable(X, cut = 0.7, full = True)    

           v1       v2  v1.target  v2.target      corr drop
    52    nox      dis   0.578860   0.526551  0.880015  nox
    25   crim      nox   0.562681   0.578860  0.821465  nox
    63    age      dis   0.525682   0.526551  0.801610  dis
    51    nox      age   0.578860   0.525682  0.795153  nox
    42  indus      nox   0.549707   0.578860  0.791189  nox
    ..    ...      ...        ...        ...       ...  ...
    8     lon      tax   0.242329   0.486066  0.050237     
    22    lat    lstat   0.159767   0.522203  0.039065     
    14    lat    indus   0.159767   0.549707  0.021472     
    18    lat      dis   0.159767   0.526551  0.012832     
    20    lat  ptratio   0.159767   0.391352  0.005332       
    
    ---------- 
    '''    

    # Get correlation matrix and upper triagle
    corr_mtx = X.corr(method = methodx).abs()
    avg_corr = corr_mtx.mean(axis = 1)
    up = corr_mtx.where(np.triu(np.ones(corr_mtx.shape), k=1).astype(np.bool))
        
    ct = pd.DataFrame(columns=(['v1', 'v2', 'v1.target', 
                                 'v2.target','corr', 'drop' ]))
    
    for row in range(len(up)-1):
        col_idx = row + 1
        for col in range (col_idx, len(up)):
            drop = ' '
            if(corr_mtx.iloc[row, col] > cut):
                if(avg_corr.iloc[row] > avg_corr.iloc[col]): 
                    drop = corr_mtx.columns[row]
                else: 
                    drop = corr_mtx.columns[col]
            
            # Populate results table
            s = pd.Series([ corr_mtx.index[row],
            up.columns[col],
            avg_corr[row],
            avg_corr[col],
            up.iloc[row,col],
            drop],
            index = ct.columns)
    
            ct = ct.append(s, ignore_index = True)
    
    ct.sort_values('corr', ascending = False, inplace=True)
    
    if full:    return ct
    else:       return ct.drop('drop', axis = 1)


def calcdrop(ct):
    
    '''
    ----------   
    
    Parameters
    ----------
    ct:    results table from correlation functions

    Returns
    -------
    List of columns to drop

    Example 
    -------
    No example - Function is called by correlation functions

    ---------- 
    '''    
    # All variables with correlation > cutoff
    all_corr_vars = list(set(ct['v1'].tolist() + ct['v2'].tolist()))
    
    # All unique variables in drop column
    poss_drop = list(set(ct['drop'].tolist()))

    # Keep any variable not in drop column
    keep = list(set(all_corr_vars).difference(set(poss_drop)))
     
    # Drop any variables in same row as a keep variable
    p = ct[ ct['v1'].isin(keep)  | ct['v2'].isin(keep) ][['v1', 'v2']]
    q = list(set(p['v1'].tolist() + p['v2'].tolist()))
    drop = (list(set(q).difference(set(keep))))

    # Remove drop variables from possible drop 
    poss_drop = list(set(poss_drop).difference(set(drop)))
    
    # subset ct dataframe to include possible drop pairs
    m = ct[ ct['v1'].isin(poss_drop)  | ct['v2'].isin(poss_drop) ][['v1', 'v2','drop']]
        
    # remove rows that are decided (drop), take set and add to drops
    more_drop = set(list(m[~m['v1'].isin(drop) & ~m['v2'].isin(drop)]['drop']))
    for item in more_drop:
        drop.append(item)
      
        
    return drop


def skewstats(X):
    '''
    ----------   
    
    Parameters
    ----------
    X:     dataframe to analyze

    Returns
    -------
    Dataframe with the following columns:
        
    index       Variable name
    dtype       Python dtype
    skewness    Skewness statistic calculated by skew function
    
    magnitude   
        2-high              Skewness less than -1 or greater than 1     
        1-medium            Skewness between -1 and -0.5 or 0.5 and 1   
        0-approx_symmetric  Skewness between -0.5 and 0.5             
    
    Example 
    -------
    import exploretransform as et
    df, X, y = et.loadboston()
    et.skewstats(df)
    
               dtype  skewness           magnitude
    cmedv    float64  1.107616              2-high
    crim     float64  5.207652              2-high
    zn       float64  2.219063              2-high
    dis      float64  1.008779              2-high
    b        float64 -2.881798              2-high
    nox      float64  0.727144            1-medium
    age      float64 -0.597186            1-medium
    tax        int64  0.667968            1-medium
    ptratio  float64 -0.799945            1-medium
    lstat    float64  0.903771            1-medium
    lon      float64 -0.204775  0-approx_symmetric
    lat      float64 -0.086421  0-approx_symmetric
    indus    float64  0.294146  0-approx_symmetric
    rm       float64  0.402415  0-approx_symmetric

    ---------- 
    '''    
    
    
    # input checks
    if isinstance(X, (pd.core.frame.DataFrame)): pass
    else: return "Function only accetps dataframes" 
    
    d = X.select_dtypes('number')
    if len(d.columns) == 0: return "Dataframe has no numeric columns" 

    

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


def ascores(X, y):
    
    '''
    ----------   
    
    Parameters
    ----------
    X:     numeric dataframe to compute association measure with y
    y:     series containing target values

    Returns
    -------
    Dataframe with the following association scores:
        
    pearson:    pearson correlation
    kendall:    kendall correlation
    spearman:   spearman correlation
    mic:        maximal information coefficient
    dcor:       distance correlation 
            
    
    Example 
    -------
    import exploretransform as et
    df, X, y = et.loadboston()
    X = X.select_dtypes('number')
    et.ascores(X, y)
    
              pearson   kendall  spearman       mic      dcor
    lon      0.322947  0.278908  0.420940  0.379753  0.435849
    lat      0.006826  0.013724  0.021420  0.234796  0.167030
    crim     0.389582  0.406992  0.562982  0.375832  0.528595
    zn       0.360386  0.340738  0.438768  0.290145  0.404253
    indus    0.484754  0.420263  0.580004  0.414140  0.543948
    nox      0.429300  0.398342  0.565899  0.442515  0.523653
    rm       0.696304  0.485182  0.635092  0.461610  0.711034
    age      0.377999  0.391067  0.551747  0.414676  0.480248
    dis      0.249315  0.313745  0.446392  0.316136  0.382746
    tax      0.471979  0.418005  0.566999  0.336899  0.518158
    ptratio  0.505655  0.397146  0.554168  0.371628  0.520320
    b        0.334861  0.126766  0.186011  0.272469  0.385468
    lstat    0.740836  0.671445  0.857447  0.615427  0.781028


    ---------- 
    
    '''     
    # Convert any ints to float for dcor calculation
    if len(X.select_dtypes(int).columns) > 0:
        for col in X.select_dtypes(int).columns:
            X.loc[:, col] = X[col].astype('float')
        
    r = pd.DataFrame()
    mine = MINE(alpha=0.6, c=15) 
    
    for col in X.columns:
        mine.compute_score(X[col], y)
        r.loc[col, 'pearson'] = abs(stats.pearsonr(X[col], y)[0])
        r.loc[col, 'kendall'] = abs(stats.kendalltau(X[col], y)[0])
        r.loc[col, 'spearman'] = abs(stats.spearmanr(X[col], y)[0])
        r.loc[col, 'mic'] = mine.mic()
        r.loc[col, 'dcor'] = distance_correlation(X[col], y)
    
    return r


class ColumnSelect( BaseEstimator, TransformerMixin ):
    '''
    ----------   
    
    Parameters
    ----------
    X                dataframe
    feature_names    list of column names to select

    Returns
    -------
    dataframe X subsetted by column using feature_names
                 
    Example 
    -------
    import exploretransform as et
    df, X, y = et.loadboston()
    colnames = ['lat', 'lon']
    et.ColumnSelect(colnames).fit_transform(X)

         lat      lon
    0    42.2550 -70.9550
    1    42.2875 -70.9500
    2    42.2830 -70.9360
    3    42.2930 -70.9280
    4    42.2980 -70.9220
    ..       ...      ...
    501  42.2312 -70.9860
    502  42.2275 -70.9910
    503  42.2260 -70.9948
    504  42.2240 -70.9875
    505  42.2210 -70.9825
    
    ---------- 
    '''    
    def __init__( self, feature_names):
        self.feature_names = feature_names
       
    def fit( self, X, y = None ):
        return self 
    
    def transform( self, X,  y = None ):
        return X[self.feature_names]


class CategoricalOtherLevel( BaseEstimator, TransformerMixin ):
    
    '''
    ----------   
    
    Parameters
    ----------
    colname     name of column to create "other" level
    threshold*  any categories occuring less than this percentage will be in 
                "other"
                
                *Note:  using threshold = 0 will create an "other" category
                with no occurances in the training set.  In the test set, any 
                novel categories not seen in train will be assigned "other"

    Returns
    -------
    dataframe X with transformed column "colname"
                 
    Example 
    -------
    import exploretransform as et
    df, X, y = et.loadboston()
    colnames = ['town', 'lat']
    cs = et.ColumnSelect(colnames).fit_transform(X)
    h = et.CategoricalOtherLevel('town', 0.015).fit_transform(cs)
    print(h.head(15))
    
         town      lat
    0   other  42.2550
    1   other  42.2875
    2   other  42.2830
    3   other  42.2930
    4   other  42.2980
    5   other  42.3040
    6   other  42.2970
    7   other  42.3100
    8   other  42.3120
    9   other  42.3160
    10  other  42.3160
    11  other  42.3170
    12  other  42.3060
    13   Lynn  42.2920
    14   Lynn  42.2870
    
    ---------- 
    '''    
    
    def __init__(self, colname, threshold):
        self.colname = colname
        self.threshold = threshold
        self.notothers = pd.Series()
        
    
    def fit( self, X, y = None):

        
        # get frequency table
        f = freq(X[self.colname])[[self.colname, 'perc']]
        
        # get  (not "others") to create lookup table
        self.notothers = f[ f['perc'] > (self.threshold * 100) ][self.colname]
        
        return self
    
    def transform( self, X, y = None):
        # if srs in o then replace with "other"
        for i in X.index:
            if sum(self.notothers.str.contains(X[self.colname][i])):
                pass
            else:
                X.at[i, self.colname] = 'other'
        return X
        

class CorrelationFilter( BaseEstimator, TransformerMixin ):
     
    '''
    ----------   
    
    Parameters
    ----------
    cut         correlation cutoff
    
    methodx     used to calculate correlations amount predictors
    methody*    used to calculate correlations between predictors & target
                *(unused in exploretransform v1.0.0) 
    
    pearson     standard correlation coefficient
    kendall     Kendall Tau correlation coefficient
    spearman    Spearman rank correlation
    callable    callable with input two 1d ndarrays and returning a float. Note 
                that the returned matrix from corr will have 1 along the 
                diagonals and will be symmetric regardless of the callable's 
                behavior.

    Returns
    -------
    Dataframe with columns removed using logic from corrtable() and calcdrop()
                              
    Example 
    -------

    import exploretransform as et
    df, X, y = et.loadboston()
    colnames = X.select_dtypes('number').columns
    cs = et.ColumnSelect(colnames).fit_transform(X)
    cf = et.CorrelationFilter(cut = 0.5).fit_transform(cs)
    print(cf)
    
             lon      lat     crim    zn     rm  ptratio       b
    0   -70.9550  42.2550  0.00632  18.0  6.575     15.3  396.90
    1   -70.9500  42.2875  0.02731   0.0  6.421     17.8  396.90
    2   -70.9360  42.2830  0.02729   0.0  7.185     17.8  392.83
    3   -70.9280  42.2930  0.03237   0.0  6.998     18.7  394.63
    4   -70.9220  42.2980  0.06905   0.0  7.147     18.7  396.90
    ..       ...      ...      ...   ...    ...      ...     ...
    501 -70.9860  42.2312  0.06263   0.0  6.593     21.0  391.99
    502 -70.9910  42.2275  0.04527   0.0  6.120     21.0  396.90
    503 -70.9948  42.2260  0.06076   0.0  6.976     21.0  396.90
    504 -70.9875  42.2240  0.10959   0.0  6.794     21.0  393.45
    505 -70.9825  42.2210  0.04741   0.0  6.030     21.0  396.90
    
    ---------- 
    '''    
      
    def __init__(self, cut = 0.9, methodx = 'pearson', methody = None):
        self.cut = cut
        self.methodx = methodx
        self.methody = methody
        self.ct = pd.DataFrame()
        self.names = []
     
    def fit( self, X, y = None ):
        self.ct = corrtable(X, y, 
                               cut = self.cut, 
                               methodx = self.methodx, 
                               methody = self.methody, 
                               full = True)
        
        self.names = calcdrop(self.ct)
        return self
    
    def transform( self, X, y = None ):
        return X.drop(self.names, axis = 1)
    
