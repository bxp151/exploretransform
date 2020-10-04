import inspect, os.path
filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
import exploretransform as et
import pandas as pd
   


###############################################################################
#  nested(obj, retloc = False)
###############################################################################
       
def test_nested_typechk():
    d0 = 'string'
    assert et.nested(d0) == "Function only accepts: List, Series, or Dataframe"
    
def test_nested_locs():
    d1 = pd.Series([1,[1,2,3],3])
    d2 = pd.DataFrame({'first' : [1, 2, 3, (1,2,3), 4, 5, 6],
      'second': [2, 4, 5, [1,3,4], 6, 7, 8]}
      , columns = ['first', 'second'])
    assert et.nested(d1, retloc=True) == [1]
    assert et.nested(d2, retloc=True) == [(3, 0), (3, 1)]
        
def test_nested_bool():
    d3 = pd.Series([1,2,3])
    assert et.nested(d3) == False
    
###############################################################################
#  loadboston()
###############################################################################

def test_loadboston_df():
    d0,X,y = et.loadboston()
    d1 = pd.read_pickle(path + '/data/loadboston.pkl')
    assert d0.equals(d1)

###############################################################################
#  explore(X)
###############################################################################

def test_explore_type():
    d0 = 'string'
    assert et.explore(d0) == "Function only accetps dataframes"

def test_explore_nest():
    d1 = pd.DataFrame({'first' : [1, 2, 3, (1,2,3), 4, 5, 6],
      'second': [2, 4, 5, [1,3,4], 6, 7, 8]} ,columns = ['first', 'second'])
    assert et.explore(d1) == "Please collapse any nested values in your dataframe"
    

def test_explore_pos():
    d0,X,y = et.loadboston()
    # Adding NA and inf values to dataframe
    d0.at[0, 'town'] = None
    d0.at[0,'lon'] = float('inf')
    d1 = pd.read_pickle(path + "/data/explore.pkl") 
    assert et.explore(d0).equals(d1)

###############################################################################
#  peek(X) 
###############################################################################

def test_peek_pos():
    d0,X,y = et.loadboston()
    d1 = pd.read_pickle(path + "/data/peek.pkl") 
    assert et.peek(d0).equals(d1)
    
def test_peek_typechk():
    d0 = 'string'
    assert et.peek(d0) == "Function only accetps dataframes"

def test_peek_nest():
    d0 = pd.DataFrame({'first' : [1, 2, 3, (1,2,3), 4, 5, 6],
      'second': [2, 4, 5, [1,3,4], 6, 7, 8]} ,columns = ['first', 'second'])
    assert et.peek(d0) == "Please collapse any nested values in your dataframe"

###############################################################################
#  plotfreq(freqdf) 
###############################################################################  

# Manually compare outputs:
# et.plotfreq(et.freq(X['town'])) to (path + "/data/plotfreq.jpg")

###############################################################################
#  def freq(srs)
############################################################################### 

def test_freq_typechk():
    d0 = [1,2,3]
    assert et.freq(d0) == "Function only accetps series"


def test_freq():
    d0,X,y = et.loadboston()
    d1 = pd.read_pickle(path + "/data/freq.pkl") 
    d2 = et.freq(d0['rad'])
    assert d1.equals(d2)

###############################################################################
#  corrtable(X, y = None, cut = 0.9, methodx = 'spearman', methody = None, full = False)
############################################################################### 

def test_corrtable():
    df,d0,y = et.loadboston()
    d1 = pd.read_pickle(path + "/data/corrtable.pkl")
    assert et.corrtable(d0,cut = 0.5).equals(d1)   

###############################################################################
#  def calcdrop(ct)
############################################################################### 

def test_calcdrop_pos():
    df,d0,y = et.loadboston()
    d1 = et.corrtable(d0,cut = 0.7, full = True)
    d2 = et.calcdrop(d1)
    d3 = ['nox', 'indus', 'dis', 'crim']
    assert set(d2) == set(d3)

###############################################################################
#  skewstats(X)
############################################################################### 

def test_skewstats_typechk():
    d0 = 'string'
    assert et.skewstats(d0) == "Function only accetps dataframes"

def test_skewstats_nonum():
    df,d0,y = et.loadboston()
    d0 = d0[['town', 'rad']]
    assert et.skewstats(d0) == "Dataframe has no numeric columns"
    

def test_skewstats_pos():
    df,d0,y = et.loadboston()
    d1 = pd.read_pickle(path + "/data/skewstats.pkl")
    assert et.skewstats(d0).equals(d1)

###############################################################################
#  ascores(X, y)
############################################################################### 

def test_ascores_pos():
    df,d0,d1 = et.loadboston()
    d0 = d0.select_dtypes('number')
    d2 = pd.read_pickle(path + "/data/ascores.pkl")
    assert et.ascores(d0, d1).equals(d2)

###############################################################################
# class ColumnSelect( BaseEstimator, TransformerMixin ):
###############################################################################

def test_ColumnSelect_pos():
    df,d0,y = et.loadboston()
    assert et.ColumnSelect('lon').fit_transform(d0).equals(d0['lon'])

###############################################################################
# class CategoricalOtherLevel( BaseEstimator, TransformerMixin )
###############################################################################

def test_CategoricalOtherLevel_pos():
    df,d0,y = et.loadboston()
    d1 = pd.read_pickle(path + "/data/categoricalotherlevel.pkl")
    assert et.CategoricalOtherLevel('town', 0.015).fit_transform(d0).equals(d1)

###############################################################################
# class CorrelationFilter( BaseEstimator, TransformerMixin )
###############################################################################

def test_CorrelationFilter_pos():
    df,d0,y = et.loadboston()
    d1 = pd.read_pickle(path + "/data/correlationfilter.pkl")
    assert et.CorrelationFilter(cut = 0.7).fit_transform(d0).equals(d1)