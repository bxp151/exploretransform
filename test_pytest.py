# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
base = '/Users/bxp151/ml/000_special_projects/01_exploretransform/exploretransform/'
os.chdir(base)
import exploretransform as et
import pandas as pd

###############################################################################
#  nested()
###############################################################################

def nested_load_data():
    
    d0 = [1,2,3]
    d1 = [1,[1,2,3]]
    d2 = pd.Series([1,2,3])
    d3 = pd.Series([1,[1,2,3],3])
    d4 = pd.DataFrame({'first' : [1, 2, 3, 4, 5, 6],
              'second': [2, 4, 5, 6, 7, 8]}
              , columns = ['first', 'second'])
    d5 = pd.DataFrame({'first' : [1, 2, 3, (1,2,3), 4, 5, 6],
              'second': [2, 4, 5, [1,3,4], 6, 7, 8]}
              , columns = ['first', 'second'])

    return [d0,d1,d2,d3,d4,d5]
    
       
def test_nested_typechk():
    assert et.nested(str()) == "Function only accepts: List, Series, or Dataframe"
    
def test_nested_locs():
    d = nested_load_data()
    assert et.nested(d[3], retloc=True) == [1]
    assert et.nested(d[5], retloc=True) == [(3, 0), (3, 1)]
        
def test_nested_bool():
    d = nested_load_data()
    assert et.nested(d[2]) == False


###############################################################################
#  nested()
###############################################################################




'''
Signature: et.nested(obj, retloc=False)

Docstring:
----------   

Parameters
----------
obj:    a list, series, or dataframe
locs:   True or False

Returns
-------

locs = 'True'   Returns locations of nested objects:
                For dataframes, it returns tuples
                For other objects it returns a list of indicies
locs = 'False'  Returns true if any nested objects reside in passed object

Example 
-------
a = [1,2,3]
b = [a,a]
c = (1,2,3)
d = pd.DataFrame({'first' : [1, 2, 3, (1,2,3), 4, 5, 6],
              'second': [2, 4, 5, [1,3,4], 6, 7, 8]}
             , columns = ['first', 'second'])

checkNested(d, locs = True)
[(3, 0), (3, 1)]

checkNested(d)
Out[59]: False

checkNested(b)
Out[60]: True

checkNested(c)
Out[61]: False

checkNested(d)
Out[62]: True


---------- 
'''






