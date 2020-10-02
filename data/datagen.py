#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 14:10:33 2020

@author: bxp151
"""
import os.path
path = '/Users/bxp151/ml/000_special_projects/01_exploretransform/exploretransform'
os.chdir(path)
import exploretransform as et
# import pandas as pd

df, X, y = et.loadboston()

# loadboston()
# df.to_pickle(path + "/data/loadboston.pkl", compression = None)

# describe()
# df.at[0, 'town'] = None
# df.at[0,'lon'] = float('inf')
# df = et.describe(df)
# df.to_pickle(path + "/data/describe.pkl", compression = None)

# glimpse
# et.glimpse(df).to_pickle(path + "/data/glimpse.pkl", compression = None)

# freq
# f = et.freq(X['rad'])
# f.to_pickle(path + "/data/freq.pkl", compression = None)

# plotfreq
et.plotfreq(et.freq(X['town']))


# corrtable
# et.corrtable(X,cut = 0.5).to_pickle(path + "/data/corrtable.pkl", compression=None)

# calcdrop
# et.calcdrop(et.corrtable(X,cut = 0.7, full = True))

# skewstatus
#et.skewstats(X).to_pickle(path + "/data/skewstats.pkl", compression = None)
