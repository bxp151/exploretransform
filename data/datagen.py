#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generated data for pytest
"""
import os.path
path = '/Users/bxp151/ml/000_special_projects/01_exploretransform/exploretransform'
os.chdir(path)
# import inspect, os.path
# filename = inspect.getframeinfo(inspect.currentframe()).filename
# path = os.path.dirname(os.path.abspath(filename))
import exploretransform as et
import plotnine as pn
import pandas as pd

df, X, y = et.loadboston()

# loadboston()
# df.to_pickle(path + "/data/loadboston.pkl", compression = None)

# explore()
# df.at[0, 'town'] = None
# df.at[0,'lon'] = float('inf')
# df = et.explore(df)
# df.to_pickle(path + "/data/explore.pkl", compression = None)

# peek
et.peek(df).to_pickle(path + "/data/peek.pkl", compression = None)

# freq
# f = et.freq(X['rad'])
# f.to_pickle(path + "/data/freq.pkl", compression = None)

# plotfreq
# p = et.plotfreq(et.freq(X['town']))
# p.labels


# corrtable
# et.corrtable(X,cut = 0.5).to_pickle(path + "/data/corrtable.pkl", compression=None)

# calcdrop
# et.calcdrop(et.corrtable(X,cut = 0.7, full = True))

# skewstatus
#et.skewstats(X).to_pickle(path + "/data/skewstats.pkl", compression = None)

# ascore
# X = X.select_dtypes('number')
# et.ascores(X, y).to_pickle(path + "/data/ascores.pkl", compression=None)

# CategoricalOtherLevel
# col = et.CategoricalOtherLevel('town', 0.015).fit_transform(X)
# col.to_pickle(path + "/data/categoricalotherlevel.pkl", compression = None)

# CorrelationFilter
# cf = et.CorrelationFilter(cut = 0.7).fit_transform(X)
# cf.to_pickle(path + "/data/correlationfilter.pkl", compression = None)
