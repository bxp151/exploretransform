#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 06:58:11 2020

@author: bxp151
"""

import wget
import os
base = '/Users/bxp151/ml/000_special_projects/01_exploretransform/exploretransform/'
os.chdir(base + 'data')

source = 'http://lib.stat.cmu.edu/datasets/boston_corrected.txt'
wget.download(source)

