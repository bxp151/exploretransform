#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 11:00:23 2020

@author: bxp151
"""
import setuptools 

with open("README.md", "r") as fh:
    long_description = fh.read()

# This call to setup() does all the work
setuptools.setup(
    name="exploretransform",
    version="1.0.0",
    author="Brian Pietracatella",
    author_email="bxp151@yahoo.com",
    description="Explore and transform your data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bxp151/exploretransform",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_required = '>=3.4'
)