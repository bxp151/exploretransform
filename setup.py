#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 11:00:23 2020

@author: bxp151
"""
import setuptools 


long_description = open('README.md').read()

# This call to setup() does all the work
setuptools.setup(
    name="exploretransform",
    version="1.0.5",
    author="Brian Pietracatella",
    author_email="bpietrac@gmail.com",
    description="Explore and transform your data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    url="https://github.com/bxp151/exploretransform",
    install_requires=['numpy', 'pandas', 'plotnine', 'scipy', 'sklearn',
                      'minepy', 'dcor'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["exploretransform"],
    python_requires = '>=3.6.9'
)