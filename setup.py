#!/usr/bin/env python

import os
from setuptools import setup


local_dir = os.path.dirname(__file__)
if len(local_dir) == 0:
        local_dir = '.'


setup(
    name='ml',
    version='0.0.1',
    description='machine learning utilitites',
    packages=[
        'ml.nets',
        'ml.utils',
        'ml.model',
    ]
)
