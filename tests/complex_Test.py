#!/usr/bin/env python
"""
complex_Test.py
================

"""
import os, numpy as np
from opticks.ana.fold import Fold 
SIZE = np.array([1280, 720])

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))
pass


