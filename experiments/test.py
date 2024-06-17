''' Test `lab` Loading
'''
import sys
import os
sys.path.append(os.getcwd())
import libs.data
from libs.utils import path_load_test
import libs
# This will print Hello World!
path_load_test() # <- open this comment
print(libs)
