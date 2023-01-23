import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import special, optimize, integrate, stats
from scipy.interpolate import UnivariateSpline, RectBivariateSpline, interp1d, interp2d, BarycentricInterpolator
from time import time
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from timeit import timeit
from time import time
from copy import copy
import sys

from astropy.io import fits   # for saving/reeading maps
import colormaps as cmaps  # for viridis and plasma colormaps

# for faster FFT
import pyfftw
import importlib
pyfftw.interfaces.cache.enable() # so subsequent FFTs use the wisdom from the first one
## however, this wisdom is only kept for 0.1sec, or for x seconds if using:
##pyfftw.interfaces.cache.set_keepalive_time(x)

##################################################################################
# for pretty plots

from matplotlib import rc
#rc('font',**{'size':'20','family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
rc('font',**{'size':'22','family':'serif','serif':['CMU serif']})
rc('mathtext', **{'fontset':'cm'})
rc('text', usetex=True)
rc('text.latex', preamble='\\usepackage{amsmath}, \\usepackage{amssymb}')
#rc('font', size=20)
rc('legend',**{'fontsize':'18'})

##################################################################################

import basic_functions
importlib.reload(basic_functions)
from basic_functions import *

