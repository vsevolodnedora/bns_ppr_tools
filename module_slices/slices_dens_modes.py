"""
    description
"""

from __future__ import division
# from sys import path
# path.append('modules/')
# import os.path
# import click
# import h5py
# from argparse import ArgumentParser
# from math import pi, log10
# import sys
# from scidata.utils import locate
# import scidata.carpet.hdf5 as h5
# from scidata.carpet.interp import Interpolator

# from glob import glob
# from _curses import raw
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib import ticker
# import matplotlib.pyplot as plt
# from matplotlib import rc
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# import scivis.units as ut # for tmerg
# import statsmodels.formula.api as smf
# import scipy.optimize as opt
# from math import pi, sqrt
# import matplotlib as mpl
# import pandas as pd
import numpy as np
# from glob import glob
# import itertools
# import os.path
# import cPickle
# import click
# import time
# import copy
import h5py
# import csv
import os
#
# import time



# import scidata.xgraph as xg


# from scipy import interpolate
# cmap = plt.get_cmap("viridis")
# from sklearn.linear_model import LinearRegression-
# from scipy.optimize import fmin
# from matplotlib.ticker import AutoMinorLocator, FixedLocator, NullFormatter, \
#     MultipleLocator
# from matplotlib.colors import LogNorm, Normalize

# from utils import *
# from uutils import Tools, Printcolor
# from module_preanalysis.module_preanalysis import LOAD_ITTIME
# from plotting_methods import PLOT_MANY_TASKS


# from uutils import Printcolor, REFLEVEL_LIMITS

# from plotting.plotting_methods import PLOT_MANY_TASKS


from uutils import Printcolor
# from module_preanalysis.module_preanalysis import LOAD_ITTIME
# from slices_methods import COMPUTE_STORE

def compute_density_modes(o_slice, rls, outdir, rewrite=True):


    if not len(rls) == 1:
        raise NameError("for task 'dm' please set one reflevel: --rl ")
    #
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    #
    rl = rls[0]
    #
    mmax = 8
    #
    fname = "rho_modes.h5"
    fpath = outdir + fname
    #
    if True:#try:
        if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
            if os.path.isfile(fpath): os.remove(fpath)
            Printcolor.print_colored_string(["task:", "rho modes", "rl:", str(rl), "mmax:", str(mmax), ":", "computing"],
                                 ["blue", "green", "blue", "green", "blue", "green", "", "green"])
            times, iterations, xcs, ycs, modes = o_slice.get_rho_modes_for_rl(rl=rl, mmax=mmax)
            dfile = h5py.File(fpath, "w")
            dfile.create_dataset("times", data=times)  # times that actually used
            dfile.create_dataset("iterations", data=iterations)  # iterations for these times
            dfile.create_dataset("xc", data=xcs)  # x coordinate of the center of mass
            dfile.create_dataset("yc", data=ycs)  # y coordinate of the center of mass
            for m in range(mmax + 1):
                group = dfile.create_group("m=%d" % m)
                group["int_phi"] = np.zeros(0, )  # NOT USED (suppose to be data for every 'R' in disk and NS)
                group["int_phi_r"] = np.array(modes[m]).flatten()  # integrated over 'R' data
            dfile.close()
        else:
            Printcolor.print_colored_string(["task:", "rho modes", "rl:", str(rl), "mmax:", str(mmax), ":", "skipping"],
                                 ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
    # except:
    #     Printcolor.print_colored_string(["task:", "rho modes", "rl:", str(rl), "mmax:", str(mmax), ":", "failed"],
    #                          ["blue", "green", "blue", "green", "blue", "green", "", "red"])