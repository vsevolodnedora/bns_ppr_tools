
#
from __future__ import division
from sys import path

from dask.array.ma import masked_array

path.append('modules/')

from _curses import raw
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# import units as ut # for tmerg
import statsmodels.formula.api as smf
from math import pi, log10, sqrt
import scipy.optimize as opt
import matplotlib as mpl
import pandas as pd
import numpy as np
import itertools
import os.path
import cPickle
import math
import time
import copy
import h5py
import csv
import os
import functools
from scipy import interpolate
from scidata.utils import locate
import scidata.carpet.hdf5 as h5
import scidata.xgraph as xg
from matplotlib.mlab import griddata

from matplotlib.ticker import AutoMinorLocator, FixedLocator, NullFormatter, \
    MultipleLocator
from matplotlib.colors import LogNorm, Normalize
from matplotlib.colors import Normalize, LogNorm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib import patches


from preanalysis import LOAD_INIT_DATA
from outflowed import EJECTA_PARS
from preanalysis import LOAD_ITTIME
from plotting_methods import PLOT_MANY_TASKS
from profile import LOAD_PROFILE_XYXZ, LOAD_RES_CORR, LOAD_DENSITY_MODES
from mkn_interface import COMPUTE_LIGHTCURVE, COMBINE_LIGHTCURVES
from combine import TEX_TABLES, COMPARISON_TABLE, TWO_SIMS, THREE_SIMS, ADD_METHODS_ALL_PAR
import units as ut # for tmerg
from utils import *


for letter in "kusi":
    print(letter),


'''==================================================| SETTINGS |===================================================='''

simulations = {"BLh":
                 {
                      "q=1.8": ["BLh_M10201856_M0_LK_SR"],  # 27 ms # Prompt collapse
                      "q=1.7": ["BLh_M10651772_M0_LK_SR"],  # 25 ms # stable #          [SHORT]
                      "q=1.4": ["BLh_M11461635_M0_LK_SR",   # 71 ms                     [LONG]
                                "BLh_M16351146_M0_LK_LR"],  # 49 ms #
                      "q=1.3": ["BLh_M11841581_M0_LK_SR"],  # 21 ms
                      "q=1":   ["BLh_M13641364_M0_LK_SR"]   # 102 ms                    [LONG]
                 },
              "DD2":
                  {
                      "q=1": ["DD2_M13641364_M0_LR",        # 50 ms     3D
                              "DD2_M13641364_M0_SR",        # 104 ms    3D              [LONG]
                              "DD2_M13641364_M0_HR",        # 19 ms     no3D

                              "DD2_M13641364_M0_LR_R04",    # 101 ms    no3D
                              "DD2_M13641364_M0_SR_R04",    # 119 ms    3D (last 3)     [LONG]
                              "DD2_M13641364_M0_HR_R04",    # 18 ms     no3D

                              "DD2_M13641364_M0_LK_LR_R04", # 130 ms    no3D
                              "DD2_M13641364_M0_LK_SR_R04", # 120 ms    3D              [LONG]
                              "DD2_M13641364_M0_LK_HR_R04", # 77 ms     3D
                              ],
                      "q=1.1": ["DD2_M14321300_M0_LR",      # 50 ms     3D
                                "DD2_M14351298_M0_LR"],     # 38 ms     3D
                                # DD2_M14321300_M0_SR [absent
                                # DD2_M14321300_M0_HR [absent]

                      "q=1.2": ["DD2_M14861254_M0_LR",      # 40 ms     3D
                                # DD2_M14861254_M0_SR [absent]
                                "DD2_M14861254_M0_HR",      # 70 ms     3D

                                "DD2_M14971246_M0_LR",      # 46 ms     3D
                                "DD2_M14971245_M0_SR",      # 98 ms     3D              [LONG]
                                "DD2_M14971245_M0_HR",      # 63 ms     3D (till 45)

                                # DD2_M15091235_M0_LK_LR [absent]
                                "DD2_M15091235_M0_LK_SR",   # 115 ms    3D              [LONG]
                                "DD2_M15091235_M0_LK_HR",   # 28 ms     no3D
                                ],

                      "q=1.4": [#DD2_M11461635_M0_LK_LR [absent]
                                "DD2_M11461635_M0_LK_SR",   # 65 ms     3D              [LONG]
                                "DD2_M16351146_M0_LK_LR"]   # 47 ms     3D
                  },
              "LS220":
                  {
                      "q=1": ["LS220_M13641364_M0_HR",      # 41 MS     no3D
                              #"LS220_M13641364_M0_LK_HR", # TOO short. 3ms
                              "LS220_M13641364_M0_LK_SR",   # 43 ms     no3D
                              "LS220_M13641364_M0_LK_SR_restart",   # 39 ms     3D      [SHORT]
                              "LS220_M13641364_M0_LR",      # 48 ms     3D
                              "LS220_M13641364_M0_SR"],     # 51 ms     3D              [SHORT]
                      "q=1.1": ["LS220_M14001330_M0_HR",    # 38 ms     no3D
                                "LS220_M14001330_M0_SR",    # 37 ms     no3D            [SHORT]
                                "LS220_M14351298_M0_HR",    # 38 ms     no3D
                                "LS220_M14351298_M0_SR"],   # 39 ms     np3D            [SHORT]
                      "q=1.2": ["LS220_M14691268_M0_HR",    # 43 ms     np3D
                                "LS220_M14691268_M0_LK_HR", # 24 ms     no3D
                                "LS220_M14691268_M0_LK_SR", # 107 ms    3D              [LONG]
                                "LS220_M14691268_M0_LR",    # 43 ms     no3D
                                "LS220_M14691268_M0_SR"],   # 50 ms     no3D            [SHORT]
                      "q=1.4": ["LS220_M16351146_M0_LK_LR", # 29 ms     3D
                                "LS220_M11461635_M0_LK_SR"],# 56 ms     3D (BH)         [SHORT]
                      "q=1.7": ["LS220_M10651772_M0_LK_SR", # 18 ms     3D *3 of them)  [SHORT]
                                "LS220_M10651772_M0_LK_LR"] # 15 ms     3D (4 of them)
                  },
              "SFHo":
                  {
                      "q=1": ["SFHo_M13641364_M0_HR",       # 23 ms     3D
                              "SFHo_M13641364_M0_LK_HR",    # 24 ms     np3D
                              "SFHo_M13641364_M0_LK_SR",    # 23 ms     no3D
                              "SFHo_M13641364_M0_LK_SR_2019pizza",  # 37 ms no3D        [SHORT]
                              "SFHo_M13641364_M0_SR"],      # 22 ms     3D              [SHORT]
                      "q=1.1":["SFHo_M14521283_M0_HR",      # 29 ms     3D
                               "SFHo_M14521283_M0_LK_HR",   # 27 ms     no3D
                               "SFHo_M14521283_M0_LK_SR",   # 25 ms     no3D
                               "SFHo_M14521283_M0_LK_SR_2019pizza", # 26 ms no3D        [SHORT]
                               "SFHo_M14521283_M0_SR"],     # 33 ms     3D              [SHORT]
                      "q=1.4":["SFHo_M11461635_M0_LK_SR",   # 70 ms     3D              [LONG]
                               "SFHo_M16351146_M0_LK_LR"],  # 31 ms     3D
                      "q=1.7":["SFHo_M10651772_M0_LK_LR"]   # 26 ms     3D
                  },
              "SLy4":
                  {
                      "q=1": [#"SLy4_M13641364_M0_HR",      # precollapse
                              # "SLy4_M13641364_M0_LK_HR",  # crap, absent tarball data
                              "SLy4_M13641364_M0_LK_LR",    # 21 ms     3D
                              "SLy4_M13641364_M0_LK_SR",    # 24 ms     no3D            [SHORT]
                              # "SLy4_M13641364_M0_LR",     #
                              "SLy4_M13641364_M0_SR"],      # 35 ms     3D              [SHORT]
                      "q=1.1":[#"SLy4_M14521283_M0_HR",     # unphysical and premerger
                               "SLy4_M14521283_M0_LR",      # 28 ms     no3D
                               "SLy4_M14521283_M0_SR"],     # 34 ms     3D              [SHORT]
                      "q=1.4":["SLy4_M11461635_M0_LK_SR"],  # 62 ms     3D              [LONG]
                      #"q=1.7":[#"SLy4_M10651772_M0_LK_LR",   # 22 ms     3D
                               #"SLy4_M10651772_M0_LK_SR"   # 32 ms     no3D ! [not even geo etracts]
                       #        ]   # 32 ms     no3D !
                  }
              }

long_sims_dd2 = ["DD2_M13641364_M0_SR", "DD2_M13641364_M0_SR_R04", "DD2_M13641364_M0_LK_SR_R04",
             "DD2_M14971245_M0_SR", "DD2_M13641364_M0_SR_R04", "DD2_M11461635_M0_LK_SR"]
#
long_sims = ["BLh_M11461635_M0_LK_SR", "BLh_M13641364_M0_LK_SR",
             "LS220_M14691268_M0_LK_SR", "SFHo_M11461635_M0_LK_SR", "SLy4_M11461635_M0_LK_SR"]
#
short_sims = ["BLh_M10651772_M0_LK_SR", "LS220_M13641364_M0_LK_SR_restart", "LS220_M13641364_M0_SR",
              "LS220_M14001330_M0_SR0", "LS220_M14351298_M0_SR", "LS220_M14691268_M0_SR",
              "LS220_M11461635_M0_LK_SR", "LS220_M10651772_M0_LK_SR",
              "SFHo_M13641364_M0_LK_SR_2019pizza", "SFHo_M13641364_M0_SR", "SFHo_M14521283_M0_LK_SR_2019pizza",
              "SFHo_M14521283_M0_SR", "SLy4_M13641364_M0_LK_SR", "SLy4_M13641364_M0_SR", "SLy4_M14521283_M0_SR"
              ]
#
def plot_2ejecta_1disk_timehists():
    # columns
    sims = ["DD2_M11461635_M0_LK_SR"]
    # rows
    #
    v_ns = ["vel_inf", "Y_e", "theta", "temperature"]
    masks2 = ["bern_geoend" for i in v_ns]
    masks1 = ["geo" for i in v_ns]
    v_ns_diks = ["Ye", "velz", "theta", "temp"]
    det = 0
    norm_to_m = 0
    _fpath = "slices/" + "rho_modes.h5"
    #
    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + "all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (5.0, 10.0)  # <->, |]
    o_plot.gen_set["figname"] = "timecorr_ej_disk_DD2_M11461635_M0_LK_SR.png"
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.03  # w
    o_plot.gen_set["subplots_adjust_w"] = 0.01
    o_plot.set_plot_dics = []
    #
    i_col = 1
    for sim in sims:
        #
        o_data = ADD_METHODS_ALL_PAR(sim)
        #
        i_row = 1
        # Time of the merger
        fpath = Paths.ppr_sims + sim + "/" + "waveforms/" + "tmerger.dat"
        if not os.path.isfile(fpath):
            raise IOError("File does not exist: {}".format(fpath))
        tmerg = float(np.loadtxt(fname=fpath, unpack=True)) * Constants.time_constant  # ms

        # Total Ejecta Mass
        for v_n, mask1, ls in zip(["Mej_tot", "Mej_tot"], ["geo", "bern_geoend"], ["--", "-"]):
            # Time to end dynamical ejecta
            fpath = Paths.ppr_sims + sim + "/" + "outflow_{}/".format(det) + mask1 + '/' + "total_flux.dat"
            if not os.path.isfile(fpath):
                raise IOError("File does not exist: {}".format(fpath))
            timearr, mass = np.loadtxt(fname=fpath, unpack=True, usecols=(0, 2))
            tend = float(timearr[np.where(mass >= (mass.max() * 0.98))][0]) * 1e3  # ms
            tend = tend - tmerg
            # print(time*1e3); exit(1)
            # Dybamical
            timearr = (timearr * 1e3) - tmerg
            mass = mass * 1e2
            plot_dic = {
                'task': 'line', 'ptype': 'cartesian',
                'position': (i_row, i_col),
                'xarr': timearr, 'yarr': mass,
                'v_n_x': "time", 'v_n_y': "mass",
                'color': "black", 'ls': ls, 'lw': 0.8, 'ds': 'default', 'alpha': 1.0,
                'ymin': 0.05, 'ymax': 2.9, 'xmin': timearr.min(), 'xmax': timearr.max(),
                'xlabel': Labels.labels("t-tmerg"), 'ylabel': "M $[M_{\odot}]$",
                'label': None, 'yscale': 'linear',
                'fontsize': 14,
                'labelsize': 14,
                'fancyticks': True,
                'minorticks': True,
                'sharex': True,  # removes angular citkscitks
                'sharey': True,
                'title': {"text": sim.replace('_', '\_'), 'fontsize': 12},
                'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
            }
            if sim == sims[0]:
                plot_dic["sharey"] = False
                if mask1 == "geo":
                    plot_dic['label'] = r"$M_{\rm{ej}}$ $[10^{-2} M_{\odot}]$"
                else:
                    plot_dic['label'] = r"$M_{\rm{ej}}^{\rm{w}}$ $[10^{-2} M_{\odot}]$"

            o_plot.set_plot_dics.append(plot_dic)
        # Total Disk Mass
        timedisk_massdisk = o_data.get_disk_mass()
        timedisk = timedisk_massdisk[:, 0]
        massdisk = timedisk_massdisk[:, 1]
        timedisk = (timedisk * 1e3) - tmerg
        massdisk = massdisk * 1e1
        plot_dic = {
            'task': 'line', 'ptype': 'cartesian',
            'position': (i_row, i_col),
            'xarr': timedisk, 'yarr': massdisk,
            'v_n_x': "time", 'v_n_y': "mass",
            'color': "black", 'ls': ':', 'lw': 0.8, 'ds': 'default', 'alpha': 1.0,
            'ymin': 0.05, 'ymax': 3.0, 'xmin': timearr.min(), 'xmax': timearr.max(),
            'xlabel': Labels.labels("t-tmerg"), 'ylabel': "M $[M_{\odot}]$",
            'label': None, 'yscale': 'linear',
            'fontsize': 14,
            'labelsize': 14,
            'fancyticks': True,
            'minorticks': True,
            'sharex': True,  # removes angular citkscitks
            'sharey': True,
            # 'title': {"text": sim.replace('_', '\_'), 'fontsize': 12},
            'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
        }
        if sim == sims[0]:
            plot_dic["sharey"] = False
            plot_dic['label'] = r"$M_{\rm{disk}}$ $[10^{-1} M_{\odot}]$"
            plot_dic['legend'] = {'loc': 'best', 'ncol': 1, 'fontsize': 9, 'framealpha': 0.}
        o_plot.set_plot_dics.append(plot_dic)
        #
        i_row = i_row + 1

        # DEBSITY MODES
        o_dm = LOAD_DENSITY_MODES(sim)
        o_dm.gen_set['fname'] = Paths.ppr_sims + sim + "/" + _fpath
        #
        mags1 = o_dm.get_data(1, "int_phi_r")
        mags1 = np.abs(mags1)
        # if sim == "DD2_M13641364_M0_SR": print("m1", mags1)#; exit(1)
        if norm_to_m != None:
            # print('Normalizing')
            norm_int_phi_r1d = o_dm.get_data(norm_to_m, 'int_phi_r')
            # print(norm_int_phi_r1d); exit(1)
            mags1 = mags1 / abs(norm_int_phi_r1d)[0]
        times = o_dm.get_grid("times")
        #
        assert len(times) > 0
        # if sim == "DD2_M13641364_M0_SR": print("m0", abs(norm_int_phi_r1d)); exit(1)
        #
        times = (times * 1e3) - tmerg  # ms
        #
        densmode_m1 = {
            'task': 'line', 'ptype': 'cartesian',
            'xarr': times, 'yarr': mags1,
            'position': (i_row, i_col),
            'v_n_x': 'times', 'v_n_y': 'int_phi_r abs',
            'ls': '-', 'color': 'black', 'lw': 0.8, 'ds': 'default', 'alpha': 1.,
            'label': None, 'ylabel': None, 'xlabel': Labels.labels("t-tmerg"),
            'xmin': timearr.min(), 'xmax': timearr.max(), 'ymin': 1e-4, 'ymax': 1e0,
            'xscale': None, 'yscale': 'log', 'legend': {},
            'fontsize': 14,
            'labelsize': 14,
            'fancyticks': True,
            'minorticks': True,
            'sharex': True,  # removes angular citkscitks
            'sharey': True
        }
        #
        mags2 = o_dm.get_data(2, "int_phi_r")
        mags2 = np.abs(mags2)
        print(mags2)
        if norm_to_m != None:
            # print('Normalizing')
            norm_int_phi_r1d = o_dm.get_data(norm_to_m, 'int_phi_r')
            # print(norm_int_phi_r1d); exit(1)
            mags2 = mags2 / abs(norm_int_phi_r1d)[0]
        # times = (times - tmerg) * 1e3 # ms
        # print(abs(norm_int_phi_r1d)); exit(1)
        densmode_m2 = {
            'task': 'line', 'ptype': 'cartesian',
            'xarr': times, 'yarr': mags2,
            'position': (i_row, i_col),
            'v_n_x': 'times', 'v_n_y': 'int_phi_r abs',
            'ls': '-', 'color': 'gray', 'lw': 0.5, 'ds': 'default', 'alpha': 1.,
            'label': None, 'ylabel': r'$C_m/C_0$', 'xlabel': Labels.labels("t-tmerg"),
            'xmin': timearr.min(), 'xmax': timearr.max(), 'ymin': 1e-4, 'ymax': 9e-1,
            'xscale': None, 'yscale': 'log',
            'legend': {},
            'fontsize': 14,
            'labelsize': 14,
            'fancyticks': True,
            'minorticks': True,
            'sharex': True,  # removes angular citkscitks
            'sharey': True,
            'title': {}  # {'text': "Density Mode Evolution", 'fontsize': 14}
            # 'sharex': True
        }
        #
        if sim == sims[0]:
            densmode_m1['label'] = r"$m=1$"
            densmode_m2['label'] = r"$m=2$"
        if sim == sims[0]:
            densmode_m1["sharey"] = False
            densmode_m1['label'] = r"$m=1$"
            densmode_m1['legend'] = {'loc': 'upper center', 'ncol': 2, 'fontsize': 9, 'framealpha': 0.,
                                     'borderayespad': 0.}
        if sim == sims[0]:
            densmode_m2["sharey"] = False
            densmode_m2['label'] = r"$m=2$"
            densmode_m2['legend'] = {'loc': 'upper center', 'ncol': 2, 'fontsize': 9, 'framealpha': 0.,
                                     'borderayespad': 0.}

        o_plot.set_plot_dics.append(densmode_m2)
        o_plot.set_plot_dics.append(densmode_m1)
        i_row = i_row + 1

        # TIME CORR EJECTA
        for v_n, mask1, mask2 in zip(v_ns, masks1, masks2):
            # Time to end dynamical ejecta
            fpath = Paths.ppr_sims + sim + "/" + "outflow_{}/".format(det) + mask1 + '/' + "total_flux.dat"
            if not os.path.isfile(fpath):
                raise IOError("File does not exist: {}".format(fpath))
            timearr, mass = np.loadtxt(fname=fpath, unpack=True, usecols=(0, 2))
            tend = float(timearr[np.where(mass >= (mass.max() * 0.98))][0]) * 1e3  # ms
            tend = tend - tmerg
            # print(time*1e3); exit(1)
            # Dybamical
            #
            fpath = Paths.ppr_sims + sim + "/" + "outflow_{}/".format(det) + mask1 + '/' + "timecorr_{}.h5".format(v_n)
            if not os.path.isfile(fpath):
                raise IOError("File does not exist: {}".format(fpath))
            # loadind data
            dfile = h5py.File(fpath, "r")
            timearr = np.array(dfile["time"]) - tmerg
            v_n_arr = np.array(dfile[v_n])
            mass = np.array(dfile["mass"])
            timearr, v_n_arr = np.meshgrid(timearr, v_n_arr)
            # mass = np.maximum(mass, mass.min())
            #
            corr_dic2 = {  # relies on the "get_res_corr(self, it, v_n): " method of data object
                'task': 'corr2d', 'dtype': 'corr', 'ptype': 'cartesian',
                'xarr': timearr, 'yarr': v_n_arr, 'zarr': mass,
                'position': (i_row, i_col),
                'v_n_x': "time", 'v_n_y': v_n, 'v_n': 'mass', 'normalize': True,
                'cbar': {},
                'cmap': 'inferno_r',
                'xlabel': Labels.labels("time"), 'ylabel': Labels.labels(v_n, alternative=True),
                'xmin': timearr.min(), 'xmax': timearr.max(), 'ymin': None, 'ymax': None, 'vmin': 1e-4, 'vmax': 1e-1,
                'xscale': "linear", 'yscale': "linear", 'norm': 'log',
                'mask_below': None, 'mask_above': None,
                'title': {},  # {"text": o_corr_data.sim.replace('_', '\_'), 'fontsize': 14},
                # 'text': {'text': lbl.replace('_', '\_'), 'coords': (0.05, 0.9), 'color': 'white', 'fs': 12},
                'axvline': {"x": tend, "linestyle": "--", "color": "black", "linewidth": 1.},
                'mask': "x>{}".format(tend),
                'fancyticks': True,
                'minorticks': True,
                'sharex': True,  # removes angular citkscitks
                'sharey': True,
                'fontsize': 14,
                'labelsize': 14
            }
            if sim == sims[0]:
                corr_dic2["sharey"] = False
            if v_n == v_ns[-1]:
                corr_dic2["sharex"] = False

            if v_n == "vel_inf":
                corr_dic2['ymin'], corr_dic2['ymax'] = 0., 0.45
            elif v_n == "Y_e":
                corr_dic2['ymin'], corr_dic2['ymax'] = 0.05, 0.45
            elif v_n == "temperature":
                corr_dic2['ymin'], corr_dic2['ymax'] = 0.1, 1.8

            o_plot.set_plot_dics.append(corr_dic2)

            # WIND
            fpath = Paths.ppr_sims + sim + "/" + "outflow_{}/".format(det) + mask2 + '/' + "timecorr_{}.h5".format(v_n)
            if not os.path.isfile(fpath):
                raise IOError("File does not exist: {}".format(fpath))
            # loadind data
            dfile = h5py.File(fpath, "r")
            timearr = np.array(dfile["time"]) - tmerg
            v_n_arr = np.array(dfile[v_n])
            mass = np.array(dfile["mass"])
            timearr, v_n_arr = np.meshgrid(timearr, v_n_arr)
            # print(timearr);exit(1)
            # mass = np.maximum(mass, mass.min())
            #
            corr_dic2 = {  # relies on the "get_res_corr(self, it, v_n): " method of data object
                'task': 'corr2d', 'dtype': 'corr', 'ptype': 'cartesian',
                'xarr': timearr, 'yarr': v_n_arr, 'zarr': mass,
                'position': (i_row, i_col),
                'v_n_x': "time", 'v_n_y': v_n, 'v_n': 'mass', 'normalize': True,
                'cbar': {},
                'cmap': 'inferno_r',
                'xlabel': Labels.labels("time"), 'ylabel': Labels.labels(v_n, alternative=True),
                'xmin': timearr.min(), 'xmax': timearr.max(), 'ymin': None, 'ymax': None, 'vmin': 1e-4, 'vmax': 1e-1,
                'xscale': "linear", 'yscale': "linear", 'norm': 'log',
                'mask_below': None, 'mask_above': None,
                'title': {},  # {"text": o_corr_data.sim.replace('_', '\_'), 'fontsize': 14},
                # 'text': {'text': lbl.replace('_', '\_'), 'coords': (0.05, 0.9), 'color': 'white', 'fs': 12},
                'mask': "x<{}".format(tend),
                'fancyticks': True,
                'minorticks': True,
                'sharex': True,  # removes angular citkscitks
                'sharey': True,
                'fontsize': 14,
                'labelsize': 14
            }
            if sim == sims[0]:
                corr_dic2["sharey"] = False
            if v_n == v_ns[-1] and len(v_ns_diks) == 0:
                corr_dic2["sharex"] = False

            if v_n == "vel_inf":
                corr_dic2['ymin'], corr_dic2['ymax'] = 0., 0.45
            elif v_n == "Y_e":
                corr_dic2['ymin'], corr_dic2['ymax'] = 0.05, 0.45
            elif v_n == "theta":
                corr_dic2['ymin'], corr_dic2['ymax'] = 0, 85
            elif v_n == "temperature":
                corr_dic2['ymin'], corr_dic2['ymax'] = 0, 1.8

            if sim == sims[-1] and v_n == v_ns[-1]:
                corr_dic2['cbar'] = {'location': 'right .02 0.', 'label': Labels.labels("mass"),
                                     # 'right .02 0.' 'fmt': '%.1e',
                                     'labelsize': 14,  # 'aspect': 6.,
                                     'fontsize': 14}

            o_plot.set_plot_dics.append(corr_dic2)
            i_row = i_row + 1

        # DISK
        if len(v_ns_diks) > 0:
            d3_corr = LOAD_RES_CORR(sim)
            iterations = d3_corr.list_iterations
            #
            for v_n in v_ns_diks:
                # Loading 3D data
                print("v_n:{}".format(v_n))
                times = []
                bins = []
                values = []
                for it in iterations:
                    fpath = Paths.ppr_sims + sim + "/" + "profiles/" + str(it) + "/" + "hist_{}.dat".format(v_n)
                    if os.path.isfile(fpath):
                        times.append(d3_corr.get_time_for_it(it, "prof"))
                        print("\tLoading it:{} t:{}".format(it, times[-1]))
                        data = np.loadtxt(fpath, unpack=False)
                        bins = data[:, 0]
                        values.append(data[:, 1])
                    else:
                        print("\tFile not found it:{}".format(fpath))

                assert len(times) > 0
                times = np.array(times) * 1e3
                bins = np.array(bins)
                values = np.reshape(np.array(values), newshape=(len(times), len(bins))).T
                #
                times = times - tmerg
                #
                values = values / np.sum(values)
                values = np.maximum(values, 1e-10)
                #
                def_dic = {'task': 'colormesh', 'ptype': 'cartesian',  # 'aspect': 1.,
                           'xarr': times, "yarr": bins, "zarr": values,
                           'position': (i_row, i_col),  # 'title': '[{:.1f} ms]'.format(time_),
                           'cbar': {},
                           'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                           'xlabel': Labels.labels("t-tmerg"), 'ylabel': Labels.labels(v_n, alternative=True),
                           'xmin': timearr.min(), 'xmax': timearr.max(), 'ymin': bins.min(), 'ymax': bins.max(),
                           'vmin': 1e-6,
                           'vmax': 1e-2,
                           'fill_vmin': False,  # fills the x < vmin with vmin
                           'xscale': None, 'yscale': None,
                           'mask': None, 'cmap': 'inferno_r', 'norm': "log",
                           'fancyticks': True,
                           'minorticks': True,
                           'title': {},
                           # "text": r'$t-t_{merg}:$' + r'${:.1f}$'.format((time_ - tmerg) * 1e3), 'fontsize': 14
                           # 'sharex': True,  # removes angular citkscitks
                           'text': {},
                           'fontsize': 14,
                           'labelsize': 14,
                           'sharex': True,
                           'sharey': True,
                           }
                if sim == sims[-1] and v_n == v_ns_diks[-1]:
                    def_dic['cbar'] = {'location': 'right .02 0.',  # 'label': Labels.labels("mass"),
                                       # 'right .02 0.' 'fmt': '%.1e',
                                       'labelsize': 14,  # 'aspect': 6.,
                                       'fontsize': 14}
                if v_n == v_ns[0]:
                    def_dic['text'] = {'coords': (1.0, 1.05), 'text': sim.replace("_", "\_"), 'color': 'black',
                                       'fs': 16}
                if v_n == "Ye":
                    def_dic['ymin'] = 0.05
                    def_dic['ymax'] = 0.45
                if v_n == "velz":
                    def_dic['ymin'] = -.25
                    def_dic['ymax'] = .25
                elif v_n == "temp":
                    # def_dic['yscale'] = "log"
                    def_dic['ymin'] = 1e-1
                    def_dic['ymax'] = 2.5e1
                elif v_n == "theta":
                    def_dic['ymin'] = 0
                    def_dic['ymax'] = 85
                    def_dic["yarr"] = 90 - (def_dic["yarr"] / np.pi * 180.)
                #
                if v_n == v_ns_diks[-1]:
                    def_dic["sharex"] = False
                if sim == sims[0]:
                    def_dic["sharey"] = False

                o_plot.set_plot_dics.append(def_dic)
                i_row = i_row + 1

        i_col = i_col + 1
    o_plot.main()
    exit(1)
if __name__  == '__main__':
    plot_2ejecta_1disk_timehists()