from __future__ import division
import os.path
import h5py
from sys import path
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# # from dask.array.ma import masked_array
#
# # path.append('modules/')
#
# from _curses import raw
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib import ticker
# # import matplotlib.pyplot as plt
# from matplotlib import rc
# # plt.rc('text', usetex=True)
# # plt.rc('font', family='serif')
# # import units as ut # for tmerg
# import statsmodels.formula.api as smf
# from math import pi, log10, sqrt
# import scipy.optimize as opt
# import matplotlib as mpl
# import pandas as pd
# import numpy as np
# import itertools
# import cPickle
# import math
# import time
# import copy

# import csv
# import os
# import functools
# from scipy import interpolate
# from scidata.utils import locate
# import scidata.carpet.hdf5 as h5
# import scidata.xgraph as xg
# from matplotlib.mlab import griddata

# from matplotlib.ticker import AutoMinorLocator, FixedLocator, NullFormatter, \
#     MultipleLocator
# from matplotlib.colors import LogNorm, Normalize
# from matplotlib.colors import Normalize, LogNorm
# from matplotlib.collections import PatchCollection
# from matplotlib.patches import Rectangle
# from matplotlib import patches


from preanalysis import LOAD_INIT_DATA
from outflowed import EJECTA_PARS
from preanalysis import LOAD_ITTIME
from plotting_methods import PLOT_MANY_TASKS
from profile import LOAD_PROFILE_XYXZ, LOAD_RES_CORR, LOAD_DENSITY_MODES
from mkn_interface import COMPUTE_LIGHTCURVE, COMBINE_LIGHTCURVES
from combine import TEX_TABLES, COMPARISON_TABLE, TWO_SIMS, THREE_SIMS, ADD_METHODS_ALL_PAR, ALL_SIMULATIONS_TABLE
import units as ut # for tmerg
from utils import *


for letter in "kusi":
    print(letter),


'''==================================================| SETTINGS |===================================================='''

resolutions = {"HR":123., "SR":185.,"LR":246.}

simulations2 = {
    "BLh":
        {
           "q=1.8":{
               "BLh_M10201856_M0_LK":
                   # 20 ms | BH | 3D | PC     # 25 ms | BH | 3D(4) | PC  # 27 ms | BH | 3D(3) | PC
                   ["BLh_M10201856_M0_LK_HR", "BLh_M10201856_M0_LK_LR", "BLh_M10201856_M0_LK_SR"],
               "BLh_M10201856_M0":
                   # 57 ms | BH | no3D       # 65 ms | BH | 3D(7)      # 37 ms | BH | 3D (5) | PC
                   ["BLh_M10201856_M0_HR", "BLh_M10201856_M0_LR", "BLh_M10201856_M0_SR"]
           },
           "q=1.7":{
               "BLh_M10651772_M0_LK":
                   # 30 ms | stable | 3D      # 74 ms | stable | 3D |
                   ["BLh_M10651772_M0_LK_SR", "BLh_M10651772_M0_LK_LR"]
           },
           "q=1.5":{
               "BLh_M11041699_M0_LK":
                   # 56ms | stable | 3D
                   ["BLh_M11041699_M0_LK_LR"],
               "BLh_M11041699_M0":
                   # 27/40 ms | stable | 3D |missing
                   ["BLh_M11041699_M0_LR"]
           },
           "q=1.4":{
               "BLh_M11461635_M0_LK":
                   # 71ms | stable | 3D       # 49 | stable (wrong merger time) |
                   ["BLh_M11461635_M0_LK_SR", "BLh_M16351146_M0_LK_LR"]
           },
           "q=1.3":{
               "BLh_M11841581_M0_LK":
                   # 75ms | stable | no3D     # 21ms | stable | 3D(5)
                   ["BLh_M11841581_M0_LK_LR", "BLh_M11841581_M0_LK_SR"],
               "BLh_M11841581_M0":
                    # 28 ms | stable | 3D
                   ["BLh_M11841581_M0_LR"]
           },
           "q=1.2":{
               "BLh_M12591482_M0":
                   # 27ms | stable | 3D
                   ["BLh_M12591482_M0_LR"],
               "BLh_M12591482_M0_LK":
                   # 81ms | stable | no3D
                   ["BLh_M12591482_M0_LK_LR"]
           },
           "q=1":{
               "BLh_M13641364_M0_LK":
                   # 102 ms | stable | 3D     # 54 ms | stable | 3D(7)
                   ["BLh_M13641364_M0_LK_SR", "BLh_M13641364_M0_LK_LR"],
               "BLh_M13641364_M0":
                   # 47 ms | stable | 3D
                   ["BLh_M13641364_M0_LR"]
           }
        },
    "DD2":
        {
           "q=1":{
               "DD2_M13641364_M0":
                   # 50ms | stable | 3D    # 110ms | stable | 3D   # 19 ms | stable | no3D
                   ["DD2_M13641364_M0_LR", "DD2_M13641364_M0_SR", "DD2_M13641364_M0_HR"],
               "DD2_M13641364_M0_R04":
                   # 101ms | stable | 3D(3)    # 120ms | stable | 3D(5)   # 17ms | stable | no3D
                   ["DD2_M13641364_M0_LR_R04", "DD2_M13641364_M0_SR_R04", "DD2_M13641364_M0_HR_R04"],
               "DD2_M13641364_M0_LK_R04":
                    # 130ms | stable | 3D(3)       # 120ms | stable | 3D(51ms+) # 82ms | stable | 3D
                   ["DD2_M13641364_M0_LK_LR_R04", "DD2_M13641364_M0_LK_SR_R04", "DD2_M13641364_M0_LK_HR_R04"]
           },
           "q=1.1":{
               "DD2_M14321300_M0":
                   # 51ms | stable | 3D
                   ["DD2_M14321300_M0_LR"],
               "DD2_M14351298_M0":
                     # 38ms | stable | 3D
                    ["DD2_M14351298_M0_LR"]
           },
           "q=1.2":{
               "DD2_M14861254_M0":
                    # 40ms | stable | 3D    # 70ms | stable | 3D
                   ["DD2_M14861254_M0_LR", "DD2_M14861254_M0_HR"],
               "DD2_M14971246_M0":
                    # 47ms | stable | 3D   # 99ms | stable | 3D   # 63ms | stable | 3D
                   ["DD2_M14971246_M0_LR", "DD2_M14971245_M0_SR", "DD2_M14971245_M0_HR"],
               "DD2_M15091235_M0_LK":
                   # 113ms | stable | 3D(60ms+) # 28ms | stable | no3D
                   ["DD2_M15091235_M0_LK_SR", "DD2_M15091235_M0_LK_HR"]
           },
           "q=1.4":{
               "DD2_M11461635_M0_LK":
                    # # 47ms | stable | 3D     # 65ms | stable | 3D (wrong init.data)
                   ["DD2_M16351146_M0_LK_LR"]  #, "DD2_M11461635_M0_LK_SR"]
           }
        },
    "LS220":
        {
           "q=1":{
               "LS220_M13641364_M0":
                   # 50ms | BH | 3D          # 41ms | BH | no3D      # 49ms | BH | 3D | wrong BH time
                   ["LS220_M13641364_M0_SR", "LS220_M13641364_M0_HR", "LS220_M13641364_M0_LR"],
               "LS220_M13641364_M0_LK":
                   # 38ms | BH | 3D
                   ["LS220_M13641364_M0_LK_SR_restart"]
           },
           "q=1.1":{
               "LS220_M14001330_M0":
                   # 38ms | BH | no3D        # 37ms | BH | no3D
                   ["LS220_M14001330_M0_HR", "LS220_M14001330_M0_SR"],
               "LS220_M14351298_M0":
                   # 38ms | stable | no 3D    # 39ms | BH | no3D
                   ["LS220_M14351298_M0_HR", "LS220_M14351298_M0_SR"]
           },
           "q=1.2":{
               "LS220_M14691268_M0":
                    # 49ms | stable | no3D   # 43ms | BH | no3D       # 43ms | stable | np3D
                   ["LS220_M14691268_M0_SR", "LS220_M14691268_M0_HR", "LS220_M14691268_M0_LR"],
               "LS220_M14691268_M0_LK":
                    # 24ms | stable | no3D       # 107ms | long-lived BH | 3D(60ms+)
                   ["LS220_M14691268_M0_LK_HR", "LS220_M14691268_M0_LK_SR"],
           },
           "q=1.4":{
               "LS220_M16351146_M0_LK":
                    # 30ms | BH | 3D            # missing 38ms | BH | 3D
                   ["LS220_M16351146_M0_LK_LR", "LS220_M11461635_M0_LK_SR"]
           },
           "q=1.7":{
               "LS220_M10651772_M0_LK":
                   # missing 23 ms | BH | PC    # 16ms | BH | 3D | PC
                   ["LS220_M10651772_M0_LK_SR", "LS220_M10651772_M0_LK_LR"]
           }
        },
    "SFHo":
        {
           "q=1":{
               "SFHo_M13641364_M0":
                    # 22ms | BH | 3D        # 23ms | BH | 3D
                   ["SFHo_M13641364_M0_SR", "SFHo_M13641364_M0_HR"],
               "SFHo_M13641364_M0_LK":
                    # 29ms | BH | no3D
                   ["SFHo_M13641364_M0_LK_SR"],
               "SFHo_M13641364_M0_LK_p2019":
                    # 24ms | BH | no3D          # 37ms | BH | no3D
                   ["SFHo_M13641364_M0_LK_HR", "SFHo_M13641364_M0_LK_SR_2019pizza"]
           },
           "q=1.1":{
               "SFHo_M14521283_M0":
                    # 29ms | BH | 3D        # 32ms | BH | 3D
                   ["SFHo_M14521283_M0_HR", "SFHo_M14521283_M0_SR"],
               "SFHo_M14521283_M0_LK_p2019":
                    # 26ms | BH | no3D         # 26ms | BH | no3D
                   ["SFHo_M14521283_M0_LK_HR", "SFHo_M14521283_M0_LK_SR_2019pizza"],
               "SFHo_M14521283_M0_LK_SR":
                   # 24ms | BH | no3D
                   ["SFHo_M14521283_M0_LK_SR"]
           },
           "q=1.4":{
               # "SFHo_M11461635_M0_LK": # [ wrong init. data]
               #      # 65(missing5)ms | stable | 3D
               #     ["SFHo_M11461635_M0_LK_SR"],
               "SFHo_M16351146_M0_LK_p2019":
                    # 31ms | BH | 3D
                   ["SFHo_M16351146_M0_LK_LR"]
           },
           # "q=1.7":{
           #     "SFHo_M10651772_M0_LK": # [wrong init.data]
           #          # 21ms | BH | 3D | PC      # 26ms | stable | 3D [might be wrong]
           #         ["SFHo_M10651772_M0_LK_SR","SFHo_M10651772_M0_LK_LR"]
           # }
        },
    "SLy4":
        {
           "q=1":{
               "SLy4_M13641364_M0_LK":
                   # 21ms | BH | no3D         # 24ms | BH | no3D
                  ["SLy4_M13641364_M0_LK_LR", "SLy4_M13641364_M0_LK_SR"],
               "SLy4_M13641364_M0":
                  # 36ms | BH | 3D
                  ["SLy4_M13641364_M0_SR"]
        },
           "q=1.1":{
               "SLy4_M14521283_M0":
                   # 28ms | BH | extracting 3D # 34ms | BH | 3D
                   ["SLy4_M14521283_M0_LR", "SLy4_M14521283_M0_SR"] # extracting profiles
           },
           # "q=1.4":{
           #     "SLy4_M11461635_M0_LK": # [wrong init. data]
           #         # 67ms | stable | 3D [ might be wrong! ]
           #         ["SLy4_M11461635_M0_LK_SR"]
           # },
           "q=1.8":{
               "SLy4_M10201856_M0_LK":
                   # 17ms | BH | 3D | PC
                   ["SLy4_M10201856_M0_LK_SR"]
           }
    }
}

# print("Simulations: ")

# for eos in simulations2.keys():
#     for q in simulations2[eos].keys():
#         for unique in simulations2[eos][q].keys():
#             for sim in simulations2[eos][q][unique]:
#                 print(sim + " "),
# print("\ndone")

# ./analyze.sh BLh_M10651772_M0_LK_SR /data1/numrel/WhiskyTHC/Backup/2018/GW170817/ /data01/numrel/vsevolod.nedora/postprocessed4/







simulations = {"BLh":
                 {
                      "q=1.8": ["BLh_M10201856_M0_LK_HR",   # 21 ms # promped
                                "BLh_M10201856_M0_LK_LR",   # 22 ms # promped
                                "BLh_M10201856_M0_LK_SR",   # 28 ms # promped
                                "BLh_M10201856_M0_HR",      # 40 ms # promped
                                "BLh_M10201856_M0_LR",      # 40 ms # promped
                                "BLh_M10201856_M0_SR"       # 23 ms # promped
                                ],
                      "q=1.7": ["BLh_M10651772_M0_LK_SR",   # 25 ms # stable #          [SHORT]
                                #"BLh_M10651772_M0_LK_HR",  # Failed at merger
                                "BLh_M10651772_M0_LK_LR"    # 70 ms # stable
                                ],
                      "q=1.5": [
                                "BLh_M11041699_M0_LK_LR",   # 55 ms # stable
                                "BLh_M11041699_M0_LR"       # 28 ms # stable
                                ],
                      "q=1.4": ["BLh_M11461635_M0_LK_SR",   # 71 ms # stable             [LONG]
                                "BLh_M16351146_M0_LK_LR"    # 49 ms #
                                ],
                      "q=1.3": [
                                "BLh_M11841581_M0_LK_LR",   # 74 ms
                                "BLh_M11841581_M0_LK_SR",   # 21 ms                     [SHORT]
                                # "BLh_M11841581_M0_LK_HR"  # failed!
                                "BLh_M11841581_M0_LR"       # 28 ms
                                ],
                      "q=1.2": [
                                "BLh_M12591482_M0_LR",      # 27 ms
                                "BLh_M12591482_M0_LK_LR"    # 80 ms
                      ],
                      "q=1":   ["BLh_M13641364_M0_LK_SR",   # 105 ms                     [LONG]
                                # "BLh_M13641364_M0_LK_HR", # too short
                                "BLh_M13641364_M0_LK_LR",   # 55ms
                                "BLh_M13641364_M0_LR"       # 48 ms
                                ]
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
                                "DD2_M16351146_M0_LK_LR"    # 47 ms     3D
                                ]
                  },
              "LS220":
                  {
                      "q=1": ["LS220_M13641364_M0_HR",      # 41 MS     no3D
                              #"LS220_M13641364_M0_LK_HR", # TOO short. 3ms
                              "LS220_M13641364_M0_LK_SR",   # 43 ms     no3D
                              "LS220_M13641364_M0_LK_SR_restart",   # 39 ms     3D      [SHORT]
                              "LS220_M13641364_M0_LR",      # 48 ms     3D
                              "LS220_M13641364_M0_SR"       # 51 ms     3D              [SHORT]
                              ],
                      "q=1.1": ["LS220_M14001330_M0_HR",    # 38 ms     no3D
                                "LS220_M14001330_M0_SR",    # 37 ms     no3D            [SHORT]
                                "LS220_M14351298_M0_HR",    # 38 ms     no3D
                                "LS220_M14351298_M0_SR"     # 39 ms     np3D            [SHORT]
                                ],
                      "q=1.2": ["LS220_M14691268_M0_HR",    # 43 ms     np3D
                                "LS220_M14691268_M0_LK_HR", # 24 ms     no3D
                                "LS220_M14691268_M0_LK_SR", # 107 ms    3D              [LONG]
                                "LS220_M14691268_M0_LR",    # 43 ms     no3D
                                "LS220_M14691268_M0_SR"     # 50 ms     no3D            [SHORT]
                                ],
                      "q=1.4": ["LS220_M16351146_M0_LK_LR", # 29 ms     3D
                                "LS220_M11461635_M0_LK_SR"  # 56 ms     3D (BH)            [SHORT]
                                ],
                      "q=1.7": ["LS220_M10651772_M0_LK_SR", # 18 ms     3D *3 of them)  [SHORT]
                                "LS220_M10651772_M0_LK_LR"  # 15 ms     3D (4 of them)
                                ]
                  },
              "SFHo":
                  {
                      "q=1": ["SFHo_M13641364_M0_HR",       # 23 ms     3D
                              "SFHo_M13641364_M0_LK_HR",    # 24 ms     np3D
                              "SFHo_M13641364_M0_LK_SR",    # 23 ms     no3D
                              "SFHo_M13641364_M0_LK_SR_2019pizza",  # 37 ms no3D        [SHORT]
                              "SFHo_M13641364_M0_SR"        # 22 ms     3D              [SHORT]
                              ],
                      "q=1.1":["SFHo_M14521283_M0_HR",      # 29 ms     3D
                               "SFHo_M14521283_M0_LK_HR",   # 27 ms     no3D
                               "SFHo_M14521283_M0_LK_SR",   # 25 ms     no3D
                               "SFHo_M14521283_M0_LK_SR_2019pizza", # 26 ms no3D        [SHORT]
                               "SFHo_M14521283_M0_SR"       # 33 ms     3D              [SHORT]
                               ],
                      "q=1.4":["SFHo_M11461635_M0_LK_SR",   # 70 ms     3D              [LONG]
                               "SFHo_M16351146_M0_LK_LR"    # 31 ms     3D
                               ],
                      "q=1.7":["SFHo_M10651772_M0_LK_LR"    # 26 ms     3D
                               ]
                  },
              "SLy4":
                  {
                      "q=1": [#"SLy4_M13641364_M0_HR",      # precollapse
                              # "SLy4_M13641364_M0_LK_HR",  # crap, absent tarball data
                              "SLy4_M13641364_M0_LK_LR",    # 21 ms     3D
                              "SLy4_M13641364_M0_LK_SR",    # 24 ms     no3D            [SHORT]
                              # "SLy4_M13641364_M0_LR",     #
                              "SLy4_M13641364_M0_SR"        # 35 ms     3D              [SHORT]
                                ],
                      "q=1.1":[#"SLy4_M14521283_M0_HR",     # unphysical and premerger
                               "SLy4_M14521283_M0_LR",      # 28 ms     no3D
                               "SLy4_M14521283_M0_SR"       # 34 ms     3D              [SHORT]
                                ],
                      "q=1.4":["SLy4_M11461635_M0_LK_SR"    # 62 ms     3D              [LONG]
                                ],
                      # "q=1.7":[#"SLy4_M10651772_M0_LK_LR", # 22 ms     3D
                               #"SLy4_M10651772_M0_LK_SR"   # 32 ms     no3D ! [not even geo etracts]
                      #        ]   # 32 ms     no3D !
                  }
              }
#
long_simulations = {
            "BLh":{
                "q=1.4": ["BLh_M11461635_M0_LK_SR"],
                "q=1": ["BLh_M13641364_M0_LK_SR"]
            },
            "DD2":{
                "q=1":["DD2_M13641364_M0_SR", "DD2_M13641364_M0_SR_R04", "DD2_M13641364_M0_LK_SR_R04"],
                "q=1.2":["DD2_M14971245_M0_SR", "DD2_M15091235_M0_LK_SR"],
                "q=1.4":["DD2_M11461635_M0_LK_SR"]
            },
            "LS220":{
                "q=1.2":["LS220_M14691268_M0_LK_SR"]
            },
            "SFHo":{
                "q=1.4":["SFHo_M11461635_M0_LK_SR"]
            },
            "SLy4":{
                "q=1.4":["SLy4_M11461635_M0_LK_SR"]
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

def print_selected_simulations():
    _fpath = "slices/" + "rho_modes.h5"
    data = {}
    for eos in simulations2.keys():
        data[eos] = {}
        for q in simulations2[eos].keys():
            data[eos][q] = {}
            for u_sim in simulations2[eos][q].keys():
                # data[eos][q][u_sim] = {}
                # select sim
                sim = ''
                for sim in simulations2[eos][q][u_sim]:
                    if sim.__contains__("SR"):
                        o_par = ADD_METHODS_ALL_PAR(sim)
                        tcoll = o_par.get_par("tcoll_gw")
                        tend = o_par.get_par("tend") * 1e3
                        if np.isinf(tcoll) and tend > 50: # No BH forming Long-Lived cases
                            data[eos][q][u_sim] = {}
                            o_dm = LOAD_DENSITY_MODES(sim)
                            o_dm.gen_set['fname'] = Paths.ppr_sims + sim + "/" + _fpath
                            tmerg = o_par.get_par("tmerg")
                            mags1 = o_dm.get_data(1, "int_phi_r")
                            mags1 = np.abs(mags1)
                            times = o_dm.get_grid("times")
                            times = (times - tmerg) * 1e3
                            #
                            data[eos][q][u_sim]["mag"] = mags1
                            data[eos][q][u_sim]["times"] = times
                        #
    Printcolor.green("Density modes Data collected")
    Printcolor.blue("Simulations are:")
    for eos in data.keys():
        Printcolor.blue("\t{}".format(eos))
        for q in data[eos].keys():
            Printcolor.blue("\t\t{}".format(q))
            for u_sim in data[eos][q].keys():
                Printcolor.green("\t\t\t{} \t {:.1f}".format(u_sim, data[eos][q][u_sim]['times'][-1]))
    #
    # exit(1)


'''===================================================| EJECTA |====================================================='''

def __get_value(o_init, o_par, det=None, mask=None, v_n=None):

    if v_n in o_init.list_v_ns and mask == None:
        value = o_init.get_par(v_n)
    elif not v_n in o_init.list_v_ns and mask == None:
        value = o_par.get_par(v_n)
    elif v_n == "Mej_tot_scaled":
        ma = __get_value(o_init, o_par, None, None, "Mb1")
        mb = __get_value(o_init, o_par, None, None, "Mb2")
        mej = __get_value(o_init, o_par, det, mask, "Mej_tot")
        return mej / (ma + mb)
    elif v_n == "Mej_tot_scaled2":
        # M1 * M2 / (M1 + M2) ^ 2
        ma = __get_value(o_init, o_par, None, None, "Mb1")
        mb = __get_value(o_init, o_par, None, None, "Mb2")
        eta = ma * mb / (ma + mb) ** 2
        mej = __get_value(o_init, o_par, det, mask, "Mej_tot")
        return mej / (eta * (ma + mb))

    elif not v_n in o_init.list_v_ns and mask != None:
        value = o_par.get_outflow_par(det, mask, v_n)
    else:
        raise NameError("unrecognized: v_n_x:{} mask_x:{} det:{} combination"
                        .format(v_n, mask, det))
    if value == None or np.isinf(value) or np.isnan(value):
        raise ValueError("sim: {} det:{} mask:{} v_n:{} --> value:{} wrong!"
                         .format(o_par.sim,det,mask,v_n, value))
    return value
def plot_last_disk_mass_with_lambda2(v_n_x, v_n_y, v_n_col, mask_x=None, mask_y=None, mask_col=None, det=None, plot_legend=True):

    simulations = long_simulations

    data = {"BLh":{}, "DD2":{}, "LS220":{}, "SFHo":{}, "SLy4":{}}

    all_all_x_arr = []
    all_all_y_arr = []
    for eos in simulations.keys():
        all_x_arr = []
        all_y_arr = []
        all_col_arr = []
        all_res_arr = []
        all_lk_arr = []
        all_bh_arr = []
        for q in simulations[eos].keys():
            data[eos][q] = {}
            #
            x_arr = []
            y_arr = []
            col_arr = []
            res_arr = []
            lk_arr = []
            bh_arr = []
            for sim in simulations[eos][q]:
                o_init = LOAD_INIT_DATA(sim)
                o_par = ADD_METHODS_ALL_PAR(sim)
                #
                x_arr.append(__get_value(o_init, o_par, det, mask_x, v_n_x))
                y_arr.append(__get_value(o_init, o_par, det, mask_y, v_n_y))
                col_arr.append(__get_value(o_init, o_par, det, mask_col, v_n_col))
                #
                res = o_init.get_par("res")
                if res == "HR": res_arr.append("v")
                if res == "SR": res_arr.append("d")
                if res == "LR": res_arr.append("^")
                #
                lk = o_init.get_par("vis")
                if lk == "LK": lk_arr.append("gray")
                else: lk_arr.append("none")
                tcoll = o_par.get_par("tcoll_gw")
                if not np.isinf(tcoll): bh_arr.append("x")
                else: bh_arr.append(None)

                #
            #
            data[eos][q][v_n_x] = x_arr
            data[eos][q][v_n_y] = y_arr
            data[eos][q][v_n_col] = col_arr
            data[eos][q]["res"] = res_arr
            data[eos][q]["vis"] = lk_arr
            data[eos][q]["tcoll"] = bh_arr
            #
            all_x_arr = all_x_arr + x_arr
            all_y_arr = all_y_arr + y_arr
            all_col_arr = all_col_arr + col_arr
            all_res_arr = all_res_arr + res_arr
            all_lk_arr = all_lk_arr + lk_arr
            all_bh_arr = all_bh_arr + bh_arr
            all_all_x_arr = all_all_x_arr + all_x_arr
            all_all_y_arr = all_all_y_arr + all_y_arr
        #
        data[eos][v_n_x + 's'] = all_x_arr
        data[eos][v_n_y + 's'] = all_y_arr
        data[eos][v_n_col+'s'] = all_col_arr
        data[eos]["res" + 's'] = all_res_arr
        data[eos]["vis" + 's'] = all_lk_arr
        data[eos]["tcoll" + 's'] = all_bh_arr

    print(" =============================== ")
    all_all_x_arr, all_all_y_arr = UTILS.x_y_z_sort(all_all_x_arr, all_all_y_arr)
    UTILS.fit_polynomial(np.log10(np.array(all_all_x_arr)),np.array(all_all_y_arr), 1, 100)
    print("ave: {}".format(np.sum(all_all_y_arr)/len(all_all_y_arr)) )
    print(" =============================== ")

    #
    #
    figname = ''
    if mask_x == None:
        figname = figname + v_n_x + '_'
    else:
        figname = figname + v_n_x + '_' + mask_x + '_'
    if mask_y == None:
        figname = figname + v_n_y + '_'
    else:
        figname = figname + v_n_y + '_' + mask_y + '_'
    if mask_col == None:
        figname = figname + v_n_col + '_'
    else:
        figname = figname + v_n_col + '_' + mask_col + '_'
    if det == None:
        figname = figname + ''
    else: figname = figname + str(det)
    figname = figname + '.png'
    #
    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + "all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = figname
    o_plot.gen_set["sharex"] = True
    o_plot.gen_set["sharey"] = False
    o_plot.gen_set["subplots_adjust_h"] = 0.0
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []
    #

    #
    i_col = 1
    for eos in ["SLy4", "SFHo", "BLh", "LS220", "DD2"]:
        print(eos)

        # LEGEND

        if eos == "DD2" and plot_legend:
            for res in ["HR", "LR", "SR"]:
                marker_dic_lr = {
                    'task': 'line', 'ptype': 'cartesian',
                    'position': (1, i_col),
                    'xarr': [-1], "yarr": [-1],
                    'xlabel': None, "ylabel": None,
                    'label': res,
                    'marker': 'd', 'color':'gray', 'ms': 8, 'alpha': 1.,
                    'sharey': False,
                    'sharex': False,  # removes angular citkscitks
                    'fontsize': 14,
                    'labelsize': 14
                }
                if res == "HR": marker_dic_lr['marker'] = "v"
                if res == "SR": marker_dic_lr['marker'] = "d"
                if res == "LR": marker_dic_lr['marker'] = "^"
                # if res == "BH": marker_dic_lr['marker'] = "x"
                if res == "SR":
                    if v_n_y == "Ye_ave":loc = 'lower right'
                    else: loc = 'upper right'
                    marker_dic_lr['legend'] = {'loc': loc, 'ncol': 1, 'fontsize': 12, 'shadow': False,
                                               'framealpha': 0.5, 'borderaxespad': 0.0}
                o_plot.set_plot_dics.append(marker_dic_lr)
        #
        xarr = np.array(data[eos][v_n_x+'s'])
        yarr = np.array(data[eos][v_n_y+'s'])
        colarr = data[eos][v_n_col+'s']
        marker = data[eos]["res"+'s']
        edgecolor = data[eos]["vis"+'s']
        bh_marker = data[eos]["tcoll" + 's']
        #
        # UTILS.fit_polynomial(xarr, yarr, 1, 100)
        #
        if v_n_y == "Mej_tot":
            yarr = yarr * 1e2
        #
        #
        #
        dic_bh = {
            'task': 'scatter', 'ptype': 'cartesian',  # 'aspect': 1.,
            'xarr': xarr, "yarr": yarr, "zarr": colarr,
            'position': (1, i_col),  # 'title': '[{:.1f} ms]'.format(time_),
            'cbar': {},
            'v_n_x': v_n_x, 'v_n_y': v_n_y, 'v_n': v_n_col,
            'xlabel': None, "ylabel": None, 'label': eos,
            'xmin': 300, 'xmax': 900, 'ymin': 0.03, 'ymax': 0.3, 'vmin': 1.0, 'vmax': 1.5,
            'fill_vmin': False,  # fills the x < vmin with vmin
            'xscale': None, 'yscale': None,
            'cmap': 'viridis', 'norm': None, 'ms': 80, 'marker': bh_marker, 'alpha': 1.0, "edgecolors": edgecolor,
            'fancyticks': True,
            'minorticks': True,
            'title': {},
            'legend': {},
            'sharey': False,
            'sharex': False,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14
        }
        #
        if mask_y != None and mask_y.__contains__("bern"):
               o_plot.set_plot_dics.append(dic_bh)
        #

        #

        #
        print("marker: {}".format(marker))
        dic = {
            'task': 'scatter', 'ptype': 'cartesian',  # 'aspect': 1.,
            'xarr': xarr, "yarr": yarr, "zarr": colarr,
            'position': (1, i_col),  # 'title': '[{:.1f} ms]'.format(time_),
            'cbar': {},
            'v_n_x': v_n_x, 'v_n_y': v_n_y, 'v_n': v_n_col,
            'xlabel': None, "ylabel": Labels.labels(v_n_y),
            'xmin': 300, 'xmax': 900, 'ymin': 0.03, 'ymax': 0.3, 'vmin': 1.0, 'vmax': 1.8,
            'fill_vmin': False,  # fills the x < vmin with vmin
            'xscale': None, 'yscale': None,
            'cmap': 'viridis', 'norm': None, 'ms': 80, 'marker': marker, 'alpha': 0.8, "edgecolors": edgecolor,
            'tick_params': {"axis":'both', "which":'both', "labelleft":True,
                            "labelright":False, #"tick1On":True, "tick2On":True,
                            "labelsize":12,
                            "direction":'in',
                            "bottom":True, "top":True, "left":True, "right":True},
            'yaxiscolor':{'bottom':'black', 'top':'black', 'right':'black', 'left':'black'},
            'minorticks': True,
            'title': {"text":eos, "fontsize":12},
            'label': "xxx",
            'legend': {},
            'sharey': False,
            'sharex': False,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14
        }
        #
        if v_n_y == "Mdisk3Dmax":
            dic['ymin'], dic['ymax'] = 0.03, 0.30
        if v_n_y == "Mej_tot" and mask_y == "geo":
            dic['ymin'], dic['ymax'] = 0, 1.0
        if v_n_y == "Mej_tot" and mask_y == "bern_geoend":
            dic['ymin'], dic['ymax'] = 0, 3.2
        if v_n_y == "Ye_ave"and mask_y == "geo":
            dic['ymin'], dic['ymax'] = 0.01, 0.30
        if v_n_y == "Ye_ave"and mask_y == "bern_geoend":
            dic['ymin'], dic['ymax'] = 0.1, 0.4
        if v_n_y == "vel_inf_ave"and mask_y == "geo":
            dic['ymin'], dic['ymax'] = 0.1, 0.3
        if v_n_y == "vel_inf_ave"and mask_y == "bern_geoend":
            dic['ymin'], dic['ymax'] = 0.05, 0.25
        #
        if eos == "SLy4":
            dic['xmin'], dic['xmax'] = 380, 420
            dic['xticks'] = [390, 410]
        if eos == "SFHo":
            dic['xmin'], dic['xmax'] = 390, 430
            dic['xticks'] = [400, 420]
        if eos == "BLh":
            dic['xmin'], dic['xmax'] = 510, 550
            dic['xticks'] = [520, 540]
        if eos == "LS220":
            dic['xmin'], dic['xmax'] = 690, 730
            dic['xticks'] = [700, 720]
        if eos == "DD2":
            dic['xmin'], dic['xmax'] = 820, 860
            dic['xticks'] = [830, 850]
        if eos == "SLy4":
            dic['tick_params']['right'] = False
            dic['yaxiscolor']["right"] = "lightgray"
        elif eos == "DD2":
            dic['tick_params']['left'] = False
            dic['yaxiscolor']["left"] = "lightgray"
        else:
            dic['tick_params']['left'] = False
            dic['tick_params']['right'] = False
            dic['yaxiscolor']["left"] = "lightgray"
            dic['yaxiscolor']["right"] = "lightgray"

        #
        # if eos != "SLy4" and eos != "DD2":
        #     dic['yaxiscolor'] = {'left':'lightgray','right':'lightgray', 'label': 'black'}
        #     dic['ytickcolor'] = {'left':'lightgray','right':'lightgray'}
        #     dic['yminortickcolor'] = {'left': 'lightgray', 'right': 'lightgray'}
        # elif eos == "DD2":
        #     dic['yaxiscolor'] = {'left': 'lightgray', 'right': 'black', 'label': 'black'}
        #     # dic['ytickcolor'] = {'left': 'lightgray'}
        #     # dic['yminortickcolor'] = {'left': 'lightgray'}
        # elif eos == "SLy4":
        #     dic['yaxiscolor'] = {'left': 'black', 'right': 'lightgray', 'label': 'black'}
        #     # dic['ytickcolor'] = {'right': 'lightgray'}
        #     # dic['yminortickcolor'] = {'right': 'lightgray'}

        #
        if eos != "SLy4":
            dic['sharey'] = True
        if eos == "BLh":
            dic['xlabel'] = Labels.labels(v_n_x)
        if eos == 'DD2':
            dic['cbar'] = {'location': 'right .03 .0', 'label': Labels.labels(v_n_col),  # 'fmt': '%.1f',
                     'labelsize': 14, 'fontsize': 14}
        #
        i_col = i_col + 1
        o_plot.set_plot_dics.append(dic)
        #

    #
    o_plot.main()
    exit(0)

def __get_val_err(sims, o_inits, o_pars, v_n, det=0,mask="geo", error=0.2):

    if v_n == "nsims":
        return len(sims), len(sims), len(sims)
    elif v_n == "pizzaeos":
        pizza_eos = ''
        for sim, o_init, o_par in zip(sims, o_inits, o_pars):
            _pizza_eos = o_init.get_par("pizza_eos")
            if pizza_eos != '' and pizza_eos != _pizza_eos:
                raise NameError("sim:{} pizza_eos:{} \n sim:{} pizza_eos: {} \n MISMATCH"
                                .format(sim, pizza_eos, sims[0], _pizza_eos))
        pizza_eos = _pizza_eos
        return pizza_eos, pizza_eos, pizza_eos
    if len(sims) == 0:
        raise ValueError("no simualtions passed")
    _resols, _values = [], []
    assert len(sims) == len(o_inits)
    assert len(sims) == len(o_pars)
    for sim, o_init, o_par in zip(sims, o_inits, o_pars):
        _val = __get_value(o_init, o_par, det, mask, v_n)
        # print(sim, _val)
        _res = "fuck"
        for res in resolutions.keys():
            if sim.__contains__(res):
                _res = res
                break
        if _res == "fuck":
            raise NameError("fuck")
        _resols.append(resolutions[_res])
        _values.append(_val)
    if len(sims) == 1:
        return _values[0], _values[0] - error * _values[0], _values[0] + error * _values[0]
    elif len(sims) == 2:
        delta = np.abs(_values[0] - _values[1])
        if _resols[0] < _resols[1]:
            return _values[0], _values[0] - delta, _values[0] + delta
        else:
            return _values[1], _values[1] - delta, _values[1] + delta
    elif len(sims) == 3:
        _resols_, _values_ = UTILS.x_y_z_sort(_resols, _values) # 123, 185, 236
        delta1 = np.abs(_values_[0] - _values_[1])
        delta2 = np.abs(_values_[1] - _values_[2])
        # print(_values, _values_); exit(0)
        return _values_[1], _values_[1] - delta1, _values_[1] + delta2
    else:
        raise ValueError("Too many simulations")

def __get_is_prompt_coll(sims, o_inits, o_pars, delta_t = 3.):

    isprompt = False
    isbh = False
    for sim, o_init, o_par in zip(sims, o_inits, o_pars):
        tcoll = o_par.get_par("tcoll_gw")
        if np.isinf(tcoll):
            pass
        else:
            isbh = True
            tmerg = o_par.get_par("tmerg")
            assert tcoll > tmerg
            if float(tcoll - tmerg) < delta_t * 1e-3:
                isprompt = True

    return isbh, isprompt

def __get_custom_descrete_colormap(n):
    # n = 5
    import matplotlib.colors as col
    from_list = col.LinearSegmentedColormap.from_list
    cm = from_list(None, plt.cm.Set1(range(0, n)), n)
    x = np.arange(99)
    y = x % 11
    z = x % n
    return cm

def plot_summary_quntity():
    """
    Plot unique simulations point by point with error bars
    :return:
    """
    v_n_x = "Lambda"
    v_n_y = "Mej_tot"
    v_n_col = "q"
    det = 0
    do_plot_error_bar = True
    mask_x, mask_y, mask_col = None, "geo", None
    data = {}
    error = 0.2 # in * 100 percent

    # collect data
    for eos in simulations2.keys():
        data[eos] = {}
        for q in simulations2[eos]:
            data[eos][q] = {}
            for u_sim in simulations2[eos][q]:
                data[eos][q][u_sim] = {}
                sims = simulations2[eos][q][u_sim]
                o_inits = [LOAD_INIT_DATA(sim) for sim in sims]
                o_pars = [ADD_METHODS_ALL_PAR(sim) for sim in sims]
                x_coord, x_err1, x_err2 = __get_val_err(sims, o_inits, o_pars, v_n_x, det, mask_x, error)
                y_coord, y_err1, y_err2 = __get_val_err(sims, o_inits, o_pars, v_n_y, det, mask_y, error)
                col_coord, col_err1, col_err2 = __get_val_err(sims, o_inits, o_pars, v_n_col, det, mask_col, error)
                data[eos][q][u_sim]["lserr"] = len(sims)
                data[eos][q][u_sim]["x"] = x_coord
                data[eos][q][u_sim]["xe1"] = x_err1
                data[eos][q][u_sim]["xe2"] = x_err2
                data[eos][q][u_sim]["y"] = y_coord
                data[eos][q][u_sim]["ye1"] = y_err1
                data[eos][q][u_sim]["ye2"] = y_err2
                data[eos][q][u_sim]["c"] = col_coord
                data[eos][q][u_sim]["ce1"] = col_err1
                data[eos][q][u_sim]["ce2"] = col_err2
                Printcolor.blue("Processing {} ({} sims) x:[{:.1f}, v:{:.1f} ^{:.1f}] y:[{:.5f}, v{:.5f} ^{:.5f}] col:{:.1f}"
                                .format(u_sim, len(sims), x_coord, x_err1, x_err2, y_coord, y_err1, y_err2, col_coord))
    Printcolor.green("Data is collaected")

    # stuck data for scatter plot
    for eos in simulations2.keys():
        for v_n in ["x", "y", "c"]:
            arr = []
            for q in simulations2[eos].keys():
                for u_sim in simulations2[eos][q]:
                    arr.append(data[eos][q][u_sim][v_n])
            data[eos][v_n+"s"] = arr
    Printcolor.green("Data is stacked")
    # plot the scatter points
    figname = ''
    if mask_x == None:
        figname = figname + v_n_x + '_'
    else:
        figname = figname + v_n_x + '_' + mask_x + '_'
    if mask_y == None:
        figname = figname + v_n_y + '_'
    else:
        figname = figname + v_n_y + '_' + mask_y + '_'
    if mask_col == None:
        figname = figname + v_n_col + '_'
    else:
        figname = figname + v_n_col + '_' + mask_col + '_'
    if det == None:
        figname = figname + ''
    else:
        figname = figname + str(det)
    figname = figname + '3.png'
    #
    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + "all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = figname
    o_plot.gen_set["sharex"] = True
    o_plot.gen_set["sharey"] = False
    o_plot.gen_set["subplots_adjust_h"] = 0.0
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []


    # ERROR BAR


    # ACTUAL PLOT
    i_col = 1
    for eos in ["SLy4", "SFHo", "BLh", "LS220", "DD2"]:
        print(eos)
        # Error Bar
        if do_plot_error_bar:
            for q in simulations2[eos].keys():
                for u_sim in simulations2[eos][q].keys():
                    x = data[eos][q][u_sim]["x"]
                    x1 = data[eos][q][u_sim]["xe1"]
                    x2 = data[eos][q][u_sim]["xe2"]
                    y = data[eos][q][u_sim]["y"]
                    y1 = data[eos][q][u_sim]["ye1"]
                    y2 = data[eos][q][u_sim]["ye2"]
                    nsims = data[eos][q][u_sim]["lserr"]
                    if v_n_y == "Mej_tot":
                        y1 = y1 * 1e2
                        y2 = y2 * 1e2
                        y = y * 1e2
                    if nsims == 1: ls = ':'
                    elif nsims == 2: ls = '--'
                    elif nsims == 3: ls = '-'
                    else: raise ValueError("too many sims >3")
                    marker_dic_lr = {
                        'task': 'line', 'ptype': 'cartesian',
                        'position': (1, i_col),
                        'xarr': [x, x], "yarr": [y1, y2],
                        'xlabel': None, "ylabel": None,
                        'label': None,
                        'ls': ls, 'color': 'gray', 'lw': 1.5, 'alpha': 1., 'ds': 'default',
                        'sharey': False,
                        'sharex': False,  # removes angular citkscitks
                        'fontsize': 14,
                        'labelsize': 14
                    }
                    o_plot.set_plot_dics.append(marker_dic_lr)


        # LEGEND
        # if eos == "DD2" and plot_legend:
        #     for res in ["HR", "LR", "SR"]:
        #         marker_dic_lr = {
        #             'task': 'line', 'ptype': 'cartesian',
        #             'position': (1, i_col),
        #             'xarr': [-1], "yarr": [-1],
        #             'xlabel': None, "ylabel": None,
        #             'label': res,
        #             'marker': 'd', 'color': 'gray', 'ms': 8, 'alpha': 1.,
        #             'sharey': False,
        #             'sharex': False,  # removes angular citkscitks
        #             'fontsize': 14,
        #             'labelsize': 14
        #         }
        #         if res == "HR": marker_dic_lr['marker'] = "v"
        #         if res == "SR": marker_dic_lr['marker'] = "d"
        #         if res == "LR": marker_dic_lr['marker'] = "^"
        #         # if res == "BH": marker_dic_lr['marker'] = "x"
        #         if res == "SR":
        #             if v_n_y == "Ye_ave":
        #                 loc = 'lower right'
        #             else:
        #                 loc = 'upper right'
        #             marker_dic_lr['legend'] = {'loc': loc, 'ncol': 1, 'fontsize': 12, 'shadow': False,
        #                                        'framealpha': 0.5, 'borderaxespad': 0.0}
        #         o_plot.set_plot_dics.append(marker_dic_lr)
        #
        xarr = np.array(data[eos]["xs"])
        yarr = np.array(data[eos]["ys"])
        colarr = data[eos]["cs"]
        # marker = data[eos]["res" + 's']
        # edgecolor = data[eos]["vis" + 's']
        # bh_marker = data[eos]["tcoll" + 's']
        #
        # UTILS.fit_polynomial(xarr, yarr, 1, 100)
        #
        # print(xarr, yarr); exit(1)
        if v_n_y == "Mej_tot":
            yarr = yarr * 1e2
        #
        #
        #
        # dic_bh = {
        #     'task': 'scatter', 'ptype': 'cartesian',  # 'aspect': 1.,
        #     'xarr': xarr, "yarr": yarr, "zarr": colarr,
        #     'position': (1, i_col),  # 'title': '[{:.1f} ms]'.format(time_),
        #     'cbar': {},
        #     'v_n_x': v_n_x, 'v_n_y': v_n_y, 'v_n': v_n_col,
        #     'xlabel': None, "ylabel": None, 'label': eos,
        #     'xmin': 300, 'xmax': 900, 'ymin': 0.03, 'ymax': 0.3, 'vmin': 1.0, 'vmax': 1.5,
        #     'fill_vmin': False,  # fills the x < vmin with vmin
        #     'xscale': None, 'yscale': None,
        #     'cmap': 'viridis', 'norm': None, 'ms': 80, 'marker': bh_marker, 'alpha': 1.0, "edgecolors": edgecolor,
        #     'fancyticks': True,
        #     'minorticks': True,
        #     'title': {},
        #     'legend': {},
        #     'sharey': False,
        #     'sharex': False,  # removes angular citkscitks
        #     'fontsize': 14,
        #     'labelsize': 14
        # }
        #
        # if mask_y != None and mask_y.__contains__("bern"):
        #     o_plot.set_plot_dics.append(dic_bh)
        #

        #

        #
        # print("marker: {}".format(marker))
        dic = {
            'task': 'scatter', 'ptype': 'cartesian',  # 'aspect': 1.,
            'xarr': xarr, "yarr": yarr, "zarr": colarr,
            'position': (1, i_col),  # 'title': '[{:.1f} ms]'.format(time_),
            'cbar': {},
            'v_n_x': v_n_x, 'v_n_y': v_n_y, 'v_n': v_n_col,
            'xlabel': None, "ylabel": Labels.labels(v_n_y),
            'xmin': 300, 'xmax': 900, 'ymin': 0.03, 'ymax': 0.3, 'vmin': 1.0, 'vmax': 1.8,
            'fill_vmin': False,  # fills the x < vmin with vmin
            'xscale': None, 'yscale': None,
            'cmap': 'viridis', 'norm': None, 'ms': 80, 'marker': "d", 'alpha': 1.0, "edgecolors": None,
            'tick_params': {"axis": 'both', "which": 'both', "labelleft": True,
                            "labelright": False,  # "tick1On":True, "tick2On":True,
                            "labelsize": 12,
                            "direction": 'in',
                            "bottom": True, "top": True, "left": True, "right": True},
            'yaxiscolor': {'bottom': 'black', 'top': 'black', 'right': 'black', 'left': 'black'},
            'minorticks': True,
            'title': {"text": eos, "fontsize": 12},
            'label': "xxx",
            'legend': {},
            'sharey': False,
            'sharex': False,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14
        }
        #
        if v_n_y == "Mdisk3Dmax":
            dic['ymin'], dic['ymax'] = 0.03, 0.30
        if v_n_y == "Mej_tot" and mask_y == "geo":
            dic['ymin'], dic['ymax'] = 0, 1.5
        if v_n_y == "Mej_tot" and mask_y == "bern_geoend":
            dic['ymin'], dic['ymax'] = 0, 3.2
        if v_n_y == "Ye_ave" and mask_y == "geo":
            dic['ymin'], dic['ymax'] = 0.01, 0.30
        if v_n_y == "Ye_ave" and mask_y == "bern_geoend":
            dic['ymin'], dic['ymax'] = 0.1, 0.4
        if v_n_y == "vel_inf_ave" and mask_y == "geo":
            dic['ymin'], dic['ymax'] = 0.1, 0.3
        if v_n_y == "vel_inf_ave" and mask_y == "bern_geoend":
            dic['ymin'], dic['ymax'] = 0.05, 0.25
        #
        if eos == "SLy4":
            dic['xmin'], dic['xmax'] = 380, 420
            dic['xticks'] = [390, 410]
        if eos == "SFHo":
            dic['xmin'], dic['xmax'] = 390, 430
            dic['xticks'] = [400, 420]
        if eos == "BLh":
            dic['xmin'], dic['xmax'] = 510, 550
            dic['xticks'] = [520, 540]
        if eos == "LS220":
            dic['xmin'], dic['xmax'] = 690, 730
            dic['xticks'] = [700, 720]
        if eos == "DD2":
            dic['xmin'], dic['xmax'] = 820, 860
            dic['xticks'] = [830, 850]
        if eos == "SLy4":
            dic['tick_params']['right'] = False
            dic['yaxiscolor']["right"] = "lightgray"
        elif eos == "DD2":
            dic['tick_params']['left'] = False
            dic['yaxiscolor']["left"] = "lightgray"
        else:
            dic['tick_params']['left'] = False
            dic['tick_params']['right'] = False
            dic['yaxiscolor']["left"] = "lightgray"
            dic['yaxiscolor']["right"] = "lightgray"

        #
        # if eos != "SLy4" and eos != "DD2":
        #     dic['yaxiscolor'] = {'left':'lightgray','right':'lightgray', 'label': 'black'}
        #     dic['ytickcolor'] = {'left':'lightgray','right':'lightgray'}
        #     dic['yminortickcolor'] = {'left': 'lightgray', 'right': 'lightgray'}
        # elif eos == "DD2":
        #     dic['yaxiscolor'] = {'left': 'lightgray', 'right': 'black', 'label': 'black'}
        #     # dic['ytickcolor'] = {'left': 'lightgray'}
        #     # dic['yminortickcolor'] = {'left': 'lightgray'}
        # elif eos == "SLy4":
        #     dic['yaxiscolor'] = {'left': 'black', 'right': 'lightgray', 'label': 'black'}
        #     # dic['ytickcolor'] = {'right': 'lightgray'}
        #     # dic['yminortickcolor'] = {'right': 'lightgray'}

        #
        if eos != "SLy4":
            dic['sharey'] = True
        if eos == "BLh":
            dic['xlabel'] = Labels.labels(v_n_x)
        if eos == 'DD2':
            dic['cbar'] = {'location': 'right .03 .0', 'label': Labels.labels(v_n_col),  # 'fmt': '%.1f',
                           'labelsize': 14, 'fontsize': 14}
        #
        o_plot.set_plot_dics.append(dic)
        #

        i_col = i_col + 1

    #
    o_plot.main()
    exit(0)

def plot_summary_quntity_all_in_one():
    """
    Plot unique simulations point by point with error bars
    :return:
    """
    v_n_x = "Lambda"
    v_n_y = "Ye_ave"
    v_n_col = "q"
    det = 0
    do_plot_linear_fit = True
    do_plot_promptcoll = True
    do_plot_bh = True
    do_plot_error_bar_y = True
    do_plot_error_bar_x = False
    do_plot_old_table = True
    do_plot_annotations = False
    mask_x, mask_y, mask_col = None, "geo", None # geo_entropy_above_10
    data = {}
    error = 0.2 # in * 100 percent
    delta_t_prompt = 2. # ms

    # collect old data
    old_data = {}
    if do_plot_old_table:
        #
        if mask_x != None and mask_x != "geo":
            raise NameError("old table des not contain data for mask_x: {}".format(mask_x))
        if mask_y != None and mask_y != "geo":
            raise NameError("old table des not contain data for mask_x: {}".format(mask_y))
        if mask_col != None and mask_col != "geo":
            raise NameError("old table des not contain data for mask_x: {}".format(mask_col))
        #
        new_old_dic = {'Mej_tot':"Mej",
                       "Lambda":"Lambda",
                       "vel_inf_ave": "vej",
                       "Ye_ave": "Yeej"}
        old_tbl = ALL_SIMULATIONS_TABLE()
        old_tbl.list_neut = ["LK", "M0"]
        old_tbl.list_vis = ["L5", "L25", "L50"]
        old_tbl.list_eos.append("BHBlp")
        old_tbl.intable = Paths.output + "radice2018_summary.csv"
        old_tbl.load_input_data()
        old_all_x = []
        old_all_y = []
        old_all_col = []
        for run in old_tbl.table:
            sim = run['name']
            old_data[sim] = {}
            if not sim.__contains__("HR") \
                and not sim.__contains__("OldM0") \
                and not sim.__contains__("LR") \
                and not sim.__contains__("L5") \
                and not sim.__contains__("L25") \
                and not sim.__contains__("L50"):
                x = float(run[new_old_dic[v_n_x]])
                y = float(run[new_old_dic[v_n_y]])
                col = "gray"
                old_all_col.append(col)
                old_all_x.append(x)
                old_all_y.append(y)
                old_data[sim][v_n_x] = x
                old_data[sim][v_n_y] = y

        Printcolor.green("old data is collected")
        old_all_x = np.array(old_all_x)
        old_all_y = np.array(old_all_y)


    # exit(1)
    # collect data
    for eos in simulations2.keys():
        data[eos] = {}
        for q in simulations2[eos]:
            data[eos][q] = {}
            for u_sim in simulations2[eos][q]:
                data[eos][q][u_sim] = {}
                sims = simulations2[eos][q][u_sim]
                o_inits = [LOAD_INIT_DATA(sim) for sim in sims]
                o_pars = [ADD_METHODS_ALL_PAR(sim) for sim in sims]
                x_coord, x_err1, x_err2 = __get_val_err(sims, o_inits, o_pars, v_n_x, det, mask_x, error)
                y_coord, y_err1, y_err2 = __get_val_err(sims, o_inits, o_pars, v_n_y, det, mask_y, error)
                col_coord, col_err1, col_err2 = __get_val_err(sims, o_inits, o_pars, v_n_col, det, mask_col, error)
                data[eos][q][u_sim]["lserr"] = len(sims)
                data[eos][q][u_sim]["x"] = x_coord
                data[eos][q][u_sim]["xe1"] = x_err1
                data[eos][q][u_sim]["xe2"] = x_err2
                data[eos][q][u_sim]["y"] = y_coord
                data[eos][q][u_sim]["ye1"] = y_err1
                data[eos][q][u_sim]["ye2"] = y_err2
                data[eos][q][u_sim]["c"] = col_coord
                data[eos][q][u_sim]["ce1"] = col_err1
                data[eos][q][u_sim]["ce2"] = col_err2
                #
                isbh, ispromtcoll = __get_is_prompt_coll(sims, o_inits, o_pars, delta_t=delta_t_prompt)
                data[eos][q][u_sim]["isprompt"] = ispromtcoll
                data[eos][q][u_sim]["isbh"] = isbh
                if isbh and not ispromtcoll: marker = 'o'
                elif isbh and ispromtcoll: marker = 's'
                else: marker = 'd'
                data[eos][q][u_sim]["marker"] = marker
                #
                pizzaeos = False
                if eos == "SFHo":
                    pizzaeos, _, _ = __get_val_err(sims, o_inits, o_pars, "pizzaeos")
                    if pizzaeos.__contains__("2019"):
                        _pizzaeos = True
                        data[eos][q][u_sim]['pizza2019'] = True
                    else:
                        _pizzaeos = False
                        data[eos][q][u_sim]['pizza2019'] = False
                #
                Printcolor.print_colored_string([u_sim, "({})".format(len(sims)),
                                                 "x:[","{:.1f}".format(x_coord),
                                                 "v:","{:.1f}".format(x_err1),
                                                 "^:","{:.1f}".format(x_err2),
                                                 "|",
                                                 "y:","{:.5f}".format(y_coord),
                                                 "v:","{:.5f}".format(y_err1),
                                                 "^:",
                                                 "{:.5f}".format(y_err2),
                                                 "] col: {} BH:".format(col_coord),
                                                 "{}".format(ispromtcoll),
                                                 "pizza2019:",
                                                 "{}".format(pizzaeos)],
                                                ["blue", "green", "blue","green","blue", "green",
                                                 "blue","green","yellow","blue","green","blue",
                                                 "green","blue","green","blue","green", "blue", "green"])

                # Printcolor.blue("Processing {} ({} sims) x:[{:.1f}, v:{:.1f} ^{:.1f}] y:[{:.5f}, v{:.5f} ^{:.5f}] col:{:.1f}"
                #                 .format(u_sim, len(sims), x_coord, x_err1, x_err2, y_coord, y_err1, y_err2, col_coord))
    Printcolor.green("Data is collaected")




    # FIT
    print(" =============================== ")
    all_x = []
    all_y = []
    for eos in data.keys():
        for q in data[eos].keys():
            for u_sim in data[eos][q].keys():
                ispc = data[eos][q][u_sim]["isprompt"]
                if not ispc:
                    all_x.append(data[eos][q][u_sim]["x"])
                    all_y.append(data[eos][q][u_sim]['y'])
    all_x = np.array(all_x)
    all_y = np.array(all_y)


    # print(all_x)
    all_x, all_y = UTILS.x_y_z_sort(all_x, all_y)
    # print(all_x);
    print("_log(lambda) as x")
    UTILS.fit_polynomial(np.log10(all_x), all_y, 1, 100)
    print("lamda as x")
    fit_x, fit_y = UTILS.fit_polynomial(all_x, all_y, 1, 100)
    # print(fit_x); exit(1)
    print("ave: {}".format(np.sum(all_y) / len(all_y)))
    print(" =============================== ")

    # stuck data for scatter plot
    for eos in simulations2.keys():
        for v_n in ["x", "y", "c", "marker"]:
            arr = []
            for q in simulations2[eos].keys():
                for u_sim in simulations2[eos][q]:
                    arr.append(data[eos][q][u_sim][v_n])
            data[eos][v_n+"s"] = arr

    Printcolor.green("Data is stacked")
    # plot the scatter points
    figname = ''
    if mask_x == None:
        figname = figname + v_n_x + '_'
    else:
        figname = figname + v_n_x + '_' + mask_x + '_'
    if mask_y == None:
        figname = figname + v_n_y + '_'
    else:
        figname = figname + v_n_y + '_' + mask_y + '_'
    if mask_col == None:
        figname = figname + v_n_col + '_'
    else:
        figname = figname + v_n_col + '_' + mask_col + '_'
    if det == None:
        figname = figname + ''
    else:
        figname = figname + str(det)
    if do_plot_old_table:
        figname = figname + '_InclOldTbl'
    figname = figname + '.png'
    #
    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + "all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = figname
    o_plot.gen_set["sharex"] = True
    o_plot.gen_set["sharey"] = False
    o_plot.gen_set["subplots_adjust_h"] = 0.0
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []


    # FOR LEGENDS
    if do_plot_promptcoll:
        x = -1.
        y = -1.
        marker_dic_lr = {
            'task': 'line', 'ptype': 'cartesian',
            'position': (1, 1),
            'xarr': [x], "yarr": [y],
            'xlabel': None, "ylabel": None,
            'label': "Prompt collapse",
            'marker': 's', 'color': 'gray', 'ms': 10., 'alpha': 0.4,
            'sharey': False,
            'sharex': False,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14
        }
        # if  eos == "BLh" and u_sim == simulations2[eos][q].keys()[-1]:
        #     print('-0--------------------')
        marker_dic_lr['legend'] = {'loc': 'upper left', 'ncol': 1, 'shadow': False, 'framealpha': 0.,
                                   'borderaxespad': 0., 'fontsize': 11}
        o_plot.set_plot_dics.append(marker_dic_lr)
    if do_plot_bh:
        x = -1.
        y = -1.
        marker_dic_lr = {
            'task': 'line', 'ptype': 'cartesian',
            'position': (1, 1),
            'xarr': [x], "yarr": [y],
            'xlabel': None, "ylabel": None,
            'label': "BH formation",
            'marker': 'o', 'color': 'gray', 'ms': 10., 'alpha': 0.4,
            'sharey': False,
            'sharex': False,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14
        }
        # if  eos == "BLh" and u_sim == simulations2[eos][q].keys()[-1]:
        #     print('-0--------------------')
        marker_dic_lr['legend'] = {'loc': 'upper left', 'ncol': 1, 'shadow': False, 'framealpha': 0.,
                                   'borderaxespad': 0., 'fontsize': 11}
        o_plot.set_plot_dics.append(marker_dic_lr)
    if do_plot_bh:
        x = -1.
        y = -1.
        marker_dic_lr = {
            'task': 'line', 'ptype': 'cartesian',
            'position': (1, 1),
            'xarr': [x], "yarr": [y],
            'xlabel': None, "ylabel": None,
            'label': "Long Lived",
            'marker': 'd', 'color': 'gray', 'ms': 10., 'alpha': 0.4,
            'sharey': False,
            'sharex': False,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14
        }
        # if  eos == "BLh" and u_sim == simulations2[eos][q].keys()[-1]:
        #     print('-0--------------------')
        marker_dic_lr['legend'] = {'loc': 'upper right', 'ncol': 1, 'shadow': False, 'framealpha': 0.,
                                   'borderaxespad': 0., 'fontsize': 11}
        o_plot.set_plot_dics.append(marker_dic_lr)
    if do_plot_old_table:
        x = -1.
        y = -1.
        marker_dic_lr = {
            'task': 'line', 'ptype': 'cartesian',
            'position': (1, 1),
            'xarr': [x], "yarr": [y],
            'xlabel': None, "ylabel": None,
            'label': "Radice+2018",
            'marker': '*', 'color': 'gray', 'ms': 10., 'alpha': 0.4,
            'sharey': False,
            'sharex': False,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14
        }
        # if  eos == "BLh" and u_sim == simulations2[eos][q].keys()[-1]:
        #     print('-0--------------------')
        marker_dic_lr['legend'] = {'loc': 'upper right', 'ncol': 1, 'shadow': False, 'framealpha': 0.,
                                   'borderaxespad': 0., 'fontsize': 11}
        o_plot.set_plot_dics.append(marker_dic_lr)

    # FOR FITS
    if do_plot_linear_fit:
        if v_n_y == "Mej_tot" or v_n_y == "Mej_tot_scaled":
            fit_y = fit_y * 1e2
        if v_n_x == "Mej_tot" or v_n_x == "Mej_tot_scaled":
            fit_x = fit_x * 1e2
        # print(fit_x, fit_y)
        linear_fit = {
            'task': 'line', 'ptype': 'cartesian',
            'position': (1, 1),
            'xarr': fit_x, "yarr": fit_y,
            'xlabel': None, "ylabel": None,
            'label': "Linear fit",
            'ls': '-', 'color': 'black', 'lw': 1., 'alpha': 1., 'ds':'default',
            'sharey': False,
            'sharex': False,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14
        }
        o_plot.set_plot_dics.append(linear_fit)


        #
    if do_plot_old_table:

        if v_n_y == "Mej_tot" or v_n_y == "Mej_tot_scaled":
            old_all_y = old_all_y * 1e2
        if v_n_x == "Mej_tot" or v_n_x == "Mej_tot_scaled":
            old_all_x = old_all_x * 1e2
        dic = {
            'task': 'scatter', 'ptype': 'cartesian',  # 'aspect': 1.,
            'xarr': old_all_x, "yarr": old_all_y, "zarr": old_all_col,
            'position': (1, 1),  # 'title': '[{:.1f} ms]'.format(time_),
            'cbar': {},
            'v_n_x': v_n_x, 'v_n_y': v_n_y, 'v_n': v_n_col,
            'xlabel': None, "ylabel": Labels.labels(v_n_y, mask_y),
            'xmin': 300, 'xmax': 900, 'ymin': 0.03, 'ymax': 0.3, 'vmin': 1.0, 'vmax': 1.9,
            'fill_vmin': False,  # fills the x < vmin with vmin
            'xscale': None, 'yscale': None,
            'cmap': 'tab10', 'norm': None, 'ms': 60, 'marker': '*', 'alpha': 0.7, "edgecolors": None,
            'tick_params': {"axis": 'both', "which": 'both', "labelleft": True,
                            "labelright": False,  # "tick1On":True, "tick2On":True,
                            "labelsize": 12,
                            "direction": 'in',
                            "bottom": True, "top": True, "left": True, "right": True},
            'yaxiscolor': {'bottom': 'black', 'top': 'black', 'right': 'black', 'left': 'black'},
            'minorticks': True,
            'title': {},  # {"text": eos, "fontsize": 12},
            'label': None,
            'legend': {},
            'sharey': False,
            'sharex': False,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14
        }
        o_plot.set_plot_dics.append(dic)
    if do_plot_annotations:
        for eos in ["SFHo"]:
            print(eos)
            for q in simulations2[eos].keys():
                for u_sim in simulations2[eos][q].keys():
                    x = data[eos][q][u_sim]["x"]
                    y = data[eos][q][u_sim]["y"]
                    y1 = data[eos][q][u_sim]["ye1"]
                    y2 = data[eos][q][u_sim]["ye2"]
                    if data[eos][q][u_sim]["pizza2019"]:
                        if v_n_x == "Mej_tot" or v_n_x == "Mej_tot_scaled":
                            x = x * 1e2
                        if v_n_y == "Mej_tot" or v_n_y == "Mej_tot_scaled":
                            y1 = y1 * 1e2
                            y2 = y2 * 1e2
                            y = y * 1e2
                        marker_dic_lr = {
                            'task': 'line', 'ptype': 'cartesian',
                            'position': (1, 1),
                            'xarr': [x], "yarr": [y],
                            'xlabel': None, "ylabel": None,
                            'label': None,
                            'marker': '2', 'color': 'blue', 'ms':15, 'alpha':1.,
                            # 'ls': ls, 'color': 'gray', 'lw': 1.5, 'alpha': 1., 'ds': 'default',
                            'sharey': False,
                            'sharex': False,  # removes angular citkscitks
                            'fontsize': 14,
                            'labelsize': 14
                        }
                        o_plot.set_plot_dics.append(marker_dic_lr)

    # PLOTS
    i_col = 1
    for eos in ["SLy4", "SFHo", "BLh", "LS220", "DD2"]:
        print(eos)
        # Error Bar
        if do_plot_error_bar_y:
            for q in simulations2[eos].keys():
                for u_sim in simulations2[eos][q].keys():
                    x = data[eos][q][u_sim]["x"]
                    y = data[eos][q][u_sim]["y"]
                    y1 = data[eos][q][u_sim]["ye1"]
                    y2 = data[eos][q][u_sim]["ye2"]
                    nsims = data[eos][q][u_sim]["lserr"]
                    if v_n_x == "Mej_tot" or v_n_x == "Mej_tot_scaled":
                        x = x * 1e2
                    if v_n_y == "Mej_tot" or v_n_y == "Mej_tot_scaled":
                        y1 = y1 * 1e2
                        y2 = y2 * 1e2
                        y = y * 1e2
                    if nsims == 1: ls = ':'
                    elif nsims == 2: ls = '--'
                    elif nsims == 3: ls = '-'
                    else: raise ValueError("too many sims >3")
                    marker_dic_lr = {
                        'task': 'line', 'ptype': 'cartesian',
                        'position': (1, i_col),
                        'xarr': [x, x], "yarr": [y1, y2],
                        'xlabel': None, "ylabel": None,
                        'label': None,
                        'ls': ls, 'color': 'gray', 'lw': 1.5, 'alpha': 0.6, 'ds': 'default',
                        'sharey': False,
                        'sharex': False,  # removes angular citkscitks
                        'fontsize': 14,
                        'labelsize': 14
                    }
                    o_plot.set_plot_dics.append(marker_dic_lr)
        if do_plot_error_bar_x:
            for q in simulations2[eos].keys():
                for u_sim in simulations2[eos][q].keys():
                    x = data[eos][q][u_sim]["x"]
                    x1 = data[eos][q][u_sim]["xe1"]
                    x2 = data[eos][q][u_sim]["xe2"]
                    y = data[eos][q][u_sim]["y"]
                    nsims = data[eos][q][u_sim]["lserr"]
                    if v_n_y == "Mej_tot" or v_n_y == "Mej_tot_scaled":
                        y = y * 1e2
                    if v_n_x == "Mej_tot" or v_n_x == "Mej_tot_scaled":
                        x1 = x1 * 1e2
                        x2 = x2 * 1e2
                        x = x * 1e2
                    if nsims == 1: ls = ':'
                    elif nsims == 2: ls = '--'
                    elif nsims == 3: ls = '-'
                    else: raise ValueError("too many sims >3")
                    marker_dic_lr = {
                        'task': 'line', 'ptype': 'cartesian',
                        'position': (1, i_col),
                        'xarr': [x1, x2], "yarr": [y, y],
                        'xlabel': None, "ylabel": None,
                        'label': None,
                        'ls': ls, 'color': 'gray', 'lw': 1.5, 'alpha': 1., 'ds': 'default',
                        'sharey': False,
                        'sharex': False,  # removes angular citkscitks
                        'fontsize': 14,
                        'labelsize': 14
                    }
                    o_plot.set_plot_dics.append(marker_dic_lr)
        # if do_plot_promptcoll:
        #     for q in simulations2[eos].keys():
        #         for u_sim in simulations2[eos][q].keys():
        #             x = data[eos][q][u_sim]["x"]
        #             y = data[eos][q][u_sim]["y"]
        #             isprompt = data[eos][q][u_sim]["isprompt"]
        #             if v_n_y == "Mej_tot" or v_n_y == "Mej_tot_scaled":
        #                 y = y * 1e2
        #             if v_n_x == "Mej_tot" or v_n_x == "Mej_tot_scaled":
        #                 x = x * 1e2
        #             if isprompt:
        #                 marker_dic_lr = {
        #                     'task': 'line', 'ptype': 'cartesian',
        #                     'position': (1, i_col),
        #                     'xarr': [x], "yarr": [y],
        #                     'xlabel': None, "ylabel": None,
        #                     'label': None,
        #                     'marker': 's', 'color': 'gray', 'ms': 10., 'alpha': 0.4,
        #                     'sharey': False,
        #                     'sharex': False,  # removes angular citkscitks
        #                     'fontsize': 14,
        #                     'labelsize': 14
        #                 }
        #                 # if  eos == "BLh" and u_sim == simulations2[eos][q].keys()[-1]:
        #                 #     print('-0--------------------')
        #                 marker_dic_lr['legend'] = {'loc':'upper left', 'ncol':1, 'shadow': False, 'framealpha':0., 'borderaxespad':0., 'fontsize':11}
        #                 o_plot.set_plot_dics.append(marker_dic_lr)
        # if do_plot_bh:
        #     for q in simulations2[eos].keys():
        #         for u_sim in simulations2[eos][q].keys():
        #             x = data[eos][q][u_sim]["x"]
        #             y = data[eos][q][u_sim]["y"]
        #             isbh = data[eos][q][u_sim]["isbh"]
        #             if v_n_y == "Mej_tot" or v_n_y == "Mej_tot_scaled":
        #                 y = y * 1e2
        #             if v_n_x == "Mej_tot" or v_n_x == "Mej_tot_scaled":
        #                 x = x * 1e2
        #             if isbh:
        #                 marker_dic_lr = {
        #                     'task': 'line', 'ptype': 'cartesian',
        #                     'position': (1, i_col),
        #                     'xarr': [x], "yarr": [y],
        #                     'xlabel': None, "ylabel": None,
        #                     'label': None,
        #                     'marker': 'o', 'color': 'gray', 'ms': 10., 'alpha': 0.4,
        #                     'sharey': False,
        #                     'sharex': False,  # removes angular citkscitks
        #                     'fontsize': 14,
        #                     'labelsize': 14
        #                 }
        #                 # if  eos == "BLh" and u_sim == simulations2[eos][q].keys()[-1]:
        #                 #     print('-0--------------------')
        #                 marker_dic_lr['legend'] = {'loc':'upper left', 'ncol':1, 'shadow': False, 'framealpha':0., 'borderaxespad':0., 'fontsize':11}
        #                 o_plot.set_plot_dics.append(marker_dic_lr)

        # LEGEND
        # if eos == "DD2" and plot_legend:
        #     for res in ["HR", "LR", "SR"]:
        #         marker_dic_lr = {
        #             'task': 'line', 'ptype': 'cartesian',
        #             'position': (1, i_col),
        #             'xarr': [-1], "yarr": [-1],
        #             'xlabel': None, "ylabel": None,
        #             'label': res,
        #             'marker': 'd', 'color': 'gray', 'ms': 8, 'alpha': 1.,
        #             'sharey': False,
        #             'sharex': False,  # removes angular citkscitks
        #             'fontsize': 14,
        #             'labelsize': 14
        #         }
        #         if res == "HR": marker_dic_lr['marker'] = "v"
        #         if res == "SR": marker_dic_lr['marker'] = "d"
        #         if res == "LR": marker_dic_lr['marker'] = "^"
        #         # if res == "BH": marker_dic_lr['marker'] = "x"
        #         if res == "SR":
        #             if v_n_y == "Ye_ave":
        #                 loc = 'lower right'
        #             else:
        #                 loc = 'upper right'
        #             marker_dic_lr['legend'] = {'loc': loc, 'ncol': 1, 'fontsize': 12, 'shadow': False,
        #                                        'framealpha': 0.5, 'borderaxespad': 0.0}
        #         o_plot.set_plot_dics.append(marker_dic_lr)
        #
        xarr = np.array(data[eos]["xs"])
        yarr = np.array(data[eos]["ys"])
        colarr = data[eos]["cs"]
        markers = data[eos]['markers']
        # marker = data[eos]["res" + 's']
        # edgecolor = data[eos]["vis" + 's']
        # bh_marker = data[eos]["tcoll" + 's']
        #
        # UTILS.fit_polynomial(xarr, yarr, 1, 100)
        #
        # print(xarr, yarr); exit(1)
        if v_n_y == "Mej_tot" or v_n_y == "Mej_tot_scaled":
            yarr = yarr * 1e2
        if v_n_x == "Mej_tot" or v_n_x == "Mej_tot_scaled":
            xarr = xarr * 1e2

        #
        #
        #
        # dic_bh = {
        #     'task': 'scatter', 'ptype': 'cartesian',  # 'aspect': 1.,
        #     'xarr': xarr, "yarr": yarr, "zarr": colarr,
        #     'position': (1, i_col),  # 'title': '[{:.1f} ms]'.format(time_),
        #     'cbar': {},
        #     'v_n_x': v_n_x, 'v_n_y': v_n_y, 'v_n': v_n_col,
        #     'xlabel': None, "ylabel": None, 'label': eos,
        #     'xmin': 300, 'xmax': 900, 'ymin': 0.03, 'ymax': 0.3, 'vmin': 1.0, 'vmax': 1.5,
        #     'fill_vmin': False,  # fills the x < vmin with vmin
        #     'xscale': None, 'yscale': None,
        #     'cmap': 'viridis', 'norm': None, 'ms': 80, 'marker': bh_marker, 'alpha': 1.0, "edgecolors": edgecolor,
        #     'fancyticks': True,
        #     'minorticks': True,
        #     'title': {},
        #     'legend': {},
        #     'sharey': False,
        #     'sharex': False,  # removes angular citkscitks
        #     'fontsize': 14,
        #     'labelsize': 14
        # }
        #
        # if mask_y != None and mask_y.__contains__("bern"):
        #     o_plot.set_plot_dics.append(dic_bh)
        #

        #

        #
        # print("marker: {}".format(marker))
        dic = {
            'task': 'scatter', 'ptype': 'cartesian',  # 'aspect': 1.,
            'xarr': xarr, "yarr": yarr, "zarr": colarr,
            'position': (1, i_col),  # 'title': '[{:.1f} ms]'.format(time_),
            'cbar': {},
            'v_n_x': v_n_x, 'v_n_y': v_n_y, 'v_n': v_n_col,
            'xlabel': None, "ylabel": Labels.labels(v_n_y, mask_y),
            'xmin': 300, 'xmax': 900, 'ymin': 0.03, 'ymax': 0.3, 'vmin': 1.0, 'vmax': 1.9,
            'fill_vmin': False,  # fills the x < vmin with vmin
            'xscale': None, 'yscale': None,
            'cmap': 'tab10', 'norm': None, 'ms': 60, 'markers': markers, 'alpha': 0.6, "edgecolors": None,
            'tick_params': {"axis": 'both', "which": 'both', "labelleft": True,
                            "labelright": False,  # "tick1On":True, "tick2On":True,
                            "labelsize": 12,
                            "direction": 'in',
                            "bottom": True, "top": True, "left": True, "right": True},
            'yaxiscolor': {'bottom': 'black', 'top': 'black', 'right': 'black', 'left': 'black'},
            'minorticks': True,
            'title': {},#{"text": eos, "fontsize": 12},
            'label': None,
            'legend': {},
            'sharey': False,
            'sharex': False,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14
        }
        #

        if v_n_y == "q":
            dic['ymin'], dic['ymax'] = 0.9, 2.0
        if v_n_col == "nsims":
            dic['vmin'], dic['vmax'] = 1, 3.9
            dic['cmap'] = __get_custom_descrete_colormap(3)
            # dic['cmap'] = 'RdYlBu'

        if v_n_y == "Mdisk3Dmax":
            dic['ymin'], dic['ymax'] = 0.03, 0.30
        if v_n_y == "Mb":
            dic['ymin'], dic['ymax'] = 2.8, 3.4
        if v_n_y == "Mej_tot" and mask_y == "geo":
            dic['ymin'], dic['ymax'] = 0, 1.2
        if v_n_y == "Mej_tot_scaled" and mask_y == "geo":
            dic['ymin'], dic['ymax'] = 0, 0.5

        if v_n_y == "Mej_tot_scaled2" and mask_y == "geo":
            dic['ymin'], dic['ymax'] = 0, 1.
        if v_n_y == "Mej_tot_scaled2" and mask_y == "geo_entropy_above_10":
            dic['ymin'], dic['ymax'] = 0, 0.01
        if v_n_y == "Mej_tot_scaled2" and mask_y == "geo_entropy_below_10":
            dic['ymin'], dic['ymax'] = 0, 0.02

        if v_n_y == "Mej_tot" and mask_y == "bern_geoend":
            if dic['yscale'] == "log":
                dic['ymin'], dic['ymax'] = 1e-3, 2e0
            else:
                dic['ymin'], dic['ymax'] = 0, 3.2
        if v_n_y == "Mej_tot" and mask_y == "geo_entropy_above_10":
            if dic['yscale'] == "log":
                dic['ymin'], dic['ymax'] = 1e-3, 2e0
            else:
                dic['ymin'], dic['ymax'] = 0, .6
        if v_n_y == "Mej_tot" and mask_y == "geo_entropy_below_10":
            if dic['yscale'] == "log":
                dic['ymin'], dic['ymax'] = 1e-2, 2e0
            else:
                dic['ymin'], dic['ymax'] = 0, 1.2
        if v_n_y == "Mej_tot_scaled" and mask_y == "bern_geoend":
            dic['ymin'], dic['ymax'] = 0, 3.

        if v_n_y == "Ye_ave" and mask_y == "geo":
            dic['ymin'], dic['ymax'] = 0.01, 0.35
        if v_n_y == "Ye_ave" and mask_y == "bern_geoend":
            dic['ymin'], dic['ymax'] = 0.1, 0.4
        if v_n_y == "vel_inf_ave" and mask_y == "geo":
            dic['ymin'], dic['ymax'] = 0.1, 0.3
        if v_n_y == "vel_inf_ave" and mask_y == "bern_geoend":
            dic['ymin'], dic['ymax'] = 0.05, 0.25
        #

        #
        if v_n_x == "Mdisk3Dmax":
            dic['xmin'], dic['xmax'] = 0.03, 0.30
        if v_n_x == "Mb":
            dic['xmin'], dic['xmax'] = 2.8, 3.4
        if v_n_x == "Mej_tot" and mask_x == "geo":
            dic['xmin'], dic['xmax'] = 0, 1.5
        if v_n_x == "Mej_tot_scaled" and mask_x == "geo":
            dic['xmin'], dic['xmax'] = 0, 0.5
        if v_n_x == "Mej_tot" and mask_x == "bern_geoend":
            dic['xmin'], dic['xmax'] = 0, 3.2
        if v_n_x == "Mej_tot" and mask_x == "geo_entropy_above_10":
            if dic['xscale'] == "log":
                dic['xmin'], dic['xmax'] = 1e-3, 2e0
            else:
                dic['xmin'], dic['xmax'] = 0, .6
        if v_n_x == "Mej_tot" and mask_x == "geo_entropy_below_10":
            if dic['xscale'] == "log":
                dic['xmin'], dic['xmax'] = 1e-2, 2e0
            else:
                dic['xmin'], dic['xmax'] = 0, 1.2
        if v_n_x == "Mej_tot_scaled" and mask_x == "bern_geoend":
            dic['xmin'], dic['xmax'] = 0, 3.
        if v_n_x == "Ye_ave" and mask_x == "geo":
            dic['xmin'], dic['xmax'] = 0.01, 0.30
        if v_n_x == "Ye_ave" and mask_x == "bern_geoend":
            dic['xmin'], dic['xmax'] = 0.1, 0.4
        if v_n_x == "vel_inf_ave" and mask_x == "geo":
            dic['xmin'], dic['xmax'] = 0.1, 0.3
        if v_n_x == "vel_inf_ave" and mask_x == "bern_geoend":
            dic['xmin'], dic['xmax'] = 0.05, 0.25


        #
        # if eos == "SLy4":
        #     dic['xmin'], dic['xmax'] = 380, 420
        #     dic['xticks'] = [390, 410]
        # if eos == "SFHo":
        #     dic['xmin'], dic['xmax'] = 390, 430
        #     dic['xticks'] = [400, 420]
        # if eos == "BLh":
        #     dic['xmin'], dic['xmax'] = 510, 550
        #     dic['xticks'] = [520, 540]
        # if eos == "LS220":
        #     dic['xmin'], dic['xmax'] = 690, 730
        #     dic['xticks'] = [700, 720]
        # if eos == "DD2":
        #     dic['xmin'], dic['xmax'] = 820, 860
        #     dic['xticks'] = [830, 850]
        # if eos == "SLy4":
        #     dic['tick_params']['right'] = False
        #     dic['yaxiscolor']["right"] = "lightgray"
        # elif eos == "DD2":
        #     dic['tick_params']['left'] = False
        #     dic['yaxiscolor']["left"] = "lightgray"
        # else:
        #     dic['tick_params']['left'] = False
        #     dic['tick_params']['right'] = False
        #     dic['yaxiscolor']["left"] = "lightgray"
        #     dic['yaxiscolor']["right"] = "lightgray"

        #
        # if eos != "SLy4" and eos != "DD2":
        #     dic['yaxiscolor'] = {'left':'lightgray','right':'lightgray', 'label': 'black'}
        #     dic['ytickcolor'] = {'left':'lightgray','right':'lightgray'}
        #     dic['yminortickcolor'] = {'left': 'lightgray', 'right': 'lightgray'}
        # elif eos == "DD2":
        #     dic['yaxiscolor'] = {'left': 'lightgray', 'right': 'black', 'label': 'black'}
        #     # dic['ytickcolor'] = {'left': 'lightgray'}
        #     # dic['yminortickcolor'] = {'left': 'lightgray'}
        # elif eos == "SLy4":
        #     dic['yaxiscolor'] = {'left': 'black', 'right': 'lightgray', 'label': 'black'}
        #     # dic['ytickcolor'] = {'right': 'lightgray'}
        #     # dic['yminortickcolor'] = {'right': 'lightgray'}

        #
        # if eos != "SLy4":
        #     dic['sharey'] = True
        if eos == "BLh":
            dic['xlabel'] = Labels.labels(v_n_x, mask_x)
        if eos == 'DD2':
            dic['cbar'] = {'location': 'right .03 .0', 'label': Labels.labels(v_n_col),  # 'fmt': '%.1f',
                           'labelsize': 14, 'fontsize': 14}
            if v_n_col == "nsims":
                dic['cbar']['fmt'] = '%d'
        #
        o_plot.set_plot_dics.append(dic)
        #

        # i_col = i_col + 1

        if do_plot_old_table:
            if v_n_x == 'Lambda':
                dic['xmin'], dic['xmax'] = 5, 1500

    # LEGEND



    #
    o_plot.main()
    exit(0)

def plot_summary_quntity_all_in_one2():
    """
    Plot unique simulations point by point with error bars
    :return:
    """
    v_n_x = "Lambda"
    v_n_y = "Mej_tot"
    v_n_col = "q"
    det = 0
    do_plot_error_bar = True
    mask_x, mask_y, mask_col = None, "geo", None
    data = {}
    error = 0.2 # in * 100 percent

    # collect data
    for eos in simulations2.keys():
        data[eos] = {}
        for q in simulations2[eos]:
            data[eos][q] = {}
            for u_sim in simulations2[eos][q]:
                data[eos][q][u_sim] = {}
                sims = simulations2[eos][q][u_sim]
                o_inits = [LOAD_INIT_DATA(sim) for sim in sims]
                o_pars = [ADD_METHODS_ALL_PAR(sim) for sim in sims]
                x_coord, x_err1, x_err2 = __get_val_err(sims, o_inits, o_pars, v_n_x, det, mask_x, error)
                y_coord, y_err1, y_err2 = __get_val_err(sims, o_inits, o_pars, v_n_y, det, mask_y, error)
                col_coord, col_err1, col_err2 = __get_val_err(sims, o_inits, o_pars, v_n_col, det, mask_col, error)
                data[eos][q][u_sim]["lserr"] = len(sims)
                data[eos][q][u_sim]["x"] = x_coord
                data[eos][q][u_sim]["xe1"] = x_err1
                data[eos][q][u_sim]["xe2"] = x_err2
                data[eos][q][u_sim]["y"] = y_coord
                data[eos][q][u_sim]["ye1"] = y_err1
                data[eos][q][u_sim]["ye2"] = y_err2
                data[eos][q][u_sim]["c"] = col_coord
                data[eos][q][u_sim]["ce1"] = col_err1
                data[eos][q][u_sim]["ce2"] = col_err2
                Printcolor.blue("Processing {} ({} sims) x:[{:.1f}, v:{:.1f} ^{:.1f}] y:[{:.5f}, v{:.5f} ^{:.5f}] col:{:.1f}"
                                .format(u_sim, len(sims), x_coord, x_err1, x_err2, y_coord, y_err1, y_err2, col_coord))
    Printcolor.green("Data is collaected")



    # stuck data for scatter plot
    for eos in simulations2.keys():
        for v_n in ["x", "y", "c"]:
            arr = []
            for q in simulations2[eos].keys():
                for u_sim in simulations2[eos][q]:
                    arr.append(data[eos][q][u_sim][v_n])
            data[eos][v_n+"s"] = arr

    Printcolor.green("Data is stacked")
    # plot the scatter points
    figname = ''
    if mask_x == None:
        figname = figname + v_n_x + '_'
    else:
        figname = figname + v_n_x + '_' + mask_x + '_'
    if mask_y == None:
        figname = figname + v_n_y + '_'
    else:
        figname = figname + v_n_y + '_' + mask_y + '_'
    if mask_col == None:
        figname = figname + v_n_col + '_'
    else:
        figname = figname + v_n_col + '_' + mask_col + '_'
    if det == None:
        figname = figname + ''
    else:
        figname = figname + str(det)
    figname = figname + '4.png'
    #
    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + "all2/dyn_ejecta/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = figname
    o_plot.gen_set["sharex"] = True
    o_plot.gen_set["sharey"] = False
    o_plot.gen_set["subplots_adjust_h"] = 0.0
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []


    # ERROR BAR


    # ACTUAL PLOT
    i_col = 1
    for eos in ["SLy4", "SFHo", "BLh", "LS220", "DD2"]:
        print(eos)
        # Error Bar
        if do_plot_error_bar:
            for q in simulations2[eos].keys():
                for u_sim in simulations2[eos][q].keys():
                    x = data[eos][q][u_sim]["x"]
                    x1 = data[eos][q][u_sim]["xe1"]
                    x2 = data[eos][q][u_sim]["xe2"]
                    y = data[eos][q][u_sim]["y"]
                    y1 = data[eos][q][u_sim]["ye1"]
                    y2 = data[eos][q][u_sim]["ye2"]
                    nsims = data[eos][q][u_sim]["lserr"]
                    if v_n_y == "Mej_tot":
                        y1 = y1 * 1e2
                        y2 = y2 * 1e2
                        y = y * 1e2
                    if nsims == 1: ls = ':'
                    elif nsims == 2: ls = '--'
                    elif nsims == 3: ls = '-'
                    else: raise ValueError("too many sims >3")
                    marker_dic_lr = {
                        'task': 'line', 'ptype': 'cartesian',
                        'position': (1, i_col),
                        'xarr': [x, x], "yarr": [y1, y2],
                        'xlabel': None, "ylabel": None,
                        'label': None,
                        'ls': ls, 'color': 'gray', 'lw': 1.5, 'alpha': 1., 'ds': 'default',
                        'sharey': False,
                        'sharex': False,  # removes angular citkscitks
                        'fontsize': 14,
                        'labelsize': 14
                    }
                    o_plot.set_plot_dics.append(marker_dic_lr)


        # LEGEND
        # if eos == "DD2" and plot_legend:
        #     for res in ["HR", "LR", "SR"]:
        #         marker_dic_lr = {
        #             'task': 'line', 'ptype': 'cartesian',
        #             'position': (1, i_col),
        #             'xarr': [-1], "yarr": [-1],
        #             'xlabel': None, "ylabel": None,
        #             'label': res,
        #             'marker': 'd', 'color': 'gray', 'ms': 8, 'alpha': 1.,
        #             'sharey': False,
        #             'sharex': False,  # removes angular citkscitks
        #             'fontsize': 14,
        #             'labelsize': 14
        #         }
        #         if res == "HR": marker_dic_lr['marker'] = "v"
        #         if res == "SR": marker_dic_lr['marker'] = "d"
        #         if res == "LR": marker_dic_lr['marker'] = "^"
        #         # if res == "BH": marker_dic_lr['marker'] = "x"
        #         if res == "SR":
        #             if v_n_y == "Ye_ave":
        #                 loc = 'lower right'
        #             else:
        #                 loc = 'upper right'
        #             marker_dic_lr['legend'] = {'loc': loc, 'ncol': 1, 'fontsize': 12, 'shadow': False,
        #                                        'framealpha': 0.5, 'borderaxespad': 0.0}
        #         o_plot.set_plot_dics.append(marker_dic_lr)
        #
        xarr = np.array(data[eos]["xs"])
        yarr = np.array(data[eos]["ys"])
        colarr = data[eos]["cs"]
        # marker = data[eos]["res" + 's']
        # edgecolor = data[eos]["vis" + 's']
        # bh_marker = data[eos]["tcoll" + 's']
        #
        # UTILS.fit_polynomial(xarr, yarr, 1, 100)
        #
        # print(xarr, yarr); exit(1)
        if v_n_y == "Mej_tot":
            yarr = yarr * 1e2
        if v_n_x == "Mej_tot":
            xarr = xarr * 1e2
        #
        #
        #
        # dic_bh = {
        #     'task': 'scatter', 'ptype': 'cartesian',  # 'aspect': 1.,
        #     'xarr': xarr, "yarr": yarr, "zarr": colarr,
        #     'position': (1, i_col),  # 'title': '[{:.1f} ms]'.format(time_),
        #     'cbar': {},
        #     'v_n_x': v_n_x, 'v_n_y': v_n_y, 'v_n': v_n_col,
        #     'xlabel': None, "ylabel": None, 'label': eos,
        #     'xmin': 300, 'xmax': 900, 'ymin': 0.03, 'ymax': 0.3, 'vmin': 1.0, 'vmax': 1.5,
        #     'fill_vmin': False,  # fills the x < vmin with vmin
        #     'xscale': None, 'yscale': None,
        #     'cmap': 'viridis', 'norm': None, 'ms': 80, 'marker': bh_marker, 'alpha': 1.0, "edgecolors": edgecolor,
        #     'fancyticks': True,
        #     'minorticks': True,
        #     'title': {},
        #     'legend': {},
        #     'sharey': False,
        #     'sharex': False,  # removes angular citkscitks
        #     'fontsize': 14,
        #     'labelsize': 14
        # }
        #
        # if mask_y != None and mask_y.__contains__("bern"):
        #     o_plot.set_plot_dics.append(dic_bh)
        #

        #

        #
        # print("marker: {}".format(marker))
        dic = {
            'task': 'scatter', 'ptype': 'cartesian',  # 'aspect': 1.,
            'xarr': xarr, "yarr": yarr, "zarr": colarr,
            'position': (1, i_col),  # 'title': '[{:.1f} ms]'.format(time_),
            'cbar': {},
            'v_n_x': v_n_x, 'v_n_y': v_n_y, 'v_n': v_n_col,
            'xlabel': None, "ylabel": Labels.labels(v_n_y),
            'xmin': 300, 'xmax': 900, 'ymin': 0.03, 'ymax': 0.3, 'vmin': 1.0, 'vmax': 1.9,
            'fill_vmin': False,  # fills the x < vmin with vmin
            'xscale': None, 'yscale': None,
            'cmap': 'Dark2', 'norm': None, 'ms': 80, 'marker': "d", 'alpha': 1.0, "edgecolors": None,
            'tick_params': {"axis": 'both', "which": 'both', "labelleft": True,
                            "labelright": False,  # "tick1On":True, "tick2On":True,
                            "labelsize": 12,
                            "direction": 'in',
                            "bottom": True, "top": True, "left": True, "right": True},
            'yaxiscolor': {'bottom': 'black', 'top': 'black', 'right': 'black', 'left': 'black'},
            'minorticks': True,
            'title': {"text": eos, "fontsize": 12},
            'label': "xxx",
            'legend': {},
            'sharey': False,
            'sharex': False,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14
        }

        if v_n_x == "q":
            dic["xmin"] = 0.9
            dic["xmax"] = 1.9
        if v_n_col == "Lambda":
            dic["vmin"] = 200
            dic["vmax"] = 900

        #
        if v_n_y == "Mdisk3Dmax":
            dic['ymin'], dic['ymax'] = 0.03, 0.30
        if v_n_y == "Mej_tot" and mask_y == "geo":
            dic['ymin'], dic['ymax'] = 0, 1.5
        if v_n_y == "Mej_tot" and mask_y == "bern_geoend":
            dic['ymin'], dic['ymax'] = 0, 3.2
        if v_n_y == "Ye_ave" and mask_y == "geo":
            dic['ymin'], dic['ymax'] = 0.01, 0.30
        if v_n_y == "Ye_ave" and mask_y == "bern_geoend":
            dic['ymin'], dic['ymax'] = 0.1, 0.4
        if v_n_y == "vel_inf_ave" and mask_y == "geo":
            dic['ymin'], dic['ymax'] = 0.1, 0.3
        if v_n_y == "vel_inf_ave" and mask_y == "bern_geoend":
            dic['ymin'], dic['ymax'] = 0.05, 0.25
        #
        # if eos == "SLy4":
        #     dic['xmin'], dic['xmax'] = 380, 420
        #     dic['xticks'] = [390, 410]
        # if eos == "SFHo":
        #     dic['xmin'], dic['xmax'] = 390, 430
        #     dic['xticks'] = [400, 420]
        # if eos == "BLh":
        #     dic['xmin'], dic['xmax'] = 510, 550
        #     dic['xticks'] = [520, 540]
        # if eos == "LS220":
        #     dic['xmin'], dic['xmax'] = 690, 730
        #     dic['xticks'] = [700, 720]
        # if eos == "DD2":
        #     dic['xmin'], dic['xmax'] = 820, 860
        #     dic['xticks'] = [830, 850]
        # if eos == "SLy4":
        #     dic['tick_params']['right'] = False
        #     dic['yaxiscolor']["right"] = "lightgray"
        # elif eos == "DD2":
        #     dic['tick_params']['left'] = False
        #     dic['yaxiscolor']["left"] = "lightgray"
        # else:
        #     dic['tick_params']['left'] = False
        #     dic['tick_params']['right'] = False
        #     dic['yaxiscolor']["left"] = "lightgray"
        #     dic['yaxiscolor']["right"] = "lightgray"

        #
        # if eos != "SLy4" and eos != "DD2":
        #     dic['yaxiscolor'] = {'left':'lightgray','right':'lightgray', 'label': 'black'}
        #     dic['ytickcolor'] = {'left':'lightgray','right':'lightgray'}
        #     dic['yminortickcolor'] = {'left': 'lightgray', 'right': 'lightgray'}
        # elif eos == "DD2":
        #     dic['yaxiscolor'] = {'left': 'lightgray', 'right': 'black', 'label': 'black'}
        #     # dic['ytickcolor'] = {'left': 'lightgray'}
        #     # dic['yminortickcolor'] = {'left': 'lightgray'}
        # elif eos == "SLy4":
        #     dic['yaxiscolor'] = {'left': 'black', 'right': 'lightgray', 'label': 'black'}
        #     # dic['ytickcolor'] = {'right': 'lightgray'}
        #     # dic['yminortickcolor'] = {'right': 'lightgray'}

        #
        # if eos != "SLy4":
        #     dic['sharey'] = True
        if eos == "BLh":
            dic['xlabel'] = Labels.labels(v_n_x)
        if eos == 'DD2':
            dic['cbar'] = {'location': 'right .03 .0', 'label': Labels.labels(v_n_col),  # 'fmt': '%.1f',
                           'labelsize': 14, 'fontsize': 14}
        #
        o_plot.set_plot_dics.append(dic)
        #

        # i_col = i_col + 1

    #
    o_plot.main()
    exit(0)
#  ---

def plot_total_fluxes_for_long_sims_bern(mask):

    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + "all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = "totfluxes_{}.png".format(mask)
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.3
    o_plot.gen_set["subplots_adjust_w"] = 0.01
    o_plot.set_plot_dics = []

    det = 0

    # sims = ["DD2_M13641364_M0_LK_SR_R04", "BLh_M13641364_M0_LK_SR", "LS220_M13641364_M0_LK_SR", "SLy4_M13641364_M0_LK_SR", "SFHo_M13641364_M0_LK_SR"]
    # lbls = ["DD2", "BLh", "LS220", "SLy4", "SFHo"]
    # masks= [mask, mask, mask, mask, mask]
    # colors=["black", "gray", "red", "blue", "green"]
    # lss   =["-", "-", "-", "-", "-"]
    #
    # sims += ["DD2_M15091235_M0_LK_SR", "LS220_M14691268_M0_LK_SR", "SFHo_M14521283_M0_LK_SR"]
    # lbls += ["DD2 151 124", "LS220 150 127", "SFHo 145 128"]
    # masks+= [mask, mask, mask, mask, mask]
    # colors+=["black", "red", "green"]
    # lss   +=["--", "--", "--"]
    # sims = ["BLh_M10651772_M0_LK_LR", "BLh_M11041699_M0_LK_LR", "BLh_M11841581_M0_LK_LR", "BLh_M12591482_M0_LK_LR",
    #         "BLh_M13641364_M0_LK_LR", "DD2_M13641364_M0_LR_R04"]
    sims = ["BLh_M11461635_M0_LK_SR", "BLh_M13641364_M0_LK_SR", "DD2_M13641364_M0_SR",
            "DD2_M13641364_M0_SR_R04", "DD2_M13641364_M0_LK_SR_R04",
            "DD2_M14971245_M0_SR", "DD2_M15091235_M0_LK_SR", "DD2_M11461635_M0_LK_SR",
            "LS220_M14691268_M0_LK_SR",
            "SFHo_M11461635_M0_LK_SR",
            "SLy4_M11461635_M0_LK_SR"]
    lbls = [sim.replace('_', '\_') for sim in sims]
    masks= [mask for sim in sims]
    colors=["green", "green", "green",
            "blue", "blue",
            "cyan", "cyan", "cyan",
            "red",
            "orange",
            "purple"]
    lss   =["-", "--", ":",
            "-", '--',
            "-", "--", ":",
            "-",
            "-",
            "-"]

    # sims += ["DD2_M15091235_M0_LK_SR", "LS220_M14691268_M0_LK_SR"]
    # lbls += ["DD2 151 124", "LS220 150 127"]
    # masks+= [mask, mask]
    # colors+=["blue", "red"]
    # lss   +=["--", "--"]


    i_x_plot = 1
    for sim, lbl, mask, color, ls in zip(sims, lbls, masks, colors, lss):

        fpath = Paths.ppr_sims + sim + "/" + "outflow_{}/".format(det) + mask + '/' + "total_flux.dat"
        if not os.path.isfile(fpath):
            raise IOError("File does not exist: {}".format(fpath))

        timearr, massarr = np.loadtxt(fpath, usecols=(0, 2), unpack=True)

        fpath = Paths.ppr_sims + sim + "/" + "waveforms/" + "tmerger.dat"
        if not os.path.isfile(fpath):
            raise IOError("File does not exist: {}".format(fpath))
        tmerg = np.float(np.loadtxt(fpath, unpack=True))
        timearr = timearr - (tmerg * Constants.time_constant * 1e-3)

        plot_dic = {
            'task': 'line', 'ptype': 'cartesian',
            'position': (1, 1),
            'xarr': timearr * 1e3, 'yarr': massarr * 1e2,
            'v_n_x': "time", 'v_n_y': "mass",
            'color': color, 'ls': ls, 'lw': 0.8, 'ds': 'default', 'alpha': 1.0,
            'xmin': 0, 'xmax': 110, 'ymin': 0, 'ymax': 2.5,
            'xlabel': Labels.labels("t-tmerg"), 'ylabel': Labels.labels("ejmass"),
            'label': lbl, 'yscale': 'linear',
            'fancyticks': True, 'minorticks': True,
            'fontsize': 14,
            'labelsize': 14,
            'legend': {'loc': 'best', 'ncol': 1, 'fontsize': 11,
                       "bbox_to_anchor":(1.1,1.1)} # 'loc': 'best', 'ncol': 2, 'fontsize': 18
        }
        if mask == "geo": plot_dic["ymax"] = 1.

        if sim >= sims[-1]:
            plot_dic['legend'] = {'loc': 'best', 'ncol': 1, 'fontsize': 11,
                       "bbox_to_anchor":(1.,1.)}

        o_plot.set_plot_dics.append(plot_dic)




        #
        #


        i_x_plot += 1
    o_plot.main()
    exit(1)


'''====================================================| DISK |======================================================'''


''' LK comparison '''
#
def plot_den_unb_vel_z():

    # tmp = d3class.get_data(688128, 3, "xy", "ang_mom_flux")
    # print(tmp.min(), tmp.max())
    # print(tmp)
    # exit(1) # dens_unb_geo

    """ --- --- --- """


    '''sly4 '''
    simlist = ["SLy4_M13641364_M0_SR", "SLy4_M13641364_M0_SR", "SLy4_M13641364_M0_SR", "SLy4_M13641364_M0_SR"]
    # itlist = [434176, 475136, 516096, 565248]
    # itlist = [606208, 647168, 696320, 737280]
    # itlist = [434176, 516096, 647168, 737280]
    ''' ls220 '''
    simlist = ["LS220_M14691268_M0_LK_SR", "LS220_M14691268_M0_LK_SR", "LS220_M14691268_M0_LK_SR"]#, "LS220_M14691268_M0_LK_SR"]
    itlist = [1515520, 1728512, 1949696]#, 2162688]
    ''' dd2 '''
    simlist = ["DD2_M13641364_M0_LK_SR_R04", "DD2_M13641364_M0_LK_SR_R04", "DD2_M13641364_M0_LK_SR_R04"]#, "DD2_M13641364_M0_LK_SR_R04"]
    itlist = [1111116,1741554,2213326]#,2611022]
    #
    simlist = ["DD2_M13641364_M0_LK_SR_R04", "BLh_M13641364_M0_LK_SR", "LS220_M14691268_M0_LK_SR", "SLy4_M13641364_M0_SR"]
    itlist = [2611022, 1974272, 1949696, 737280]

    # equal post-merger time
    simlist = ["DD2_M13641364_M0_SR_R04", "DD2_M13641364_M0_LK_SR_R04"]
    itlist = [2116290, 2058718]

    # equal post-merger time
    simlist = ["LS220_M13641364_M0_SR", "LS220_M13641364_M0_LK_SR_restart"]
    itlist = [737280, 696320]

    # equal post-merger time (pre-black hole)
    # simlist = ["LS220_M13641364_M0_SR", "LS220_M13641364_M0_LK_SR_restart"]
    # itlist = [540672, 499712]

    #
    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + 'all2/'
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4*len(simlist), 6.0)  # <->, |] # to match hists with (8.5, 2.7)
    o_plot.gen_set["figname"] = "disk_structure_last_ls220_lk.png"
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = -0.35
    o_plot.gen_set["subplots_adjust_w"] = 0.05
    o_plot.set_plot_dics = []
    #
    rl = 3
    #
    o_plot.gen_set["figsize"] = (4.2*len(simlist), 8.0)  # <->, |] # to match hists with (8.5, 2.7)

    plot_x_i = 1
    for sim, it in zip(simlist, itlist):
        print("sim:{} it:{}".format(sim, it))
        d3class = LOAD_PROFILE_XYXZ(sim)
        d1class = ADD_METHODS_ALL_PAR(sim)

        t = d3class.get_time_for_it(it, d1d2d3prof="prof")
        tmerg = d1class.get_par("tmerg")
        time = t - tmerg
        xmin, xmax, ymin, ymax, zmin, zmax = UTILS.get_xmin_xmax_ymin_ymax_zmin_zmax(rl)



        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        mask = "x>0"
        #
        v_n = "rho"
        data_arr = d3class.get_data(it, rl, "xz", v_n)
        x_arr = d3class.get_data(it, rl, "xz", "x")
        z_arr = d3class.get_data(it, rl, "xz", "z")
        # print(data_arr); exit(1)

        contour_dic_xz = {
            'task': 'contour',
            'ptype': 'cartesian', 'aspect': 1.,
            'xarr': x_arr, "yarr": z_arr, "zarr": data_arr, 'levels': [1.e13 / 6.176e+17],
            'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
            'colors': ['white'], 'lss': ["-"], 'lws': [1.],
            'v_n_x': 'x', 'v_n_y': 'y', 'v_n': 'rho',
            'xscale': None, 'yscale': None,
            'fancyticks': True,
            'sharey': False,
            'sharex': True,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14}
        o_plot.set_plot_dics.append(contour_dic_xz)

        rho_dic_xz = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                      'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                      'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                      'cbar': {},
                      'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                      'xmin': xmin, 'xmax': xmax, 'ymin': zmin, 'ymax': zmax, 'vmin': 1e-9, 'vmax': 1e-5,
                      'fill_vmin': False,  # fills the x < vmin with vmin
                      'xscale': None, 'yscale': None,
                      'mask': mask, 'cmap': 'Greys', 'norm': "log",
                      'fancyticks': True,
                      'minorticks':True,
                      'title': {"text": sim.replace('_', '\_') + " {:.1f}ms".format(time * 1e3), 'fontsize': 12},
                      #'title': {"text": r'$t-t_{merg}:$' + r'${:.1f}$ [ms]'.format((t - tmerg) * 1e3), 'fontsize': 14},
                      'sharey': False,
                      'sharex': True,  # removes angular citkscitks
                      'fontsize': 14,
                      'labelsize': 14
                      }
        #
        data_arr = d3class.get_data(it, rl, "xy", v_n)
        x_arr = d3class.get_data(it, rl, "xy", "x")
        y_arr = d3class.get_data(it, rl, "xy", "y")

        contour_dic_xy = {
            'task': 'contour',
            'ptype': 'cartesian', 'aspect': 1.,
            'xarr': x_arr, "yarr": y_arr, "zarr": data_arr, 'levels': [1.e13 / 6.176e+17],
            'position': (2, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
            'colors': ['white'], 'lss': ["-"], 'lws': [1.],
            'v_n_x': 'x', 'v_n_y': 'y', 'v_n': 'rho',
            'xscale': None, 'yscale': None,
            'fancyticks': True,
            'sharey': False,
            'sharex': True,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14}
        o_plot.set_plot_dics.append(contour_dic_xy)

        rho_dic_xy = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                      'xarr': x_arr, "yarr": y_arr, "zarr": data_arr,
                      'position': (2, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                      'cbar': {},
                      'v_n_x': 'x', 'v_n_y': 'y', 'v_n': v_n,
                      'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'vmin': 1e-9, 'vmax': 1e-5,
                      'fill_vmin': False,  # fills the x < vmin with vmin
                      'xscale': None, 'yscale': None,
                      'mask': mask, 'cmap': 'Greys', 'norm': "log",
                      'fancyticks': True,
                      'minorticks': True,
                      'title': {},
                      'sharey': False,
                      'sharex': False,  # removes angular citkscitks
                      'fontsize': 14,
                      'labelsize': 14
                      }
        #
        if plot_x_i == 1:
            rho_dic_xy['cbar'] = {'location': 'bottom -.05 .00', 'label': r'$\rho$ [GEO]',  # 'fmt': '%.1e',
                          'labelsize': 14,
                          'fontsize': 14}
        if plot_x_i > 1:
            rho_dic_xz['sharey'] = True
            rho_dic_xy['sharey'] = True

        o_plot.set_plot_dics.append(rho_dic_xz)
        o_plot.set_plot_dics.append(rho_dic_xy)

        # ----------------------------------------------------------------------
        v_n = "dens_unb_bern"
        #
        data_arr = d3class.get_data(it, rl, "xz", v_n)
        x_arr = d3class.get_data(it, rl, "xz", "x")
        z_arr = d3class.get_data(it, rl, "xz", "z")
        dunb_dic_xz = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                      'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                      'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                      'cbar': {},
                      'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                      'xmin': xmin, 'xmax': xmax, 'ymin': zmin, 'ymax': zmax, 'vmin': 1e-10, 'vmax': 1e-7,
                      'fill_vmin': False,  # fills the x < vmin with vmin
                      'xscale': None, 'yscale': None,
                      'mask': mask, 'cmap': 'Blues', 'norm': "log",
                      'fancyticks': True,
                       'minorticks': True,
                       'title': {},#{"text": r'$t-t_{merg}:$' + r'${:.1f}$ [ms]'.format((t - tmerg) * 1e3), 'fontsize': 14},
                      'sharex': True,  # removes angular citkscitks
                      'sharey': False,
                      'fontsize': 14,
                      'labelsize': 14
                      }
        #
        data_arr = d3class.get_data(it, rl, "xy", v_n)
        x_arr = d3class.get_data(it, rl, "xy", "x")
        y_arr = d3class.get_data(it, rl, "xy", "y")
        dunb_dic_xy = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                      'xarr': x_arr, "yarr": y_arr, "zarr": data_arr,
                      'position': (2, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                      'cbar': {},
                      'fill_vmin': False,  # fills the x < vmin with vmin
                      'v_n_x': 'x', 'v_n_y': 'y', 'v_n': v_n,
                      'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'vmin': 1e-10, 'vmax': 1e-7,
                      'xscale': None, 'yscale': None,
                      'mask': mask, 'cmap': 'Blues', 'norm': "log",
                      'fancyticks': True,
                       'minorticks': True,
                       'title': {},
                      'sharey': False,
                      'sharex': False,  # removes angular citkscitks
                      'fontsize': 14,
                      'labelsize': 14
                      }
        #
        if plot_x_i == 2:
            dunb_dic_xy['cbar'] = {'location': 'bottom -.05 .00', 'label': r'$D_{\rm{unb}}$ [GEO]',  # 'fmt': '%.1e',
                          'labelsize': 14,
                          'fontsize': 14}
        if plot_x_i > 1:
            dunb_dic_xz['sharey'] = True
            dunb_dic_xy['sharey'] = True

        o_plot.set_plot_dics.append(dunb_dic_xz)
        o_plot.set_plot_dics.append(dunb_dic_xy)

        # ----------------------------------------------------------------------
        mask = "x<0"
        #
        v_n = "Ye"
        cmap = "bwr_r"
        #
        data_arr = d3class.get_data(it, rl, "xz", v_n)
        x_arr = d3class.get_data(it, rl, "xz", "x")
        z_arr = d3class.get_data(it, rl, "xz", "z")
        ye_dic_xz = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                       'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                       'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                       'cbar': {},
                       'fill_vmin': False,  # fills the x < vmin with vmin
                       'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                       'xmin': xmin, 'xmax': xmax, 'ymin': zmin, 'ymax': zmax, 'vmin': 0.05, 'vmax': 0.5,
                       'xscale': None, 'yscale': None,
                       'mask': mask, 'cmap': cmap, 'norm': None,
                       'fancyticks': True,
                       'minorticks': True,
                       'title': {},#{"text": r'$t-t_{merg}:$' + r'${:.1f}$ [ms]'.format((t - tmerg) * 1e3), 'fontsize': 14},
                       'sharey': False,
                       'sharex': True,  # removes angular citkscitks
                       'fontsize': 14,
                       'labelsize': 14
                       }
        #
        data_arr = d3class.get_data(it, rl, "xy", v_n)
        x_arr = d3class.get_data(it, rl, "xy", "x")
        y_arr = d3class.get_data(it, rl, "xy", "y")
        ye_dic_xy = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                       'xarr': x_arr, "yarr": y_arr, "zarr": data_arr,
                       'position': (2, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                       'cbar': {},
                       'fill_vmin': False,  # fills the x < vmin with vmin
                       'v_n_x': 'x', 'v_n_y': 'y', 'v_n': v_n,
                       'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'vmin': 0.01, 'vmax': 0.5,
                       'xscale': None, 'yscale': None,
                       'mask': mask, 'cmap': cmap, 'norm': None,
                       'fancyticks': True,
                       'minorticks': True,
                       'title': {},
                       'sharey': False,
                       'sharex': False,  # removes angular citkscitks
                       'fontsize': 14,
                       'labelsize': 14
                       }
        #
        if plot_x_i == 3:
            ye_dic_xy['cbar'] = {'location': 'bottom -.05 .00', 'label': r'$Y_e$',   'fmt': '%.1f',
                          'labelsize': 14,
                          'fontsize': 14}
        if plot_x_i > 1:
            ye_dic_xz['sharey'] = True
            ye_dic_xy['sharey'] = True

        o_plot.set_plot_dics.append(ye_dic_xz)
        o_plot.set_plot_dics.append(ye_dic_xy)

        # ----------------------------------------------------------
        tcoll = d1class.get_par("tcoll_gw")
        if not np.isnan(tcoll) and t >= tcoll:
            print(tcoll, t)
            v_n = "lapse"
            mask = "z>0.15"
            data_arr = d3class.get_data(it, rl, "xz", v_n)
            x_arr = d3class.get_data(it, rl, "xz", "x")
            z_arr = d3class.get_data(it, rl, "xz", "z")
            lapse_dic_xz = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                            'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                            'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                            'cbar': {},
                            'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                            'xmin': xmin, 'xmax': xmax, 'ymin': zmin, 'ymax': zmax, 'vmin': 0., 'vmax': 0.15,
                            'fill_vmin': False,  # fills the x < vmin with vmin
                            'xscale': None, 'yscale': None,
                            'mask': mask, 'cmap': 'Greys', 'norm': None,
                            'fancyticks': True,
                            'minorticks': True,
                            'title': {},#,{"text": r'$t-t_{merg}:$' + r'${:.1f}$ [ms]'.format((t - tmerg) * 1e3),
                                      #'fontsize': 14},
                            'sharey': False,
                            'sharex': True,  # removes angular citkscitks
                            'fontsize': 14,
                            'labelsize': 14
                            }
            #
            data_arr = d3class.get_data(it, rl, "xy", v_n)
            # print(data_arr.min(), data_arr.max()); exit(1)
            x_arr = d3class.get_data(it, rl, "xy", "x")
            y_arr = d3class.get_data(it, rl, "xy", "y")
            lapse_dic_xy = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                            'xarr': x_arr, "yarr": y_arr, "zarr": data_arr,
                            'position': (2, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                            'cbar': {},
                            'v_n_x': 'x', 'v_n_y': 'y', 'v_n': v_n,
                            'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'vmin': 0, 'vmax': 0.15,
                            'fill_vmin': False,  # fills the x < vmin with vmin
                            'xscale': None, 'yscale': None,
                            'mask': mask, 'cmap': 'Greys', 'norm': None,
                            'fancyticks': True,
                            'minorticks': True,
                            'title': {},
                            'sharey': False,
                            'sharex': False,  # removes angular citkscitks
                            'fontsize': 14,
                            'labelsize': 14
                            }
            #
            # if plot_x_i == 1:
            #     rho_dic_xy['cbar'] = {'location': 'bottom -.05 .00', 'label': r'$\rho$ [GEO]',  # 'fmt': '%.1e',
            #                           'labelsize': 14,
            #                           'fontsize': 14}
            if plot_x_i > 1:
                lapse_dic_xz['sharey'] = True
                lapse_dic_xy['sharey'] = True

            o_plot.set_plot_dics.append(lapse_dic_xz)
            o_plot.set_plot_dics.append(lapse_dic_xy)


        plot_x_i += 1




    o_plot.main()

    exit(0)
#
def plot_total_fluxes_sims_disk_hist():
    #
    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + 'all2/'
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (15.0, 3.0)  # <->, |]
    o_plot.gen_set["figname"] = "disk_hists_dd2_lk.png"
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.3
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []
    averages = {}
    #
    # LS220_M13641364_M0_SR tbh = 29.5 ms
    # LS220_M13641364_M0_LK_SR_restart tbh 25.6 ms
    #
    simlist = ["DD2_M13641364_M0_SR_R04", "DD2_M13641364_M0_LK_SR_R04", "LS220_M13641364_M0_SR", "LS220_M13641364_M0_LK_SR_restart",
               "LS220_M13641364_M0_SR", "LS220_M13641364_M0_LK_SR_restart"]
    itlist = [2116290, 2058718, 737280, 696320,
              540672, 499712] # 540672 540672
    lbls = [sim.replace('_', '\_') for sim in simlist]
    colors=["blue", "blue", "red", "red",
            "green", "green"]
    alphas=[1., 1., 1., 1.,
            1., 1.]
    lss   =["-", ":", "-", ":",
            "-", ":"]
    lws =  [0.8, 0.5, 0.8, 0.5,
            0.8, 0.5]
    #
    v_ns = ["Ye", "theta", "entr", "r", "temp", "press", "rho"]
    i_x_plot = 1
    for v_n in v_ns:
        for sim, it, lbl, alpha, color, ls, lw in zip(simlist, itlist, lbls, alphas, colors, lss, lws):
            #
            d3_corr = LOAD_RES_CORR(sim)
            d1class = ADD_METHODS_ALL_PAR(sim)

            # for it, in itlist:
            fpath = Paths.ppr_sims + sim + "/profiles/" + str(it) + "/" + "hist_{}.dat".format(v_n)
            assert int(it) in d3_corr.list_iterations
            #
            t = d3_corr.get_time(it)
            tmerg = d1class.get_par("tmerg")
            time = t - tmerg
            #
            if not os.path.isfile(fpath):
                raise IOError("file not found: {}".format(fpath))
            #
            data = np.loadtxt(fpath, unpack=False)
            #
            default_dic = {
                'task': 'hist1d', 'ptype': 'cartesian',
                'position': (1, i_x_plot),
                'data': data, 'normalize': True,
                'v_n_x': 'var', 'v_n_y': 'mass',
                'color': color, 'ls': ls, 'lw': lw, 'ds': 'steps', 'alpha': alpha,
                'ymin': 1e-4, 'ymax': 1e-1,
                'xlabel': None, 'ylabel': "mass",
                'label': lbl + " {:.1f}ms".format(time * 1e3), 'yscale': 'log',
                'fancyticks': True, 'minorticks': True,
                'fontsize': 14,
                'labelsize': 14,
                'legend': {},  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
                'sharex': False,
                'sharey': False,
            }
            #
            if v_n == "r":
                default_dic['v_n_x'] = 'r'
                default_dic['xlabel'] = 'cylindrical radius'
                default_dic['xmin'] = 10.
                default_dic['xmax'] = 95.
            elif v_n == "theta":
                default_dic['v_n_x'] = 'theta'
                default_dic['xlabel'] = r'$\theta$'
                default_dic['xmin'] = 0
                default_dic['xmax'] = 85.
            elif v_n == "entr":
                default_dic['v_n_x'] = 'entropy'
                default_dic['xlabel'] = "entropy"
                default_dic['xmin'] = 0.
                default_dic['xmax'] = 30.
            elif v_n == "Ye":
                default_dic['v_n_x'] = 'Ye'
                default_dic['xlabel'] = 'Ye'
                default_dic['xmin'] = 0.05
                default_dic['xmax'] = 0.45
            elif v_n == "temp":
                default_dic['v_n_x'] = "temp"
                default_dic["xlabel"] = "temp"
                default_dic['xmin'] = 1.e-1
                default_dic['xmax'] = 8.e1
                default_dic['xscale'] = "log"
            elif v_n == "rho":
                default_dic['v_n_x'] = "rho"
                default_dic["xlabel"] = "rho"
                default_dic['xmin'] = 5.e-10
                default_dic['xmax'] = 2.e-5
                default_dic['xscale'] = "log"
            elif v_n == "dens_unb_bern":
                default_dic['v_n_x'] = "temp"
                default_dic["xlabel"] = "temp"
                default_dic['xmin'] = 5.e-11
                default_dic['xmax'] = 5.e-5
                default_dic['xscale'] = "log"
            elif v_n == "press":
                default_dic['v_n_x'] = "press"
                default_dic["xlabel"] = "press"
                default_dic['xmin'] = 1e-13
                default_dic['xmax'] = 1e-5
                default_dic['xscale'] = "log"
            else:
                raise NameError("hist v_n:{} is not recognized".format(v_n))
            #
            if v_n != v_ns[0]:
                default_dic["sharey"] = True
            if v_n == v_ns[-1] and sim == simlist[-1]:
                default_dic['legend'] = {'loc': 'upper right',
                                         'bbox_to_anchor': (1.,1.2),
                                         'ncol': 3, "fontsize": 9,
                                         "framealpha": 0., "borderaxespad": 0., "shadow": False}  #

            o_plot.set_plot_dics.append(default_dic)

        i_x_plot += 1


    o_plot.main()




    exit(1)
#
''' ------------- '''

def tmp_plot_disk_mass_evol_SR():
    # 11
    sims = ["SFHo_M10651772_M0_LK_SR", "SLy4_M10651772_M0_LK_SR", "SLy4_M10201856_M0_LK_SR"]
    #
    colors = ["blue", "red", "green"]
    #
    lss=["-", "-", "-"]
    #
    lws = [1., 1., 1.]
    alphas=[1., 1., 1.]
    #
    # ----

    from scipy import interpolate

    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + "all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = "tmp_disk_mass_evol_SR.png"
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.3
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []

    for sim, color, ls, lw, alpha in zip(sims, colors, lss, lws, alphas):
        print("{}".format(sim))
        o_data = ADD_METHODS_ALL_PAR(sim)
        data = o_data.get_disk_mass()
        tmerg = o_data.get_par("tmerg")
        tarr = (data[:, 0] - tmerg) * 1e3
        marr = data[:, 1]

        if sim == "DD2_M13641364_M0_LK_SR_R04":
            tarr = tarr[3:] # 3ms, 6ms, 51ms.... Removing initial profiles
            marr = marr[3:] #
        #
        tcoll = o_data.get_par("tcoll_gw")
        if not np.isnan(tcoll) and tcoll < tarr[-1]:
            tcoll = (tcoll - tmerg) * 1e3
            print(tcoll, tarr[0])
            mcoll = interpolate.interp1d(tarr,marr,kind="linear")(tcoll)
            tcoll_dic = {
                'task': 'line', 'ptype': 'cartesian',
                'position': (1, 1),
                'xarr': [tcoll], 'yarr': [mcoll],
                'v_n_x': "time", 'v_n_y': "mass",
                'color': color, 'marker': "x", 'ms': 5., 'alpha': alpha,
                'xmin': -10, 'xmax': 100, 'ymin': 0, 'ymax': .3,
                'xlabel': Labels.labels("t-tmerg"), 'ylabel': Labels.labels("diskmass"),
                'label': None, 'yscale': 'linear',
                'fancyticks': True, 'minorticks': True,
                'fontsize': 14,
                'labelsize': 14,
                'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
            }
            o_plot.set_plot_dics.append(tcoll_dic)
        #
        plot_dic = {
            'task': 'line', 'ptype': 'cartesian',
            'position': (1, 1),
            'xarr': tarr, 'yarr': marr,
            'v_n_x': "time", 'v_n_y': "mass",
            'color': color, 'ls': ls, 'lw': 0.8, 'ds': 'steps', 'alpha': 1.0,
            'xmin': -10, 'xmax': 30, 'ymin': 0, 'ymax': .25,
            'xlabel': Labels.labels("t-tmerg"), 'ylabel': Labels.labels("diskmass"),
            'label': str(sim).replace('_', '\_'), 'yscale': 'linear',
            'fancyticks': True, 'minorticks': True,
            'fontsize': 14,
            'labelsize': 14,
            'legend': {'bbox_to_anchor':(1.1,1.05),
                'loc': 'lower right', 'ncol': 2, 'fontsize': 8}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
        }
        if sim == sims[-1]:
            plot_dic['legend'] = {'bbox_to_anchor':(1.1,1.05),
                'loc': 'lower right', 'ncol': 2, 'fontsize': 10}
        o_plot.set_plot_dics.append(plot_dic)

        print(sim)
        print(tarr)
        print(marr)
        print('----')

    o_plot.main()
    exit(1)
#
def plot_disk_mass_evol_SR():
    # 11
    sims = ["DD2_M13641364_M0_LK_SR_R04", "BLh_M13641364_M0_LK_SR"] + \
           ["DD2_M15091235_M0_LK_SR", "LS220_M14691268_M0_LK_SR"] + \
           ["DD2_M13641364_M0_SR", "LS220_M13641364_M0_SR", "SFHo_M13641364_M0_SR", "SLy4_M13641364_M0_SR"] + \
           ["DD2_M14971245_M0_SR", "SFHo_M14521283_M0_SR", "SLy4_M14521283_M0_SR"]
    #
    colors = ["blue", "black"] + \
           ["blue", "red"] + \
           ["blue", "red", "green", "orange"] + \
           ["blue", "green", "orange"]
    #
    lss=["-", "-"] + \
        ["--", "--"] + \
        [":", ":", ":", ":"] + \
        ["-.", "-."]
    #
    lws = [1., 1.] + \
        [1., 1.] + \
        [1., 1., 1., 1.] + \
        [1., 1.]
    alphas=[1., 1.] + \
        [1., 1.] + \
        [1., 1., 1., 1.] + \
        [1., 1.]
    #
    # ----

    from scipy import interpolate

    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + "all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = "disk_mass_evol_SR.png"
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.3
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []

    for sim, color, ls, lw, alpha in zip(sims, colors, lss, lws, alphas):
        print("{}".format(sim))
        o_data = ADD_METHODS_ALL_PAR(sim)
        data = o_data.get_disk_mass()
        tmerg = o_data.get_par("tmerg")
        tarr = (data[:, 0] - tmerg) * 1e3
        marr = data[:, 1]

        if sim == "DD2_M13641364_M0_LK_SR_R04":
            tarr = tarr[3:] # 3ms, 6ms, 51ms.... Removing initial profiles
            marr = marr[3:] #
        #
        tcoll = o_data.get_par("tcoll_gw")
        if not np.isnan(tcoll) and tcoll < tarr[-1]:
            tcoll = (tcoll - tmerg) * 1e3
            print(tcoll, tarr[0])
            mcoll = interpolate.interp1d(tarr,marr,kind="linear")(tcoll)
            tcoll_dic = {
                'task': 'line', 'ptype': 'cartesian',
                'position': (1, 1),
                'xarr': [tcoll], 'yarr': [mcoll],
                'v_n_x': "time", 'v_n_y': "mass",
                'color': color, 'marker': "x", 'ms': 5., 'alpha': alpha,
                'xmin': -10, 'xmax': 100, 'ymin': 0, 'ymax': .3,
                'xlabel': Labels.labels("t-tmerg"), 'ylabel': Labels.labels("diskmass"),
                'label': None, 'yscale': 'linear',
                'fancyticks': True, 'minorticks': True,
                'fontsize': 14,
                'labelsize': 14,
                'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
            }
            o_plot.set_plot_dics.append(tcoll_dic)
        #
        plot_dic = {
            'task': 'line', 'ptype': 'cartesian',
            'position': (1, 1),
            'xarr': tarr, 'yarr': marr,
            'v_n_x': "time", 'v_n_y': "mass",
            'color': color, 'ls': ls, 'lw': 0.8, 'ds': 'steps', 'alpha': 1.0,
            'xmin': -10, 'xmax': 100, 'ymin': 0, 'ymax': .35,
            'xlabel': Labels.labels("t-tmerg"), 'ylabel': Labels.labels("diskmass"),
            'label': str(sim).replace('_', '\_'), 'yscale': 'linear',
            'fancyticks': True, 'minorticks': True,
            'fontsize': 14,
            'labelsize': 14,
            'legend': {'bbox_to_anchor':(1.1,1.05),
                'loc': 'lower right', 'ncol': 2, 'fontsize': 8}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
        }
        if sim == sims[-1]:
            plot_dic['legend'] = {'bbox_to_anchor':(1.1,1.05),
                'loc': 'lower right', 'ncol': 2, 'fontsize': 8}
        o_plot.set_plot_dics.append(plot_dic)

    o_plot.main()
    exit(1)
#
def plot_disk_mass_evol_LR():

    sims = ["BLh_M16351146_M0_LK_LR", "BLh_M13641364_M0_LK_LR", "SLy4_M10651772_M0_LK_LR",  "SFHo_M10651772_M0_LK_LR", "SFHo_M16351146_M0_LK_LR",
            "LS220_M10651772_M0_LK_LR", "LS220_M16351146_M0_LK_LR", "DD2_M16351146_M0_LK_LR"] + \
           ["DD2_M13641364_M0_LR", "LS220_M13641364_M0_LR"] + \
           ["DD2_M14971246_M0_LR", "DD2_M14861254_M0_LR", "DD2_M14351298_M0_LR", "DD2_M14321300_M0_LR"]
    #
    colors = ["black", "gray", "orange", "pink", "olive", "red", "purple", "blue"] + \
            ["blue", "red"] + \
            ["green", "blue", "lightblue", "cyan"]
    #
    lss = ["-", "-", "-", "-", "-", "-", "-", "-"] +\
          ['--', '--', '--'] + \
          [":", ":", ":", ":"]
    #
    lws = [1., 1., 1., 1., 1., 1., 1., 1.] + \
          [1., 1.] + \
          [1., 1., 1., 1.]
    #
    alphas = [1., 1., 1., 1., 1., 1., 1., 1.] + \
          [1., 1.] + \
          [1., 1., 1., 1.]


    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + "all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = "disk_mass_evol_LR.png"
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.3
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []

    from scipy import interpolate

    for sim, color, ls, lw, alpha in zip(sims, colors, lss, lws, alphas):
        print("{}".format(sim))
        o_data = ADD_METHODS_ALL_PAR(sim)
        data = o_data.get_disk_mass()
        assert len(data) > 0
        tmerg = o_data.get_par("tmerg")
        tarr = (data[:, 0] - tmerg) * 1e3
        marr = data[:, 1]

        if sim == "DD2_M13641364_M0_LK_SR_R04":
            tarr = tarr[3:]  # 3ms, 6ms, 51ms.... Removing initial profiles
            marr = marr[3:]  #
        #
        tcoll = o_data.get_par("tcoll_gw")
        if not np.isnan(tcoll) and tcoll < tarr[-1]:
            tcoll = (tcoll - tmerg) * 1e3
            print(tcoll, tarr[0])
            mcoll = interpolate.interp1d(tarr, marr, kind="linear")(tcoll)
            tcoll_dic = {
                'task': 'line', 'ptype': 'cartesian',
                'position': (1, 1),
                'xarr': [tcoll], 'yarr': [mcoll],
                'v_n_x': "time", 'v_n_y': "mass",
                'color': color, 'marker': "x", 'ms': 5., 'alpha': alpha,
                'xmin': -10, 'xmax': 40, 'ymin': 0, 'ymax': .3,
                'xlabel': Labels.labels("t-tmerg"), 'ylabel': Labels.labels("diskmass"),
                'label': None, 'yscale': 'linear',
                'fancyticks': True, 'minorticks': True,
                'fontsize': 14,
                'labelsize': 14,
                'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
            }
            o_plot.set_plot_dics.append(tcoll_dic)
        #
        plot_dic = {
            'task': 'line', 'ptype': 'cartesian',
            'position': (1, 1),
            'xarr': tarr, 'yarr': marr,
            'v_n_x': "time", 'v_n_y': "mass",
            'color': color, 'ls': ls, 'lw': 0.8, 'ds': 'steps', 'alpha': 1.0,
            'xmin': -10, 'xmax': 40, 'ymin': 0, 'ymax': .35,
            'xlabel': Labels.labels("t-tmerg"), 'ylabel': Labels.labels("diskmass"),
            'label': str(sim).replace('_', '\_'), 'yscale': 'linear',
            'fancyticks': True, 'minorticks': True,
            'fontsize': 14,
            'labelsize': 14,
            'legend': {'bbox_to_anchor': (1.1, 1.05),
                       'loc': 'lower right', 'ncol': 2, 'fontsize': 8}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
        }
        if sim == sims[-1]:
            plot_dic['legend'] = {'bbox_to_anchor': (1.1, 1.05),
                                  'loc': 'lower right', 'ncol': 2, 'fontsize': 8}
        o_plot.set_plot_dics.append(plot_dic)


    o_plot.main()
    exit(1)
#
def plot_total_fluxes_sims_disk_hist_last():
    #
    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + 'all2/'
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (15.0, 3.0)  # <->, |]
    o_plot.gen_set["figname"] = "disk_hists_last_it.png"
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.3
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []
    averages = {}
    #
    simlist = ["DD2_M13641364_M0_LK_SR_R04", "BLh_M13641364_M0_LK_SR", "LS220_M14691268_M0_LK_SR", "SLy4_M13641364_M0_SR"]
    itlist = [2611022, 1974272, 1949696, 737280]
    lbls = [sim.replace('_', '\_') for sim in simlist]
    colors=["blue", "orange", "red", "green",]
    alphas=[1., 1., 1., 1.,]
    lss   =["-","-", "-", "-"]
    lws =  [0.8, 0.8, 0.8, 0.8]
    #
    v_ns = ["Ye", "theta", "entr", "r", "temp", "press", "rho"]
    i_x_plot = 1
    for v_n in v_ns:
        for sim, it, lbl, alpha, color, ls, lw in zip(simlist, itlist, lbls, alphas, colors, lss, lws):
            #
            d3_corr = LOAD_RES_CORR(sim)
            assert int(it) in d3_corr.list_iterations
            time = d3_corr.get_time(it)
            #
            for it, time in zip([it], [time]):
                fpath = Paths.ppr_sims + sim + "/profiles/" + str(it) + "/" + "hist_{}.dat".format(v_n)
                #
                if not os.path.isfile(fpath):
                    raise IOError("file not found: {}".format(fpath))
                #
                data = np.loadtxt(fpath, unpack=False)
                #
                default_dic = {
                    'task': 'hist1d', 'ptype': 'cartesian',
                    'position': (1, i_x_plot),
                    'data': data, 'normalize': True,
                    'v_n_x': 'var', 'v_n_y': 'mass',
                    'color': color, 'ls': ls, 'lw': lw, 'ds': 'steps', 'alpha': alpha,
                    'ymin': 1e-4, 'ymax': 1e-1,
                    'xlabel': None, 'ylabel': "mass",
                    'label': lbl, 'yscale': 'log',
                    'fancyticks': True, 'minorticks': True,
                    'fontsize': 14,
                    'labelsize': 14,
                    'legend': {}, # 'loc': 'best', 'ncol': 2, 'fontsize': 18
                    'sharex': False,
                    'sharey': False,
                }
                #
                if v_n == "r":
                    default_dic['v_n_x'] = 'r'
                    default_dic['xlabel'] = 'cylindrical radius'
                    default_dic['xmin'] = 10.
                    default_dic['xmax'] = 95.
                elif v_n == "theta":
                    default_dic['v_n_x'] = 'theta'
                    default_dic['xlabel'] = r'$\theta$'
                    default_dic['xmin'] = 0
                    default_dic['xmax'] = 85.
                elif v_n == "entr":
                    default_dic['v_n_x'] = 'entropy'
                    default_dic['xlabel'] = "entropy"
                    default_dic['xmin'] = 0.
                    default_dic['xmax'] = 45.
                elif v_n == "Ye":
                    default_dic['v_n_x'] = 'Ye'
                    default_dic['xlabel'] = 'Ye'
                    default_dic['xmin'] = 0.05
                    default_dic['xmax'] = 0.45
                elif v_n == "temp":
                    default_dic['v_n_x'] = "temp"
                    default_dic["xlabel"] = "temp"
                    default_dic['xmin'] = 1.e-1
                    default_dic['xmax'] = 8.e1
                    default_dic['xscale'] = "log"
                elif v_n == "rho":
                    default_dic['v_n_x'] = "rho"
                    default_dic["xlabel"] = "rho"
                    default_dic['xmin'] = 5.e-10
                    default_dic['xmax'] = 2.e-5
                    default_dic['xscale'] = "log"
                elif v_n == "dens_unb_bern":
                    default_dic['v_n_x'] = "temp"
                    default_dic["xlabel"] = "temp"
                    default_dic['xmin'] = 5.e-11
                    default_dic['xmax'] = 5.e-5
                    default_dic['xscale'] = "log"
                elif v_n == "press":
                    default_dic['v_n_x'] = "press"
                    default_dic["xlabel"] = "press"
                    default_dic['xmin'] = 1e-13
                    default_dic['xmax'] = 1e-5
                    default_dic['xscale'] = "log"
                else: raise NameError("hist v_n:{} is not recognized".format(v_n))
                #
                if v_n != v_ns[0]:
                    default_dic["sharey"] = True
                if v_n == v_ns[1] and sim == simlist[-1]:
                    default_dic['legend'] = {'loc': 'upper right', 'ncol': 1, "fontsize": 8,
                                             "framealpha":0.,"borderaxespad":0.,"shadow":False}  #

                o_plot.set_plot_dics.append(default_dic)

        i_x_plot += 1


    o_plot.main()




    exit(1)
#
def plot_2ejecta_1disk_timehists():
    # columns
    sims = ["DD2_M13641364_M0_SR", "BLh_M13641364_M0_LK_SR", "BLh_M11461635_M0_LK_SR", "LS220_M13641364_M0_SR",
             "SLy4_M13641364_M0_SR"]#, "DD2_M15091235_M0_LK_SR"] # , "DD2_M15091235_M0_LK_SR"

    # rows
    #
    v_ns = ["vel_inf", "Y_e", "theta", "temperature"]
    masks2 = ["bern_geoend" for i in v_ns]
    masks1 = ["geo" for i in v_ns]
    v_ns_diks = ["Ye", "velz", "theta", "r", "temp", "press"]
    det = 0
    norm_to_m = 0
    _fpath = "slices/" + "rho_modes.h5"
    #
    cmap_ejecta = "cubehelix_r"
    cmap_disk = 'RdYlBu_r'
    #
    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + "all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (3.*len(sims), 14.0)  # <->, |]
    o_plot.gen_set["figname"] = "timecorr_ej_disk2.png"
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
                'cmap': cmap_ejecta,#'inferno_r',
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
                'cmap': cmap_ejecta,
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
                           'mask': None, 'cmap': cmap_disk, 'norm': "log", # 'inferno_r',
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
                elif v_n == "press":
                    def_dic['yscale'] = "log"
                    def_dic['ymin'] = 5.e-8
                    def_dic['ymax'] = 2.e-6
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

"""======================================================| REMNANT |================================================="""

def plot_desity_modes2():
    #
    _fpath = "slices/" + "rho_modes.h5" #"profiles/" + "density_modes_lap15.h5"
    sims = ["DD2_M13641364_M0_LK_SR_R04", "DD2_M15091235_M0_LK_SR", "BLh_M13641364_M0_LK_SR", "BLh_M11461635_M0_LK_SR"]
    lbls = ["DD2 q=1 LK" , "DD2 q=1.2 LK", "BLh q=1 LK" , "BLh q=1.4 LK"]
    ls_m1 = ["-", "-", "-", "-"]
    ls_m2 = [":", ":", ":", ":"]
    colors = ["green", "blue", "orange", "red"]
    lws_m1 = [.8, .8, .8, .8]
    lws_m2 = [.5, .5, .5, .5]
    alphas = [1., 1., 1., 1.]
    #
    norm_to_m = 0
    # Load and parse the data
    data = {}
    for sim in sims:
        data[sim] = {}
        o_dm = LOAD_DENSITY_MODES(sim)
        o_dm.gen_set['fname'] = Paths.ppr_sims + sim + "/" + _fpath
        o_par = ADD_METHODS_ALL_PAR(sim)
        # -- m=1
        mags1 = o_dm.get_data(1, "int_phi_r")
        mags1 = np.abs(mags1)
        if norm_to_m != None:
            # print('Normalizing')
            norm_int_phi_r1d = o_dm.get_data(norm_to_m, 'int_phi_r')
            # print(norm_int_phi_r1d); exit(1)
            mags1 = mags1 / abs(norm_int_phi_r1d)[0]
        #
        tmerg = o_par.get_par("tmerg")
        times = o_dm.get_grid("times")
        times = (times - tmerg) * 1e3  # ms
        # -- m=2
        mags2 = o_dm.get_data(2, "int_phi_r")
        mags2 = np.abs(mags2)
        if norm_to_m != None:
            # print('Normalizing')
            norm_int_phi_r1d = o_dm.get_data(norm_to_m, 'int_phi_r')
            # print(norm_int_phi_r1d); exit(1)
            mags2 = mags2 / abs(norm_int_phi_r1d)[0]
        #
        data[sim]["times"] = np.array(times)
        data[sim]["m1"] = np.array(mags1)
        data[sim]["m2"] = np.array(mags2)
    #
    Printcolor.green("Density modes Data is collected.")
    # interpolate - smooth the data
    from scipy import interpolate
    for sim in sims:
        times = data[sim]["times"]
        mags1 = data[sim]["m1"]
        mags2 = data[sim]["m2"]
        intmags1 = interpolate.interp1d(times, mags1, kind="cubic")(times[::20])
        intmags2 = interpolate.interp1d(times, mags2, kind="cubic")(times[::20])
        data[sim]["int_times"] = times[::20]
        data[sim]["int_m1"] = intmags1
        data[sim]["int_m2"] = intmags2
    # plot the data
    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + "all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (9.0, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = "density_modes.png"
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = False
    o_plot.gen_set["subplots_adjust_h"] = 0.2
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []
    #

    #

    for sim, lbl, ls1, ls2, color, lw1, lw2, alpha in zip(sims, lbls, ls_m1, ls_m2, colors, lws_m1, lws_m2, alphas):
        # for labels
        densmode_m1 = {
            'task': 'line', 'ptype': 'cartesian',
            'xarr': data[sim]["int_times"], 'yarr': data[sim]["int_m1"],
            'position': (1, 1),
            'v_n_x': 'times', 'v_n_y': 'int_phi_r abs',
            'ls': ls1, 'color': 'gray', 'lw': lw1, 'ds': 'default', 'alpha': alpha,
            'label': None, 'ylabel': None, 'xlabel': Labels.labels("t-tmerg"),
            'xmin': -10, 'xmax': 110, 'ymin': 1e-4, 'ymax': 5e-1,
            'xscale': None, 'yscale': 'log', 'legend': {},
            'fancyticks': True, 'minorticks': True,
            'fontsize': 14,
            'labelsize': 14
        }
        # for labels
        densmode_m2 = {
            'task': 'line', 'ptype': 'cartesian',
            'xarr': data[sim]["int_times"], 'yarr': data[sim]["int_m2"],
            'position': (1, 1),
            'v_n_x': 'times', 'v_n_y': 'int_phi_r abs',
            'ls': ls2, 'color': 'gray', 'lw': lw2, 'ds': 'default', 'alpha': alpha,
            'label': None, 'ylabel': r'$C_m/C_0$', 'xlabel': Labels.labels("t-tmerg"),
            'xmin': 0, 'xmax': 110, 'ymin': 1e-4, 'ymax': 5e-1,
            'xscale': None, 'yscale': 'log',
            'fancyticks': True, 'minorticks': True,
            'legend': {},
            'fontsize': 14,
            'labelsize': 14,
            'title': {'text': "Density Mode Evolution", 'fontsize': 14}
            # 'sharex': True
        }
        # for labels
        if sim == sims[0]:
            densmode_m1['label'] = r"$m=1$"
            densmode_m2['label'] = r"$m=2$"
            o_plot.set_plot_dics.append(densmode_m1)
            o_plot.set_plot_dics.append(densmode_m2)

        # for actual plot
        densmode_m1 = {
            'task': 'line', 'ptype': 'cartesian',
            'xarr': data[sim]["int_times"], 'yarr': data[sim]["int_m1"],
            'position': (1, 1),
            'v_n_x': 'times', 'v_n_y': 'int_phi_r abs',
            'ls': ls1, 'color': color, 'lw': lw1, 'ds': 'default', 'alpha': alpha,
            'label': lbl, 'ylabel': None, 'xlabel': Labels.labels("t-tmerg"),
            'xmin': -10, 'xmax': 110, 'ymin': 1e-4, 'ymax': 5e-1,
            'xscale': None, 'yscale': 'log',
            'fancyticks': True, 'minorticks': True,
            'legend': {'loc': 'upper right', 'ncol': 3, 'fontsize': 12, 'shadow': False, 'framealpha': 0.5,
                       'borderaxespad': 0.0},
            'fontsize': 14,
            'labelsize': 14
        }
        densmode_m2 = {
            'task': 'line', 'ptype': 'cartesian',
            'xarr': data[sim]["int_times"], 'yarr': data[sim]["int_m2"],
            'position': (1, 1),
            'v_n_x': 'times', 'v_n_y': 'int_phi_r abs',
            'ls': ls2, 'color': color, 'lw': lw2, 'ds': 'default', 'alpha': alpha,
            'label': None, 'ylabel': r'$C_m/C_0$', 'xlabel': Labels.labels("t-tmerg"),
            'xmin': 0, 'xmax': 110, 'ymin': 1e-3, 'ymax': 1e-1,
            'xscale': None, 'yscale': 'log',
            'fancyticks': True, 'minorticks': True,
            # 'legend2': {'loc': 'lower right', 'ncol': 1, 'fontsize': 12, 'shadow':False, 'framealpha': 1.0, 'borderaxespad':0.0},
            'fontsize': 14,
            'labelsize': 14,
            'title': {'text': "Density Mode Evolution", 'fontsize': 14}
            # 'sharex': True
        }
        #
        o_plot.set_plot_dics.append(densmode_m1)
        o_plot.set_plot_dics.append(densmode_m2)
    o_plot.main()
    exit(1)

def plot_center_of_mass_movement():

    _fpath = "slices/" + "rho_modes.h5" #"profiles/" + "density_modes_lap15.h5"
    sims = ["DD2_M13641364_M0_LK_SR_R04", "DD2_M15091235_M0_LK_SR", "BLh_M13641364_M0_LK_SR", "BLh_M11461635_M0_LK_SR"]
    lbls = ["DD2 q=1 LK" , "DD2 q=1.2 LK", "BLh q=1 LK" , "BLh q=1.4 LK"]
    lss = ["-", "-", "-", "-"]
    colors = ["green", "blue", "orange", "red"]
    lws = [.8, .8, .8, .8]
    alphas = [1., 1., 1., 1.]
    #
    norm_to_m = 0
    # Load and parse the data
    data = {}
    for sim in sims:
        data[sim] = {}
        o_dm = LOAD_DENSITY_MODES(sim)
        o_dm.gen_set['fname'] = Paths.ppr_sims + sim + "/" + _fpath
        o_par = ADD_METHODS_ALL_PAR(sim)
        # -- m=1
        x = o_dm.get_grid("xc")
        y = o_dm.get_grid("yc")
        # print("x:{}".format(x))
        # print("y:{}".format(y))
        times = o_dm.get_grid("times")
        #
        data[sim]["times"] = np.array(times)
        data[sim]["xc"] = np.array(x)
        data[sim]["yc"] = np.array(y)

    #
    Printcolor.green("Density modes Data is collected.")
    # interpolate - smooth the data
    from scipy import interpolate
    for sim in sims:
        times = data[sim]["times"]
        mags1 = data[sim]["xc"]
        mags2 = data[sim]["yc"]
        intmags1 = interpolate.interp1d(times, mags1, kind="cubic")(times[::20])
        intmags2 = interpolate.interp1d(times, mags2, kind="cubic")(times[::20])
        data[sim]["int_times"] = times[::20]
        data[sim]["int_xc"] = intmags1
        data[sim]["int_yc"] = intmags2
    # plot the data
    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + "all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = "center_of_mass.png"
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = False
    o_plot.gen_set["subplots_adjust_h"] = 0.2
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []
    #

    #

    for sim, lbl, ls, color, lw, alpha in zip(sims, lbls, lss, colors, lws, alphas):

        # for actual plot
        densmode_m1 = {
            'task': 'line', 'ptype': 'cartesian',
            'xarr': data[sim]["xc"], 'yarr': data[sim]["yc"],
            'position': (1, 1),
            'v_n_x': 'times', 'v_n_y': 'int_phi_r abs',
            'ls': ls, 'color': color, 'lw': lw, 'ds': 'default', 'alpha': alpha,
            'label': lbl, 'ylabel': r'$y$ [GEO]', 'xlabel': r"$x$ [GEO]",
            'xmin': -10, 'xmax': 10, 'ymin': -10, 'ymax': 10,
            'xscale': None, 'yscale': None,
            'fancyticks': True, 'minorticks': True,
            'legend': {'loc': 'upper right', 'ncol': 2, 'fontsize': 10, 'shadow': False, 'framealpha': 0.5,
                       'borderaxespad': 0.0},
            'fontsize': 14,
            'labelsize': 14,
            'title': {'text': "Center of mass", 'fontsize': 14}
        }
        #
        o_plot.set_plot_dics.append(densmode_m1)
    o_plot.main()
    exit(1)

def plot_den_unb__vel_z_sly4_evol():

    # tmp = d3class.get_data(688128, 3, "xy", "ang_mom_flux")
    # print(tmp.min(), tmp.max())
    # print(tmp)
    # exit(1) # dens_unb_geo

    """ --- --- --- """


    '''sly4 '''
    simlist = ["SLy4_M13641364_M0_SR", "SLy4_M13641364_M0_SR", "SLy4_M13641364_M0_SR", "SLy4_M13641364_M0_SR"]
    # itlist = [434176, 475136, 516096, 565248]
    # itlist = [606208, 647168, 696320, 737280]
    # itlist = [434176, 516096, 647168, 737280]
    ''' ls220 '''
    simlist = ["LS220_M14691268_M0_LK_SR", "LS220_M14691268_M0_LK_SR", "LS220_M14691268_M0_LK_SR"]#, "LS220_M14691268_M0_LK_SR"]
    itlist = [1515520, 1728512, 1949696]#, 2162688]
    ''' dd2 '''
    simlist = ["DD2_M13641364_M0_LK_SR_R04", "DD2_M13641364_M0_LK_SR_R04", "DD2_M13641364_M0_LK_SR_R04"]#, "DD2_M13641364_M0_LK_SR_R04"]
    itlist = [1111116,1741554,2213326]#,2611022]
    #
    simlist = ["DD2_M13641364_M0_LK_SR_R04", "BLh_M13641364_M0_LK_SR", "LS220_M14691268_M0_LK_SR", "SLy4_M13641364_M0_SR"]
    itlist = [2611022, 1974272, 1949696, 737280]
    #
    simlist = ["BLh_M13641364_M0_LK_SR"]
    itlist = [737280]
    #
    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + 'all2/'
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4*len(simlist), 6.0)  # <->, |] # to match hists with (8.5, 2.7)
    o_plot.gen_set["figname"] = "remnant.png".format(simlist[0])#"DD2_1512_slices.png" # LS_1412_slices
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = -0.35
    o_plot.gen_set["subplots_adjust_w"] = 0.05
    o_plot.set_plot_dics = []
    #
    rl = 6
    #
    o_plot.gen_set["figsize"] = (4.2*len(simlist), 8.0)  # <->, |] # to match hists with (8.5, 2.7)

    plot_x_i = 1
    for sim, it in zip(simlist, itlist):
        print("sim:{} it:{}".format(sim, it))
        d3class = LOAD_PROFILE_XYXZ(sim)
        d1class = ADD_METHODS_ALL_PAR(sim)

        t = d3class.get_time_for_it(it, d1d2d3prof="prof")
        tmerg = d1class.get_par("tmerg")
        xmin, xmax, ymin, ymax, zmin, zmax = UTILS.get_xmin_xmax_ymin_ymax_zmin_zmax(rl)



        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        mask = "x>0"
        #
        v_n = "rho"
        data_arr = d3class.get_data(it, rl, "xz", v_n)
        x_arr = d3class.get_data(it, rl, "xz", "x")
        z_arr = d3class.get_data(it, rl, "xz", "z")
        # print(data_arr); exit(1)

        contour_dic_xz = {
            'task': 'contour',
            'ptype': 'cartesian', 'aspect': 1.,
            'xarr': x_arr, "yarr": z_arr, "zarr": data_arr, 'levels': [1.e13 / 6.176e+17],
            'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
            'colors': ['white'], 'lss': ["-"], 'lws': [1.],
            'v_n_x': 'x', 'v_n_y': 'y', 'v_n': 'rho',
            'xscale': None, 'yscale': None,
            'fancyticks': True,
            'sharey': False,
            'sharex': True,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14}
        o_plot.set_plot_dics.append(contour_dic_xz)

        rho_dic_xz = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                      'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                      'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                      'cbar': {},
                      'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                      'xmin': xmin, 'xmax': xmax, 'ymin': zmin, 'ymax': zmax, 'vmin': 1e-9, 'vmax': 1e-5,
                      'fill_vmin': False,  # fills the x < vmin with vmin
                      'xscale': None, 'yscale': None,
                      'mask': mask, 'cmap': 'Greys', 'norm': "log",
                      'fancyticks': True,
                      'minorticks':True,
                      'title': {"text": sim.replace('_', '\_'), 'fontsize': 12},
                      #'title': {"text": r'$t-t_{merg}:$' + r'${:.1f}$ [ms]'.format((t - tmerg) * 1e3), 'fontsize': 14},
                      'sharey': False,
                      'sharex': True,  # removes angular citkscitks
                      'fontsize': 14,
                      'labelsize': 14
                      }
        #
        data_arr = d3class.get_data(it, rl, "xy", v_n)
        x_arr = d3class.get_data(it, rl, "xy", "x")
        y_arr = d3class.get_data(it, rl, "xy", "y")

        contour_dic_xy = {
            'task': 'contour',
            'ptype': 'cartesian', 'aspect': 1.,
            'xarr': x_arr, "yarr": y_arr, "zarr": data_arr, 'levels': [1.e13 / 6.176e+17],
            'position': (2, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
            'colors': ['white'], 'lss': ["-"], 'lws': [1.],
            'v_n_x': 'x', 'v_n_y': 'y', 'v_n': 'rho',
            'xscale': None, 'yscale': None,
            'fancyticks': True,
            'sharey': False,
            'sharex': True,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14}
        o_plot.set_plot_dics.append(contour_dic_xy)

        rho_dic_xy = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                      'xarr': x_arr, "yarr": y_arr, "zarr": data_arr,
                      'position': (2, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                      'cbar': {},
                      'v_n_x': 'x', 'v_n_y': 'y', 'v_n': v_n,
                      'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'vmin': 1e-9, 'vmax': 1e-5,
                      'fill_vmin': False,  # fills the x < vmin with vmin
                      'xscale': None, 'yscale': None,
                      'mask': mask, 'cmap': 'Greys', 'norm': "log",
                      'fancyticks': True,
                      'minorticks': True,
                      'title': {},
                      'sharey': False,
                      'sharex': False,  # removes angular citkscitks
                      'fontsize': 14,
                      'labelsize': 14
                      }
        #
        if v_n == "rho" and rl == 6:
            rho_dic_xy['vmin'], rho_dic_xy['vmax'] = 1e-6, 1e-3
            rho_dic_xz['vmin'], rho_dic_xz['vmax'] = 1e-6, 1e-3

        if plot_x_i == 1:
            rho_dic_xy['cbar'] = {'location': 'bottom -.05 .00', 'label': r'$\rho$ [GEO]',  # 'fmt': '%.1e',
                          'labelsize': 14,
                          'fontsize': 14}
        if plot_x_i > 1:
            rho_dic_xz['sharey'] = True
            rho_dic_xy['sharey'] = True

        o_plot.set_plot_dics.append(rho_dic_xz)
        o_plot.set_plot_dics.append(rho_dic_xy)

        # ----------------------------------------------------------------------
        v_n = "dens_unb_bern"
        #
        data_arr = d3class.get_data(it, rl, "xz", v_n)
        x_arr = d3class.get_data(it, rl, "xz", "x")
        z_arr = d3class.get_data(it, rl, "xz", "z")
        dunb_dic_xz = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                      'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                      'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                      'cbar': {},
                      'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                      'xmin': xmin, 'xmax': xmax, 'ymin': zmin, 'ymax': zmax, 'vmin': 1e-10, 'vmax': 1e-7,
                      'fill_vmin': False,  # fills the x < vmin with vmin
                      'xscale': None, 'yscale': None,
                      'mask': mask, 'cmap': 'Blues', 'norm': "log",
                      'fancyticks': True,
                       'minorticks': True,
                       'title': {},#{"text": r'$t-t_{merg}:$' + r'${:.1f}$ [ms]'.format((t - tmerg) * 1e3), 'fontsize': 14},
                      'sharex': True,  # removes angular citkscitks
                      'sharey': False,
                      'fontsize': 14,
                      'labelsize': 14
                      }
        #
        data_arr = d3class.get_data(it, rl, "xy", v_n)
        x_arr = d3class.get_data(it, rl, "xy", "x")
        y_arr = d3class.get_data(it, rl, "xy", "y")
        dunb_dic_xy = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                      'xarr': x_arr, "yarr": y_arr, "zarr": data_arr,
                      'position': (2, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                      'cbar': {},
                      'fill_vmin': False,  # fills the x < vmin with vmin
                      'v_n_x': 'x', 'v_n_y': 'y', 'v_n': v_n,
                      'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'vmin': 1e-10, 'vmax': 1e-7,
                      'xscale': None, 'yscale': None,
                      'mask': mask, 'cmap': 'Blues', 'norm': "log",
                      'fancyticks': True,
                       'minorticks': True,
                       'title': {},
                      'sharey': False,
                      'sharex': False,  # removes angular citkscitks
                      'fontsize': 14,
                      'labelsize': 14
                      }
        #
        if plot_x_i == 2:
            dunb_dic_xy['cbar'] = {'location': 'bottom -.05 .00', 'label': r'$D_{\rm{unb}}$ [GEO]',  # 'fmt': '%.1e',
                          'labelsize': 14,
                          'fontsize': 14}
        if plot_x_i > 1:
            dunb_dic_xz['sharey'] = True
            dunb_dic_xy['sharey'] = True


        o_plot.set_plot_dics.append(dunb_dic_xz)
        o_plot.set_plot_dics.append(dunb_dic_xy)

        # ----------------------------------------------------------------------
        mask = "x<0"
        #
        v_n = "Ye"
        cmap = "bwr_r"
        #
        data_arr = d3class.get_data(it, rl, "xz", v_n)
        x_arr = d3class.get_data(it, rl, "xz", "x")
        z_arr = d3class.get_data(it, rl, "xz", "z")
        ye_dic_xz = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                       'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                       'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                       'cbar': {},
                       'fill_vmin': False,  # fills the x < vmin with vmin
                       'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                       'xmin': xmin, 'xmax': xmax, 'ymin': zmin, 'ymax': zmax, 'vmin': 0.05, 'vmax': 0.5,
                       'xscale': None, 'yscale': None,
                       'mask': mask, 'cmap': cmap, 'norm': None,
                       'fancyticks': True,
                       'minorticks': True,
                       'title': {},#{"text": r'$t-t_{merg}:$' + r'${:.1f}$ [ms]'.format((t - tmerg) * 1e3), 'fontsize': 14},
                       'sharey': False,
                       'sharex': True,  # removes angular citkscitks
                       'fontsize': 14,
                       'labelsize': 14
                       }
        #
        data_arr = d3class.get_data(it, rl, "xy", v_n)
        x_arr = d3class.get_data(it, rl, "xy", "x")
        y_arr = d3class.get_data(it, rl, "xy", "y")
        ye_dic_xy = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                       'xarr': x_arr, "yarr": y_arr, "zarr": data_arr,
                       'position': (2, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                       'cbar': {},
                       'fill_vmin': False,  # fills the x < vmin with vmin
                       'v_n_x': 'x', 'v_n_y': 'y', 'v_n': v_n,
                       'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'vmin': 0.01, 'vmax': 0.5,
                       'xscale': None, 'yscale': None,
                       'mask': mask, 'cmap': cmap, 'norm': None,
                       'fancyticks': True,
                       'minorticks': True,
                       'title': {},
                       'sharey': False,
                       'sharex': False,  # removes angular citkscitks
                       'fontsize': 14,
                       'labelsize': 14
                       }
        #
        if plot_x_i == 3:
            ye_dic_xy['cbar'] = {'location': 'bottom -.05 .00', 'label': r'$Y_e$',   'fmt': '%.1f',
                          'labelsize': 14,
                          'fontsize': 14}
        if plot_x_i > 1:
            ye_dic_xz['sharey'] = True
            ye_dic_xy['sharey'] = True

        if v_n == "rho" and rl == 6:
            rho_dic_xy['vmin'], rho_dic_xy['vmax'] = .05, 0.3
            rho_dic_xz['vmin'], rho_dic_xz['vmax'] = .05, 0.3
        o_plot.set_plot_dics.append(ye_dic_xz)
        o_plot.set_plot_dics.append(ye_dic_xy)

        # ----------------------------------------------------------
        tcoll = d1class.get_par("tcoll_gw")
        if not np.isnan(tcoll) and t >= tcoll:
            print(tcoll, t)
            v_n = "lapse"
            mask = "z>0.15"
            data_arr = d3class.get_data(it, rl, "xz", v_n)
            x_arr = d3class.get_data(it, rl, "xz", "x")
            z_arr = d3class.get_data(it, rl, "xz", "z")
            lapse_dic_xz = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                            'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                            'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                            'cbar': {},
                            'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                            'xmin': xmin, 'xmax': xmax, 'ymin': zmin, 'ymax': zmax, 'vmin': 0., 'vmax': 0.15,
                            'fill_vmin': False,  # fills the x < vmin with vmin
                            'xscale': None, 'yscale': None,
                            'mask': mask, 'cmap': 'Greys', 'norm': None,
                            'fancyticks': True,
                            'minorticks': True,
                            'title': {},#,{"text": r'$t-t_{merg}:$' + r'${:.1f}$ [ms]'.format((t - tmerg) * 1e3),
                                      #'fontsize': 14},
                            'sharey': False,
                            'sharex': True,  # removes angular citkscitks
                            'fontsize': 14,
                            'labelsize': 14
                            }
            #
            data_arr = d3class.get_data(it, rl, "xy", v_n)
            # print(data_arr.min(), data_arr.max()); exit(1)
            x_arr = d3class.get_data(it, rl, "xy", "x")
            y_arr = d3class.get_data(it, rl, "xy", "y")
            lapse_dic_xy = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                            'xarr': x_arr, "yarr": y_arr, "zarr": data_arr,
                            'position': (2, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                            'cbar': {},
                            'v_n_x': 'x', 'v_n_y': 'y', 'v_n': v_n,
                            'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'vmin': 0, 'vmax': 0.15,
                            'fill_vmin': False,  # fills the x < vmin with vmin
                            'xscale': None, 'yscale': None,
                            'mask': mask, 'cmap': 'Greys', 'norm': None,
                            'fancyticks': True,
                            'minorticks': True,
                            'title': {},
                            'sharey': False,
                            'sharex': False,  # removes angular citkscitks
                            'fontsize': 14,
                            'labelsize': 14
                            }
            #
            # if plot_x_i == 1:
            #     rho_dic_xy['cbar'] = {'location': 'bottom -.05 .00', 'label': r'$\rho$ [GEO]',  # 'fmt': '%.1e',
            #                           'labelsize': 14,
            #                           'fontsize': 14}
            if plot_x_i > 1:
                lapse_dic_xz['sharey'] = True
                lapse_dic_xy['sharey'] = True

            o_plot.set_plot_dics.append(lapse_dic_xz)
            o_plot.set_plot_dics.append(lapse_dic_xy)


        plot_x_i += 1





    o_plot.main()

    exit(0)

if __name__  == '__main__':
    plot_summary_quntity_all_in_one()

    plot_den_unb__vel_z_sly4_evol()
    plot_center_of_mass_movement()
    plot_desity_modes2()
    # plot_total_fluxes_for_long_sims_bern("bern_geoend")
    # tmp_plot_disk_mass_evol_SR()
    # plot_summary_quntity()
    plot_summary_quntity_all_in_one()
    # plot_summary_quntity_all_in_one2()

    #### --- CSV TALBE OF ALL SIMULATIOSN
    # o_sims = ALL_SIMULATIONS_TABLE()
    # exit(1)


    # plot_last_disk_mass_with_lambda2(v_n_x="Lambda", v_n_y="Mej_tot", v_n_col="q",
    #                                  mask_x=None,mask_y="geo",mask_col=None,det=0, plot_legend=True)
    # plot_last_disk_mass_with_lambda2(v_n_x="Lambda", v_n_y="Ye_ave", v_n_col="q",
    #                                  mask_x=None,mask_y="geo",mask_col=None,det=0, plot_legend=True)
    # plot_last_disk_mass_with_lambda2(v_n_x="Lambda", v_n_y="vel_inf_ave", v_n_col="q",
    #                                  mask_x=None,mask_y="geo",mask_col=None,det=0, plot_legend=True)
    # plot_last_disk_mass_with_lambda2(v_n_x="Lambda", v_n_y="Ye_ave", v_n_col="q",
    #                                  mask_x=None,mask_y="geo",mask_col=None,det=0, plot_legend=True)

    # plot_last_disk_mass_with_lambda2(v_n_x="Lambda", v_n_y="Mej_tot", v_n_col="q",
    #                                  mask_x=None,mask_y="bern_geoend",mask_col=None,det=0, plot_legend=True)
    # plot_last_disk_mass_with_lambda2(v_n_x="Lambda", v_n_y="Ye_ave", v_n_col="q",
    #                                  mask_x=None,mask_y="bern_geoend",mask_col=None,det=0, plot_legend=True)
    # plot_last_disk_mass_with_lambda2(v_n_x="Lambda", v_n_y="vel_inf_ave", v_n_col="q",
    #                                  mask_x=None,mask_y="bern_geoend",mask_col=None,det=0, plot_legend=True)

    # plot_total_fluxes_sims_disk_hist()
    # plot_den_unb_vel_z()
    # plot_disk_mass_evol_SR()
    # plot_disk_mass_evol_LR()
    # plot_2ejecta_1disk_timehists()
    # plot_total_fluxes_sims_disk_hist_last()

    #
    #
    #

    ### --- OVERALL ---
    tbl1 = TEX_TABLES()

    tbl1.print_mult_table([simulations["BLh"]["q=1"], simulations["BLh"]["q=1.3"], simulations["BLh"]["q=1.4"], simulations["BLh"]["q=1.7"], simulations["BLh"]["q=1.8"],
                          simulations["DD2"]["q=1"], simulations["DD2"]["q=1.1"], simulations["DD2"]["q=1.2"], simulations["DD2"]["q=1.4"],
                          simulations["LS220"]["q=1"], simulations["LS220"]["q=1.1"], simulations["LS220"]["q=1.2"], simulations["LS220"]["q=1.4"], simulations["LS220"]["q=1.7"],
                          simulations["SFHo"]["q=1"], simulations["SFHo"]["q=1.1"], simulations["SFHo"]["q=1.4"], simulations["SFHo"]["q=1.7"],
                          simulations["SLy4"]["q=1"], simulations["SLy4"]["q=1.1"], simulations["SLy4"]["q=1.4"]],
                         [r"\hline", r"\hline", r"\hline", r"\hline",
                          r"\hline\hline",
                          r"\hline", r"\hline", r"\hline",
                          r"\hline\hline",
                          r"\hline", r"\hline", r"\hline", r"\hline",
                          r"\hline\hline",
                          r"\hline", r"\hline", r"\hline",
                          r"\hline\hline",
                          r"\hline", r"\hline", r"\hline"])
    exit(1)




    tbl = COMPARISON_TABLE()
    exit(1)
    ### --- resulution effect on simulations with viscosity
    tbl.print_mult_table([["DD2_M13641364_M0_LK_SR_R04", "DD2_M13641364_M0_LK_LR_R04", "DD2_M13641364_M0_LK_HR_R04"], # HR too short
                         ["DD2_M15091235_M0_LK_SR", "DD2_M15091235_M0_LK_HR"],          # no
                         ["LS220_M14691268_M0_LK_SR", "LS220_M14691268_M0_LK_HR"],      # no
                         ["SFHo_M13641364_M0_LK_SR", "SFHo_M13641364_M0_LK_HR"],        # no
                         ["SFHo_M14521283_M0_LK_SR", "SFHo_M14521283_M0_LK_HR"]],       # no
                         [r"\hline",
                          r"\hline",
                          r"\hline",
                          r"\hline",
                          r"\hline"],
                         comment=r"{Resolution effect to on the outflow properties and disk mass on the simulations with "
                         r"subgird turbulence. Here the $t_{\text{disk}}$ "
                         r"is the maximum postmerger time, for which the 3D is available for both simulations "
                         r"For that time, the disk mass is interpolated using linear inteprolation. The "
                         r"$\Delta t_{\text{wind}}$ is the maximum common time window between the time at "
                         r"which dynamical ejecta reaches 98\% of its total mass and the end of the simulation "
                         r"Cases where $t_{\text{disk}}$ or $\Delta t_{\text{wind}}$ is N/A indicate the absence "
                         r"of the ovelap between 3D data fro simulations or absence of this data entirely and "
                         r"absence of overlap between the time window in which the spiral-wave wind is computed "
                         r"which does not allow to do a proper, one-to-one comparison. $\Delta$ is a estimated "
                         r"change as $|value_1 - value_2|/value_1$ in percentage }",
                         label=r"{tbl:res_effect_vis}"
                         )
    exit(0)