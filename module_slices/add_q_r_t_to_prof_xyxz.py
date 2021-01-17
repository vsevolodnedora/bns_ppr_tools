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
import sys
# from scidata.utils import locate
# import scidata.carpet.hdf5 as h5
# from scidata.carpet.interp import Interpolator

from glob import glob
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


from uutils import Printcolor, REFLEVEL_LIMITS

from plotting.plotting_methods import PLOT_MANY_TASKS


from uutils import Paths, Printcolor
# from module_preanalysis.preanalysis import LOAD_ITTIME
from slices_methods import COMPUTE_STORE

from module_preanalysis.it_time import LOAD_ITTIME

def add_q_r_t_to_prof_xyxz(v_ns,
                           rls,
                           planes,
                           iterations,
                           sim,
                           indir,
                           pprdir,
                           path_to_sliced_profiles,
                           overwrite
                           ):
    """
        this function loops over the avilalbe profiles.xy.h5 and profiles.xz.h5
        and adds to theid data module_slices of neutrino data from carpet files 'file.xy.h5'
    """


    # glob_sim = "LS220_M14691268_M0_LK_SR"
    glob_profxyxz_path = path_to_sliced_profiles#Paths.ppr_sims+glob_sim+'/profiles/'
    #glob_nlevels = 7
    # glob_overwrite = False

    ititme = LOAD_ITTIME(sim, pprdir=pprdir)
    _, profit, proft = ititme.get_ittime("profiles", d1d2d3prof="prof")
    #
    if len(profit) == 0:
        Printcolor.yellow("No profiles found. Q R T values are not added to prof.xy.h5")
        return 0
    #
    d2data = COMPUTE_STORE(sim, indir=indir, pprdir=pprdir)
    #
    # assert len(glob_reflevels) > 0
    assert len(v_ns) > 0
    #
    for it in iterations:
        for plane in planes:
            fpath = glob_profxyxz_path + str(int(it)) + '/' + "module_profile.{}.h5".format(plane)
            if os.path.isfile(fpath):
                try:
                    dfile = h5py.File(glob_profxyxz_path + str(int(it)) + '/' + "module_profile.{}.h5".format(plane), "a")

                    Printcolor.print_colored_string(
                        ["task:", "addm0", "it:", "{}".format(it), "plane", plane, ':', "Adding"], ["blue", "green", "blue", "green","blue", "green",  "", "green"]
                    )
                    for rl in rls:
                        gname = "reflevel=%d" % rl
                        for v_n in v_ns:
                            if (v_n in dfile[gname] and overwrite) or not v_n in dfile[gname]:
                                if v_n in dfile[gname]:
                                        del dfile[gname][v_n]
                                #
                                prof_rho = dfile[gname]["rho"]
                                rho_arr = d2data.get_data(it, plane, "rho")[rl][3:-3, 3:-3]
                                nu_arr = d2data.get_data(it, plane, v_n)[rl][3:-3, 3:-3]
                                assert rho_arr.shape == nu_arr.shape

                                if prof_rho.shape != nu_arr.shape:
                                    Printcolor.yellow("Size Mismatch. Profile:{} 2D data:{} Filling with nans..."
                                                      .format(prof_rho.shape, nu_arr.shape))
                                    px, py, pz = dfile[gname]["x"], dfile[gname]["y"], dfile[gname]["z"]
                                    nx, nz = d2data.get_grid_v_n_rl(it, plane, rl, "x")[3:-3, 3:-3], \
                                           d2data.get_grid_v_n_rl(it, plane, rl, "z")[3:-3, 3:-3]
                                    # print("mismatch prof_rho:{} nu:{}".format(prof_rho.shape, nu_arr.shape))
                                    # print("mismatch prof x:{} prof z:{}".format(px.shape, pz.shape))
                                    # print("mismatch x:{} z:{}".format(nx.shape, nz.shape))
                                    # arr = np.full(prof_rho[:,0,:].shape, 1)

                                    # tst = np.where((px>=nx.min()) | (px<=nx.max()), arr, nu_arr)
                                    # print(tst)

                                    tmp = np.full(prof_rho.shape, np.nan)
                                    # for ipx in range(len(px)):

                                    for ipx in range(len(px[:, 0])):
                                        for ipz in range(len(pz[0, :])):
                                            if px[ipx] in nx and pz[ipz] in nz:
                                                # print("found: {} {}".format(px[ipx], py[ipz]))
                                                # print(px[(px[ipx] == nx)&(pz[ipz] == nz)])
                                                # print(pz[(px[ipx] == nx) & (pz[ipz] == nz)])
                                                # print(nu_arr[(px[ipx] == nx)&(pz[ipz] == nz)])
                                                # print("x:{} z:{}".format(px[ipx, 0], pz[0,  ipz]))
                                                # print(nu_arr[(px[ipx, 0] == nx)&(pz[0, ipz] == nz)])
                                                # print(float(nu_arr[(px[ipx, 0] == nx) & (pz[0, ipz] == nz)]))
                                                tmp[ipx, ipz] = float(nu_arr[(px[ipx, 0] == nx) & (pz[0, ipz] == nz)])
                                                # print("x:{} z:{} filling with:{}".format(px[ipx, 0], pz[0, ipz], tmp[ipx, ipz]))
                                    #
                                    nu_arr = tmp
                                            # else:
                                                # print("wrong: {}".format(px[ipx], py[ipz]))
                                    # print(tmp)
                                    # print(tmp.shape)
                                    # exit(1)

                                    # UTILS.find_nearest_index()
                                    #
                                    #
                                    #
                                    # for ix in range(len(arr[:, 0])):
                                    #     for iz in range(len(arr[0, :])):
                                    #         x = np.round(px[ix, iz], decimals=1)
                                    #         z = np.round(py[ix, iz], decimals=1)
                                    #
                                    #
                                    #
                                    #         if x in np.round(nx, decimals=1) and z in np.round(nz, decimals=1):
                                    #             arr[ix, iz] = nu_arr[np.where((np.round(nx, decimals=1) == x) & (np.round(nz, decimals=1) == z))]
                                    #             print('\t\treplacing {} {}'.format(ix, iz))
                                    # print(arr)
                                    #
                                    # exit(1)
                                    #
                                    #
                                    # ileft, iright = np.where(px<nx.min()), np.where(px>nx.max())
                                    # print(ileft)  # (axis=0 -- array, axis=1 -- array)
                                    # print(iright)
                                    # ilower, iupper = np.where(pz<nz.min()), np.where(pz>nz.max())
                                    # print(ilower)
                                    # print(iupper)
                                    #
                                    # #
                                    # import copy
                                    # tmp = copy.deepcopy(nu_arr)
                                    # for axis in range(len(ileft)):
                                    #     for element in ileft[axis]:
                                    #         tmp = np.insert(tmp, 0, np.full(len(tmp[0,:]), np.nan), axis=0)
                                    #
                                    # # tmp = copy.deepcopy(nu_arr)
                                    # for axis in range(len(iright)):
                                    #     print("\taxis:{} indexes:{}".format(axis, iright[axis]))
                                    #     for element in iright[axis]:
                                    #         tmp = np.insert(tmp, -1, np.full(len(tmp[0,:]), np.nan), axis=0)
                                    #     print(tmp.shape)
                                    #
                                    # print(prof_rho.shape)
                                    # print(tmp.shape)

                                    # indexmap = np.where((px<nx.min()) | (px>nx.max()), arr, 0)
                                    # arr[indexmap] = nu_arr
                                    # print(indexmap)
                                    # print(arr)
                                    # print(indexmap.shape)

                                    # insert coordinates
                                    # exit(1)


                                    # arr = np.full(prof_rho.shape, np.nan)



                                    # exit(1)
                                    #
                                    #
                                    #
                                    #
                                    # arr = np.full(prof_rho.shape,np.nan)
                                    # for ix in range(len(arr[:, 0])):
                                    #     for iz in range(len(arr[0,:])):
                                    #         x = px[ix, iz]
                                    #         z = py[ix, iz]
                                    #         if x in nx and z in nz:
                                    #             arr[ix, iz] = nu_arr[np.where((nx == x)&(nz == z))]
                                    #             print('\t\treplacing {} {}'.format(ix, iz))
                                    # print(arr);
                                    #
                                    # exit(1)

                                print("\t{} nu:{} prof_rho:{}".format(rl, nu_arr.shape, prof_rho.shape))
                                # nu_arr = nu_arr[3:-3, 3:-3]
                                # hydro_arr = d3data.get_data(it, rl, plane, "rho")
                                # assert nu_arr.shape == hydro_arr.shape
                                gname = "reflevel=%d" % rl
                                dfile[gname].create_dataset(v_n, data=np.array(nu_arr, dtype=np.float32))
                            else:
                                Printcolor.print_colored_string(["\trl:", str(rl), "v_n:", v_n, ':',
                                     "skipping"],
                                    ["blue", "green","blue", "green", "", "blue"]
                                )
                    dfile.close()
                except KeyboardInterrupt:
                    exit(1)
                except ValueError:
                    Printcolor.print_colored_string(
                        ["task:", "addm0", "it:", "{}".format(it), "plane", plane, ':', "ValueError"],
                        ["blue", "green", "blue", "green","blue", "green", "", "red"]
                    )
                except IOError:
                    Printcolor.print_colored_string(
                        ["task:", "addm0", "it:", "{}".format(it), "plane", plane, ':', "IOError"],
                        ["blue", "green", "blue", "green","blue", "green", "", "red"]
                    )
                except:
                    Printcolor.print_colored_string(
                        ["task:", "addm0", "it:", "{}".format(it), "plane", plane, ':', "FAILED"],
                        ["blue", "green", "blue", "green", "blue", "green", "", "red"]
                    )
            else:
                Printcolor.print_colored_string(
                    ["task:", "adding neutrino data to prof. slice", "it:", "{}".format(it), ':', "IOError: module_profile.{}.h5 does not exist".format(plane)],
                    ["blue", "green", "blue", "green", "", "red"]
                )
    # for it in profit:
    #     #
    #     fpathxy = glob_profxyxz_path + str(int(it)) + '/' + "module_profile.xy.h5"
    #     fpathxz = glob_profxyxz_path + str(int(it)) + '/' + "module_profile.xz.h5"