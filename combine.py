#
from __future__ import division
from sys import path
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
from profile import LOAD_PROFILE_XYXZ, LOAD_RES_CORR
import units as ut # for tmerg
from utils import *
#

for letter in "kusi":
    print(letter),


''' lissts of all the simulations '''
simulations = {"BLh":
                 {
                      "q=1.8": ["BLh_M10201856_M0_LK_SR"], # Prompt collapse
                      "q=1.7": ["BLh_M10651772_M0_LK_SR"], # stable
                      "q=1.4": ["BLh_M16351146_M0_LK_LR"],
                      "q=1.3": ["BLh_M11841581_M0_LK_SR"],
                      "q=1":   ["BLh_M13641364_M0_LK_SR"]
                 },
              "DD2":
                  {
                      "q=1": ["DD2_M13641364_M0_HR_R04", "DD2_M13641364_M0_LK_HR_R04",
                              "DD2_M13641364_M0_LK_LR_R04", "DD2_M13641364_M0_LK_SR_R04",
                              "DD2_M13641364_M0_LR", "DD2_M13641364_M0_LR_R04",
                              "DD2_M13641364_M0_SR", "DD2_M13641364_M0_SR_R04"],
                      "q=1.1": ["DD2_M14321300_M0_LR", "DD2_M14351298_M0_LR"],
                      "q=1.2": ["DD2_M14861254_M0_HR", "DD2_M14861254_M0_LR",
                                "DD2_M14971245_M0_HR", "DD2_M14971245_M0_SR",
                                "DD2_M14971246_M0_LR", "DD2_M15091235_M0_LK_HR",
                                "DD2_M15091235_M0_LK_SR"],
                      "q=1.4": ["DD2_M16351146_M0_LK_LR"]
                  },
              "LS220":
                  {
                      "q=1": ["LS220_M13641364_M0_HR", #"LS220_M13641364_M0_LK_HR", # TOO short. 3ms
                              "LS220_M13641364_M0_LK_SR", "LS220_M13641364_M0_LK_SR_restart",
                              "LS220_M13641364_M0_LR", "LS220_M13641364_M0_SR"],
                      "q=1.1": ["LS220_M14001330_M0_HR", "LS220_M14001330_M0_SR",
                                "LS220_M14351298_M0_HR", "LS220_M14351298_M0_SR"],
                      "q=1.2": ["LS220_M14691268_M0_HR", "LS220_M14691268_M0_LK_HR",
                                "LS220_M14691268_M0_LK_SR", "LS220_M14691268_M0_LR",
                                "LS220_M14691268_M0_SR"],
                      "q=1.4": ["LS220_M16351146_M0_LK_LR"]
                  },
              "SFHo":
                  {
                      "q=1": ["SFHo_M13641364_M0_HR", "SFHo_M13641364_M0_LK_HR",
                              "SFHo_M13641364_M0_LK_SR", #"SFHo_M13641364_M0_LK_SR_2019pizza", # failed
                              "SFHo_M13641364_M0_SR"],
                      "q=1.1":["SFHo_M14521283_M0_HR", "SFHo_M14521283_M0_LK_HR",
                               "SFHo_M14521283_M0_LK_SR", "SFHo_M14521283_M0_LK_SR_2019pizza",
                               "SFHo_M14521283_M0_SR"],
                      "q=1.4":["SFHo_M16351146_M0_LK_LR"]
                  },
              "SLy4":
                  {
                      "q=1": [#"SLy4_M13641364_M0_HR", # precollapse
                              # "SLy4_M13641364_M0_LK_HR", # crap, absent tarball data
                              "SLy4_M13641364_M0_LK_LR", "SLy4_M13641364_M0_LK_SR",
                              "SLy4_M13641364_M0_LR", "SLy4_M13641364_M0_SR"],
                      "q=1.1":[#"SLy4_M14521283_M0_HR", unphysical and premerger
                               "SLy4_M14521283_M0_LR",
                               "SLy4_M14521283_M0_SR"]
                  }
              }

sims_err_lk_onoff = {

    "def": {"sims":["DD2_M13641364_M0_LK_SR_R04", "DD2_M15091235_M0_LK_SR", "LS220_M14691268_M0_LK_SR", "SFHo_M14521283_M0_LK_SR"],
            "lbls": ["DD2 136 136 LK", "DD2 151 123 LK", "LS220 147 127 LK", "SFHo 145 128 LK"],
            "colors":["black", 'gray', 'red', "green"],
            "lss":["-", '-', '-', '-'],
            "lws":[1.,1.,1.,1.]},
    "comp":{"sims":["DD2_M13641364_M0_SR_R04", "DD2_M14971245_M0_SR", "LS220_M14691268_M0_SR", "SFHo_M14521283_M0_SR"],
            "lbls": ["DD2 136 136", "DD2 150 125", "LS220 147 127", "SFHo 145 128"],
            "colors":["black", 'gray', 'red', "green"],
            "lss":["--", '--', '--', '--'],
            "lws":[1.,1.,1.,1.]},
}

''' ==============================| LOAD OUTFLOW & COLLATED & Mdisk |=============================================== '''

class LOAD_FILES(LOAD_ITTIME):

    list_outflowed_files = [
        "total_flux.dat",
        "hist_temperature.dat",
        "hist_theta.dat",
        "hist_Y_e.dat",
        "hist_log_rho.dat", #hist_rho.dat",
        "hist_entropy.dat",
        "hist_vel_inf.dat",
        "hist_vel_inf_bern.dat",

        "mass_averages.dat",

        "ejecta_profile.dat",
        "ejecta_profile_bern.dat",

        "corr_vel_inf_bern_theta.h5",
        "corr_vel_inf_theta.h5",
        "corr_ye_entropy.h5",
        "corr_ye_theta.h5"
    ]

    list_collated_files = [
        "dens_unbnd.norm1.asc",
        "dens_unbnd_bernoulli.norm1.asc",
        "dens.norm1.asc",
    ]

    list_gw_files = [
        "waveform_l2_m2.dat",
        "tmerger.dat",
        "tcoll.dat"
    ]

    list_3d_data_files = [
        "disk_mass.txt"
    ]

    def __init__(self, sim, add_mask):

        LOAD_ITTIME.__init__(self, sim)

        self.sim = sim

        self.list_detectors = [0, 1]

        self.list_masks = ["geo", "bern", "bern_geoend", "Y_e04_geoend"]
        if add_mask != None and not add_mask in self.list_masks:
            self.list_masks.append(add_mask)

        self.matrix_outflow_data = [[[np.zeros(0,)
                                    for i in range(len(self.list_outflowed_files))]
                                    for k in range(len(self.list_masks))]
                                    for j in range(len(self.list_detectors))]

        self.matrix_gw_data = [np.zeros(0,)
                                     for i in range(len(self.list_gw_files))]

        self.matrix_collated_data = [np.zeros(0,)
                                     for i in range(len(self.list_collated_files))]

        self.matrix_3d_data = [np.zeros(0,)
                               for i in range(len(self.list_3d_data_files))]

    def check_v_n(self, v_n):
        if not v_n in self.list_outflowed_files:
            if not v_n in self.list_gw_files:
                if not v_n in self.list_collated_files:
                    if not v_n in self.list_3d_data_files:
                        raise NameError("v_n: {} is not in the any list: \n"
                                        "outflow: [ {} ], \ngw: [ {} ], \ncolated: [ {} ], \n3D: [ {} ]"
                                        .format(v_n,
                                                self.list_outflowed_files,
                                                self.list_gw_files,
                                                self.list_collated_files,
                                                self.list_3d_data_files))

    def check_det(self, det):
        if not det in self.list_detectors:
            raise NameError("det: {} is not in the list: {}"
                            .format(det, self.list_detectors))

    def check_mask(self, mask):
        if not mask in self.list_masks:
            return NameError("mask: {} is not in the list {} "
                             .format(mask, self.list_masks))

    def i_fn_outflow(self, v_n):
        return int(self.list_outflowed_files.index(v_n))

    def i_fn_col(self, v_n):
        return int(self.list_collated_files.index(v_n))

    def i_fn_gw(self, v_n):
        return int(self.list_gw_files.index(v_n))

    def i_fn_3d(self, v_n):
        return int(self.list_3d_data_files.index(v_n))

    def i_det(self, det):
        return int(self.list_detectors.index(int(det)))

    def i_mask(self, mask):
        return int(self.list_masks.index(mask))

    # --------------------------- LOADING DATA METHODS ----------------------------

    def load_outflow_data(self, det, mask, v_n):

        fpath = Paths.ppr_sims+self.sim+'/'+ "outflow_{:d}".format(det) + '/' + mask + '/' + v_n
        if not os.path.isfile(fpath):
            raise IOError("File not found for det:{} mask:{} v_n:{} -> {}"
                          .format(det, mask, v_n, fpath))
        # loading acii file
        if not v_n.__contains__(".h5"):
            data = np.loadtxt(fpath)
            return data
        # loading correlation files
        if v_n.__contains__(".h5"):
            dfile = h5py.File(fpath, "r")
            v_ns = []
            for _v_n in dfile:
                v_ns.append(_v_n)
            assert len(v_ns) == 3
            assert "mass" in v_ns
            mass = np.array(dfile["mass"])
            v_ns.remove('mass')
            xarr = np.array(dfile[v_ns[0]])
            yarr = np.array(dfile[v_ns[1]])
            table = UTILS.combine(xarr, yarr, mass)
            return table
        raise ValueError("Loading data method for ourflow data is not found")

    def load_collated_data(self, v_n):
        fpath = Paths.ppr_sims + self.sim + '/' + "collated/" + v_n
        if not os.path.isfile(fpath):
            raise IOError("File not found: collated v_n:{} -> {}"
                          .format(v_n, fpath))

        data = np.loadtxt(fpath)
        return data

    def load_gw_data(self, v_n):
        fpath =  Paths.ppr_sims + self.sim + '/' + "waveforms/" + v_n
        if not os.path.isfile(fpath):
            raise IOError("File not found: gw v_n:{} -> {}"
                          .format(v_n, fpath))

        data = np.loadtxt(fpath)
        return np.array([data])

    def load_3d_data(self, v_n):
        # ispar, itpar, tpar = self.get_ittime("profiles", "prof")

        if not os.path.isdir(Paths.ppr_sims+self.sim+'/' + 'profiles/'):
            # print("No dir: {}".format())
            return np.zeros(0,)

        list_iterations = Paths.get_list_iterations_from_res_3d(self.sim, "profiles/")
        if len(list_iterations) == 0:
            return np.zeros(0,)

        # empty = np.array(list_iterations)
        # empty.fill(np.nan)
        # return empty

        time_arr = []
        data_arr = []
        for it in list_iterations:
            fpath = Paths.ppr_sims+self.sim+'/'+"profiles/" + str(int(it))  + '/' + v_n
            time_ = self.get_time_for_it(it, "prof")
            time_arr.append(time_)
            if os.path.isfile(fpath):
                data_ = np.float(np.loadtxt(fpath, unpack=True))
                data_arr.append(data_)
            else:
                data_arr.append(np.nan)

        # if len(data_arr) ==0:
        #     return np.zeros(0,)

        data_arr = np.array(data_arr)
        time_arr = np.array(time_arr)
        # print(data_arr.shape, time_arr.shape)

        assert len(data_arr) == len(time_arr)
        res = np.vstack((time_arr, data_arr))
        return res

    # -----------------------------------------------------------------------------
    def is_outflow_data_loaded(self, det, mask, v_n):
        data = self.matrix_outflow_data[self.i_det(det)][self.i_mask(mask)][self.i_fn_outflow(v_n)]
        if len(data) == 0:
            data = self.load_outflow_data(det, mask, v_n)
            self.matrix_outflow_data[self.i_det(det)][self.i_mask(mask)][self.i_fn_outflow(v_n)] = data

        data = self.matrix_outflow_data[self.i_det(det)][self.i_mask(mask)][self.i_fn_outflow(v_n)]
        if len(data) == 0:
            raise ValueError("Loading outflow data has failed. Array is empty. det:{} mask:{} v_n:{}"
                             .format(det, mask, v_n))

    def get_outflow_data(self, det, mask, v_n):
        self.check_v_n(v_n)
        self.check_det(det)
        self.check_mask(mask)
        self.is_outflow_data_loaded(det, mask, v_n)
        data = self.matrix_outflow_data[self.i_det(det)][self.i_mask(mask)][self.i_fn_outflow(v_n)]
        return data
    # --- --- #
    def is_collated_loaded(self, v_n):
        data = self.matrix_collated_data[self.i_fn_col(v_n)]
        if len(data) == 0:
            data = self.load_collated_data(v_n)
            self.matrix_collated_data[self.i_fn_col(v_n)] = data
        # --- #
        data = self.matrix_collated_data[self.i_fn_col(v_n)]
        if len(data) == 0:
            raise ValueError("Failed loading collated data. Array is empty. v_n:{}"
                             .format(v_n))

    def get_collated_data(self, v_n):
        self.check_v_n(v_n)
        self.is_collated_loaded(v_n)
        data = self.matrix_collated_data[self.i_fn_col(v_n)]
        return data
    # --- --- #
    def is_gw_data_loaded(self, v_n):
        data = self.matrix_gw_data[self.i_fn_gw(v_n)]
        if len(data) == 0:
            data = self.load_gw_data(v_n)
            self.matrix_gw_data[self.i_fn_gw(v_n)] = data

        data = self.matrix_gw_data[self.i_fn_gw(v_n)]
        # print(data)
        if len(data) == 0:
            raise ValueError("Failed to load GW data v_n:{}")

    def get_gw_data(self, v_n):
        self.check_v_n(v_n)
        self.is_gw_data_loaded(v_n)
        data = self.matrix_gw_data[self.i_fn_gw(v_n)]
        return data
    # --- --- #
    def is_3d_data_loaded(self, v_n):
        data = self.matrix_3d_data[self.i_fn_3d(v_n)]
        if len(data) == 0:
            data = self.load_3d_data(v_n)
            self.matrix_3d_data[self.i_fn_3d(v_n)] = data

        # data = self.matrix_3d_data[self.i_fn_3d(v_n)]
        # if len(data) == 0:
        #     raise ValueError("Failed to load 3D ")

    def get_3d_data(self, v_n):
        self.check_v_n(v_n)
        self.is_3d_data_loaded(v_n)
        data = self.matrix_3d_data[self.i_fn_3d(v_n)]
        return data
    # -------------------------------------------------------------------------------


class COMPUTE_ARR(LOAD_FILES):

    def __init__(self, sim, add_mask=None):
        LOAD_FILES.__init__(self, sim, add_mask)

    def get_outflow_hist(self, det, mask, v_n):
        data = self.get_outflow_data(det, mask, "hist_{}.dat".format(v_n))
        return data.T

    def get_outflow_corr(self, det, mask, v_n1_vn2):
        data = self.get_outflow_data(det, mask, "corr_{}.h5".format(v_n1_vn2))
        return data

    def get_disk_mass(self):
        data = self.get_3d_data("disk_mass.txt")
        return data.T

    def get_total_mass(self):
        tmp = self.get_collated_data("dens.norm1.asc")
        t_total_mass, dens = tmp[:, 1], tmp[:, 2]
        t_total_mass = t_total_mass * Constants.time_constant / 1000  # [s]
        m_tot = dens * Constants.volume_constant ** 3
        return np.vstack((t_total_mass, m_tot)).T

    def get_tot_unb_mass(self):
        tmp2 = self.get_collated_data("dens_unbnd.norm1.asc")
        t_unb_mass, dens_unb = tmp2[:, 1], tmp2[:, 2]
        t_unb_mass *= Constants.time_constant / 1000
        unb_mass = dens_unb * (Constants.volume_constant ** 3)
        return np.vstack((t_unb_mass, unb_mass)).T

    def get_unb_bern_mass(self):
        tmp2 = self.get_collated_data("dens_unbnd_bernoulli.norm1.asc")
        t_unb_mass, dens_unb = tmp2[:, 1], tmp2[:, 2]
        t_unb_mass *= Constants.time_constant / 1000
        unb_mass = dens_unb * (Constants.volume_constant ** 3)
        return np.vstack((t_unb_mass, unb_mass)).T


class ALL_PAR(COMPUTE_ARR):

    def __init__(self, sim, add_mask=None):
        COMPUTE_ARR.__init__(self, sim, add_mask)

    def get_outflow_par(self, det, mask, v_n):

        if v_n == "Mej_tot":
            data = np.array(self.get_outflow_data(det, mask, "total_flux.dat"))
            mass_arr = data[:, 2]
            res = mass_arr[-1]
        elif v_n == "Ye_ave":
            mej = self.get_outflow_par(det, mask, "Mej_tot")
            hist = self.get_outflow_hist(det, mask, "Y_e").T
            res = EJECTA_PARS.compute_ave_ye(mej, hist)
        elif v_n == "s_ave":
            mej = self.get_outflow_par(det, mask, "Mej_tot")
            hist = self.get_outflow_hist(det, mask, "entropy").T
            res = EJECTA_PARS.compute_ave_s(mej, hist)
        elif v_n == "vel_inf_ave":
            mej = self.get_outflow_par(det, mask, "Mej_tot")
            hist = self.get_outflow_hist(det, mask, "vel_inf").T
            res = EJECTA_PARS.compute_ave_vel_inf(mej, hist)
        elif v_n == "E_kin_ave":
            mej = self.get_outflow_par(det, mask, "Mej_tot")
            hist = self.get_outflow_hist(det, mask, "vel_inf").T
            res = EJECTA_PARS.compute_ave_ekin(mej, hist)
        elif v_n == "theta_rms":
            hist = self.get_outflow_hist(det, mask, "theta").T
            res = EJECTA_PARS.compute_ave_theta_rms(hist)
        elif v_n == "tend":
            data = np.array(self.get_outflow_data(det, mask, "total_flux.dat"))
            mass_arr = data[:, 0]
            res = mass_arr[-1]
        elif v_n == "t98mass":
            data = np.array(self.get_outflow_data(det, mask, "total_flux.dat"))
            mass_arr = data[:, 2]
            time_arr = data[:, 0]
            fraction = 0.98
            i_t98mass = int(np.where(mass_arr >= fraction * mass_arr[-1])[0][0])
            # print(i_t98mass)
            assert i_t98mass < len(time_arr)
            res = time_arr[i_t98mass]
        else:
            raise NameError("no method for estimation det:{} mask:{} v_n:{}"
                            .format(det, mask, v_n))
        return res

    def get_par(self, v_n):

        if v_n == "tcoll_gw":
            try:
                data = self.get_gw_data("tcoll.dat")
            except IOError:
                Printcolor.yellow("\tWarning! No tcoll.dat found for sim:{}".format(self.sim))
                return np.inf
            Printcolor.yellow("\tWarning! using defauled M_Inf=2.748, R_GW=400.0 for retardet time")
            ret_time = PHYSICS.get_retarded_time(data, M_Inf=2.748, R_GW=400.0)
            # tcoll = ut.conv_time(ut.cactus, ut.cgs, ret_time)
            return float(ret_time * Constants.time_constant * 1e-3)
        elif v_n == "tend":
            total_mass = self.get_total_mass()
            t, Mtot = total_mass[:, 0], total_mass[:, 1]
            # print(t)
            return t[-1]
        elif v_n == "tmerg" or v_n == "tmerger":
            try:
                data = self.get_gw_data("tmerger.dat")
            except IOError:
                Printcolor.yellow("\tWarning! No tmerger.dat found for sim:{}".format(self.sim))
                return np.nan
            Printcolor.yellow("\tWarning! using defauled M_Inf=2.748, R_GW=400.0 for retardet time")
            ret_time = PHYSICS.get_retarded_time(data, M_Inf=2.748, R_GW=400.0)
            # tmerg = ut.conv_time(ut.cactus, ut.cgs, ret_time)
            return float(ret_time * Constants.time_constant * 1e-3)
        elif v_n == "tcoll" or v_n == "Mdisk":
            total_mass = self.get_total_mass()
            unb_mass = self.get_tot_unb_mass()
            t, Mtot = total_mass[:, 0]*Constants.time_constant*1e-3, total_mass[:, 1]
            _, Munb = unb_mass[:, 0]*Constants.time_constant*1e-3, unb_mass[:, 1]
            # print(Mtot.min()); exit(1)
            if Mtot[-1] > 1.0:
                Mdisk = np.nan
                tcoll = np.inf
                Printcolor.yellow("Warning: Mtot[-1] > 1 Msun. -> Either no disk or wrong .ascii")
            else:
                i_BH = np.argmin(Mtot > 1.0)
                tcoll = t[i_BH]  # *1000 #* utime
                i_disk = i_BH + int(1.0 / (t[1] * 1000))  #
                # my solution to i_disk being out of bound:
                if i_disk > len(Mtot): i_disk = len(Mtot) - 1
                if i_disk > len(Munb): i_disk = len(Munb) - 1
                Mdisk = Mtot[i_disk] - Munb[i_disk]
            if v_n == "tcoll":
                return tcoll
            else:
                return Mdisk
        elif v_n == "Munb_tot":
            unb_mass = self.get_tot_unb_mass()
            _, Munb = unb_mass[:, 0], unb_mass[:, 1]
            print(unb_mass.shape)
            return Munb[-1]
        elif v_n == "Munb_bern_tot":
            unb_mass = self.get_unb_bern_mass()
            _, Munb = unb_mass[:, 0], unb_mass[:, 1]
            return Munb[-1]
        elif v_n == "Mdisk3D":
            dislmasses = self.get_disk_mass()
            if len(dislmasses) > 0:
                return dislmasses[-1, 1]
            else:
                return np.nan
        elif v_n == "tdisk3D":
            dislmasses = self.get_disk_mass()
            if len(dislmasses) > 0:
                return dislmasses[-1, 0]
            else:
                return np.nan
        else:
            raise NameError("no parameter found for v_n:{}".format(v_n))


class ADD_METHODS_ALL_PAR(ALL_PAR):

    def __init__(self, sim, add_mask=None):
        ALL_PAR.__init__(self, sim, add_mask)

    def get_int_par(self, v_n, t):

        if v_n == "Mdisk3D":
            dislmasses = self.get_disk_mass()
            if len(dislmasses) == 0:
                Printcolor.red("no disk mass data found (empty get_disk_mass()): "
                               "{}".format(self.sim))
                return np.nan
                # raise ValueError("no disk mass data found")
            tarr = dislmasses[:,0]
            marr = dislmasses[:,1]
            if t > tarr.max():
                raise ValueError("t: {} is above DiskMass time array max: {}"
                                 .format(t, tarr.max()))
            if t < tarr.min():
                raise ValueError("t: {} is below DiskMass time array min: {}"
                                 .format(t, tarr.min()))
            f = interpolate.interp1d(tarr, marr, kind="linear", bounds_error=True)
            return f(t)

""" =======================================| 2 SIM CLASSES |========================================================="""

class TWO_SIMS():

    def __init__(self, sim1, sim2):
        self.sim1 = sim1
        self.sim2 = sim2
        self.o_par1 = ADD_METHODS_ALL_PAR(self.sim1)
        self.o_par2 = ADD_METHODS_ALL_PAR(self.sim2)
        self.outflow_tasks = ["totflux", "hist"]

    def compute_outflow_new_mask(self, det, sim, mask, rewrite):

        # get_tmax60 # ms
        print("\tAdding mask:{}".format(mask))
        o_outflow = EJECTA_PARS(sim, add_mask=mask)

        if not os.path.isdir(Paths.ppr_sims + sim +"/" +"outflow_{:d}/".format(det) + mask + '/'):
            os.mkdir(Paths.ppr_sims + sim +"/" +"outflow_{:d}/".format(det) + mask + '/')

        Printcolor.blue("Creating new outflow mask dir:{}"
                        .format(sim +"/" +"outflow_{:d}/".format(det) + mask + '/'))

        for task in self.outflow_tasks:
            if task == "hist":
                from outflowed import outflowed_historgrams
                outflowed_historgrams(o_outflow, [det], [mask], o_outflow.list_hist_v_ns, rewrite=rewrite)
            elif task == "corr":
                from outflowed import outflowed_correlations
                outflowed_correlations(o_outflow, [det], [mask], o_outflow.list_corr_v_ns, rewrite=rewrite)
            elif task == "totflux":
                from outflowed import outflowed_totmass
                outflowed_totmass(o_outflow, [det], [mask], rewrite=rewrite)
            elif task == "timecorr":
                from outflowed import outflowed_timecorr
                outflowed_timecorr(o_outflow, [det], [mask], o_outflow.list_hist_v_ns, rewrite=rewrite)
            else:
                raise NameError("method for computing outflow with new mask is not setup for task:{}".format(task))

    def get_post_geo_delta_t(self, det):

        # o_par1 = ALL_PAR(self.sim1)
        # o_par2 = ALL_PAR(self.sim2)

        # tmerg1 = self.o_par1.get_par("tmerger")
        # tmerg2 = self.o_par2.get_par("tmerger")

        t98geomass1 = self.o_par1.get_outflow_par(det, "geo", "t98mass")
        t98geomass2 = self.o_par2.get_outflow_par(det, "geo", "t98mass")

        tend1 = self.o_par1.get_outflow_par(det, "geo", "tend")
        tend2 = self.o_par2.get_outflow_par(det, "geo", "tend")

        if tend1 < t98geomass1:
            Printcolor.red("tend1:{} < t98geomass1:{}".format(tend1, t98geomass1))
            return np.nan
        if tend2 < t98geomass2:
            Printcolor.red("tend2:{} < t98geomass2:{}".format(tend2, t98geomass2))
            return np.nan

        if tend1 < t98geomass2:
            Printcolor.red("Delta t does not overlap tend1:{} < t98geomass2:{}".format(tend1, t98geomass2))
            return np.nan
        if tend2 < t98geomass1:
            Printcolor.red("Delta t does not overlap tend2:{} < t98geomass1:{}".format(tend2, t98geomass1))
            return np.nan
        # assert tmerg1 < t98geomass1
        # assert tmerg2 < t98geomass2

        # tend1 = tend1 - tmerg1
        # tend2 = tend2 - tmerg2
        # t98geomass1 = t98geomass1 - tmerg1
        # t98geomass2 = t98geomass2 - tmerg2

        delta_t1 = tend1 - t98geomass1
        delta_t2 = tend2 - t98geomass2

        print("\tTime window for bernoulli ")
        print("\t{} {:.2f} [ms]".format(self.sim1, delta_t1*1e3))
        print("\t{} {:.2f} [ms]".format(self.sim2, delta_t2*1e3))
        # exit(1)

        delta_t = np.min([delta_t1, delta_t2])

        if delta_t < 0.005:
            return np.nan# ms

        return delta_t

    def get_tmax_d3_data(self):

        isd3_1, itd3_1, td3_1 = self.o_par1.get_ittime("profiles", "prof")
        isd3_2, itd3_2, td3_2 = self.o_par2.get_ittime("profiles", "prof")

        if len(td3_1) == 0:
            Printcolor.red("D3 data not found for sim1:{}".format(self.sim1))
            return np.nan
        if len(td3_2) == 0:
            Printcolor.red("D3 data not found for sim2:{}".format(self.sim2))
            return np.nan

        tmerg1 = self.o_par1.get_par("tmerger")
        tmerg2 = self.o_par2.get_par("tmerger")

        Printcolor.blue("\ttd3_1[-1]:{} tmerg1:{} -> {}".format(td3_1[-1], tmerg1, td3_1[-1] - tmerg1))
        Printcolor.blue("\ttd3_2[-1]:{} tmerg2:{} -> {}".format(td3_2[-1], tmerg2, td3_2[-1] - tmerg2))

        td3_1 = np.array(td3_1 - tmerg1)
        td3_2 = np.array(td3_2 - tmerg2)

        if td3_1.min() > td3_2.max():
            Printcolor.red("D3 data does not overlap. sim1 has min:{} that is > than sim2 max: {}"
                           .format(td3_1.min(), td3_2.max()))
            return np.nan

        if td3_1.max() < td3_2.min():
            Printcolor.red("D3 data does not overlap. sim1 has max:{} that is < than sim2 min: {}"
                           .format(td3_1.max(), td3_2.min()))
            return np.nan

        tmax = np.min([td3_1.max(), td3_2.max()])
        Printcolor.blue("\ttmax for D3 data: {}".format(tmax))
        return float(tmax)

    def get_outflow_par_err(self, det, new_mask, v_n):

        o_par1 = ALL_PAR(self.sim1, add_mask=new_mask)
        o_par2 = ALL_PAR(self.sim2, add_mask=new_mask)

        val1 = o_par1.get_outflow_par(det, new_mask, v_n)
        val2 = o_par2.get_outflow_par(det, new_mask, v_n)

        # err = np.abs(val1 - val2) / val1

        return val1, val2

    # --- --- --- --- ---

    def get_outflow_pars(self, det, mask, v_n, rewrite=False):

        if mask == "geo":
            self.compute_outflow_new_mask(det, self.sim1, mask, rewrite=rewrite)
            self.compute_outflow_new_mask(det, self.sim2, mask, rewrite=rewrite)
            return self.get_outflow_par_err(det, mask, v_n)

        elif mask.__contains__("bern_"):
            delta_t = self.get_post_geo_delta_t(det)
            if not np.isnan(delta_t):
                mask = "bern_geoend" + "_length{:.0f}".format(delta_t * 1e5)  # [1e2 ms]
                self.compute_outflow_new_mask(det, self.sim1, mask, rewrite=rewrite)
                self.compute_outflow_new_mask(det, self.sim2, mask, rewrite=rewrite)
                return self.get_outflow_par_err(det, mask, v_n)
            else:
                return np.nan, np.nan
        else:
            raise NameError("No method exists for mask:{} ".format(mask))

    def get_3d_pars(self, v_n):
        td3 = self.get_tmax_d3_data()
        if not np.isnan(td3):
            tmerg1 = self.o_par1.get_par("tmerger")
            tmerg2 = self.o_par2.get_par("tmerger")
            print("\n{} and {}".format(td3+tmerg1, td3+tmerg2))
            val1 = self.o_par1.get_int_par(v_n, td3+tmerg1)
            val2 = self.o_par2.get_int_par(v_n, td3+tmerg2)
            return val1, val2
        else:
            return np.nan, np.nan

""" ==========================================| TABLES |============================================================="""

class TEX_TABLES:

    def __init__(self):
        self.sim_list = []

        # setting up parameters
        self.init_data_v_ns = ["EOS", "q", "note", "res", "vis"]
        self.init_data_prec = ["", ".1f", "", "", ""]
        #
        self.col_d3_gw_data_v_ns = ['tend', "tdisk3D", "Mdisk3D", 'tcoll_gw', "Mdisk"]
        self.col_d3_gw_data_prec = [".1f", ".1f",       ".2f",       ".2f",     ".2f"]
        #
        self.outflow_data_v_ns = ['Mej_tot', 'Ye_ave', 'vel_inf_ave', 'theta_rms',
                                  'Mej_tot', 'Ye_ave', 'vel_inf_ave', 'theta_rms']
        self.outflow_data_prec = [".2f", ".2f", ".2f", ".2f",
                                  ".2f", ".2f", ".2f", ".2f"]
        self.outflow_data_mask = ["geo", "geo", "geo", "geo",
                                  "bern_geoend", "bern_geoend", "bern_geoend", "bern_geoend"]


        pass

    # --- UNITS --- #

    @staticmethod
    def get_unit_lbl(v_n):
        if v_n in ["M1", "M2"]: return "$[M_{\odot}]$"
        elif v_n in ["Mej_tot"]: return "$[10^{-2} M_{\odot}]$"
        elif v_n in ["Mdisk3D", "Mdisk"]: return "$[M_{\odot}]$"
        elif v_n in ["vel_inf_ave"]: return "$[c]$"
        elif v_n in ["tcoll_gw", "tmerg_gw", "tmerg", "tcoll", "tend", "tdisk3D"]: return "[ms]"
        else:
            return " "

    # --- LABELS --- #
    @staticmethod
    def get_other_lbl(v_n):
        if v_n == "Mdisk3D": return r"$M_{\text{disk}} ^{\text{last}}$"
        elif v_n == "Mdisk": return r"$M_{\text{disk}} ^{\text{BH}}$"
        elif v_n == "M1": return "$M_a$"
        elif v_n == "M2": return "$M_b$"
        elif v_n == "tcoll_gw" or v_n == "tcoll": return r"$t_{\text{BH}}$"
        elif v_n == "tend": return r"$t_{\text{end}}$"
        elif v_n == "tdisk3D": return r"$t_{\text{disk}}$"
        elif v_n == "q": return r"$M_a/M_b$"
        elif v_n == "EOS": return r"EOS"
        elif v_n == "res": return r"res"
        elif v_n == "vis": return "LK"
        elif v_n == "note": return r"note"
        else:
            raise NameError("No label found for other v_n: {}".format(v_n))
    @staticmethod
    def get_outflow_lbl(v_n, mask):

        if v_n == "theta_rms" and mask=="geo": return "$\\langle \\theta_{\\text{ej}} \\rangle$"
        elif v_n == "Mej_tot" and mask=="geo": return "$M_{\\text{ej}}$"
        elif v_n == "Ye_ave" and mask=="geo": return "$\\langle Y_e \\rangle$"
        elif v_n == "vel_inf_ave" and mask=="geo": return "$\\langle \\upsilon_{\\text{ej}} \\rangle$"
        elif v_n == "theta_rms" and mask=="bern_geoend": return "$\\langle \\theta_{\\text{ej}}^{\\text{w}} \\rangle$"
        elif v_n == "Mej_tot" and mask=="bern_geoend": return "$M_{\\text{ej}}^{\\text{w}}$"
        elif v_n == "Ye_ave" and mask=="bern_geoend": return "$\\langle Y_e ^{\\text{w}}  \\rangle$"
        elif v_n == "vel_inf_ave" and mask=="bern_geoend": return "$\\langle \\upsilon_{\\text{ej}}^{\\text{w}} \\rangle$"
        elif v_n == "res": return "res"
        else:
            raise NameError("No label found for outflow v_n: {} and mask: {} ".format(v_n, mask))

    # --- DATA --- #

    def get_inital_data_val(self, o_data, v_n, prec):
        #
        if v_n == "note":
            eos = o_data.get_par("EOS")
            if eos == "DD2":
                val = o_data.get_par("run")
            elif eos == "SFHo":
                pizza = o_data.get_par("pizza_eos")
                if pizza.__contains__("2019"):
                    val = "pz19"
                else:
                    val = ""
            elif eos == "LS220":
                val = ""
            elif eos == "SLy4":
                val = ""
            elif eos == "BLh":
                val = ""
            else:
                raise NameError("no notes for EOS:{}".format(eos))

        else:
            val = o_data.get_par(v_n)
        #

        #
        if prec == "":
            return str(val)
        else:
            return ("%{}".format(prec) % float(val))

    def get_col_gw_d3_val(self, o_data, v_n, prec):

        val = o_data.get_par(v_n)

        if v_n == "tcoll_gw":
            tmerg = float(o_data.get_par("tmerger"))
            if np.isinf(val):
                tend = o_data.get_par("tend")
                # print(o_data.sim, tend, tmerg)
                assert tend > tmerg
                return str(r"$>{:.1f}$".format((tend-tmerg) * 1e3))
            else:
                print(val)
                # tcoll = o_data.get_par("tcoll")
                assert val > tmerg
                val = (val-tmerg) * 1e3

        if v_n == "tend":
            tmerg = o_data.get_par("tmerg")
            val = (val - tmerg) * 1e3

        if v_n == "Mdisk3D" or v_n == "Mdisk":
            if np.isnan(val):
                return r"N/A"
            else:
                val = val# * 1e2

        if v_n == "tdisk3D":

            if np.isnan(val):
                return r"N/A"
            else:
                tmerg = o_data.get_par("tmerg")
                val = (val - tmerg) * 1e3# * 1e2

        if prec == "":
            return str(val)
        else:
            return ("%{}".format(prec) % val)

    def get_ouflow_data(self, o_data, v_n, mask, prec):

        val = o_data.get_outflow_par(0, mask, v_n)
        if v_n == "Mej_tot":
            val = val * 1e2
            if mask == "bern_geoend":
                tcoll = o_data.get_par("tcoll")
                if np.isinf(tcoll):
                    return("$>~%{}$".format(prec) % val)
                else:
                    return("$\propto~%{}$".format(prec) % val)

        if prec == "":
            return str(val)
        else:
            return ("%{}".format(prec) % val)

    # --- MAIN --- #

    def get_table_size_head(self):

        print("\n")
        size = '{'
        head = ''
        i = 0

        all_v_ns = self.init_data_v_ns + self.col_d3_gw_data_v_ns + self.outflow_data_v_ns
        if len(self.init_data_v_ns) > 0:
            for init_data_v in self.init_data_v_ns:
                v_n = self.get_other_lbl(init_data_v)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if init_data_v != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1
        if len(self.col_d3_gw_data_v_ns) > 0:
            for other_v_n in self.col_d3_gw_data_v_ns:
                v_n = self.get_other_lbl(other_v_n)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if other_v_n != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1
        if len(self.outflow_data_v_ns) > 0:
            for outflow_v_n, outflow_mask in zip(self.outflow_data_v_ns, self.outflow_data_mask):
                v_n = self.get_outflow_lbl(outflow_v_n, outflow_mask)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if outflow_v_n != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1

        size = size + '}'

        head = head + ' \\\\'  # = \\

        return size, head

    def get_unit_bar(self):

        all_v_ns = self.init_data_v_ns + self.col_d3_gw_data_v_ns + self.outflow_data_v_ns

        unit_bar = ''
        for i, v_n in enumerate(all_v_ns):
            unit = self.get_unit_lbl(v_n)

            unit_bar = unit_bar + '{}'.format(unit)
            # if v_ns.index(v_n) != len(v_ns): unit_bar = unit_bar + ' & '
            if i != len(all_v_ns) - 1: unit_bar = unit_bar + ' & '

        unit_bar = unit_bar + ' \\\\ '

        return unit_bar

    def get_rows(self, sim_list):

        all_v_ns = self.init_data_v_ns + self.col_d3_gw_data_v_ns + self.outflow_data_v_ns

        rows = []
        for i, sim in enumerate(sim_list):
            row = ''
            j = 0
            # add init_data_val:
            if len(self.init_data_v_ns) > 0:
                o_init_data = LOAD_INIT_DATA(sim)
                for init_data_v, init_data_p in zip(self.init_data_v_ns, self.init_data_prec):
                    print("\tPrinting Initial Data {}".format(init_data_v))
                    val = self.get_inital_data_val(o_init_data, v_n=init_data_v, prec=init_data_p)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1

            # add coll gw d3 data:
            if len(self.col_d3_gw_data_v_ns) > 0 or len(self.outflow_data_v_ns) > 0:
                o_data = ALL_PAR(sim)
                for other_v_n, other_prec in zip(self.col_d3_gw_data_v_ns, self.col_d3_gw_data_prec):
                    print("\tPrinting Initial Data {}".format(other_v_n))
                    val = self.get_col_gw_d3_val(o_data, v_n=other_v_n, prec=other_prec)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1

                # add outflow data:
                for outflow_v_n, outflow_prec, outflow_mask in zip(self.outflow_data_v_ns, self.outflow_data_prec,
                                                                   self.outflow_data_mask):
                    print("\tPrinting Outflow Data {} (mask: {})".format(outflow_v_n, outflow_mask))
                    val = self.get_ouflow_data(o_data, v_n=outflow_v_n, mask=outflow_mask, prec=outflow_prec)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1
            row = row + ' \\\\'  # = \\
            rows.append(row)

        return rows

    def print_intro_table(self):

        size, head = self.get_table_size_head()
        unit_bar = self.get_unit_bar()

        print('\\begin{table*}[t]')
        print('\\begin{center}')
        print('\\begin{tabular}' + '{}'.format(size))
        print('\\hline')
        print(head)
        print(unit_bar)
        print('\\hline\\hline')

    def print_end_table(self):
        print('\\hline')
        print('\\end{tabular}')
        print('\\end{center}')
        print('\\caption{I am your table! }')
        print('\\label{tbl:1}')
        print('\\end{table*}')


    def print_one_table(self, sim_list, print_head=True, print_end=True):

        # setting up parameters
        init_data_v_ns = self.init_data_v_ns
        init_data_prec = self.init_data_prec
        #
        col_d3_gw_data_v_ns = self.col_d3_gw_data_v_ns
        col_d3_gw_data_prec = self.col_d3_gw_data_prec
        #
        outflow_data_v_ns = self.outflow_data_v_ns
        outflow_data_prec = self.outflow_data_prec
        outflow_data_mask = self.outflow_data_mask
        #
        assert len(init_data_prec) == len(init_data_v_ns)
        assert len(col_d3_gw_data_prec) == len(col_d3_gw_data_v_ns)
        assert len(outflow_data_mask) == len(outflow_data_prec)
        assert len(outflow_data_prec) == len(outflow_data_v_ns)
        #
        all_v_ns = init_data_v_ns + col_d3_gw_data_v_ns + outflow_data_v_ns
        #
        rows = []
        for i, sim in enumerate(sim_list):
            row = ''
            j = 0
            # add init_data_val:
            if len(init_data_v_ns) > 0:
                o_init_data = LOAD_INIT_DATA(sim)
                for init_data_v, init_data_p in zip(init_data_v_ns, init_data_prec):
                    print("\tPrinting Initial Data {}".format(init_data_v))
                    val = self.get_inital_data_val(o_init_data, v_n=init_data_v, prec=init_data_p)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1

            # add coll gw d3 data:
            if len(col_d3_gw_data_v_ns) > 0 or len(outflow_data_v_ns) > 0:
                o_data = ALL_PAR(sim)
                for other_v_n, other_prec in zip(col_d3_gw_data_v_ns, col_d3_gw_data_prec):
                    print("\tPrinting Initial Data {}".format(other_v_n))
                    val = self.get_col_gw_d3_val(o_data, v_n=other_v_n, prec=other_prec)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1

                # add outflow data:
                for outflow_v_n, outflow_prec, outflow_mask in zip(outflow_data_v_ns, outflow_data_prec, outflow_data_mask):
                    print("\tPrinting Outflow Data {} (mask: {})".format(outflow_v_n, outflow_mask))
                    val = self.get_ouflow_data(o_data, v_n=outflow_v_n, mask=outflow_mask, prec=outflow_prec)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1
            row = row + ' \\\\'  # = \\
            rows.append(row)

        # --- HEAD --- #

        print("\n")
        size = '{'
        head = ''
        i = 0
        if len(init_data_v_ns) > 0:
            for init_data_v in init_data_v_ns:
                v_n = self.get_other_lbl(init_data_v)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if init_data_v != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1
        if len(col_d3_gw_data_v_ns) > 0:
            for other_v_n in col_d3_gw_data_v_ns:
                v_n = self.get_other_lbl(other_v_n)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if other_v_n != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1
        if len(outflow_data_v_ns) > 0:
            for outflow_v_n, outflow_mask in zip(outflow_data_v_ns, outflow_data_mask):
                v_n = self.get_outflow_lbl(outflow_v_n, outflow_mask)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if outflow_v_n != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1

        size = size + '}'

        # --- UNIT BAR --- #

        unit_bar = ''
        for i, v_n in enumerate(all_v_ns):
            unit = self.get_unit_lbl(v_n)

            unit_bar = unit_bar + '{}'.format(unit)
            # if v_ns.index(v_n) != len(v_ns): unit_bar = unit_bar + ' & '
            if i != len(all_v_ns) - 1: unit_bar = unit_bar + ' & '

        head = head + ' \\\\'  # = \\
        unit_bar = unit_bar + ' \\\\ '

        # ====================== PRINT TABLE ================== #



        if print_head:
            print('\\begin{table*}[t]')
            print('\\begin{center}')
            print('\\begin{tabular}' + '{}'.format(size))
            print('\\hline')
            print(head)
            print(unit_bar)
            print('\\hline\\hline')

        for row in rows:
            print(row)

        if print_end:
            print('\\hline')
            print('\\end{tabular}')
            print('\\end{center}')
            print('\\caption{I am your table! }')
            print('\\label{tbl:1}')
            print('\\end{table*}')

        exit(0)

    def print_mult_table(self, list_simgroups, separateors):

        assert len(list_simgroups) == len(separateors)

        group_rows = []
        for sim_group in list_simgroups:
            rows = self.get_rows(sim_group)
            group_rows.append(rows)

        print("data colleted. Printing...")

        self.print_intro_table()
        for i in range(len(list_simgroups)):
            for row in group_rows[i]:
                print(row)
            print(separateors[i])
            # print("\\hline")

        self.print_end_table()

class COMPARISON_TABLE:

    def __init__(self):
        self.sim_list = []

        # setting up parameters
        self.init_data_v_ns = ["EOS", "q", "note", "res", "vis"]
        self.init_data_prec = ["", ".1f", "", "", ""]
        #
        self.col_d3_gw_data_v_ns = ["Mdisk3D", "tdisk3D"]
        self.col_d3_gw_data_prec = [".2f", ".1f"]
        #
        self.outflow_data_v_ns = ['Mej_tot', 'Ye_ave', 'vel_inf_ave', 'theta_rms', 'delta_t',
                                  'Mej_tot', 'Ye_ave', 'vel_inf_ave', 'theta_rms']
        self.outflow_data_prec = [".2f", ".2f", ".2f", ".2f", ".1f",
                                  ".2f", ".2f", ".2f", ".2f"]
        self.outflow_data_mask = ["geo", "geo", "geo", "geo", "bern_geoend",
                                  "bern_geoend", "bern_geoend", "bern_geoend", "bern_geoend"]
        #

        pass

    # --- UNITS --- #

    @staticmethod
    def get_unit_lbl(v_n):
        if v_n in ["M1", "M2"]:
            return "$[M_{\odot}]$"
        elif v_n in ["Mej_tot"]:
            return "$[10^{-2} M_{\odot}]$"
        elif v_n in ["Mdisk3D", "Mdisk"]:
            return "$[M_{\odot}]$"
        elif v_n in ["vel_inf_ave"]:
            return "$[c]$"
        elif v_n in ["tcoll_gw", "tmerg_gw", "tmerg", "tcoll", "tend", "tdisk3D", "delta_t"]:
            return "[ms]"
        else:
            return " "

    # --- LABELS --- #
    @staticmethod
    def get_other_lbl(v_n):
        if v_n == "Mdisk3D":
            return r"$M_{\text{disk}} ^{\text{last}}$"
        elif v_n == "Mdisk":
            return r"$M_{\text{disk}} ^{\text{BH}}$"
        elif v_n == "M1":
            return "$M_a$"
        elif v_n == "M2":
            return "$M_b$"
        elif v_n == "tcoll_gw" or v_n == "tcoll":
            return r"$t_{\text{BH}}$"
        elif v_n == "tend":
            return r"$t_{\text{end}}$"
        elif v_n == "tdisk3D":
            return r"$t_{\text{disk}}$"
        elif v_n == "q":
            return r"$M_a/M_b$"
        elif v_n == "EOS":
            return r"EOS"
        elif v_n == "res":
            return r"res"
        elif v_n == "vis":
            return "LK"
        elif v_n == "note":
            return r"note"
        else:
            raise NameError("No label found for other v_n: {}".format(v_n))

    @staticmethod
    def get_outflow_lbl(v_n, mask):

        if v_n == "theta_rms" and mask == "geo":
            return "$\\langle \\theta_{\\text{ej}} \\rangle$"
        elif v_n == "Mej_tot" and mask == "geo":
            return "$M_{\\text{ej}}$"
        elif v_n == "Ye_ave" and mask == "geo":
            return "$\\langle Y_e \\rangle$"
        elif v_n == "vel_inf_ave" and mask == "geo":
            return "$\\langle \\upsilon_{\\text{ej}} \\rangle$"
        elif v_n == "theta_rms" and mask == "bern_geoend":
            return "$\\langle \\theta_{\\text{ej}}^{\\text{w}} \\rangle$"
        elif v_n == "Mej_tot" and mask == "bern_geoend":
            return "$M_{\\text{ej}}^{\\text{w}}$"
        elif v_n == "Ye_ave" and mask == "bern_geoend":
            return "$\\langle Y_e ^{\\text{w}}  \\rangle$"
        elif v_n == "vel_inf_ave" and mask == "bern_geoend":
            return "$\\langle \\upsilon_{\\text{ej}}^{\\text{w}} \\rangle$"
        elif v_n == "delta_t" and mask.__contains__("geoend"):
            return r"$\Delta t_{\text{wind}}$"
        elif v_n == "res":
            return "res"
        else:
            raise NameError("No label found for outflow v_n: {} and mask: {} ".format(v_n, mask))

    # --- DATA --- #

    def get_inital_data_val(self, o_data, v_n, prec):
        #
        if v_n == "note":
            eos = o_data.get_par("EOS")
            if eos == "DD2":
                val = o_data.get_par("run")
            elif eos == "SFHo":
                pizza = o_data.get_par("pizza_eos")
                if pizza.__contains__("2019"):
                    val = "pz19"
                else:
                    val = ""
            elif eos == "LS220":
                val = ""
            elif eos == "SLy4":
                val = ""
            elif eos == "BLh":
                val = ""
            else:
                raise NameError("no notes for EOS:{}".format(eos))

        else:
            val = o_data.get_par(v_n)
        #

        #
        if prec == "":
            return str(val)
        else:
            return ("%{}".format(prec) % float(val))

    def get_col_gw_d3_val(self, o_data, v_n, prec):

        val = o_data.get_par(v_n)

        if v_n == "tcoll_gw":
            tmerg = float(o_data.get_par("tmerger"))
            if np.isinf(val):
                tend = o_data.get_par("tend")
                # print(o_data.sim, tend, tmerg)
                assert tend > tmerg
                return str(r"$>{:.1f}$".format((tend - tmerg) * 1e3))
            else:
                print(val)
                # tcoll = o_data.get_par("tcoll")
                assert val > tmerg
                val = (val - tmerg) * 1e3

        if v_n == "tend":
            tmerg = o_data.get_par("tmerg")
            val = (val - tmerg) * 1e3

        if v_n == "Mdisk3D" or v_n == "Mdisk":
            if np.isnan(val):
                return r"N/A"
            else:
                val = val  # * 1e2

        if v_n == "tdisk3D":

            if np.isnan(val):
                return r"N/A"
            else:
                tmerg = o_data.get_par("tmerg")
                val = (val - tmerg) * 1e3  # * 1e2

        if prec == "":
            return str(val)
        else:
            return ("%{}".format(prec) % val)

    def get_ouflow_data(self, o_data, v_n, mask, prec):

        val = o_data.get_outflow_par(0, mask, v_n)
        if v_n == "Mej_tot":
            val = val * 1e2
            if mask == "bern_geoend":
                tcoll = o_data.get_par("tcoll")
                if np.isinf(tcoll):
                    return ("$>~%{}$".format(prec) % val)
                else:
                    return ("$\propto~%{}$".format(prec) % val)

        if prec == "":
            return str(val)
        else:
            return ("%{}".format(prec) % val)

    # ---- Comparison Data ---

    def get_comp_other_data(self, o_2sim, v_n, prec):

        val1, val2 = o_2sim.get_3d_pars(v_n)

        if v_n == "Mdisk3D" or v_n == "Mdisk":
            if np.isnan(val1) or np.isnan(val2):
                err = "N/A"
            else:
                err = 100 * (val1 - val2) / val1
                err = "{:.0f}".format(err)
            #
            if np.isnan(val1):
                res1 = "N/A"
            else:
                res1 = "%{}".format(prec) % val1
            #
            if np.isnan(val2):
                res2 = "N/A"
            else:
                res2 = "%{}".format(prec) % val2
            #
            return res1, res2, err

        elif v_n == "tdisk3D" or v_n == "tdisk3D":

            val = o_2sim.get_tmax_d3_data()
            if np.isnan(val):
                res1 = res2 = "N/A"
            else:
                res1 = res2 = "%{}".format(prec) % (val * 1e3)
            #
            return res1, res2, " "

        else:
            raise NameError("np method for comp_other_data v_n:{} is set".format(v_n))

    def get_comp_ouflow_data(self, o_2sim, v_n, mask, prec):

        if v_n == "delta_t" and mask.__contains__("geoend"):
            val = o_2sim.get_post_geo_delta_t(0)
            if np.isnan(val):
                res1 = res2 = "N/A"
            else:
                res1 = res2 = "%{}".format(prec) % (val * 1e3)
            return res1, res2, ""

        val1, val2 = o_2sim.get_outflow_pars(0, mask, v_n, rewrite=False)

        if v_n == "Mej_tot" or v_n == "Mej_tot":
            if np.isnan(val1) or np.isnan(val2):
                err = "N/A"
            else:
                err = 100 * (val1 - val2) / val1
                err = "{:.0f}".format(err)
            #
            if np.isnan(val1):
                res1 = "N/A"
            else:
                res1 = "%{}".format(prec) % (val1 * 1e2)
            #
            if np.isnan(val2):
                res2 = "N/A"
            else:
                res2 = "%{}".format(prec) % (val2 * 1e2)
            #
            return res1, res2, err


        elif v_n in ['Ye_ave', 'vel_inf_ave', 'theta_rms']:
            if np.isnan(val1) or np.isnan(val2):
                err = "N/A"
            else:
                err = 100 * (val1 - val2) / val1
                err = "{:.0f}".format(err)
            #
            if np.isnan(val1):
                res1 = "N/A"
            else:
                res1 = "%{}".format(prec) % val1
            #
            if np.isnan(val2):
                res2 = "N/A"
            else:
                res2 = "%{}".format(prec) % val2
            #
            return res1, res2, err
        else:
            raise NameError("no method setup for geting a str(val1, val2, err) for v_n:{} mask:{}"
                            .format(v_n, mask))

    # --- MAIN --- #

    def get_table_size_head(self):

        print("\n")
        size = '{'
        head = ''
        i = 0

        all_v_ns = self.init_data_v_ns + self.col_d3_gw_data_v_ns + self.outflow_data_v_ns
        if len(self.init_data_v_ns) > 0:
            for init_data_v in self.init_data_v_ns:
                v_n = self.get_other_lbl(init_data_v)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if init_data_v != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1
        if len(self.col_d3_gw_data_v_ns) > 0:
            for other_v_n in self.col_d3_gw_data_v_ns:
                v_n = self.get_other_lbl(other_v_n)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if other_v_n != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1
        if len(self.outflow_data_v_ns) > 0:
            for outflow_v_n, outflow_mask in zip(self.outflow_data_v_ns, self.outflow_data_mask):
                v_n = self.get_outflow_lbl(outflow_v_n, outflow_mask)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if outflow_v_n != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1

        size = size + '}'

        head = head + ' \\\\'  # = \\

        return size, head

    def get_unit_bar(self):

        all_v_ns = self.init_data_v_ns + self.col_d3_gw_data_v_ns + self.outflow_data_v_ns

        unit_bar = ''
        for i, v_n in enumerate(all_v_ns):
            unit = self.get_unit_lbl(v_n)

            unit_bar = unit_bar + '{}'.format(unit)
            # if v_ns.index(v_n) != len(v_ns): unit_bar = unit_bar + ' & '
            if i != len(all_v_ns) - 1: unit_bar = unit_bar + ' & '

        unit_bar = unit_bar + ' \\\\ '

        return unit_bar

    def get_rows(self, sim_list):

        all_v_ns = self.init_data_v_ns + self.col_d3_gw_data_v_ns + self.outflow_data_v_ns

        rows = []

        for i, sim in enumerate(sim_list):
            row = ''
            j = 0
            # add init_data_val:
            if len(self.init_data_v_ns) > 0:
                o_init_data = LOAD_INIT_DATA(sim)
                for init_data_v, init_data_p in zip(self.init_data_v_ns, self.init_data_prec):
                    print("\tPrinting Initial Data {}".format(init_data_v))
                    val = self.get_inital_data_val(o_init_data, v_n=init_data_v, prec=init_data_p)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1

            # add coll gw d3 data:
            if len(self.col_d3_gw_data_v_ns) > 0 or len(self.outflow_data_v_ns) > 0:
                o_data = ALL_PAR(sim)
                for other_v_n, other_prec in zip(self.col_d3_gw_data_v_ns, self.col_d3_gw_data_prec):
                    print("\tPrinting Initial Data {}".format(other_v_n))
                    val = self.get_col_gw_d3_val(o_data, v_n=other_v_n, prec=other_prec)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1

                # add outflow data:
                for outflow_v_n, outflow_prec, outflow_mask in zip(self.outflow_data_v_ns, self.outflow_data_prec,
                                                                   self.outflow_data_mask):
                    print("\tPrinting Outflow Data {} (mask: {})".format(outflow_v_n, outflow_mask))
                    val = self.get_ouflow_data(o_data, v_n=outflow_v_n, mask=outflow_mask, prec=outflow_prec)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1
            row = row + ' \\\\'  # = \\
            rows.append(row)

        return rows

    def get_compartison_rows(self, two_sims):

        assert len(two_sims) == 2

        all_v_ns = self.init_data_v_ns + self.col_d3_gw_data_v_ns + self.outflow_data_v_ns

        # rows = []

        # o_2sim = TWO_SIMS(two_sims[0], two_sims[1])

        # if len(self.init_data_v_ns) > 0:
        #     o_init_data = LOAD_INIT_DATA(sim)
        #     for init_data_v, init_data_p in zip(self.init_data_v_ns, self.init_data_prec):
        #         print("\tPrinting Initial Data {}".format(init_data_v))
        #         val = self.get_inital_data_val(o_init_data, v_n=init_data_v, prec=init_data_p)
        #         row = row + val
        #         if j != len(all_v_ns) - 1: row = row + ' & '
        #         j = j + 1

        row1 = ''
        row2 = ''
        row3 = ''
        j = 0
        if len(self.init_data_v_ns) > 0:
            o_init_data1 = LOAD_INIT_DATA(two_sims[0])
            o_init_data2 = LOAD_INIT_DATA(two_sims[1])
            for init_data_v, init_data_p in zip(self.init_data_v_ns, self.init_data_prec):
                print("\tPrinting Initial Data {}".format(init_data_v))
                val1 = self.get_inital_data_val(o_init_data1, v_n=init_data_v, prec=init_data_p)
                val2 = self.get_inital_data_val(o_init_data2, v_n=init_data_v, prec=init_data_p)
                row1 = row1 + val1
                row2 = row2 + val2
                if init_data_v == self.init_data_v_ns[0]:
                    row3 = row3 + r"$\Delta$"
                else:
                    row3 = row3 + ""
                if j != len(all_v_ns) - 1: row1 = row1 + ' & '
                if j != len(all_v_ns) - 1: row2 = row2 + ' & '
                if j != len(all_v_ns) - 1: row3 = row3 + ' & '
                j = j + 1

        if len(self.col_d3_gw_data_v_ns) > 0 or len(self.outflow_data_v_ns) > 0:
            o_2sim = TWO_SIMS(two_sims[0], two_sims[1])
            for other_v_n, other_prec in zip(self.col_d3_gw_data_v_ns, self.col_d3_gw_data_prec):
                print("\tPrinting Other Data {}".format(other_v_n))
                val1, val2, err = self.get_comp_other_data(o_2sim, v_n=other_v_n, prec=other_prec)
                row1 = row1 + val1
                row2 = row2 + val2
                row3 = row3 + err
                if j != len(all_v_ns) - 1: row1 = row1 + ' & '
                if j != len(all_v_ns) - 1: row2 = row2 + ' & '
                if j != len(all_v_ns) - 1: row3 = row3 + ' & '
                j = j + 1

            for outflow_v_n, outflow_prec, outflow_mask in zip(self.outflow_data_v_ns, self.outflow_data_prec,
                                                               self.outflow_data_mask):
                print("\tPrinting Outflow Data {} (mask: {})".format(outflow_v_n, outflow_mask))
                val1, val2, err = self.get_comp_ouflow_data(o_2sim, v_n=outflow_v_n, mask=outflow_mask,
                                                           prec=outflow_prec)
                row1 = row1 + val1
                row2 = row2 + val2
                row3 = row3 + err
                if j != len(all_v_ns) - 1: row1 = row1 + ' & '
                if j != len(all_v_ns) - 1: row2 = row2 + ' & '
                if j != len(all_v_ns) - 1: row3 = row3 + ' & '
                j = j + 1

            row1 = row1 + ' \\\\'  # = \\
            row2 = row2 + ' \\\\'  # = \\
            row3 = row3 + ' \\\\'  # = \\
        rows = [row1, row2, row3]

        return rows

    def print_intro_table(self):

        size, head = self.get_table_size_head()
        unit_bar = self.get_unit_bar()

        print('\\begin{table*}[t]')
        print('\\begin{center}')
        print('\\begin{tabular}' + '{}'.format(size))
        print('\\hline')
        print(head)
        print(unit_bar)
        print('\\hline\\hline')

    def print_end_table(self, comment):
        print(r'\hline')
        print(r'\end{tabular}')
        print(r'\end{center}')
        print(r'\caption{}'.format(comment))
        print(r'\label{tbl:1}')
        print(r'\end{table*}')

    def print_one_table(self, sim_list, print_head=True, print_end=True):

        # setting up parameters
        init_data_v_ns = self.init_data_v_ns
        init_data_prec = self.init_data_prec
        #
        col_d3_gw_data_v_ns = self.col_d3_gw_data_v_ns
        col_d3_gw_data_prec = self.col_d3_gw_data_prec
        #
        outflow_data_v_ns = self.outflow_data_v_ns
        outflow_data_prec = self.outflow_data_prec
        outflow_data_mask = self.outflow_data_mask
        #
        assert len(init_data_prec) == len(init_data_v_ns)
        assert len(col_d3_gw_data_prec) == len(col_d3_gw_data_v_ns)
        assert len(outflow_data_mask) == len(outflow_data_prec)
        assert len(outflow_data_prec) == len(outflow_data_v_ns)
        #
        all_v_ns = init_data_v_ns + col_d3_gw_data_v_ns + outflow_data_v_ns
        #
        rows = []
        for i, sim in enumerate(sim_list):
            row = ''
            j = 0
            # add init_data_val:
            if len(init_data_v_ns) > 0:
                o_init_data = LOAD_INIT_DATA(sim)
                for init_data_v, init_data_p in zip(init_data_v_ns, init_data_prec):
                    print("\tPrinting Initial Data {}".format(init_data_v))
                    val = self.get_inital_data_val(o_init_data, v_n=init_data_v, prec=init_data_p)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1

            # add coll gw d3 data:
            if len(col_d3_gw_data_v_ns) > 0 or len(outflow_data_v_ns) > 0:
                o_data = ALL_PAR(sim)
                for other_v_n, other_prec in zip(col_d3_gw_data_v_ns, col_d3_gw_data_prec):
                    print("\tPrinting Initial Data {}".format(other_v_n))
                    val = self.get_col_gw_d3_val(o_data, v_n=other_v_n, prec=other_prec)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1

                # add outflow data:
                for outflow_v_n, outflow_prec, outflow_mask in zip(outflow_data_v_ns, outflow_data_prec,
                                                                   outflow_data_mask):
                    print("\tPrinting Outflow Data {} (mask: {})".format(outflow_v_n, outflow_mask))
                    val = self.get_ouflow_data(o_data, v_n=outflow_v_n, mask=outflow_mask, prec=outflow_prec)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1
            row = row + ' \\\\'  # = \\
            rows.append(row)

        # --- HEAD --- #

        print("\n")
        size = '{'
        head = ''
        i = 0
        if len(init_data_v_ns) > 0:
            for init_data_v in init_data_v_ns:
                v_n = self.get_other_lbl(init_data_v)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if init_data_v != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1
        if len(col_d3_gw_data_v_ns) > 0:
            for other_v_n in col_d3_gw_data_v_ns:
                v_n = self.get_other_lbl(other_v_n)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if other_v_n != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1
        if len(outflow_data_v_ns) > 0:
            for outflow_v_n, outflow_mask in zip(outflow_data_v_ns, outflow_data_mask):
                v_n = self.get_outflow_lbl(outflow_v_n, outflow_mask)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if outflow_v_n != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1

        size = size + '}'

        # --- UNIT BAR --- #

        unit_bar = ''
        for i, v_n in enumerate(all_v_ns):
            unit = self.get_unit_lbl(v_n)

            unit_bar = unit_bar + '{}'.format(unit)
            # if v_ns.index(v_n) != len(v_ns): unit_bar = unit_bar + ' & '
            if i != len(all_v_ns) - 1: unit_bar = unit_bar + ' & '

        head = head + ' \\\\'  # = \\
        unit_bar = unit_bar + ' \\\\ '

        # ====================== PRINT TABLE ================== #

        if print_head:
            print('\\begin{table*}[t]')
            print('\\begin{center}')
            print('\\begin{tabular}' + '{}'.format(size))
            print('\\hline')
            print(head)
            print(unit_bar)
            print('\\hline\\hline')

        for row in rows:
            print(row)

        if print_end:
            print('\\hline')
            print('\\end{tabular}')
            print('\\end{center}')
            print('\\caption{I am your table! }')
            print('\\label{tbl:1}')
            print('\\end{table*}')

        exit(0)

    def print_mult_table(self, list_simgroups, separateors, comment):

        assert len(list_simgroups) == len(separateors)

        group_rows = []
        for sim_group in list_simgroups:
            rows = self.get_compartison_rows(sim_group)
            group_rows.append(rows)

        print("data colleted. Printing...")

        self.print_intro_table()
        for i in range(len(list_simgroups)):
            # print(len(group_rows[i])); exit(1)
            for i_row, row in enumerate(group_rows[i]):
                print(row)
            print(separateors[i])
            # print("\\hline")

        self.print_end_table(comment)

""" ================================================================================================================ """


class ErrorTexTables:

    def __init__(self, ind_dic, comb_dic):


        # sims:[sim1, sim2], masks:[mask1,mask2], ...
        self.ind_dic = dict(ind_dic)

        # mask:{v_n1:[val1, val2, err], v_n2:[val1, val2, err]...} mask2:{}...
        self.comb_dic = dict(comb_dic)

    @staticmethod
    def get_lbl(v_n, mask=""):

        if mask == "":
            if v_n == "Mdisk3D":
                return r"$M_{\text{disk}} ^{\text{last}}$"
            elif v_n == "Mdisk":
                return r"$M_{\text{disk}} ^{\text{BH}}$"
            elif v_n == "M1":
                return "$M_a$"
            elif v_n == "M2":
                return "$M_b$"
            elif v_n == "tcoll_gw" or v_n == "tcoll":
                return r"$t_{\text{BH}}$"
            elif v_n == "tend":
                return r"$t_{\text{end}}$"
            elif v_n == "tdisk3D":
                return r"$t_{\text{disk}}$"
            elif v_n == "q":
                return r"$M_a/M_b$"
            elif v_n == "EOS":
                return r"EOS"
            elif v_n == "res":
                return r"res"
            elif v_n == "vis":
                return "LK"
            elif v_n == "note":
                return r"note"
        else:
            if mask == "geo":
                if v_n == "theta_rms":
                    return "$\\langle \\theta_{\\text{ej}} \\rangle$"
                elif v_n == "Mej_tot":
                    return "$M_{\\text{ej}}$"
                elif v_n == "Ye_ave":
                    return "$\\langle Y_e \\rangle$"
                elif v_n == "vel_inf_ave":
                    return "$\\langle \\upsilon_{\\text{ej}} \\rangle$"
                else:
                    return v_n
            elif mask.__contains__("bern_"):
                if v_n == "theta_rms": return "$\\langle \\theta_{\\text{ej}}^{\\text{w}} \\rangle$"
                elif v_n == "Mej_tot": return "$M_{\\text{ej}}^{\\text{w}}$"
                elif v_n == "Ye_ave": return "$\\langle Y_e ^{\\text{w}}  \\rangle$"
                elif v_n == "vel_inf_ave": return "$\\langle \\upsilon_{\\text{ej}}^{\\text{w}} \\rangle$"
                else: return v_n
            else:
                raise NameError("No label for v_n: {} and mask: {}"
                                .format(v_n, mask))

    @staticmethod
    def get_unit_lbl(v_n):
        if v_n in ["M1", "M2"]: return "$[M_{\odot}]$"
        elif v_n in ["Mej_tot"]: return "$[10^{-2} M_{\odot}]$"
        elif v_n in ["Mdisk3D", "Mdisk"]: return "$[M_{\odot}]$"
        elif v_n in ["vel_inf_ave"]: return "$[c]$"
        elif v_n in ["tcoll_gw", "tmerg_gw", "tmerg", "tcoll", "tend", "tdisk3D"]: return "[ms]"
        else:
            return " "


    def print_table(self):



        comb_cols = ["sims", "t98mass"]
        ind_cols = []

        for mask in self.comb_dic.keys():
            for v_n in self.comb_dic[mask].keys():
                ind_cols.append(self.comb_dic[mask][v_n])
        cols = comb_cols + ind_cols
        #
        masks = ["" for i in range(len(cols))]
        for mask in self.comb_dic.keys():
            for v_n in self.comb_dic[mask].keys():
                ind_cols.append(self.comb_dic[mask][v_n])

        size = '{'
        head = ''
        for i, v_n in enumerate(cols):
            v_n =  self.get_lbl(v_n)
            size = size + 'c'
            head = head + '{}'.format(v_n)
            if v_n != cols[-1]: size = size + ' '
            if i != len(cols) - 1: head = head + ' & '
        size = size + '}'

        unit_bar = ''
        for v_n in cols:
            unit = self.get_unit_lbl(v_n)
            unit_bar = unit_bar + '{}'.format(unit)
            if v_n != cols[-1]: unit_bar = unit_bar + ' & '

        head = head + ' \\\\'  # = \\
        unit_bar = unit_bar + ' \\\\ '

        print('\n')

        print('\\begin{table*}[t]')
        print('\\begin{center}')
        print('\\begin{tabular}' + '{}'.format(size))
        print('\\hline')
        print(head)
        print(unit_bar)
        print('\\hline\\hline')



def __err_lk_onoff(mask, det = 0):

    simdic = sims_err_lk_onoff

    errs = {}
    for sim1, mask1, sim2, mask2 in zip(simdic["def"], mask, simdic["comp"], mask):

        errs[sim1] = {}

        print(" --------------| {} |---------------- ".format(sim1.split('_')[0]))

        # loading times
        fpath1 = Paths.ppr_sims + sim1 + "/" + "outflow_{}/".format(det) + mask1 + '/' + "total_flux.dat"
        if not os.path.isfile(fpath1):
            raise IOError("File does not exist: {}".format(fpath1))

        timearr1, massarr1 = np.loadtxt(fpath1, usecols=(0, 2), unpack=True)

        # loading tmerg
        fpath1 = Paths.ppr_sims + sim1 + "/" + "waveforms/" + "tmerger.dat"
        if not os.path.isfile(fpath1):
            raise IOError("File does not exist: {}".format(fpath1))
        tmerg1 = np.float(np.loadtxt(fpath1, unpack=True))
        timearr1 = timearr1 - (tmerg1 * Constants.time_constant * 1e-3)

        # loading times
        fpath2 = Paths.ppr_sims + sim2 + "/" + "outflow_{}/".format(det) + mask2 + '/' + "total_flux.dat"
        if not os.path.isfile(fpath2):
            raise IOError("File does not exist: {}".format(fpath2))

        timearr2, massarr2 = np.loadtxt(fpath2, usecols=(0, 2), unpack=True)

        # loading tmerg
        fpath2 = Paths.ppr_sims + sim2 + "/" + "waveforms/" + "tmerger.dat"
        if not os.path.isfile(fpath2):
            raise IOError("File does not exist: {}".format(fpath2))
        tmerg2 = np.float(np.loadtxt(fpath2, unpack=True))
        timearr2 = timearr2 - (tmerg2 * Constants.time_constant * 1e-3)

        # estimating tmax
        tmax = np.array([timearr1[-1], timearr2[-1]]).min()
        assert tmax <= timearr1.max()
        assert tmax <= timearr2.max()
        m1 = massarr1[UTILS.find_nearest_index(timearr1, tmax)]
        m2 = massarr2[UTILS.find_nearest_index(timearr2, tmax)]

        # print(" --------------| {} |---------------- ".format(sim1.split('_')[0]))
        print(" tmax:         {:.1f} [ms]".format(tmax*1e3))
        # print(" \n")
        print(" sim1:         {} ".format(sim1))
        print(" timearr1[-1]: {:.1f} [ms]".format(timearr1[-1]*1e3))
        print(" mass1[-1]     {:.2f} [1e-2Msun]".format(massarr1[-1]*1e2))
        print(" m1[tmax]      {:.2f} [1e-2Msun]".format(m1 * 1e2))
        # print(" \n")
        print(" sim1:         {} ".format(sim2))
        print(" timearr1[-1]: {:.1f} [ms]".format(timearr2[-1]*1e3))
        print(" mass1[-1]     {:.2f} [1e-2Msun]".format(massarr2[-1]*1e2))
        print(" m2[tmax]      {:.2f} [1e-2Msun]".format(m2 * 1e2))
        # print(" \n")
        print(" abs(m1-m2)/m1 {:.1f} [%]".format(100 * np.abs(m1 - m2) / m1))
        print(" ---------------------------------------- ")

        errs[sim1]["sim1"] = sim1
        errs[sim1]["sim2"] = sim2
        errs[sim1]["tmax"] = tmax*1e3
        errs[sim1]["m1"] = m1*1e2
        errs[sim1]["m2"] = m2*1e2
        errs[sim1]["err"] = 100 * np.abs(m1 - m2) / m1

    return errs

def err_lk_onoff(det=0):

    geo_errs = __err_lk_onoff("geo", det)
    bern_errs = __err_lk_onoff("bern_geoend", det)

    cols = ["sim1", "sim2", "m1_geo", "m2_geo", "tmax_geo", "err_geo",  "m1_bern", "m2_bern", "tmax_bern", "err_bern"]
    units_dic = {"sim1": "", "sim2": "", "m1_geo": "$[10^{-2} M_{\odot}]$", "m2_geo": "$[10^{-2} M_{\odot}]$", "tmax_geo": "[ms]",
                 "err_geo": r"[\%]", "m1_bern": "$[10^{-2} M_{\odot}]$", "m2_bern": "$[10^{-2} M_{\odot}]$", "tmax_bern": "[ms]",
                 "err_bern": r"[\%]"}

    lbl_dic = {"sim1": "Default Run", "sim2": "Comparison Run", "m1_geo": r"$M_{\text{ej}}^a$", "m2_geo": r"$M_{\text{ej}}^b$",
               "tmax_geo": r"$t_{\text{max}}$", "err_geo": r"$\Delta$", "m1_bern": r"$M_{\text{ej}}^a$", "m2_bern": r"$M_{\text{ej}}^b$",
               "tmax_bern": r"$t_{\text{max}}$", "err_bern": r"$\Delta$"}
    precs = ["", "", ".2f", ".2f", ".1f", "d"]

    size = '{'
    head = ''
    for i, v_n in enumerate(cols):
        v_n = lbl_dic[v_n]
        size = size + 'c'
        head = head + '{}'.format(v_n)
        if v_n != cols[-1]: size = size + ' '
        if i != len(cols) - 1: head = head + ' & '
    size = size + '}'

    unit_bar = ''
    for v_n in cols:
        if v_n in units_dic.keys():
            unit = units_dic[v_n]
        else:
            unit = v_n
        unit_bar = unit_bar + '{}'.format(unit)
        if v_n != cols[-1]: unit_bar = unit_bar + ' & '

    head = head + ' \\\\'  # = \\
    unit_bar = unit_bar + ' \\\\ '


    print('\n')

    print('\\begin{table*}[t]')
    print('\\begin{center}')
    print('\\begin{tabular}' + '{}'.format(size))
    print('\\hline')
    print(head)
    print(unit_bar)
    print('\\hline\\hline')

    for sim1, mask1, sim2, mask2 in zip(simdic["def"], mask, simdic["comp"], mask):
        row = ''
        for v_n, prec in zip(cols, precs):

            if prec != "":
                val = "%{}".format(prec) % errs[sim1][v_n]
            else:
                val = errs[sim1][v_n].replace("_", "\_")
            row = row + val
            if v_n != cols[-1]: row = row + ' & '
        row = row + ' \\\\'  # = \\
        print(row)

    print(r'\hline')
    print(r'\end{tabular}')
    print(r'\end{center}')
    print(r'\caption{' + r'Viscosity effect on the ejected material total cumulative mass. Criterion {} '
          .format(mask.replace('_', '\_')) +
          r'$\Delta = |M_{\text{ej}}^a - M_{\text{ej}}^b| / M_{\text{ej}}^a |_{tmax} $ }')
    print(r'\label{tbl:1}')
    print(r'\end{table*}')

    exit(1)





def table_err_total_fluxes_lk_on_off(mask, det=0):

    sims = ["DD2_M13641364_M0_LK_SR_R04", "DD2_M15091235_M0_LK_SR", "LS220_M14691268_M0_LK_SR",
            "SFHo_M14521283_M0_LK_SR"]
    lbls = ["DD2 136 136 LK", "DD2 151 123 LK", "LS220 147 127 LK", "SFHo 145 128 LK"]
    masks = [mask, mask, mask, mask]
    colors = ["black", 'gray', 'red', "green"]
    lss = ["-", '-', '-', '-']
    # minus LK
    sims2 = ["DD2_M13641364_M0_SR_R04", "DD2_M14971245_M0_SR", "LS220_M14691268_M0_SR", "SFHo_M14521283_M0_SR"]
    lbls2 = ["DD2 136 136", "DD2 150 125", "LS220 147 127", "SFHo 145 128"]
    masks2 = [mask, mask, mask, mask]
    colors2 = ["black", 'gray', 'red', "green"]
    lss2 = ["--", '--', '--', '--']

    sims += sims2
    lbls += lbls2
    masks += masks2
    colors += colors2
    lss += lss2

    # ---------------------

    errs = {}

    for sim1, mask1, sim2, mask2 in zip(sims, masks, sims2, masks2):

        errs[sim1] = {}

        print(" --------------| {} |---------------- ".format(sim1.split('_')[0]))

        # loading times
        fpath1 = Paths.ppr_sims + sim1 + "/" + "outflow_{}/".format(det) + mask1 + '/' + "total_flux.dat"
        if not os.path.isfile(fpath1):
            raise IOError("File does not exist: {}".format(fpath1))

        timearr1, massarr1 = np.loadtxt(fpath1, usecols=(0, 2), unpack=True)

        # loading tmerg
        fpath1 = Paths.ppr_sims + sim1 + "/" + "waveforms/" + "tmerger.dat"
        if not os.path.isfile(fpath1):
            raise IOError("File does not exist: {}".format(fpath1))
        tmerg1 = np.float(np.loadtxt(fpath1, unpack=True))
        timearr1 = timearr1 - (tmerg1 * Constants.time_constant * 1e-3)

        # loading times
        fpath2 = Paths.ppr_sims + sim2 + "/" + "outflow_{}/".format(det) + mask2 + '/' + "total_flux.dat"
        if not os.path.isfile(fpath2):
            raise IOError("File does not exist: {}".format(fpath2))

        timearr2, massarr2 = np.loadtxt(fpath2, usecols=(0, 2), unpack=True)

        # loading tmerg
        fpath2 = Paths.ppr_sims + sim2 + "/" + "waveforms/" + "tmerger.dat"
        if not os.path.isfile(fpath2):
            raise IOError("File does not exist: {}".format(fpath2))
        tmerg2 = np.float(np.loadtxt(fpath2, unpack=True))
        timearr2 = timearr2 - (tmerg2 * Constants.time_constant * 1e-3)

        # estimating tmax
        tmax = np.array([timearr1[-1], timearr2[-1]]).min()
        assert tmax <= timearr1.max()
        assert tmax <= timearr2.max()
        m1 = massarr1[UTILS.find_nearest_index(timearr1, tmax)]
        m2 = massarr2[UTILS.find_nearest_index(timearr2, tmax)]

        # print(" --------------| {} |---------------- ".format(sim1.split('_')[0]))
        print(" tmax:         {:.1f} [ms]".format(tmax*1e3))
        # print(" \n")
        print(" sim1:         {} ".format(sim1))
        print(" timearr1[-1]: {:.1f} [ms]".format(timearr1[-1]*1e3))
        print(" mass1[-1]     {:.2f} [1e-2Msun]".format(massarr1[-1]*1e2))
        print(" m1[tmax]      {:.2f} [1e-2Msun]".format(m1 * 1e2))
        # print(" \n")
        print(" sim1:         {} ".format(sim2))
        print(" timearr1[-1]: {:.1f} [ms]".format(timearr2[-1]*1e3))
        print(" mass1[-1]     {:.2f} [1e-2Msun]".format(massarr2[-1]*1e2))
        print(" m2[tmax]      {:.2f} [1e-2Msun]".format(m2 * 1e2))
        # print(" \n")
        print(" abs(m1-m2)/m1 {:.1f} [%]".format(100 * np.abs(m1 - m2) / m1))
        print(" ---------------------------------------- ")

        errs[sim1]["sim1"] = sim1
        errs[sim1]["sim2"] = sim2
        errs[sim1]["tmax"] = tmax*1e3
        errs[sim1]["m1"] = m1*1e2
        errs[sim1]["m2"] = m2*1e2
        errs[sim1]["err"] = 100 * np.abs(m1 - m2) / m1

    return errs

    # table

    # sims = ['DD2_M13641364_M0_SR', 'LS220_M13641364_M0_SR', 'SLy4_M13641364_M0_SR']
    # v_ns = ["EOS", "M1", "M2", 'Mdisk3D', 'Mej', 'Yeej', 'vej', 'Mej_bern', 'Yeej_bern', 'vej_bern']
    # precs = ["str", "1.2", "1.2", ".4", ".4", ".4", ".4", ".4", ".4", ".4"]

    print('\n')

    cols = ["sim1", "sim2", "m1", "m2", "tmax", "err"]
    units_dic = {"sim1": "", "sim2": "", "m1":"$[10^{-2} M_{\odot}]$", "m2":"$[10^{-2} M_{\odot}]$", "tmax":"[ms]", "err":r"[\%]"}
    lbl_dic = {"sim1": "Default Run", "sim2": "Comparison Run", "m1": r"$M_{\text{ej}}^a$", "m2": r"$M_{\text{ej}}^b$", "tmax":r"$t_{\text{max}}$", "err":r"$\Delta$"}
    precs = ["", "", ".2f", ".2f", ".1f", "d"]

    size = '{'
    head = ''
    for i, v_n in enumerate(cols):
        v_n = lbl_dic[v_n]
        size = size + 'c'
        head = head + '{}'.format(v_n)
        if v_n != cols[-1]: size = size + ' '
        if i != len(cols) - 1: head = head + ' & '
    size = size + '}'

    unit_bar = ''
    for v_n in cols:
        if v_n in units_dic.keys():
            unit = units_dic[v_n]
        else:
            unit = v_n
        unit_bar = unit_bar + '{}'.format(unit)
        if v_n != cols[-1]: unit_bar = unit_bar + ' & '

    head = head + ' \\\\'  # = \\
    unit_bar = unit_bar + ' \\\\ '

    print(errs[sims[0]])

    print('\n')

    print('\\begin{table*}[t]')
    print('\\begin{center}')
    print('\\begin{tabular}' + '{}'.format(size))
    print('\\hline')
    print(head)
    print(unit_bar)
    print('\\hline\\hline')

    for sim1, mask1, sim2, mask2 in zip(sims, masks, sims2, masks2):
        row = ''
        for v_n, prec in zip(cols, precs):

            if prec != "":
                val = "%{}".format(prec) % errs[sim1][v_n]
            else:
                val = errs[sim1][v_n].replace("_", "\_")
            row = row + val
            if v_n != cols[-1]: row = row + ' & '
        row = row + ' \\\\'  # = \\
        print(row)

    print(r'\hline')
    print(r'\end{tabular}')
    print(r'\end{center}')
    print(r'\caption{'+r'Viscosity effect on the ejected material total cumulative mass. Criterion {} '
          .format(mask.replace('_', '\_')) +
          r'$\Delta = |M_{\text{ej}}^a - M_{\text{ej}}^b| / M_{\text{ej}}^a |_{tmax} $ }')
    print(r'\label{tbl:1}')
    print(r'\end{table*}')

    exit(1)

"""=================================================================================================================="""

def eos_color(eos):
    if eos == 'DD2':
        return 'blue'
    elif eos == 'BHBlp':
        return 'purple'
    elif eos == 'LS220':
        return 'orange'
    elif eos == 'SFHo':
        return 'red'
    elif eos == 'SLy4':
        return 'green'
    else:
        return 'black'

def get_ms(q, qmin=1, qmax = 1.4, msmin = 5., msmax = 10.):

    k = (qmax - qmin) / (msmax - msmin)
    b = qmax - (k * msmax)

    return (q - b) / k

""" =================================================| DUMPSTER |===================================================="""

class ErrorEstimation_old:

    def __init__(self, sim1, sim2):

        self.det = 0

        self.sim1 = sim1
        self.sim2 = sim2

        pass

    # --------------------| Preparation |--------------------------- #

    def get_tmax(self):

        o_par1 = ALL_PAR(self.sim1)
        o_par2 = ALL_PAR(self.sim2)

        tmerg1 = o_par1.get_par("tmerger")
        tmerg2 = o_par2.get_par("tmerger")

        t98geomass1 = o_par1.get_outflow_par(self.det, "geo", "t98mass")
        t98geomass2 = o_par2.get_outflow_par(self.det, "geo", "t98mass")

        tend1 = o_par1.get_outflow_par(self.det, "geo", "tend")
        tend2 = o_par2.get_outflow_par(self.det, "geo", "tend")

        assert tend1 > t98geomass1
        assert tend2 > t98geomass2
        assert tmerg1 < t98geomass1
        assert tmerg2 < t98geomass2

        tend1 = tend1 - tmerg1
        tend2 = tend2 - tmerg2
        t98geomass1 = t98geomass1 - tmerg1
        t98geomass2 = t98geomass2 - tmerg2

        delta_t1 = tend1 - t98geomass1
        delta_t2 = tend2 - t98geomass2

        print("Time window for bernoulli ")
        print("\t{} {:.2f} [ms]".format(self.sim1, delta_t1*1e3))
        print("\t{} {:.2f} [ms]".format(self.sim2, delta_t2*1e3))
        exit(1)

        delta_t = np.min([delta_t1, delta_t2])

        return delta_t
        #
        #
        # assert tend1 > tmerg1
        # assert tend2 > tmerg2
        #
        # print("tend1:{} tmerg1:{} -> {}".format(tend1, tmerg1, tend1-tmerg1))
        # print("tend2:{} tmerg2:{} -> {}".format(tend2, tmerg2, tend2-tmerg2))
        # # print("tmax:{}".format)
        #
        # tend1 = tend1 - tmerg1
        # tend2 = tend2 - tmerg2
        #
        # tmax = np.min([tend1, tend2])
        # print("get_tmax = tmax:{}".format(tmax))
        #
        #
        # return tmax

    def compute_outflow_new_mask(self, sim, tasks, new_mask, rewrite):

        # get_tmax60 # ms
        print("\tAdding mask:{}".format(new_mask))
        o_outflow = EJECTA_PARS(sim, add_mask=new_mask)

        if not os.path.isdir(Paths.ppr_sims+sim+"/"+"outflow_{:d}/".format(self.det)+new_mask+'/'):
            os.mkdir(Paths.ppr_sims+sim+"/"+"outflow_{:d}/".format(self.det)+new_mask+'/')

        for task in tasks:
            if task == "hist":
                from outflowed import outflowed_historgrams
                outflowed_historgrams(o_outflow, [self.det], [new_mask], o_outflow.list_hist_v_ns, rewrite=rewrite)
            elif task == "corr":
                from outflowed import outflowed_correlations
                outflowed_correlations(o_outflow, [self.det], [new_mask], o_outflow.list_corr_v_ns, rewrite=rewrite)
            elif task == "totflux":
                from outflowed import outflowed_totmass
                outflowed_totmass(o_outflow, [self.det], [new_mask], rewrite=rewrite)
            elif task == "timecorr":
                from outflowed import outflowed_timecorr
                outflowed_timecorr(o_outflow, [self.det], [new_mask], o_outflow.list_hist_v_ns, rewrite=rewrite)
            else:
                raise NameError("method for computing outflow with new mask is not setup for task:{}".format(task))

    def main_prepare_outflow_data(self, new_mask, rewrite=False):

        # get new mask for a maximum time (postmerger)
        # compute outflow data for this new mask
        tasks = ["totflux", "hist"]
        self.compute_outflow_new_mask(self.sim1, tasks, new_mask, rewrite=rewrite)
        self.compute_outflow_new_mask(self.sim2, tasks, new_mask, rewrite=rewrite)

        return new_mask

    # --------------------| Data Comparison |--------------------------- #

    def get_outflow_par_err(self, new_mask, v_n):

        o_par1 = ALL_PAR(self.sim1, add_mask=new_mask)
        o_par2 = ALL_PAR(self.sim2, add_mask=new_mask)

        val1 = o_par1.get_outflow_par(self.det, new_mask, v_n)
        val2 = o_par2.get_outflow_par(self.det, new_mask, v_n)

        err = np.abs(val1 - val2) / val1

        return val1, val2, err

    def main(self, v_ns, rewrite):

        base_masks = ["geo", "bern_geoend"]
        new_masks = []
        ind_res_dic = {}
        comb_res_dic = {}

        tmax = self.get_tmax()

        # preparing data
        for base_mask in base_masks:
            __new_mask = base_mask + "_length{:.0f}".format(tmax * 1e5)  # 100ms
            Printcolor.print_colored_string(
                ["task:", "outflow", "det:", "{}".format(self.det), "mask:", __new_mask, ":", "starting"],
                ["blue", "green", "blue", "green", "blue", "green", "", "green"])
            # try:
            new_mask = self.main_prepare_outflow_data(__new_mask, rewrite=rewrite)
            new_masks.append(new_mask)
            # except AssertionError:
            #     Printcolor.print_colored_string(
            #         ["task:", "outflow", "det:", "{}".format(self.det), "mask:", __new_mask, ":", "Assertion Error"],
            #         ["blue", "green", "blue", "green", "blue", "green", "", "red"])
            #     break

        if len(new_masks) == 0:
            raise ValueError("non of the given base_masks:{} succeeded".format(base_masks))

        # writing resukts

        o_par1 = ALL_PAR(self.sim1)
        o_par2 = ALL_PAR(self.sim2)

        ind_res_dic["sims"] = [self.sim1, self.sim2]
        ind_res_dic["base_masks"] = base_masks
        ind_res_dic["new_masks"] = new_masks

        for mask in base_masks:
            if mask.__contains__("bern_"):
                t98mass1 = o_par1.get_outflow_par(self.det, "geo", "t98mass")
                t98mass2 = o_par2.get_outflow_par(self.det, "geo", "t98mass")

                tmerg1 = o_par1.get_par("tmerger")
                tmerg2 = o_par2.get_par("tmerger")

                ind_res_dic["t98mass"] = [t98mass1-tmerg1, t98mass2-tmerg2]


        # loading results
        for new_mask in new_masks:
            comb_res_dic[new_mask] = {}
            for v_n in v_ns:
                val1, val2, err = self.get_outflow_par_err(new_mask, v_n)
                comb_res_dic[new_mask][v_n] = [val1, val2, err]

        # printing results
        for key in ind_res_dic.keys():
            print ind_res_dic[key]

        print("sim1:{} sim2:{}".format(self.sim1, self.sim2))
        for new_mask in new_masks:
            print("\tmask:{}".format(new_mask))
            for v_n in v_ns:
                val1, val2, err = comb_res_dic[new_mask][v_n]
                print("\t\tval1:{} val2:{} err:{}".format(val1, val2, err))

        return ind_res_dic, comb_res_dic

class ErrorEstimation:

    def __init__(self, sim1, sim2):
        self.det = 0
        self.sim1 = sim1
        self.sim2 = sim2

        self.o_par1 = ADD_METHODS_ALL_PAR(self.sim1)
        self.o_par2 = ADD_METHODS_ALL_PAR(self.sim2)

    def get_post_geo_delta_t(self):

        # o_par1 = ALL_PAR(self.sim1)
        # o_par2 = ALL_PAR(self.sim2)

        tmerg1 = self.o_par1.get_par("tmerger")
        tmerg2 = self.o_par2.get_par("tmerger")

        t98geomass1 = self.o_par1.get_outflow_par(self.det, "geo", "t98mass")
        t98geomass2 = self.o_par2.get_outflow_par(self.det, "geo", "t98mass")

        tend1 = self.o_par1.get_outflow_par(self.det, "geo", "tend")
        tend2 = self.o_par2.get_outflow_par(self.det, "geo", "tend")

        assert tend1 > t98geomass1
        assert tend2 > t98geomass2
        assert tmerg1 < t98geomass1
        assert tmerg2 < t98geomass2

        tend1 = tend1 - tmerg1
        tend2 = tend2 - tmerg2
        t98geomass1 = t98geomass1 - tmerg1
        t98geomass2 = t98geomass2 - tmerg2

        delta_t1 = tend1 - t98geomass1
        delta_t2 = tend2 - t98geomass2

        print("Time window for bernoulli ")
        print("\t{} {:.2f} [ms]".format(self.sim1, delta_t1*1e3))
        print("\t{} {:.2f} [ms]".format(self.sim2, delta_t2*1e3))
        # exit(1)

        delta_t = np.min([delta_t1, delta_t2])

        return delta_t

    def get_tmax_d3_data(self):

        isd3_1, itd3_1, td3_1 = self.o_par1.get_ittime("profiles", "prof")
        isd3_2, itd3_2, td3_2 = self.o_par2.get_ittime("profiles", "prof")

        if len(td3_1) == 0:
            Printcolor.red("D3 data not found for sim1:{}".format(self.sim1))
            return np.nan
        if len(td3_2) == 0:
            Printcolor.red("D3 data not found for sim2:{}".format(self.sim2))
            return np.nan

        if td3_1.min() > td3_2.max():
            Printcolor.red("D3 data does not overlap. sim1 has min:{} that is > than sim2 max: {}"
                           .format(td3_1.min(), td3_2.max()))
            return np.nan

        if td3_1.max() < td3_2.min():
            Printcolor.red("D3 data does not overlap. sim1 has max:{} that is < than sim2 min: {}"
                           .format(td3_1.max(), td3_2.min()))
            return np.nan

        tmax = np.min([td3_1.max(), td3_2.max()])
        print("\ttmax for D3 data: {}".format(tmax))
        return float(tmax)

    def compute_outflow_new_mask(self, sim, tasks, mask, rewrite):

        # get_tmax60 # ms
        print("\tAdding mask:{}".format(mask))
        o_outflow = EJECTA_PARS(sim, add_mask=mask)

        if not os.path.isdir(Paths.ppr_sims + sim +"/" +"outflow_{:d}/".format(self.det) + mask + '/'):
            os.mkdir(Paths.ppr_sims + sim +"/" +"outflow_{:d}/".format(self.det) + mask + '/')

        for task in tasks:
            if task == "hist":
                from outflowed import outflowed_historgrams
                outflowed_historgrams(o_outflow, [self.det], [mask], o_outflow.list_hist_v_ns, rewrite=rewrite)
            elif task == "corr":
                from outflowed import outflowed_correlations
                outflowed_correlations(o_outflow, [self.det], [mask], o_outflow.list_corr_v_ns, rewrite=rewrite)
            elif task == "totflux":
                from outflowed import outflowed_totmass
                outflowed_totmass(o_outflow, [self.det], [mask], rewrite=rewrite)
            elif task == "timecorr":
                from outflowed import outflowed_timecorr
                outflowed_timecorr(o_outflow, [self.det], [mask], o_outflow.list_hist_v_ns, rewrite=rewrite)
            else:
                raise NameError("method for computing outflow with new mask is not setup for task:{}".format(task))

    def get_outflow_par_err(self, new_mask, v_n):

        o_par1 = ALL_PAR(self.sim1, add_mask=new_mask)
        o_par2 = ALL_PAR(self.sim2, add_mask=new_mask)

        val1 = o_par1.get_outflow_par(self.det, new_mask, v_n)
        val2 = o_par2.get_outflow_par(self.det, new_mask, v_n)

        # err = np.abs(val1 - val2) / val1

        return val1, val2


    def main(self, rewrite=True):

        geo_v_ns = ["Mej_tot", "Ye_ave", "s_ave", "theta_rms"]
        tasks = ["totflux", "hist"]

        self.get_tmax_d3_data()

        # d3
        v_ns = ["Mdisk3D"]
        d3_res1 = {}
        d3_res2 = {}
        td3 = self.get_tmax_d3_data()
        if not np.isnan(td3):
            for v_n in v_ns:
                d3_res1[v_n] = self.o_par1.get_int_par(v_n, td3)
                d3_res2[v_n] = self.o_par2.get_int_par(v_n, td3)
        else:
            for v_n in v_ns:
                d3_res1[v_n] = np.nan
                d3_res2[v_n] = np.nan

        print("--- {} ---".format("d3"))
        print(self.sim1),
        print([("{}: {}".format(key, val)) for key, val in d3_res1.items()])
        print(self.sim2),
        print([("{}: {}".format(key, val)) for key, val in d3_res2.items()])

        # geo
        mask = "geo"
        self.compute_outflow_new_mask(self.sim1, tasks, mask, rewrite=rewrite)
        self.compute_outflow_new_mask(self.sim2, tasks, mask, rewrite=rewrite)
        geo_res1 = {}
        geo_res2 = {}
        for v_n in geo_v_ns:
            val1, val2 = self.get_outflow_par_err(mask, v_n)
            geo_res1[v_n] = val1
            geo_res2[v_n] = val2

        print("--- {} ---".format(mask))
        print(self.sim1),
        print([("{}: {}".format(key, val)) for key, val in geo_res1.items()])
        print(self.sim2),
        print([("{}: {}".format(key, val)) for key, val in geo_res2.items()])

        # bern
        delta_t = self.get_post_geo_delta_t()
        mask = "bern_geoend" + "_length{:.0f}".format(delta_t*1e5)# [1e2 ms]
        self.compute_outflow_new_mask(self.sim1, tasks, mask, rewrite=rewrite)
        self.compute_outflow_new_mask(self.sim2, tasks, mask, rewrite=rewrite)
        bern_res1 = {}
        bern_res2 = {}
        for v_n in geo_v_ns:
            val1, val2 = self.get_outflow_par_err(mask, v_n)
            bern_res1[v_n] = val1
            bern_res2[v_n] = val2

        print("--- {} ---".format(mask))
        print(self.sim1),
        print([("{}: {}".format(key, val)) for key, val in bern_res1.items()])
        print(self.sim2),
        print([("{}: {}".format(key, val)) for key, val in bern_res2.items()])

    # ----------------------------------------------------------

    @staticmethod
    def get_lbl(v_n, mask=""):

        if mask == "":
            if v_n == "Mdisk3D":
                return r"$M_{\text{disk}} ^{\text{last}}$"
            elif v_n == "Mdisk":
                return r"$M_{\text{disk}} ^{\text{BH}}$"
            elif v_n == "M1":
                return "$M_a$"
            elif v_n == "M2":
                return "$M_b$"
            elif v_n == "tcoll_gw" or v_n == "tcoll":
                return r"$t_{\text{BH}}$"
            elif v_n == "tend":
                return r"$t_{\text{end}}$"
            elif v_n == "tdisk3D":
                return r"$t_{\text{disk}}$"
            elif v_n == "q":
                return r"$M_a/M_b$"
            elif v_n == "EOS":
                return r"EOS"
            elif v_n == "res":
                return r"res"
            elif v_n == "vis":
                return "LK"
            elif v_n == "note":
                return r"note"
        else:
            if mask == "geo":
                if v_n == "theta_rms":
                    return "$\\langle \\theta_{\\text{ej}} \\rangle$"
                elif v_n == "Mej_tot":
                    return "$M_{\\text{ej}}$"
                elif v_n == "Ye_ave":
                    return "$\\langle Y_e \\rangle$"
                elif v_n == "vel_inf_ave":
                    return "$\\langle \\upsilon_{\\text{ej}} \\rangle$"
                else:
                    return v_n
            elif mask.__contains__("bern_"):
                if v_n == "theta_rms":
                    return "$\\langle \\theta_{\\text{ej}}^{\\text{w}} \\rangle$"
                elif v_n == "Mej_tot":
                    return "$M_{\\text{ej}}^{\\text{w}}$"
                elif v_n == "Ye_ave":
                    return "$\\langle Y_e ^{\\text{w}}  \\rangle$"
                elif v_n == "vel_inf_ave":
                    return "$\\langle \\upsilon_{\\text{ej}}^{\\text{w}} \\rangle$"
                else:
                    return v_n
            else:
                raise NameError("No label for v_n: {} and mask: {}"
                                .format(v_n, mask))

    @staticmethod
    def get_unit_lbl(v_n):
        if v_n in ["M1", "M2"]:
            return "$[M_{\odot}]$"
        elif v_n in ["Mej_tot"]:
            return "$[10^{-2} M_{\odot}]$"
        elif v_n in ["Mdisk3D", "Mdisk"]:
            return "$[M_{\odot}]$"
        elif v_n in ["vel_inf_ave"]:
            return "$[c]$"
        elif v_n in ["tcoll_gw", "tmerg_gw", "tmerg", "tcoll", "tend", "tdisk3D"]:
            return "[ms]"
        else:
            return " "

    def one_tex_table(self, rewrite = True):



        size = '{'
        head = ''
        for i, v_n in enumerate(cols):
            v_n = self.get_lbl(v_n)
            size = size + 'c'
            head = head + '{}'.format(v_n)
            if v_n != cols[-1]: size = size + ' '
            if i != len(cols) - 1: head = head + ' & '
        size = size + '}'

        unit_bar = ''
        for v_n in cols:
            unit = self.get_unit_lbl(v_n)
            unit_bar = unit_bar + '{}'.format(unit)
            if v_n != cols[-1]: unit_bar = unit_bar + ' & '

        head = head + ' \\\\'  # = \\
        unit_bar = unit_bar + ' \\\\ '

        print('\n')

        print('\\begin{table*}[t]')
        print('\\begin{center}')
        print('\\begin{tabular}' + '{}'.format(size))
        print('\\hline')
        print(head)
        print(unit_bar)
        print('\\hline\\hline')

"""=================================================================================================================="""


def plot_ejecta_time_corr_properites():
    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots+"all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (11.0, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = "timecorrs_Ye_DD2_LS220_SLy_equalmass.png"
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.3
    o_plot.gen_set["subplots_adjust_w"] = 0.01
    o_plot.set_plot_dics = []

    det = 0

    sims = ["DD2_M13641364_M0_LK_SR_R04", "BLh_M13641364_M0_LK_SR", "LS220_M13641364_M0_LK_SR", "SLy4_M13641364_M0_LK_SR", "SFHo_M13641364_M0_LK_SR"]
    lbls = ["DD2_M13641364_M0_LK_SR_R04", "BLh_M13641364_M0_LK_SR", "LS220_M13641364_M0_LK_SR", "SLy4_M13641364_M0_LK_SR", "SFHo_M13641364_M0_LK_SR"]
    masks= ["bern_geoend", "bern_geoend", "bern_geoend", "bern_geoend", "bern_geoend"]
    # v_ns = ["vel_inf", "vel_inf", "vel_inf", "vel_inf", "vel_inf"]
    v_ns = ["Y_e", "Y_e", "Y_e", "Y_e", "Y_e"]


    i_x_plot = 1
    for sim, lbl, mask, v_n in zip(sims,lbls,masks,v_ns):

        fpath = Paths.ppr_sims+sim+"/"+"outflow_{}/".format(det) + mask + '/' + "timecorr_{}.h5".format(v_n)
        if not os.path.isfile(fpath):
            raise IOError("File does not exist: {}".format(fpath))

        dfile = h5py.File(fpath, "r")
        timearr = np.array(dfile["time"])
        v_n_arr = np.array(dfile[v_n])
        mass    = np.array(dfile["mass"])

        corr_dic2 = {  # relies on the "get_res_corr(self, it, v_n): " method of data object
            'task': 'corr2d', 'dtype': 'corr', 'ptype': 'cartesian',
            'xarr':timearr, 'yarr':v_n_arr, 'zarr':mass,
            'position': (1, i_x_plot),
            'v_n_x': "time", 'v_n_y': v_n, 'v_n': 'mass', 'normalize': True,
            'cbar': {},
            'cmap': 'inferno',
            'xlabel': Labels.labels("time"), 'ylabel': Labels.labels(v_n),
            'xmin': timearr[0], 'xmax': timearr[-1], 'ymin': None, 'ymax': None, 'vmin': 1e-4, 'vmax': 1e-1,
            'xscale': "linear", 'yscale': "linear", 'norm': 'log',
            'mask_below': None, 'mask_above': None,
            'title': {},  # {"text": o_corr_data.sim.replace('_', '\_'), 'fontsize': 14},
            'text':  {'text': lbl.replace('_', '\_'), 'coords': (0.05, 0.9),  'color': 'white', 'fs': 12},
            'fancyticks': True,
            'minorticks': True,
            'sharex': False,  # removes angular citkscitks
            'sharey': False,
            'fontsize': 14,
            'labelsize': 14
        }

        if i_x_plot > 1:
            corr_dic2['sharey']=True
        # if i_x_plot == 1:
        #     corr_dic2['text'] = {'text': lbl.replace('_', '\_'), 'coords': (0.1, 0.9),  'color': 'white', 'fs': 14}
        if sim == sims[-1]:
            corr_dic2['cbar'] = {
                'location': 'right .03 .0', 'label': Labels.labels("mass"),  # 'fmt': '%.1f',
                'labelsize': 14, 'fontsize': 14}
        i_x_plot += 1
        corr_dic2 = Limits.in_dic(corr_dic2)
        o_plot.set_plot_dics.append(corr_dic2)

    o_plot.main()
    exit(1)
# plot_ejecta_time_corr_properites()

# def plot_total_fluxes_q1():
#
#     o_plot = PLOT_MANY_TASKS()
#     o_plot.gen_set["figdir"] = Paths.plots + "all2/"
#     o_plot.gen_set["type"] = "cartesian"
#     o_plot.gen_set["figsize"] = (9.0, 3.6)  # <->, |]
#     o_plot.gen_set["figname"] = "totfluxes_equalmasses.png"
#     o_plot.gen_set["sharex"] = False
#     o_plot.gen_set["sharey"] = True
#     o_plot.gen_set["dpi"] = 128
#     o_plot.gen_set["subplots_adjust_h"] = 0.3
#     o_plot.gen_set["subplots_adjust_w"] = 0.01
#     o_plot.set_plot_dics = []
#
#     det = 0
#
#     sims = ["DD2_M13641364_M0_LK_SR_R04", "BLh_M13641364_M0_LK_SR", "LS220_M13641364_M0_LK_SR", "SLy4_M13641364_M0_LK_SR", "SFHo_M13641364_M0_LK_SR"]
#     lbls = ["DD2", "BLh", "LS220", "SLy4", "SFHo"]
#     masks= ["bern_geoend", "bern_geoend", "bern_geoend", "bern_geoend", "bern_geoend"]
#     colors=["black", "gray", "red", "blue", "green"]
#     lss   =["-", "-", "-", "-", "-"]
#
#     i_x_plot = 1
#     for sim, lbl, mask, color, ls in zip(sims, lbls, masks, colors, lss):
#
#         fpath = Paths.ppr_sims + sim + "/" + "outflow_{}/".format(det) + mask + '/' + "total_flux.dat"
#         if not os.path.isfile(fpath):
#             raise IOError("File does not exist: {}".format(fpath))
#
#         timearr, massarr = np.loadtxt(fpath,usecols=(0,2),unpack=True)
#
#         plot_dic = {
#             'task': 'line', 'ptype': 'cartesian',
#             'position': (1, 1),
#             'xarr': timearr * 1e3, 'yarr': massarr * 1e2,
#             'v_n_x': "time", 'v_n_y': "mass",
#             'color': color, 'ls': ls, 'lw': 0.8, 'ds': 'default', 'alpha': 1.0,
#             'ymin': 0, 'ymax': 1.5, 'xmin': 15, 'xmax': 100,
#             'xlabel': Labels.labels("time"), 'ylabel': Labels.labels("ejmass"),
#             'label': lbl, 'yscale': 'linear',
#             'fancyticks': True, 'minorticks': True,
#             'fontsize': 14,
#             'labelsize': 14,
#             'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
#         }
#         if sim == sims[-1]:
#             plot_dic['legend'] = {'loc': 'best', 'ncol': 1, 'fontsize': 14}
#
#         o_plot.set_plot_dics.append(plot_dic)
#
#         #
#         #
#
#
#         i_x_plot += 1
#     o_plot.main()
#     exit(1)
# plot_total_fluxes_q1()

# def plot_total_fluxes_qnot1():
#
#     o_plot = PLOT_MANY_TASKS()
#     o_plot.gen_set["figdir"] = Paths.plots + "all2/"
#     o_plot.gen_set["type"] = "cartesian"
#     o_plot.gen_set["figsize"] = (9.0, 3.6)  # <->, |]
#     o_plot.gen_set["figname"] = "totfluxes_unequalmasses.png"
#     o_plot.gen_set["sharex"] = False
#     o_plot.gen_set["sharey"] = True
#     o_plot.gen_set["dpi"] = 128
#     o_plot.gen_set["subplots_adjust_h"] = 0.3
#     o_plot.gen_set["subplots_adjust_w"] = 0.01
#     o_plot.set_plot_dics = []
#
#     det = 0
#
#     sims = ["DD2_M15091235_M0_LK_SR", "LS220_M14691268_M0_LK_SR", "SFHo_M14521283_M0_LK_SR"]
#     lbls = ["DD2 151 124", "LS220 150 127", "SFHo 145 128"]
#     masks= ["bern_geoend", "bern_geoend", "bern_geoend"]
#     colors=["black", "red", "green"]
#     lss   =["-", "-", "-"]
#
#     i_x_plot = 1
#     for sim, lbl, mask, color, ls in zip(sims, lbls, masks, colors, lss):
#
#         fpath = Paths.ppr_sims + sim + "/" + "outflow_{}/".format(det) + mask + '/' + "total_flux.dat"
#         if not os.path.isfile(fpath):
#             raise IOError("File does not exist: {}".format(fpath))
#
#         timearr, massarr = np.loadtxt(fpath,usecols=(0,2),unpack=True)
#
#         plot_dic = {
#             'task': 'line', 'ptype': 'cartesian',
#             'position': (1, 1),
#             'xarr': timearr * 1e3, 'yarr': massarr * 1e2,
#             'v_n_x': "time", 'v_n_y': "mass",
#             'color': color, 'ls': ls, 'lw': 0.8, 'ds': 'default', 'alpha': 1.0,
#             'ymin': 0, 'ymax': 3.0, 'xmin': 15, 'xmax': 100,
#             'xlabel': Labels.labels("time"), 'ylabel': Labels.labels("ejmass"),
#             'label': lbl, 'yscale': 'linear',
#             'fancyticks': True, 'minorticks': True,
#             'fontsize': 14,
#             'labelsize': 14,
#             'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
#         }
#         if sim == sims[-1]:
#             plot_dic['legend'] = {'loc': 'best', 'ncol': 1, 'fontsize': 14}
#
#         o_plot.set_plot_dics.append(plot_dic)
#
#         #
#         #
#
#
#         i_x_plot += 1
#     o_plot.main()
#     exit(1)
# plot_total_fluxes_qnot1()

''' ejecta mass '''

def plot_total_fluxes_q1_and_qnot1(mask):

    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + "all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (9.0, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = "totfluxes_{}.png".format(mask)
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.3
    o_plot.gen_set["subplots_adjust_w"] = 0.01
    o_plot.set_plot_dics = []

    det = 0

    sims = ["DD2_M13641364_M0_LK_SR_R04", "BLh_M13641364_M0_LK_SR", "LS220_M13641364_M0_LK_SR", "SLy4_M13641364_M0_LK_SR", "SFHo_M13641364_M0_LK_SR"]
    lbls = ["DD2", "BLh", "LS220", "SLy4", "SFHo"]
    masks= [mask, mask, mask, mask, mask]
    colors=["black", "gray", "red", "blue", "green"]
    lss   =["-", "-", "-", "-", "-"]

    sims += ["DD2_M15091235_M0_LK_SR", "LS220_M14691268_M0_LK_SR", "SFHo_M14521283_M0_LK_SR"]
    lbls += ["DD2 151 124", "LS220 150 127", "SFHo 145 128"]
    masks+= [mask, mask, mask, mask, mask]
    colors+=["black", "red", "green"]
    lss   +=["--", "--", "--"]


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
            'xmin': 0, 'xmax': 110, 'ymin': 0, 'ymax': 3.0,
            'xlabel': Labels.labels("t-tmerg"), 'ylabel': Labels.labels("ejmass"),
            'label': lbl, 'yscale': 'linear',
            'fancyticks': True, 'minorticks': True,
            'fontsize': 14,
            'labelsize': 14,
            'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
        }
        if mask == "geo": plot_dic["ymax"] = 1.

        if sim == sims[-1]:
            plot_dic['legend'] = {'loc': 'best', 'ncol': 2, 'fontsize': 14}

        o_plot.set_plot_dics.append(plot_dic)




        #
        #


        i_x_plot += 1
    o_plot.main()
    exit(1)
# plot_total_fluxes_q1_and_qnot1(mask="bern_geoend")
# plot_total_fluxes_q1_and_qnot1(mask="geo")

def plot_total_fluxes_lk_on_off(mask):

    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + "all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (9.0, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = "totfluxes_lk_{}.png".format(mask)
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.3
    o_plot.gen_set["subplots_adjust_w"] = 0.01
    o_plot.set_plot_dics = []

    det = 0
    # plus LK
    sims = ["DD2_M13641364_M0_LK_SR_R04", "DD2_M15091235_M0_LK_SR", "LS220_M14691268_M0_LK_SR", "SFHo_M14521283_M0_LK_SR"]
    lbls = ["DD2 136 136 LK", "DD2 151 123 LK", "LS220 147 127 LK", "SFHo 145 128 LK"]
    masks = [mask, mask, mask, mask]
    colors = ["black", 'gray', 'red', "green"]
    lss = ["-", '-', '-','-']
    # minus LK
    sims2 = ["DD2_M13641364_M0_SR_R04", "DD2_M14971245_M0_SR", "LS220_M14691268_M0_SR", "SFHo_M14521283_M0_SR"]
    lbls2 = ["DD2 136 136", "DD2 150 125", "LS220 147 127", "SFHo 145 128"]
    masks2 = [mask, mask, mask, mask]
    colors2 = ["black", 'gray', 'red', "green"]
    lss2 = ["--", '--', '--', '--']

    sims += sims2
    lbls += lbls2
    masks += masks2
    colors += colors2
    lss += lss2

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
        timearr = timearr - (tmerg*Constants.time_constant*1e-3)


        plot_dic = {
            'task': 'line', 'ptype': 'cartesian',
            'position': (1, 1),
            'xarr': timearr * 1e3, 'yarr': massarr * 1e2,
            'v_n_x': "time", 'v_n_y': "mass",
            'color': color, 'ls': ls, 'lw': 0.8, 'ds': 'default', 'alpha': 1.0,
            'xmin': 0, 'xmax': 110, 'ymin': 0, 'ymax': 3.0,
            'xlabel': Labels.labels("t-tmerg"), 'ylabel': Labels.labels("ejmass"),
            'label': lbl, 'yscale': 'linear',
            'fancyticks': True, 'minorticks': True,
            'fontsize': 14,
            'labelsize': 14,
            'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
        }
        if mask == "geo": plot_dic["ymax"] = 1.
        if sim == sims[-1]:
            plot_dic['legend'] = {'loc': 'best', 'ncol': 2, 'fontsize': 14}

        o_plot.set_plot_dics.append(plot_dic)

        #

        #

        i_x_plot += 1
    o_plot.main()

    errs = {}

    for sim1, mask1, sim2, mask2 in zip(sims, masks, sims2, masks2):

        errs[sim1] = {}

        print(" --------------| {} |---------------- ".format(sim1.split('_')[0]))

        # loading times
        fpath1 = Paths.ppr_sims + sim1 + "/" + "outflow_{}/".format(det) + mask1 + '/' + "total_flux.dat"
        if not os.path.isfile(fpath1):
            raise IOError("File does not exist: {}".format(fpath1))

        timearr1, massarr1 = np.loadtxt(fpath1, usecols=(0, 2), unpack=True)

        # loading tmerg
        fpath1 = Paths.ppr_sims + sim1 + "/" + "waveforms/" + "tmerger.dat"
        if not os.path.isfile(fpath1):
            raise IOError("File does not exist: {}".format(fpath1))
        tmerg1 = np.float(np.loadtxt(fpath1, unpack=True))
        timearr1 = timearr1 - (tmerg1 * Constants.time_constant * 1e-3)

        # loading times
        fpath2 = Paths.ppr_sims + sim2 + "/" + "outflow_{}/".format(det) + mask2 + '/' + "total_flux.dat"
        if not os.path.isfile(fpath2):
            raise IOError("File does not exist: {}".format(fpath2))

        timearr2, massarr2 = np.loadtxt(fpath2, usecols=(0, 2), unpack=True)

        # loading tmerg
        fpath2 = Paths.ppr_sims + sim2 + "/" + "waveforms/" + "tmerger.dat"
        if not os.path.isfile(fpath2):
            raise IOError("File does not exist: {}".format(fpath2))
        tmerg2 = np.float(np.loadtxt(fpath2, unpack=True))
        timearr2 = timearr2 - (tmerg2 * Constants.time_constant * 1e-3)

        # estimating tmax
        tmax = np.array([timearr1[-1], timearr2[-1]]).min()
        assert tmax <= timearr1.max()
        assert tmax <= timearr2.max()
        m1 = massarr1[UTILS.find_nearest_index(timearr1, tmax)]
        m2 = massarr2[UTILS.find_nearest_index(timearr2, tmax)]

        # print(" --------------| {} |---------------- ".format(sim1.split('_')[0]))
        print(" tmax:         {:.1f} [ms]".format(tmax*1e3))
        # print(" \n")
        print(" sim1:         {} ".format(sim1))
        print(" timearr1[-1]: {:.1f} [ms]".format(timearr1[-1]*1e3))
        print(" mass1[-1]     {:.2f} [1e-2Msun]".format(massarr1[-1]*1e2))
        print(" m1[tmax]      {:.2f} [1e-2Msun]".format(m1 * 1e2))
        # print(" \n")
        print(" sim1:         {} ".format(sim2))
        print(" timearr1[-1]: {:.1f} [ms]".format(timearr2[-1]*1e3))
        print(" mass1[-1]     {:.2f} [1e-2Msun]".format(massarr2[-1]*1e2))
        print(" m2[tmax]      {:.2f} [1e-2Msun]".format(m2 * 1e2))
        # print(" \n")
        print(" abs(m1-m2)/m1 {:.1f} [%]".format(100 * np.abs(m1 - m2) / m1))
        print(" ---------------------------------------- ")

        errs[sim1]["sim1"] = sim1
        errs[sim1]["sim2"] = sim2
        errs[sim1]["tmax"] = tmax*1e3
        errs[sim1]["m1"] = m1*1e2
        errs[sim1]["m2"] = m2*1e2
        errs[sim1]["err"] = 100 * np.abs(m1 - m2) / m1

    # table

    # sims = ['DD2_M13641364_M0_SR', 'LS220_M13641364_M0_SR', 'SLy4_M13641364_M0_SR']
    # v_ns = ["EOS", "M1", "M2", 'Mdisk3D', 'Mej', 'Yeej', 'vej', 'Mej_bern', 'Yeej_bern', 'vej_bern']
    # precs = ["str", "1.2", "1.2", ".4", ".4", ".4", ".4", ".4", ".4", ".4"]

    print('\n')

    cols = ["sim1", "sim2", "m1", "m2", "tmax", "err"]
    units_dic = {"sim1": "", "sim2": "", "m1":"$[10^{-2} M_{\odot}]$", "m2":"$[10^{-2} M_{\odot}]$", "tmax":"[ms]", "err":r"[\%]"}
    lbl_dic = {"sim1": "Default Run", "sim2": "Comparison Run", "m1": r"$M_{\text{ej}}^a$", "m2": r"$M_{\text{ej}}^b$", "tmax":r"$t_{\text{max}}$", "err":r"$\Delta$"}
    precs = ["", "", ".2f", ".2f", ".1f", "d"]

    size = '{'
    head = ''
    for i, v_n in enumerate(cols):
        v_n = lbl_dic[v_n]
        size = size + 'c'
        head = head + '{}'.format(v_n)
        if v_n != cols[-1]: size = size + ' '
        if i != len(cols) - 1: head = head + ' & '
    size = size + '}'

    unit_bar = ''
    for v_n in cols:
        if v_n in units_dic.keys():
            unit = units_dic[v_n]
        else:
            unit = v_n
        unit_bar = unit_bar + '{}'.format(unit)
        if v_n != cols[-1]: unit_bar = unit_bar + ' & '

    head = head + ' \\\\'  # = \\
    unit_bar = unit_bar + ' \\\\ '

    print(errs[sims[0]])

    print('\n')

    print('\\begin{table*}[t]')
    print('\\begin{center}')
    print('\\begin{tabular}' + '{}'.format(size))
    print('\\hline')
    print(head)
    print(unit_bar)
    print('\\hline\\hline')

    for sim1, mask1, sim2, mask2 in zip(sims, masks, sims2, masks2):
        row = ''
        for v_n, prec in zip(cols, precs):

            if prec != "":
                val = "%{}".format(prec) % errs[sim1][v_n]
            else:
                val = errs[sim1][v_n].replace("_", "\_")
            row = row + val
            if v_n != cols[-1]: row = row + ' & '
        row = row + ' \\\\'  # = \\
        print(row)

    print(r'\hline')
    print(r'\end{tabular}')
    print(r'\end{center}')
    print(r'\caption{'+r'Viscosity effect on the ejected material total cumulative mass. Criterion {} '
          .format(mask.replace('_', '\_')) +
          r'$\Delta = |M_{\text{ej}}^a - M_{\text{ej}}^b| / M_{\text{ej}}^a |_{tmax} $ }')
    print(r'\label{tbl:1}')
    print(r'\end{table*}')

    exit(1)
# plot_total_fluxes_lk_on_off(mask="bern_geoend")
# plot_total_fluxes_lk_on_off("geo")

def plot_total_fluxes_lk_on_resolution(mask):

    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + "all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (9.0, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = "totfluxes_lk_res_{}.png".format(mask)
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.3
    o_plot.gen_set["subplots_adjust_w"] = 0.01
    o_plot.set_plot_dics = []

    det = 0
    # HR # LS220_M13641364_M0_LK_HR
    sims_hr  = ["DD2_M13641364_M0_LK_HR_R04", "DD2_M15091235_M0_LK_HR", "",                     "LS220_M14691268_M0_LK_HR", "SFHo_M13641364_M0_LK_HR", "SFHo_M14521283_M0_LK_HR"]
    lbl_hr   = ["DD2 136 136 HR", "DD2 151 124 HR", "LS220 136 136 HR", "LS220 147 137 HR", "SFHo 136 136 HR", "SFHo 145 128 HR"]
    color_hr = ["black", "gray", "orange", "red", "green", "lightgreen"]
    masks_hr = [mask, mask, mask, mask, mask, mask]
    lss_hr   = ['--', '--', '--', '--', "--", "--"]
    # SR
    sims_sr  = ["DD2_M13641364_M0_LK_SR_R04", "DD2_M15091235_M0_LK_SR", "LS220_M13641364_M0_LK_SR", "LS220_M14691268_M0_LK_SR", "SFHo_M13641364_M0_LK_SR", "SFHo_M14521283_M0_LK_SR"]
    lbl_sr   = ["DD2 136 136 SR", "DD2 151 124 HR", "LS220 136 136 SR", "LS220 147 137 SR", "SFHo 136 136 HR", "SFHo 145 128 HR"]
    color_sr = ["black", "gray", "orange", "red", "green", "lightgreen"]
    masks_sr = [mask, mask, mask, mask, mask, mask]
    lss_sr   = ['-', '-', '-', '-', '-', '-']
    # LR
    sims_lr  = ["DD2_M13641364_M0_LK_LR_R04", "", "", "", "", ""]
    lbl_lr   = ["DD2 136 136 LR", "DD2 151 124 LR", "LS220 136 136 LR", "LS220 147 137 LR", "SFHo 136 136 LR", "SFHo 145 128 LR"]
    color_lr = ["black", "gray", "orange", "red", "green", "lightgreen"]
    masks_lr = [mask, mask, mask, mask, mask, mask]
    lss_lr   = [':', ':', ":", ":", ":", ":"]

    # plus
    sims = sims_hr + sims_lr + sims_sr
    lsls = lbl_hr + lbl_lr + lbl_sr
    colors = color_hr + color_lr  + color_sr
    masks = masks_hr + masks_lr + masks_sr
    lss = lss_hr + lss_lr + lss_sr

    i_x_plot = 1
    for sim, lbl, mask, color, ls in zip(sims, lsls, masks, colors, lss):

        if sim != "":
            fpath = Paths.ppr_sims + sim + "/" + "outflow_{}/".format(det) + mask + '/' + "total_flux.dat"
            if not os.path.isfile(fpath):
                raise IOError("File does not exist: {}".format(fpath))

            timearr, massarr = np.loadtxt(fpath, usecols=(0, 2), unpack=True)

            fpath = Paths.ppr_sims + sim + "/" + "waveforms/" + "tmerger.dat"
            if not os.path.isfile(fpath):
                raise IOError("File does not exist: {}".format(fpath))
            tmerg = np.float(np.loadtxt(fpath, unpack=True))
            timearr = timearr - (tmerg*Constants.time_constant*1e-3)


            plot_dic = {
                'task': 'line', 'ptype': 'cartesian',
                'position': (1, 1),
                'xarr': timearr * 1e3, 'yarr': massarr * 1e2,
                'v_n_x': "time", 'v_n_y': "mass",
                'color': color, 'ls': ls, 'lw': 0.8, 'ds': 'default', 'alpha': 1.0,
                'xmin': 0, 'xmax': 110, 'ymin': 0, 'ymax': 3.0,
                'xlabel': Labels.labels("t-tmerg"), 'ylabel': Labels.labels("ejmass"),
                'label': lbl, 'yscale': 'linear',
                'fancyticks': True, 'minorticks': True,
                'fontsize': 14,
                'labelsize': 14,
                'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
            }
            if mask == "geo": plot_dic["ymax"] = 1.
            # print(sim, sims[-1])
            if sim == sims[-1]:
                plot_dic['legend'] = {'loc': 'best', 'ncol': 2, 'fontsize': 12}

            o_plot.set_plot_dics.append(plot_dic)

            i_x_plot += 1
    o_plot.main()

    for sim_hr, sim_sr, sim_lr, mask_hr, mask_sr, mask_lr in \
            zip(sims_hr, sims_sr,sims_lr, masks_hr, masks_sr, masks_lr):

        def_sim = sim_sr
        def_mask = mask_sr
        def_res = "SR"

        if sims_hr != "":
            comp_res = "HR"
            comp_sim = sim_hr
            comp_mask = mask_hr
        elif sims_lr != "":
            comp_res="LR"
            comp_sim = sim_lr
            comp_mask = mask_lr
        else:
            raise ValueError("neither HR nor LR is available")

        # loading times
        fpath1 = Paths.ppr_sims + def_sim + "/" + "outflow_{}/".format(det) + def_mask + '/' + "total_flux.dat"
        if not os.path.isfile(fpath1):
            raise IOError("File does not exist: {}".format(fpath1))

        timearr1, massarr1 = np.loadtxt(fpath1, usecols=(0, 2), unpack=True)

        # loading tmerg
        fpath1 = Paths.ppr_sims + def_sim + "/" + "waveforms/" + "tmerger.dat"
        if not os.path.isfile(fpath1):
            raise IOError("File does not exist: {}".format(fpath1))
        tmerg1 = np.float(np.loadtxt(fpath1, unpack=True))
        timearr1 = timearr1 - (tmerg1 * Constants.time_constant * 1e-3)

        # loading times
        fpath2 = Paths.ppr_sims + comp_sim + "/" + "outflow_{}/".format(det) + comp_mask + '/' + "total_flux.dat"
        if not os.path.isfile(fpath2):
            raise IOError("File does not exist: {}".format(fpath2))

        timearr2, massarr2 = np.loadtxt(fpath2, usecols=(0, 2), unpack=True)

        # loading tmerg
        fpath2 = Paths.ppr_sims + comp_sim + "/" + "waveforms/" + "tmerger.dat"
        if not os.path.isfile(fpath2):
            raise IOError("File does not exist: {}".format(fpath2))
        tmerg2 = np.float(np.loadtxt(fpath2, unpack=True))
        timearr2 = timearr2 - (tmerg2 * Constants.time_constant * 1e-3)

        # estimating tmax
        tmax = np.array([timearr1[-1], timearr2[-1]]).min()
        assert tmax <= timearr1.max()
        assert tmax <= timearr2.max()
        m1 = massarr1[UTILS.find_nearest_index(timearr1, tmax)]
        m2 = massarr2[UTILS.find_nearest_index(timearr2, tmax)]

        # print(" --------------| {} |---------------- ".format(sim1.split('_')[0]))
        print(" tmax:         {:.1f} [ms]".format(tmax*1e3))
        # print(" \n")
        print(" Resolution:   {} ".format(def_res))
        print(" sim1:         {} ".format(def_sim))
        print(" timearr1[-1]: {:.1f} [ms]".format(timearr1[-1]*1e3))
        print(" mass1[-1]     {:.2f} [1e-2Msun]".format(massarr1[-1]*1e2))
        print(" m1[tmax]      {:.2f} [1e-2Msun]".format(m1 * 1e2))
        # print(" \n")
        print("\nResolution:   {} ".format(comp_res))
        print(" sim1:         {} ".format(comp_sim))
        print(" timearr1[-1]: {:.1f} [ms]".format(timearr2[-1]*1e3))
        print(" mass1[-1]     {:.2f} [1e-2Msun]".format(massarr2[-1]*1e2))
        print(" m2[tmax]      {:.2f} [1e-2Msun]".format(m2 * 1e2))
        # print(" \n")
        print(" abs(m1-m2)/m1 {:.1f} [%]".format(100 * np.abs(m1 - m2) / m1))
        print(" ---------------------------------------- ")



    #
    #     print(" --------------| {} |---------------- ".format(sim1.split('_')[0]))
    #
    #     # loading times
    #     fpath1 = Paths.ppr_sims + sim1 + "/" + "outflow_{}/".format(det) + mask1 + '/' + "total_flux.dat"
    #     if not os.path.isfile(fpath1):
    #         raise IOError("File does not exist: {}".format(fpath1))
    #
    #     timearr1, massarr1 = np.loadtxt(fpath1, usecols=(0, 2), unpack=True)
    #
    #     # loading tmerg
    #     fpath1 = Paths.ppr_sims + sim1 + "/" + "waveforms/" + "tmerger.dat"
    #     if not os.path.isfile(fpath1):
    #         raise IOError("File does not exist: {}".format(fpath1))
    #     tmerg1 = np.float(np.loadtxt(fpath1, unpack=True))
    #     timearr1 = timearr1 - (tmerg1 * Constants.time_constant * 1e-3)
    #
    #     # loading times
    #     fpath2 = Paths.ppr_sims + sim2 + "/" + "outflow_{}/".format(det) + mask2 + '/' + "total_flux.dat"
    #     if not os.path.isfile(fpath2):
    #         raise IOError("File does not exist: {}".format(fpath2))
    #
    #     timearr2, massarr2 = np.loadtxt(fpath2, usecols=(0, 2), unpack=True)
    #
    #     # loading tmerg
    #     fpath2 = Paths.ppr_sims + sim2 + "/" + "waveforms/" + "tmerger.dat"
    #     if not os.path.isfile(fpath2):
    #         raise IOError("File does not exist: {}".format(fpath2))
    #     tmerg2 = np.float(np.loadtxt(fpath2, unpack=True))
    #     timearr2 = timearr2 - (tmerg2 * Constants.time_constant * 1e-3)
    #
    #     # estimating tmax
    #     tmax = np.array([timearr1[-1], timearr2[-1]]).min()
    #     assert tmax <= timearr1.max()
    #     assert tmax <= timearr2.max()
    #     m1 = massarr1[UTILS.find_nearest_index(timearr1, tmax)]
    #     m2 = massarr2[UTILS.find_nearest_index(timearr2, tmax)]
    #
    #     # print(" --------------| {} |---------------- ".format(sim1.split('_')[0]))
    #     print(" tmax:         {:.1f} [ms]".format(tmax*1e3))
    #     # print(" \n")
    #     print(" sim1:         {} ".format(sim1))
    #     print(" timearr1[-1]: {:.1f} [ms]".format(timearr1[-1]*1e3))
    #     print(" mass1[-1]     {:.2f} [1e-2Msun]".format(massarr1[-1]*1e2))
    #     print(" m1[tmax]      {:.2f} [1e-2Msun]".format(m1 * 1e2))
    #     # print(" \n")
    #     print(" sim1:         {} ".format(sim2))
    #     print(" timearr1[-1]: {:.1f} [ms]".format(timearr2[-1]*1e3))
    #     print(" mass1[-1]     {:.2f} [1e-2Msun]".format(massarr2[-1]*1e2))
    #     print(" m2[tmax]      {:.2f} [1e-2Msun]".format(m2 * 1e2))
    #     # print(" \n")
    #     print(" abs(m1-m2)/m1 {:.1f} [%]".format(100 * np.abs(m1 - m2) / m1))
    #     print(" ---------------------------------------- ")

    exit(1)
# plot_total_fluxes_lk_on_resolution(mask="geo_geoend")
# plot_total_fluxes_lk_on_resolution(mask="geo")

def plot_total_fluxes_lk_off_resolution(mask):

    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + "all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (9.0, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = "totfluxes_res_{}.png".format(mask)
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.3
    o_plot.gen_set["subplots_adjust_w"] = 0.01
    o_plot.set_plot_dics = []

    det = 0
    # HR    "DD2_M13641364_M0_HR_R04"
    sims_hr  = ["",                          "DD2_M14971245_M0_HR", "LS220_M13641364_M0_HR", "LS220_M14691268_M0_HR", "SFHo_M13641364_M0_HR", "SFHo_M14521283_M0_HR"]
    lbl_hr   = ["DD2 136 136 HR", "DD2 150 125 HR", "LS220 136 136 HR", "LS220 147 127 HR", "SFHo 136 136 HR", "SFHo 145 128 HR"]
    color_hr = ["black", "gray", "orange", "red", "lightgreen", "green"]
    masks_hr = [mask, mask, mask, mask, mask, mask]
    lss_hr   = ['--', '--', '--', '--', '--', '--']
    # SR
    sims_sr  = ["DD2_M13641364_M0_SR_R04", "DD2_M14971245_M0_SR", "LS220_M13641364_M0_SR", "LS220_M14691268_M0_SR", "SFHo_M13641364_M0_SR", "SFHo_M14521283_M0_SR"]
    lbl_sr   = ["DD2 136 136 SR", "DD2 150 125 SR", "LS220 136 136 SR", "LS220 147 127 SR", "SFHo 136 136 SR", "SFHo 145 128 SR"]
    color_sr = ["black", "gray", "orange", "red", "lightgreen", "green"]
    masks_sr = [mask, mask, mask, mask, mask, mask]
    lss_sr   = ['-','-','-','-','-','-']
    # LR
    sims_lr  = ["DD2_M13641364_M0_LR_R04", "DD2_M14971246_M0_LR", "LS220_M13641364_M0_LR", "LS220_M14691268_M0_LR", "", ""]
    lbl_lr   = ["DD2 136 136 LR", "DD2 150 125 LR", "LS220 136 136 LR", "LS220 147 127 LR", "SFHo 136 136 LR", "SFHo 145 128 LR"]
    color_lr = ["black", "gray", "orange", "red", "lightgreen", "green"]
    masks_lr = [mask, mask, mask, mask, mask, mask]
    lss_lr   = [':', ':', ':', ':', ':', ':']


    # plus
    sims = sims_hr + sims_lr + sims_sr
    lsls = lbl_hr + lbl_lr + lbl_sr
    colors = color_hr + color_lr  + color_sr
    masks = masks_hr + masks_lr + masks_sr
    lss = lss_hr + lss_lr + lss_sr

    i_x_plot = 1
    for sim, lbl, mask, color, ls in zip(sims, lsls, masks, colors, lss):

        if sim != "":
            fpath = Paths.ppr_sims + sim + "/" + "outflow_{}/".format(det) + mask + '/' + "total_flux.dat"
            if not os.path.isfile(fpath):
                raise IOError("File does not exist: {}".format(fpath))

            timearr, massarr = np.loadtxt(fpath, usecols=(0, 2), unpack=True)

            fpath = Paths.ppr_sims + sim + "/" + "waveforms/" + "tmerger.dat"
            if not os.path.isfile(fpath):
                raise IOError("File does not exist: {}".format(fpath))
            tmerg = np.float(np.loadtxt(fpath, unpack=True))
            timearr = timearr - (tmerg*Constants.time_constant*1e-3)


            plot_dic = {
                'task': 'line', 'ptype': 'cartesian',
                'position': (1, 1),
                'xarr': timearr * 1e3, 'yarr': massarr * 1e2,
                'v_n_x': "time", 'v_n_y': "mass",
                'color': color, 'ls': ls, 'lw': 0.8, 'ds': 'default', 'alpha': 1.0,
                'xmin': 0, 'xmax': 110, 'ymin': 0, 'ymax': 3.0,
                'xlabel': Labels.labels("t-tmerg"), 'ylabel': Labels.labels("ejmass"),
                'label': lbl, 'yscale': 'linear',
                'fancyticks': True, 'minorticks': True,
                'fontsize': 14,
                'labelsize': 14,
                'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
            }
            # print(sim, sims[-1])
            if mask == "geo": plot_dic["ymax"] = 1.
            if sim == sims[-1]:
                plot_dic['legend'] = {'loc': 'best', 'ncol': 3, 'fontsize': 12}

            o_plot.set_plot_dics.append(plot_dic)

            i_x_plot += 1
    o_plot.main()

    for sim_hr, sim_sr, sim_lr, mask_hr, mask_sr, mask_lr in \
            zip(sims_hr, sims_sr, sims_lr, masks_hr, masks_sr, masks_lr):

        def_sim = sim_sr
        def_mask = mask_sr
        def_res = "SR"

        if sim_hr != "":
            comp_res = "HR"
            comp_sim = sim_hr
            comp_mask = mask_hr
        elif sim_lr != "":
            comp_res = "LR"
            comp_sim = sim_lr
            comp_mask = mask_lr
        else:
            raise ValueError("neither HR nor LR is available")

        assert comp_sim != ""

        # loading times
        fpath1 = Paths.ppr_sims + def_sim + "/" + "outflow_{}/".format(det) + def_mask + '/' + "total_flux.dat"
        if not os.path.isfile(fpath1):
            raise IOError("File does not exist: {}".format(fpath1))

        timearr1, massarr1 = np.loadtxt(fpath1, usecols=(0, 2), unpack=True)

        # loading tmerg
        fpath1 = Paths.ppr_sims + def_sim + "/" + "waveforms/" + "tmerger.dat"
        if not os.path.isfile(fpath1):
            raise IOError("File does not exist: {}".format(fpath1))
        tmerg1 = np.float(np.loadtxt(fpath1, unpack=True))
        timearr1 = timearr1 - (tmerg1 * Constants.time_constant * 1e-3)

        # loading times
        fpath2 = Paths.ppr_sims + comp_sim + "/" + "outflow_{}/".format(det) + comp_mask + '/' + "total_flux.dat"
        if not os.path.isfile(fpath2):
            raise IOError("File does not exist: {}".format(fpath2))

        timearr2, massarr2 = np.loadtxt(fpath2, usecols=(0, 2), unpack=True)

        # loading tmerg
        fpath2 = Paths.ppr_sims + comp_sim + "/" + "waveforms/" + "tmerger.dat"
        if not os.path.isfile(fpath2):
            raise IOError("File does not exist: {}".format(fpath2))
        tmerg2 = np.float(np.loadtxt(fpath2, unpack=True))
        timearr2 = timearr2 - (tmerg2 * Constants.time_constant * 1e-3)

        # estimating tmax
        tmax = np.array([timearr1[-1], timearr2[-1]]).min()
        assert tmax <= timearr1.max()
        assert tmax <= timearr2.max()
        m1 = massarr1[UTILS.find_nearest_index(timearr1, tmax)]
        m2 = massarr2[UTILS.find_nearest_index(timearr2, tmax)]

        # print(" --------------| {} |---------------- ".format(sim1.split('_')[0]))
        print(" tmax:         {:.1f} [ms]".format(tmax*1e3))
        # print(" \n")
        print(" Resolution:   {} ".format(def_res))
        print(" sim1:         {} ".format(def_sim))
        print(" timearr1[-1]: {:.1f} [ms]".format(timearr1[-1]*1e3))
        print(" mass1[-1]     {:.2f} [1e-2Msun]".format(massarr1[-1]*1e2))
        print(" m1[tmax]      {:.2f} [1e-2Msun]".format(m1 * 1e2))
        # print(" \n")
        print("\nResolution: {} ".format(comp_res))
        print(" sim1:         {} ".format(comp_sim))
        print(" timearr1[-1]: {:.1f} [ms]".format(timearr2[-1]*1e3))
        print(" mass1[-1]     {:.2f} [1e-2Msun]".format(massarr2[-1]*1e2))
        print(" m2[tmax]      {:.2f} [1e-2Msun]".format(m2 * 1e2))
        # print(" \n")
        print(" abs(m1-m2)/m1 {:.1f} [%]".format(100 * np.abs(m1 - m2) / m1))
        print(" ---------------------------------------- ")



    #
    #     print(" --------------| {} |---------------- ".format(sim1.split('_')[0]))
    #
    #     # loading times
    #     fpath1 = Paths.ppr_sims + sim1 + "/" + "outflow_{}/".format(det) + mask1 + '/' + "total_flux.dat"
    #     if not os.path.isfile(fpath1):
    #         raise IOError("File does not exist: {}".format(fpath1))
    #
    #     timearr1, massarr1 = np.loadtxt(fpath1, usecols=(0, 2), unpack=True)
    #
    #     # loading tmerg
    #     fpath1 = Paths.ppr_sims + sim1 + "/" + "waveforms/" + "tmerger.dat"
    #     if not os.path.isfile(fpath1):
    #         raise IOError("File does not exist: {}".format(fpath1))
    #     tmerg1 = np.float(np.loadtxt(fpath1, unpack=True))
    #     timearr1 = timearr1 - (tmerg1 * Constants.time_constant * 1e-3)
    #
    #     # loading times
    #     fpath2 = Paths.ppr_sims + sim2 + "/" + "outflow_{}/".format(det) + mask2 + '/' + "total_flux.dat"
    #     if not os.path.isfile(fpath2):
    #         raise IOError("File does not exist: {}".format(fpath2))
    #
    #     timearr2, massarr2 = np.loadtxt(fpath2, usecols=(0, 2), unpack=True)
    #
    #     # loading tmerg
    #     fpath2 = Paths.ppr_sims + sim2 + "/" + "waveforms/" + "tmerger.dat"
    #     if not os.path.isfile(fpath2):
    #         raise IOError("File does not exist: {}".format(fpath2))
    #     tmerg2 = np.float(np.loadtxt(fpath2, unpack=True))
    #     timearr2 = timearr2 - (tmerg2 * Constants.time_constant * 1e-3)
    #
    #     # estimating tmax
    #     tmax = np.array([timearr1[-1], timearr2[-1]]).min()
    #     assert tmax <= timearr1.max()
    #     assert tmax <= timearr2.max()
    #     m1 = massarr1[UTILS.find_nearest_index(timearr1, tmax)]
    #     m2 = massarr2[UTILS.find_nearest_index(timearr2, tmax)]
    #
    #     # print(" --------------| {} |---------------- ".format(sim1.split('_')[0]))
    #     print(" tmax:         {:.1f} [ms]".format(tmax*1e3))
    #     # print(" \n")
    #     print(" sim1:         {} ".format(sim1))
    #     print(" timearr1[-1]: {:.1f} [ms]".format(timearr1[-1]*1e3))
    #     print(" mass1[-1]     {:.2f} [1e-2Msun]".format(massarr1[-1]*1e2))
    #     print(" m1[tmax]      {:.2f} [1e-2Msun]".format(m1 * 1e2))
    #     # print(" \n")
    #     print(" sim1:         {} ".format(sim2))
    #     print(" timearr1[-1]: {:.1f} [ms]".format(timearr2[-1]*1e3))
    #     print(" mass1[-1]     {:.2f} [1e-2Msun]".format(massarr2[-1]*1e2))
    #     print(" m2[tmax]      {:.2f} [1e-2Msun]".format(m2 * 1e2))
    #     # print(" \n")
    #     print(" abs(m1-m2)/m1 {:.1f} [%]".format(100 * np.abs(m1 - m2) / m1))
    #     print(" ---------------------------------------- ")

    exit(1)
# plot_total_fluxes_lk_off_resolution(mask="bern_geoend")
# plot_total_fluxes_lk_off_resolution(mask="geo")

''' ejecta 1D histograms '''

def plot_histograms_ejecta(mask):


    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots+"all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (11.0, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = "tothist_{}.png".format(mask)
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.3
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []
    averages = {}

    det = 0

    sims = ["DD2_M13641364_M0_LK_SR_R04", "BLh_M13641364_M0_LK_SR", "LS220_M13641364_M0_LK_SR",
            "SLy4_M13641364_M0_LK_SR", "SFHo_M13641364_M0_LK_SR"]
    lbls = ["DD2", "BLh", "LS220", "SLy4", "SFHo"]
    masks = [mask, mask, mask, mask, mask]
    colors = ["black", "gray", "red", "blue", "green"]
    lss = ["-", "-", "-", "-", "-"]
    lws = [1., 1., 1., 1., 1.]

    sims += ["DD2_M15091235_M0_LK_SR", "LS220_M14691268_M0_LK_SR", "SFHo_M14521283_M0_LK_SR"]
    lbls += ["DD2 151 124", "LS220 150 127", "SFHo 145 128"]
    masks += [mask, mask, mask]
    colors += ["black", "red", "green"]
    lss += ["--", "--", "--"]
    lws += [1., 1., 1.]

    v_ns = ["theta", "Y_e", "vel_inf", "entropy"]
    i_x_plot = 1
    for v_n in v_ns:
        averages[v_n] = {}
        for sim, lbl, mask, color, ls, lw in zip(sims, lbls, masks, colors, lss, lws):

            # loading hist
            fpath = Paths.ppr_sims + sim + "/" + "outflow_{}/".format(det) + mask + '/' + "hist_{}.dat".format(v_n)
            if not os.path.isfile(fpath):
                raise IOError("File does not exist: {}".format(fpath))
            hist = np.loadtxt(fpath, usecols=(0, 1), unpack=False)

            # loading times
            fpath1 = Paths.ppr_sims + sim + "/" + "outflow_{}/".format(det) + mask + '/' + "total_flux.dat"
            if not os.path.isfile(fpath1):
                raise IOError("File does not exist: {}".format(fpath1))
            timearr1, massarr1 = np.loadtxt(fpath1, usecols=(0, 2), unpack=True)

            if v_n == "Y_e":
                ave = EJECTA_PARS.compute_ave_ye(massarr1[-1], hist)
                averages[v_n][sim] = ave
            elif v_n == "theta":
                ave = EJECTA_PARS.compute_ave_theta_rms(hist)
                averages[v_n][sim] = ave
            elif v_n == "vel_inf":
                ave = EJECTA_PARS.compute_ave_vel_inf(massarr1[-1], hist)
                averages[v_n][sim] = ave
            elif v_n == "entropy":
                ave = EJECTA_PARS.compute_ave_vel_inf(massarr1[-1], hist)
                averages[v_n][sim] = ave
            else:
                raise NameError("no averages set for v_n:{}".format(v_n))

            plot_dic = {
                'task': 'hist1d', 'ptype': 'cartesian',
                'position': (1, i_x_plot),
                'data': hist, 'normalize': True,
                'v_n_x': v_n, 'v_n_y': None,
                'color': color, 'ls': ls, 'lw': lw, 'ds': 'steps', 'alpha': 1.0,
                'xmin': None, 'xamx': None, 'ymin': 1e-3, 'ymax': 1e0,
                'xlabel': Labels.labels(v_n), 'ylabel': Labels.labels("mass"),
                'label': lbl, 'yscale': 'log',
                'fancyticks': True, 'minorticks': True,
                'fontsize': 14,
                'labelsize': 14,
                'sharex': False,
                'sharey': False,
                'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
            }
            plot_dic = Limits.in_dic(plot_dic)
            if v_n != v_ns[0]:
                plot_dic["sharey"] = True
            if v_n == v_ns[-1] and sim == sims[-1]:
                plot_dic['legend'] = {'bbox_to_anchor': (0.0, 1.0), 'loc': 'best', 'ncol':4,"fontsize":12}

            o_plot.set_plot_dics.append(plot_dic)

        i_x_plot += 1
    o_plot.main()

    for v_n in v_ns:
        print("\t{}".format(v_n))
        for sim in sims:
            print("\t\t{}".format(sim)),
            print("       {:.2f}".format(averages[v_n][sim]))


    exit(1)
# plot_histograms_ejecta("geo")
# plot_histograms_ejecta("bern_geoend")

def plot_histograms_lk_on_off(mask):


    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots+"all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (11.0, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = "tothist_lk_{}.png".format(mask)
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.3
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []
    averages = {}

    det = 0

    sims = ["DD2_M13641364_M0_LK_SR_R04", "DD2_M15091235_M0_LK_SR", "LS220_M14691268_M0_LK_SR", "SFHo_M14521283_M0_LK_SR"]
    lbls = ["DD2 136 136 LK", "DD2 151 123 LK", "LS220 147 127 LK", "SFHo 145 128 LK"]
    masks = [mask, mask, mask, mask]
    colors = ["black", 'gray', 'red', "green"]
    lss = ["-", '-', '-','-']
    lws = [1., 1., 1., 1.,]
    # minus LK
    sims2 = ["DD2_M13641364_M0_SR_R04", "DD2_M14971245_M0_SR", "LS220_M14691268_M0_SR", "SFHo_M14521283_M0_SR"]
    lbls2 = ["DD2 136 136", "DD2 150 125", "LS220 147 127", "SFHo 145 128"]
    masks2 = [mask, mask, mask, mask]
    colors2 = ["black", 'gray', 'red', "green"]
    lss2 = ["--", '--', '--', '--']
    lws2 = [1., 1., 1., 1., ]

    sims += sims2
    lbls += lbls2
    masks += masks2
    colors += colors2
    lss += lss2
    lws += lws2

    v_ns = ["theta", "Y_e", "vel_inf", "entropy"]
    i_x_plot = 1
    for v_n in v_ns:
        averages[v_n] = {}
        for sim, lbl, mask, color, ls, lw in zip(sims, lbls, masks, colors, lss, lws):

            # loading hist
            fpath = Paths.ppr_sims + sim + "/" + "outflow_{}/".format(det) + mask + '/' + "hist_{}.dat".format(v_n)
            if not os.path.isfile(fpath):
                raise IOError("File does not exist: {}".format(fpath))
            hist = np.loadtxt(fpath, usecols=(0, 1), unpack=False)

            # loading times
            fpath1 = Paths.ppr_sims + sim + "/" + "outflow_{}/".format(det) + mask + '/' + "total_flux.dat"
            if not os.path.isfile(fpath1):
                raise IOError("File does not exist: {}".format(fpath1))
            timearr1, massarr1 = np.loadtxt(fpath1, usecols=(0, 2), unpack=True)

            if v_n == "Y_e":
                ave = EJECTA_PARS.compute_ave_ye(massarr1[-1], hist)
                averages[v_n][sim] = ave
            elif v_n == "theta":
                ave = EJECTA_PARS.compute_ave_theta_rms(hist)
                averages[v_n][sim] = ave
            elif v_n == "vel_inf":
                ave = EJECTA_PARS.compute_ave_vel_inf(massarr1[-1], hist)
                averages[v_n][sim] = ave
            elif v_n == "entropy":
                ave = EJECTA_PARS.compute_ave_vel_inf(massarr1[-1], hist)
                averages[v_n][sim] = ave
            else:
                raise NameError("no averages set for v_n:{}".format(v_n))

            plot_dic = {
                'task': 'hist1d', 'ptype': 'cartesian',
                'position': (1, i_x_plot),
                'data': hist, 'normalize': True,
                'v_n_x': v_n, 'v_n_y': None,
                'color': color, 'ls': ls, 'lw': lw, 'ds': 'steps', 'alpha': 1.0,
                'xmin': None, 'xamx': None, 'ymin': 1e-3, 'ymax': 1e0,
                'xlabel': Labels.labels(v_n), 'ylabel': Labels.labels("mass"),
                'label': lbl, 'yscale': 'log',
                'fancyticks': True, 'minorticks': True,
                'fontsize': 14,
                'labelsize': 14,
                'sharex': False,
                'sharey': False,
                'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
            }
            plot_dic = Limits.in_dic(plot_dic)
            if v_n != v_ns[0]:
                plot_dic["sharey"] = True
            if v_n == v_ns[-1] and sim == sims[-1]:
                plot_dic['legend'] = {'bbox_to_anchor': (-3.00, 1.0), 'loc': 'upper left', 'ncol':4,"fontsize":12}

            o_plot.set_plot_dics.append(plot_dic)

        i_x_plot += 1
    o_plot.main()

    for v_n in v_ns:
        print(" --- v_n: {} --- ".format(v_n))
        for sim1, sim2 in zip(sims, sims2):
            val1 = averages[v_n][sim1]
            val2 = averages[v_n][sim2]
            err = 100 * (val1 - val2) / val1
            print("\t{}  :  {:.2f}".format(sim1, val1))
            print("\t{}  :  {:.2f}".format(sim2, val2))
            print("\t\tErr:\t\t{:.1f}".format(err))
        print(" -------------------- ".format(v_n))



    exit(1)
# plot_histograms_lk_on_off("geo")
# plot_histograms_lk_on_off("bern_geoend")


def plot_histograms_lk_on_resolution(mask):


    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots+"all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (11.0, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = "tothist_lk_res_{}.png".format(mask)
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.3
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []
    averages = {}

    det = 0
    # HR  "LS220_M13641364_M0_LK_HR"  -- too short
    sims_hr  = ["DD2_M13641364_M0_LK_HR_R04", "DD2_M15091235_M0_LK_HR", "", "LS220_M14691268_M0_LK_HR", "SFHo_M13641364_M0_LK_HR", "SFHo_M14521283_M0_LK_HR"]
    lbl_hr   = ["DD2 136 136 HR", "DD2 151 124 HR", "LS220 136 136 HR", "LS220 147 137 HR", "SFHo 136 136 HR", "SFHo 145 128 HR"]
    color_hr = ["black", "gray", "orange", "red", "green", "lightgreen"]
    masks_hr = [mask, mask, mask, mask, mask, mask]
    lss_hr   = ['--', '--', '--', '--', "--", "--"]
    lws_hr   = [1., 1., 1., 1., 1., 1.]
    # SR  "LS220_M13641364_M0_LK_SR"
    sims_sr  = ["DD2_M13641364_M0_LK_SR_R04", "DD2_M15091235_M0_LK_SR", "", "LS220_M14691268_M0_LK_SR", "SFHo_M13641364_M0_LK_SR", "SFHo_M14521283_M0_LK_SR"]
    lbl_sr   = ["DD2 136 136 SR", "DD2 151 124 HR", "LS220 136 136 SR", "LS220 147 137 SR", "SFHo 136 136 HR", "SFHo 145 128 HR"]
    color_sr = ["black", "gray", "orange", "red", "green", "lightgreen"]
    masks_sr = [mask, mask, mask, mask, mask, mask]
    lss_sr   = ['-', '-', '-', '-', '-', '-']
    lws_sr   = [1., 1., 1., 1., 1., 1.]
    # LR
    sims_lr  = ["DD2_M13641364_M0_LK_LR_R04", "", "", "", "", ""]
    lbl_lr   = ["DD2 136 136 LR", "DD2 151 124 LR", "LS220 136 136 LR", "LS220 147 137 LR", "SFHo 136 136 LR", "SFHo 145 128 LR"]
    color_lr = ["black", "gray", "orange", "red", "green", "lightgreen"]
    masks_lr = [mask, mask, mask, mask, mask, mask]
    lss_lr   = [':', ':', ":", ":", ":", ":"]
    lws_lr   = [1., 1., 1., 1., 1., 1.]



    # plus
    sims = sims_hr + sims_lr + sims_sr
    lbls = lbl_hr + lbl_lr + lbl_sr
    colors = color_hr + color_lr  + color_sr
    masks = masks_hr + masks_lr + masks_sr
    lss = lss_hr + lss_lr + lss_sr
    lws = lws_hr + lws_lr + lws_sr

    v_ns = ["theta", "Y_e", "vel_inf", "entropy"]
    i_x_plot = 1
    for v_n in v_ns:
        averages[v_n] = {}
        for sim, lbl, mask, color, ls, lw in zip(sims, lbls, masks, colors, lss, lws):
            if sim != "":
                # loading hist
                fpath = Paths.ppr_sims + sim + "/" + "outflow_{}/".format(det) + mask + '/' + "hist_{}.dat".format(v_n)
                if not os.path.isfile(fpath):
                    raise IOError("File does not exist: {}".format(fpath))
                hist = np.loadtxt(fpath, usecols=(0, 1), unpack=False)

                # loading times
                fpath1 = Paths.ppr_sims + sim + "/" + "outflow_{}/".format(det) + mask + '/' + "total_flux.dat"
                if not os.path.isfile(fpath1):
                    raise IOError("File does not exist: {}".format(fpath1))
                timearr1, massarr1 = np.loadtxt(fpath1, usecols=(0, 2), unpack=True)

                if v_n == "Y_e":
                    ave = EJECTA_PARS.compute_ave_ye(massarr1[-1], hist)
                    averages[v_n][sim] = ave
                elif v_n == "theta":
                    ave = EJECTA_PARS.compute_ave_theta_rms(hist)
                    averages[v_n][sim] = ave
                elif v_n == "vel_inf":
                    ave = EJECTA_PARS.compute_ave_vel_inf(massarr1[-1], hist)
                    averages[v_n][sim] = ave
                elif v_n == "entropy":
                    ave = EJECTA_PARS.compute_ave_vel_inf(massarr1[-1], hist)
                    averages[v_n][sim] = ave
                else:
                    raise NameError("no averages set for v_n:{}".format(v_n))

                plot_dic = {
                    'task': 'hist1d', 'ptype': 'cartesian',
                    'position': (1, i_x_plot),
                    'data': hist, 'normalize': True,
                    'v_n_x': v_n, 'v_n_y': None,
                    'color': color, 'ls': ls, 'lw': lw, 'ds': 'steps', 'alpha': 1.0,
                    'xmin': None, 'xamx': None, 'ymin': 1e-3, 'ymax': 1e0,
                    'xlabel': Labels.labels(v_n), 'ylabel': Labels.labels("mass"),
                    'label': lbl, 'yscale': 'log',
                    'fancyticks': True, 'minorticks': True,
                    'fontsize': 14,
                    'labelsize': 14,
                    'sharex': False,
                    'sharey': False,
                    'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
                }
                plot_dic = Limits.in_dic(plot_dic)
                if v_n != v_ns[0]:
                    plot_dic["sharey"] = True
                if v_n == v_ns[-1] and sim == sims[-1]:
                    plot_dic['legend'] = {'bbox_to_anchor': (-3.00, 1.0), 'loc': 'upper left', 'ncol':4,"fontsize":12}

                o_plot.set_plot_dics.append(plot_dic)

        i_x_plot += 1
    o_plot.main()

    for v_n in v_ns:
        print(" --- v_n: {} --- ".format(v_n))
        for sim_hr, sim_sr, sim_lr in zip(sims_hr, sims_sr, sims_lr):
            # print(sim_hr, sim_sr, sim_lr)
            if not sim_sr == "":
                assert sim_sr != ""
                def_sim = sim_sr
                def_res = "SR"

                if sim_hr != '':
                    comp_res = "HR"
                    comp_sim = sim_hr
                elif  sim_hr == '' and sim_lr != '':
                    comp_res = "LR"
                    comp_sim = sim_lr
                else:
                    raise ValueError("neither HR nor LR is available")

                # print(def_sim, comp_sim)

                assert comp_sim != ""

                val1 = averages[v_n][def_sim]
                val2 = averages[v_n][comp_sim]
                err = 100 * (val1 - val2) / val1
                print("\t{}  :  {:.2f}".format(def_sim, val1))
                print("\t{}  :  {:.2f}".format(comp_sim, val2))
                print("\t\tErr:\t\t{:.1f}".format(err))
        print(" -------------------- ".format(v_n))



    exit(1)
# plot_histograms_lk_on_resolution("geo")
# plot_histograms_lk_on_resolution("bern_geoend")

def plot_histograms_lk_off_resolution(mask):

    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots+"all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (11.0, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = "tothist_res_{}.png".format(mask)
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.3
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []
    averages = {}

    det = 0
    # HR  "LS220_M13641364_M0_LK_HR"  -- too short
    sims_hr  = ["", "DD2_M14971245_M0_HR", "LS220_M13641364_M0_HR", "LS220_M14691268_M0_HR", "SFHo_M13641364_M0_HR", "SFHo_M14521283_M0_HR"]
    lbl_hr   = ["DD2 136 136 HR", "DD2 150 125 HR", "LS220 136 136 HR", "LS220 147 127 HR", "SFHo 136 136 HR", "SFHo 145 128 HR"]
    color_hr = ["black", "gray", "orange", "red", "lightgreen", "green"]
    masks_hr = [mask, mask, mask, mask, mask, mask]
    lss_hr   = ['--', '--', '--', '--', '--', '--']
    lws_hr   = [1., 1., 1., 1., 1., 1.]
    # SR  "LS220_M13641364_M0_LK_SR"
    sims_sr  = ["DD2_M13641364_M0_SR_R04", "DD2_M14971245_M0_SR", "LS220_M13641364_M0_SR", "LS220_M14691268_M0_SR", "SFHo_M13641364_M0_SR", "SFHo_M14521283_M0_SR"]
    lbl_sr   = ["DD2 136 136 SR", "DD2 150 125 SR", "LS220 136 136 SR", "LS220 147 127 SR", "SFHo 136 136 SR", "SFHo 145 128 SR"]
    color_sr = ["black", "gray", "orange", "red", "lightgreen", "green"]
    masks_sr = [mask, mask, mask, mask, mask, mask]
    lss_sr   = ['-','-','-','-','-','-']
    lws_sr   = [1., 1., 1., 1., 1., 1.]
    # LR
    sims_lr  = ["DD2_M13641364_M0_LR_R04", "DD2_M14971246_M0_LR", "LS220_M13641364_M0_LR", "LS220_M14691268_M0_LR", "", ""]
    lbl_lr   = ["DD2 136 136 LR", "DD2 150 125 LR", "LS220 136 136 LR", "LS220 147 127 LR", "SFHo 136 136 LR", "SFHo 145 128 LR"]
    color_lr = ["black", "gray", "orange", "red", "lightgreen", "green"]
    masks_lr = [mask, mask, mask, mask, mask, mask]
    lss_lr   = [':', ':', ':', ':', ':', ':']
    lws_lr   = [1., 1., 1., 1., 1., 1.]



    # plus
    sims = sims_hr + sims_lr + sims_sr
    lbls = lbl_hr + lbl_lr + lbl_sr
    colors = color_hr + color_lr  + color_sr
    masks = masks_hr + masks_lr + masks_sr
    lss = lss_hr + lss_lr + lss_sr
    lws = lws_hr + lws_lr + lws_sr

    v_ns = ["theta", "Y_e", "vel_inf", "entropy"]
    i_x_plot = 1
    for v_n in v_ns:
        averages[v_n] = {}
        for sim, lbl, mask, color, ls, lw in zip(sims, lbls, masks, colors, lss, lws):
            if sim != "":
                # loading hist
                fpath = Paths.ppr_sims + sim + "/" + "outflow_{}/".format(det) + mask + '/' + "hist_{}.dat".format(v_n)
                if not os.path.isfile(fpath):
                    raise IOError("File does not exist: {}".format(fpath))
                hist = np.loadtxt(fpath, usecols=(0, 1), unpack=False)

                # loading times
                fpath1 = Paths.ppr_sims + sim + "/" + "outflow_{}/".format(det) + mask + '/' + "total_flux.dat"
                if not os.path.isfile(fpath1):
                    raise IOError("File does not exist: {}".format(fpath1))
                timearr1, massarr1 = np.loadtxt(fpath1, usecols=(0, 2), unpack=True)

                if v_n == "Y_e":
                    ave = EJECTA_PARS.compute_ave_ye(massarr1[-1], hist)
                    averages[v_n][sim] = ave
                elif v_n == "theta":
                    ave = EJECTA_PARS.compute_ave_theta_rms(hist)
                    averages[v_n][sim] = ave
                elif v_n == "vel_inf":
                    ave = EJECTA_PARS.compute_ave_vel_inf(massarr1[-1], hist)
                    averages[v_n][sim] = ave
                elif v_n == "entropy":
                    ave = EJECTA_PARS.compute_ave_vel_inf(massarr1[-1], hist)
                    averages[v_n][sim] = ave
                else:
                    raise NameError("no averages set for v_n:{}".format(v_n))

                plot_dic = {
                    'task': 'hist1d', 'ptype': 'cartesian',
                    'position': (1, i_x_plot),
                    'data': hist, 'normalize': True,
                    'v_n_x': v_n, 'v_n_y': None,
                    'color': color, 'ls': ls, 'lw': lw, 'ds': 'steps', 'alpha': 1.0,
                    'xmin': None, 'xamx': None, 'ymin': 1e-3, 'ymax': 1e0,
                    'xlabel': Labels.labels(v_n), 'ylabel': Labels.labels("mass"),
                    'label': lbl, 'yscale': 'log',
                    'fancyticks': True, 'minorticks': True,
                    'fontsize': 14,
                    'labelsize': 14,
                    'sharex': False,
                    'sharey': False,
                    'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
                }
                plot_dic = Limits.in_dic(plot_dic)
                if v_n != v_ns[0]:
                    plot_dic["sharey"] = True
                if v_n == v_ns[-1] and sim == sims[-1]:
                    plot_dic['legend'] = {'bbox_to_anchor': (-3.00, 1.0), 'loc': 'upper left', 'ncol':4,"fontsize":12}

                o_plot.set_plot_dics.append(plot_dic)

        i_x_plot += 1
    o_plot.main()

    for v_n in v_ns:
        print(" --- v_n: {} --- ".format(v_n))
        for sim_hr, sim_sr, sim_lr in zip(sims_hr, sims_sr, sims_lr):
            # print(sim_hr, sim_sr, sim_lr)
            if not sim_sr == "":
                assert sim_sr != ""
                def_sim = sim_sr
                def_res = "SR"

                if sim_hr != '':
                    comp_res = "HR"
                    comp_sim = sim_hr
                elif  sim_hr == '' and sim_lr != '':
                    comp_res = "LR"
                    comp_sim = sim_lr
                else:
                    raise ValueError("neither HR nor LR is available")

                # print(def_sim, comp_sim)

                assert comp_sim != ""

                val1 = averages[v_n][def_sim]
                val2 = averages[v_n][comp_sim]
                err = 100 * (val1 - val2) / val1
                print("\t{}  :  {:.2f}".format(def_sim, val1))
                print("\t{}  :  {:.2f}".format(comp_sim, val2))
                print("\t\tErr:\t\t{:.1f}".format(err))
        print(" -------------------- ".format(v_n))



    exit(1)
# plot_histograms_lk_off_resolution("geo")
# plot_histograms_lk_off_resolution("bern_geoend")

''' neutrino driven wind '''

def plot_several_q_eff(v_n, sims, iterations, figname):

    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots+"all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (12., 3.2)  # <->, |] # to match hists with (8.5, 2.7)
    o_plot.gen_set["figname"] = figname
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = False
    o_plot.gen_set["subplots_adjust_h"] = 0.2
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []

    rl = 3
    # v_n = "Q_eff_nua"

    # sims = ["LS220_M14691268_M0_LK_SR"]
    # iterations = [1302528, 1515520, 1843200]

    i_x_plot = 1
    i_y_plot = 1
    for sim in sims:

        d3class = LOAD_PROFILE_XYXZ(sim)
        d1class = ADD_METHODS_ALL_PAR(sim)

        for it in iterations:

            tmerg = d1class.get_par("tmerg")
            time_ = d3class.get_time_for_it(it, "prof")

            dens_arr = d3class.get_data(it, rl, "xz", "density")
            data_arr = d3class.get_data(it, rl, "xz", v_n)
            data_arr = data_arr / dens_arr
            x_arr = d3class.get_data(it, rl, "xz", "x")
            z_arr = d3class.get_data(it, rl, "xz", "z")

            def_dic_xz = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                          'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                          'position': (i_y_plot, i_x_plot),  # 'title': '[{:.1f} ms]'.format(time_),
                          'cbar': {},
                          'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                          'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None, 'vmin': 1e-10, 'vmax': 1e-4,
                          'fill_vmin': False,  # fills the x < vmin with vmin
                          'xscale': None, 'yscale': None,
                          'mask': None, 'cmap': 'inferno_r', 'norm': "log",
                          'fancyticks': True,
                          'minorticks': True,
                          'title': {"text": r'$t-t_{merg}:$' + r'${:.1f}$'.format((time_-tmerg)*1e3), 'fontsize': 14},
                          # 'sharex': True,  # removes angular citkscitks
                          'fontsize': 14,
                          'labelsize': 14,
                          'sharex': False,
                          'sharey': True,
                          }

            def_dic_xz["xmin"], def_dic_xz["xmax"], _, _, def_dic_xz["ymin"], def_dic_xz["ymax"] \
                = UTILS.get_xmin_xmax_ymin_ymax_zmin_zmax(rl)

            if v_n == 'Q_eff_nua':

                def_dic_xz['v_n'] = 'Q_eff_nua/D'
                def_dic_xz['vmin'] = 1e-7
                def_dic_xz['vmax'] = 1e-3
                # def_dic_xz['norm'] = None
            elif v_n == 'Q_eff_nue':

                def_dic_xz['v_n'] = 'Q_eff_nue/D'
                def_dic_xz['vmin'] = 1e-7
                def_dic_xz['vmax'] = 1e-3
                # def_dic_xz['norm'] = None
            elif v_n == 'Q_eff_nux':

                def_dic_xz['v_n'] = 'Q_eff_nux/D'
                def_dic_xz['vmin'] = 1e-10
                def_dic_xz['vmax'] = 1e-4
                # def_dic_xz['norm'] = None
                # print("v_n: {} [{}->{}]".format(v_n, def_dic_xz['zarr'].min(), def_dic_xz['zarr'].max()))
            elif v_n == "R_eff_nua":

                def_dic_xz['v_n'] = 'R_eff_nua/D'
                def_dic_xz['vmin'] = 1e2
                def_dic_xz['vmax'] = 1e6
                # def_dic_xz['norm'] = None

                print("v_n: {} [{}->{}]".format(v_n, def_dic_xz['zarr'].min(), def_dic_xz['zarr'].max()))
                # exit(1)

            if it == iterations[0]:
                def_dic_xz["sharey"] = False

            if it == iterations[-1]:
                def_dic_xz['cbar'] = {'location': 'right .02 0.', 'label': Labels.labels(v_n) + "/D",  # 'right .02 0.' 'fmt': '%.1e',
                                   'labelsize': 14, 'aspect': 6.,
                                   'fontsize': 14}

            o_plot.set_plot_dics.append(def_dic_xz)

            i_x_plot = i_x_plot + 1
        i_y_plot = i_y_plot + 1
    o_plot.main()
    exit(0)

''' disk histogram evolution0000 '''

def plot_disk_hist_evol_one_v_n(v_n, sim, figname):

    # sim = "LS220_M13641364_M0_LK_SR_restart"
    # v_n = "Ye"
    # figname = "ls220_ye_disk_hist.png"
    print(v_n)

    d3_corr = LOAD_RES_CORR(sim)
    iterations = d3_corr.list_iterations
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
    values = np.reshape(np.array(values), newshape=(len(iterations),len(bins))).T
    #
    d1class = ADD_METHODS_ALL_PAR(sim)
    tmerg = d1class.get_par("tmerg") * 1e3
    times = times - tmerg
    #
    values = values / np.sum(values)
    values = np.maximum(values, 1e-10)
    #
    if v_n in ["theta"]:
        bins = bins / np.pi * 180.
    #
    def_dic = {'task': 'colormesh', 'ptype': 'cartesian', # 'aspect': 1.,
                  'xarr': times, "yarr": bins, "zarr": values,
                  'position': (1, 1),  # 'title': '[{:.1f} ms]'.format(time_),
                  'cbar': {'location': 'right .02 0.', 'label': Labels.labels("mass"),  # 'right .02 0.' 'fmt': '%.1e',
                                   'labelsize': 14, #'aspect': 6.,
                                   'fontsize': 14},
                  'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                  'xlabel': Labels.labels("t-tmerg"), 'ylabel': Labels.labels(v_n),
                  'xmin': times.min(), 'xmax': times.max(), 'ymin': bins.min(), 'ymax': bins.max(), 'vmin': 1e-6, 'vmax': 1e-2,
                  'fill_vmin': False,  # fills the x < vmin with vmin
                  'xscale': None, 'yscale': None,
                  'mask': None, 'cmap': 'Greys', 'norm': "log",
                  'fancyticks': True,
                  'minorticks': True,
                  'title': {}, # "text": r'$t-t_{merg}:$' + r'${:.1f}$'.format((time_ - tmerg) * 1e3), 'fontsize': 14
                  # 'sharex': True,  # removes angular citkscitks
                  'fontsize': 14,
                  'labelsize': 14,
                  'sharex': False,
                  'sharey': False,
                  }
    #
    tcoll = d1class.get_par("tcoll_gw")
    if not np.isnan(tcoll):
        tcoll = (tcoll * 1e3) - tmerg
        tcoll_dic = {'task':'line', 'ptype': 'cartesian',
                     'position': (1,1),
                     'xarr':[tcoll, tcoll], 'yarr':[bins.min(), bins.max()],
                     'color': 'black', 'ls': '-', 'lw': 0.6, 'ds': 'default', 'alpha': 1.0,
                     }
        print(tcoll)
    else:
        print("No tcoll")

    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + "all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |] # to match hists with (8.5, 2.7)
    o_plot.gen_set["figname"] = figname
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = False
    o_plot.gen_set["subplots_adjust_h"] = 0.2
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []
    #
    if not np.isnan(tcoll):
        o_plot.set_plot_dics.append(tcoll_dic)
    o_plot.set_plot_dics.append(def_dic)
    #
    if v_n in ["temp", "dens_unb_bern", "rho"]:
        def_dic["yscale"] = "log"
    #
    o_plot.main()


    exit(1)

def plot_disk_hist_evol(sim, figname):

    # v_ns = ["r", "theta", "Ye", "temp", "velz", "rho", "dens_unb_bern"]

    v_ns = ["velz"]#, "temp", "rho", "dens_unb_bern"]

    d3_corr = LOAD_RES_CORR(sim)
    iterations = d3_corr.list_iterations

    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + "all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (len(v_ns)*3., 2.7)  # <->, |] # to match hists with (8.5, 2.7)
    o_plot.gen_set["figname"] = figname
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = False
    o_plot.gen_set["subplots_adjust_h"] = 0.2
    o_plot.gen_set["subplots_adjust_w"] = 0.3
    o_plot.set_plot_dics = []

    i_plot = 1
    for v_n in v_ns:
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
        values = np.reshape(np.array(values), newshape=(len(iterations), len(bins))).T
        #
        d1class = ADD_METHODS_ALL_PAR(sim)
        tmerg = d1class.get_par("tmerg") * 1e3
        times = times - tmerg
        #
        values = values / np.sum(values)
        values = np.maximum(values, 1e-10)
        #
        if v_n in ["theta"]:
            bins = bins / np.pi * 180.
        #
        def_dic = {'task': 'colormesh', 'ptype': 'cartesian',  # 'aspect': 1.,
                   'xarr': times, "yarr": bins, "zarr": values,
                   'position': (1, i_plot),  # 'title': '[{:.1f} ms]'.format(time_),
                   'cbar': {},
                   'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                   'xlabel': Labels.labels("t-tmerg"), 'ylabel': Labels.labels(v_n),
                   'xmin': times.min(), 'xmax': times.max(), 'ymin': bins.min(), 'ymax': bins.max(), 'vmin': 1e-6,
                   'vmax': 1e-2,
                   'fill_vmin': False,  # fills the x < vmin with vmin
                   'xscale': None, 'yscale': None,
                   'mask': None, 'cmap': 'Greys', 'norm': "log",
                   'fancyticks': True,
                   'minorticks': True,
                   'title': {},  # "text": r'$t-t_{merg}:$' + r'${:.1f}$'.format((time_ - tmerg) * 1e3), 'fontsize': 14
                   # 'sharex': True,  # removes angular citkscitks
                   'fontsize': 14,
                   'labelsize': 14,
                   'sharex': False,
                   'sharey': False,
                   }
        if v_n == v_ns[-1]:
            def_dic['cbar'] = {'location': 'right .02 0.', 'label': Labels.labels("mass"),  # 'right .02 0.' 'fmt': '%.1e',
                            'labelsize': 14,  # 'aspect': 6.,
                            'fontsize': 14}
        # if v_n == "velz":
        #     def_dic['ymin'] = -.3
        #     def_dic['ymax'] = .3
        #
        tcoll = d1class.get_par("tcoll_gw")
        if not np.isnan(tcoll):
            tcoll = (tcoll * 1e3) - tmerg
            tcoll_dic = {'task': 'line', 'ptype': 'cartesian',
                         'position': (1, i_plot),
                         'xarr': [tcoll, tcoll], 'yarr': [bins.min(), bins.max()],
                         'color': 'black', 'ls': '-', 'lw': 0.6, 'ds': 'default', 'alpha': 1.0,
                         }
            print(tcoll)
        else:
            print("No tcoll")

        #
        if not np.isnan(tcoll):
            o_plot.set_plot_dics.append(tcoll_dic)
        o_plot.set_plot_dics.append(def_dic)
        #
        if v_n in ["temp", "dens_unb_bern", "rho"]:
            def_dic["yscale"] = "log"
        #
        i_plot = i_plot + 1
    o_plot.main()


if __name__ == '__main__':

    ''' --- neutrinos --- '''
    # plot_several_q_eff("Q_eff_nua", ["LS220_M14691268_M0_LK_SR"], [1302528, 1515520, 1843200], "ls220_q_eff.png")
    # plot_several_q_eff("Q_eff_nua", ["DD2_M15091235_M0_LK_SR"], [1277952, 1425408, 1540096], "dd2_q_eff.png")
    #
    # plot_several_q_eff("R_eff_nua", ["LS220_M14691268_M0_LK_SR"], [1302528, 1515520, 1843200], "ls220_r_eff.png")
    # plot_several_q_eff("R_eff_nua", ["DD2_M15091235_M0_LK_SR"], [1277952, 1425408, 1540096], "dd2_r_eff.png")

    ''' disk properties '''

    plot_disk_hist_evol("LS220_M13641364_M0_LK_SR_restart", "ls220_disk_hists.png")

    plot_disk_hist_evol_one_v_n("Ye", "LS220_M13641364_M0_LK_SR_restart", "ls220_ye_disk_hist.png")
    plot_disk_hist_evol_one_v_n("temp", "LS220_M13641364_M0_LK_SR_restart", "ls220_temp_disk_hist.png")
    plot_disk_hist_evol_one_v_n("rho", "LS220_M13641364_M0_LK_SR_restart", "ls220_rho_disk_hist.png")
    plot_disk_hist_evol_one_v_n("dens_unb_bern", "LS220_M13641364_M0_LK_SR_restart", "ls220_dens_unb_bern_disk_hist.png")
    plot_disk_hist_evol_one_v_n("velz", "LS220_M13641364_M0_LK_SR_restart", "ls220_velz_disk_hist.png")

    # o_err = ErrorEstimation("DD2_M15091235_M0_LK_SR","DD2_M14971245_M0_SR")
    # o_err.main(rewrite=False)
    # # plot_total_fluxes_lk_on_off("bern_geoend")
    # exit(1)
    ''' --- COMPARISON TABLE --- '''
    tbl = COMPARISON_TABLE()

    ### effect of viscosity
    # tbl.print_mult_table([["DD2_M15091235_M0_LK_SR", "DD2_M14971245_M0_SR"],
    #                       ["DD2_M13641364_M0_LK_SR_R04", "DD2_M13641364_M0_SR_R04"],
    #                       ["LS220_M14691268_M0_LK_SR", "LS220_M14691268_M0_SR"],
    #                       ["SFHo_M14521283_M0_LK_SR", "SFHo_M14521283_M0_SR"]],
    #                      [r"\hline",
    #                       r"\hline",
    #                       r"\hline",
    #                       r"\hline"],
    #                       r"{Analysis of the viscosity effect on the outflow properties and disk mass. "
    #                       r"Here the $t_{\text{disk}}$ is the maximum postmerger time, for which the 3D is "
    #                       r"available for both simulations For that time, the disk mass is interpolated using "
    #                       r"linear inteprolation. The $\Delta t_{\text{wind}}$ is the maximum common time window "
    #                       r"between the time at which dynamical ejecta reaches 98\% of its total mass and the end of the "
    #                       r"simulation Cases where $t_{\text{disk}}$ or $\Delta t_{\text{wind}}$ is N/A indicate the absence "
    #                      r"of the ovelap between 3D data fro simulations or absence of this data entirely and "
    #                      r"absence of overlap between the time window in which the spiral-wave wind is computed "
    #                      r"which does not allow to do a proper, one-to-one comparison. $\Delta$ is a estimated "
    #                      r"change as $|value_1 - value_2|/value_1$ in percentage }"
    #                      )
    # exit(0)

    #### resulution effect on simulations with viscosity
    # tbl.print_mult_table([["DD2_M13641364_M0_LK_SR_R04", "DD2_M13641364_M0_LK_HR_R04"], # DD2_M13641364_M0_LK_LR_R04
    #                      ["DD2_M15091235_M0_LK_SR", "DD2_M15091235_M0_LK_HR"],          # no
    #                      ["LS220_M14691268_M0_LK_SR", "LS220_M14691268_M0_LK_HR"],      # no
    #                      ["SFHo_M13641364_M0_LK_SR", "SFHo_M13641364_M0_LK_HR"],        # no
    #                      ["SFHo_M14521283_M0_LK_SR", "SFHo_M14521283_M0_LK_HR"]],       # no
    #                      [r"\hline",
    #                       r"\hline",
    #                       r"\hline",
    #                       r"\hline",
    #                       r"\hline"],
    #                      r"{Resolution effec to on the outflow properties and disk mass on the simulations with "
    #                      r"subgird turbulence. Here the $t_{\text{disk}}$ "
    #                      r"is the maximum postmerger time, for which the 3D is available for both simulations "
    #                      r"For that time, the disk mass is interpolated using linear inteprolation. The "
    #                      r"$\Delta t_{\text{wind}}$ is the maximum common time window between the time at "
    #                      r"which dynamical ejecta reaches 98\% of its total mass and the end of the simulation "
    #                      r"Cases where $t_{\text{disk}}$ or $\Delta t_{\text{wind}}$ is N/A indicate the absence "
    #                      r"of the ovelap between 3D data fro simulations or absence of this data entirely and "
    #                      r"absence of overlap between the time window in which the spiral-wave wind is computed "
    #                      r"which does not allow to do a proper, one-to-one comparison. $\Delta$ is a estimated "
    #                      r"change as $|value_1 - value_2|/value_1$ in percentage }"
    #                      )
    # exit(0)
    
    # resolution effect on simulations without voscosity
    tbl.print_mult_table([["DD2_M13641364_M0_SR_R04", "DD2_M13641364_M0_LR_R04"],
                         ["DD2_M14971245_M0_SR", "DD2_M14971245_M0_HR"],
                         ["LS220_M13641364_M0_SR", "LS220_M13641364_M0_HR"],
                         ["LS220_M14691268_M0_SR", "LS220_M14691268_M0_HR"],
                         ["SFHo_M13641364_M0_SR", "SFHo_M13641364_M0_HR"],
                         ["SFHo_M14521283_M0_SR", "SFHo_M14521283_M0_HR"]],
                         [r"\hline",
                          r"\hline",
                          r"\hline",
                          r"\hline",
                          r"\hline",
                          r"\hline"],
                         r"{Resolution effec to on the outflow properties and disk mass on the simulations without "
                         r"subgird turbulence. Here the $t_{\text{disk}}$ "
                         r"is the maximum postmerger time, for which the 3D is available for both simulations "
                         r"For that time, the disk mass is interpolated using linear inteprolation. The "
                         r"$\Delta t_{\text{wind}}$ is the maximum common time window between the time at "
                         r"which dynamical ejecta reaches 98\% of its total mass and the end of the simulation "
                         r"Cases where $t_{\text{disk}}$ or $\Delta t_{\text{wind}}$ is N/A indicate the absence "
                         r"of the ovelap between 3D data fro simulations or absence of this data entirely and "
                         r"absence of overlap between the time window in which the spiral-wave wind is computed "
                         r"which does not allow to do a proper, one-to-one comparison. $\Delta$ is a estimated "
                         r"change as $|value_1 - value_2|/value_1$ in percentage }"
                         )


    exit(0)

    ''' --- OVERALL TABLE --- '''
    tbl = TEX_TABLES()

    tbl.print_mult_table([simulations["BLh"]["q=1"], simulations["BLh"]["q=1.3"], simulations["BLh"]["q=1.4"], simulations["BLh"]["q=1.7"], simulations["BLh"]["q=1.8"],
                          simulations["DD2"]["q=1"], simulations["DD2"]["q=1.1"], simulations["DD2"]["q=1.2"], simulations["DD2"]["q=1.4"],
                          simulations["LS220"]["q=1"], simulations["LS220"]["q=1.1"], simulations["LS220"]["q=1.2"], simulations["LS220"]["q=1.4"],
                          simulations["SFHo"]["q=1"], simulations["SFHo"]["q=1.1"], simulations["SFHo"]["q=1.4"],
                          simulations["SLy4"]["q=1"], simulations["SLy4"]["q=1.1"]],
                         [r"\hline", r"\hline", r"\hline", r"\hline",
                          r"\hline\hline",
                          r"\hline", r"\hline", r"\hline",
                          r"\hline\hline",
                          r"\hline", r"\hline", r"\hline",
                          r"\hline\hline",
                          r"\hline", r"\hline",
                          r"\hline\hline",
                          r"\hline", r"\hline"])




    # par = COMPUTE_PAR("LS220_M14691268_M0_LK_SR")

    # print("tcoll",par.get_par("tcoll_gw"))
    # print("Mdisk",par.get_par("Mdisk3D"))

    # o_lf = COMPUTE_PAR("SLy4_M13641364_M0_LK_SR")
    # print(o_lf.get_outflow_data(0, "geo", "corr_vel_inf_theta.h5"))
    # print(o_lf.get_collated_data("dens_unbnd.norm1.asc"))
    # print(o_lf.get_gw_data("tmerger.dat"))

    # print(o_lf.get_outflow_par(0, "geo", "Mej_tot"))
    # print(o_lf.get_outflow_par(0, "geo", "Ye_ave"))
    # print(o_lf.get_outflow_par(0, "geo", "vel_inf_ave"))
    # print(o_lf.get_outflow_par(0, "geo", "s_ave"))
    # print(o_lf.get_outflow_par(0, "geo", "theta_rms"))
    # print(o_lf.get_disk_mass())
    # print("---")
    # print(o_lf.get_par("tmerg"))
    # print(o_lf.get_par("Munb_tot"))
    # print(o_lf.get_par("Munb_tot"))
    # print(o_lf.get_par("Munb_bern_tot"))
    # print(o_lf.get_par("tcoll_gw"))
