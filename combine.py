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
import scivis.units as ut # for tmerg
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

from outflowed import EJECTA_PARS
from preanalysis import LOAD_ITTIME
from plotting_methods import PLOT_MANY_TASKS
from utils import *
#

''' ============ LOAD OUTFLOW & COLLATED & Mdisk ============== '''

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

    def __init__(self, sim):

        LOAD_ITTIME.__init__(self, sim)

        self.sim = sim

        self.list_detectors = [0, 1]

        self.list_masks = ["geo", "bern", "bern_geoend", "Y_e04_geoend"]

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

        if not os.path.isdir(Paths.ppr_sims+self.sim+'profiles/'):
            return np.zeros(0,)

        list_iterations = Paths.get_list_iterations_from_res_3d(self.sim, "profiles/")
        if len(list_iterations) == 0:
            return np.zeros(0,)

        time_arr = []
        data_arr = []
        for it in list_iterations:
            fpath = Paths.ppr_sims+self.sim+'/'+"profiles/" + str(int(it))  + '/' + v_n
            if os.path.isfile(fpath):
                data_ = np.float(np.loadtxt(fpath, unpack=True))
                time_ = self.get_time_for_it(it, "prof")
                data_arr.append(data_)
                time_arr.append(time_)

        if len(data_arr) ==0:
            return np.zeros(0,)

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

    def __init__(self, sim):
        LOAD_FILES.__init__(self, sim)

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
        t_total_mass *= Constants.time_constant / 1000  # [s]
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


class COMPUTE_PAR(COMPUTE_ARR):

    def __init__(self, sim):
        COMPUTE_ARR.__init__(self, sim)

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
                return np.nan
            Printcolor.yellow("\tWarning! using defauled M_Inf=2.748, R_GW=400.0 for retardet time")
            ret_time = PHYSICS.get_retarded_time(data, M_Inf=2.748, R_GW=400.0)
            tcoll = ut.conv_time(ut.cactus, ut.cgs, ret_time)
            return tcoll
        elif v_n == "tmerg" or v_n == "tmerger":
            try:
                data = self.get_gw_data("tmerger.dat")
            except IOError:
                Printcolor.yellow("\tWarning! No tmerger.dat found for sim:{}".format(self.sim))
                return np.nan
            Printcolor.yellow("\tWarning! using defauled M_Inf=2.748, R_GW=400.0 for retardet time")
            ret_time = PHYSICS.get_retarded_time(data, M_Inf=2.748, R_GW=400.0)
            tmerg = ut.conv_time(ut.cactus, ut.cgs, ret_time)
            return tmerg
        elif v_n == "tcoll" or v_n == "Mdisk":
            total_mass = self.get_total_mass()
            unb_mass = self.get_tot_unb_mass()
            t, Mtot = total_mass[:, 0]*Constants.time_constant*1e-3, total_mass[:, 1]
            _, Munb = unb_mass[:, 0]*Constants.time_constant*1e-3, unb_mass[:, 1]
            if Mtot[-1] > 1.0:
                Mdisk = np.nan
                tcoll = np.inf
                Printcolor.yellow("Warning: Disk mass at tcoll not estimated")
            else:
                i_BH = np.argmin(Mtot > 1.0)
                tcoll = t[i_BH]  # *1000 #* utime
                i_disk = i_BH + int(1.0 / (t[1] * 1000))  #
                # my solution to i_disk being out of bound:
                if i_disk > len(Mtot): i_disk = len(Mtot) - 1
                Mdisk = Mtot[i_disk] - Munb[i_disk]
            if v_n == "tcoll":
                return tcoll
            else:
                return Mdisk
        elif v_n == "Munb_tot":
            unb_mass = self.get_tot_unb_mass()
            _, Munb = unb_mass[:, 0]*Constants.time_constant*1e-3, unb_mass[:, 1]
            print(unb_mass.shape)
            return Munb[-1]

        elif v_n == "Munb_bern_tot":
            unb_mass = self.get_unb_bern_mass()
            _, Munb = unb_mass[:, 0] * Constants.time_constant * 1e-3, unb_mass[:, 1]
            return Munb[-1]





''' '''

class TEX_TABLE:

    def __init__(self):

        self.sim_groups = [
            [],
            [],
            []
        ]

        self.translate_dic = {
            "Mdisk3D": "$M_{\\text{disk}}$",

            "M1": "$M_a$",
            "M2": "$M_b$",

            "tcoll_gw": "$t_{\\text{BH}}$",

            "theta_rms": "$\\theta_{\\text{ej}}$",

            "Mej_tot": "$M_{\\text{ej}}$",
            "Ye_ave": "$\\langle Y_e \\rangle$",
            "vel_inf_ave": "$\\upsilon_{\\text{ej}}$",

            # "Mej_bern": "$M_{\\text{ej}}^b$",
            # "Yeej_bern": "$\\langle Y_e \\rangle^b$",
            # "vej_bern": "$\\upsilon_{\\text{ej}}^b$",
        }

        self.units_dic = {
            "Mdisk3D": "$[M_{\odot}]$",
            "EOS": " ",
            "LK": "  ",
            "theta_rms": " ",
            "tcoll_gw": "[ms]",

            "Mej_tot": "$[10^{-2} M_{\odot}]$",
            "Ye_ave": "  ",
            "vel_inf_ave": "$[c]$",

            # "Mej_bern": "$[M_{\odot}]$",
            # "Yeej_bern": "  ",
            # "vej_bern": "$[c]$",

            "M1": "$[M_{\odot}]$",
            "M2": "$[M_{\odot}]$",
        }

    def get_lbl(self, v_n, criterion="_0"):
        if v_n == "Mdisk3D": return "$M_{\\text{disk}}$"
        elif v_n == "M1": return "$M_a$"
        elif v_n == "M2": return "$M_b$"
        elif v_n == "tcoll_gw": return "$t_{\\text{BH}}$"
        elif v_n == "theta_rms" and criterion=="_0": return "$\\theta_{\\text{ej}}$"
        elif v_n == "Mej_tot" and criterion=="_0": return "$M_{\\text{ej}}$"
        elif v_n == "Ye_ave" and criterion=="_0": return "$\\langle Y_e \\rangle$"
        elif v_n == "vel_inf_ave" and criterion=="_0": return "$\\upsilon_{\\text{ej}}$"
        elif v_n == "theta_rms" and criterion=="_0_b_w": return "$\\theta_{\\text{ej}}^{\\text{w}}$"
        elif v_n == "Mej_tot" and criterion=="_0_b_w": return "$M_{\\text{ej}}^{\\text{w}}$"
        elif v_n == "Ye_ave" and criterion=="_0_b_w": return "$\\langle Y_e ^{\\text{w}}  \\rangle$"
        elif v_n == "vel_inf_ave" and criterion=="_0_b_w": return "$\\upsilon_{\\text{ej}}^{\\text{w}}$"
        elif v_n == "LK": return "LK"
        else: return v_n

    def get_value(self, sim, v_n, criterion="_0"):

        d1class = ADD_METHODS_1D(sim)
        selfclass = LOAD_INIT_DATA(sim)

        if v_n in d1class.list_parameters:
            if v_n == "tcoll_gw":
                tcoll = d1class.get_par("tcoll_gw")
                if not np.isnan(tcoll):
                    tcoll = float(tcoll - d1class.get_par("tmerger_gw"))
                    return str("$ {:.1f}$".format(tcoll * 1e3))  # ms
                else:
                    print("Failed to load 'tcoll' or 'tmerg' for {}".format(sim))
                    tlast = d1class.get_arr("t_unb_mass")[-1] - float(d1class.get_par("tmerger_gw"))
                    return str("$>{:.1f}$".format(tlast * 1e3))  # ms

            elif v_n == "Mej_tot" and criterion != "_0_b_w" :
                return "$ {:.1f}$" % (d1class.get_par(v_n, criterion) * 1e2)
            elif v_n == "Mej_tot" and criterion == "_0_b_w" and np.isnan(d1class.get_par("tcoll_gw")):
                return "$>{:.1f}$" % (d1class.get_par(v_n, criterion) * 1e2)
            elif v_n == "Mej_tot" and criterion == "_0_b_w" and not np.isnan(d1class.get_par("tcoll_gw")):
                return "$\sim{:.1f}$" % (d1class.get_par(v_n, criterion) * 1e2)

            else:
                return d1class.get_par(v_n, criterion=criterion)

        if v_n == "LK":
            if sim.__contains__("LK"):
                return("LK")
            else:
                return("  ")

        if v_n in selfclass.par_dic.keys():
            print(selfclass.par_dic.keys())
            return selfclass.get_par(v_n)

        raise NameError("v_n:{} is not recognized".format(v_n))

    def get_value2(self, sim, v_n, crit, prec):

        d1class = ADD_METHODS_1D(sim)
        selfclass = LOAD_INIT_DATA(sim)

        if v_n in d1class.list_parameters:
            if v_n == "tcoll_gw":
                tcoll = d1class.get_par("tcoll_gw")
                if not np.isnan(tcoll):
                    #tcoll = float(tcoll - d1class.get_par("tmerger_gw"))
                    return str("$ {:.1f}$".format(tcoll * 1e3))
                else:
                    print("Failed to load 'tcoll' or 'tmerg' for {}".format(sim))
                    tlast = d1class.get_arr("t_unb_mass")[-1] - float(d1class.get_par("tmerger_gw"))
                    return str("$>{:.1f}$".format(tlast * 1e3))  # ms
            elif v_n == "Mdisk3D":
                mdisk = d1class.get_par("Mdisk3D")
                if not np.isnan(mdisk):
                    return("%{}f".format(prec) % mdisk)
                else:
                    return("N/A")

            elif v_n == "Mej_tot" and crit == "_0":
                mej = d1class.get_par(v_n, criterion=crit) * 1e2
                return ("%{}f".format(prec) % mej)

            elif v_n == "Mej_tot" and crit != "_0":
                mej = d1class.get_par(v_n, criterion=crit) * 1e2
                if np.isnan(d1class.get_par("tcoll_gw")):
                    return "$>" + "%{}f".format(prec) % mej + "$"
                else:
                    return "$\sim" + "%{}f".format(prec) % mej + "$"

            elif prec != "str":
                return("%{}f".format(prec) % d1class.get_par(v_n, crit))
            elif prec == "str":
                return (str(d1class.get_par(v_n, crit)))

        if v_n in selfclass.par_dic.keys():

            if v_n == "EOS" and selfclass.get_par("EOS") == "SFHo" and sim.__contains__("2019pizza"):
                return("{}$^{}$".format(str(selfclass.get_par(v_n)), "p"))

            if prec == "str":
                return(str(selfclass.get_par(v_n)))
            else:
                return("%{}f".format(prec) % selfclass.get_par(v_n))

            # print(selfclass.par_dic.keys())
            # return selfclass.get_par(v_n)

        if v_n == "LK":
            if sim.__contains__("LK"):
                return("LK")
            else:
                return("  ")

        raise NameError("v_n:{} is not recognized\n{}\n{}"
                        .format(v_n,d1class.list_parameters, selfclass.par_dic.keys()))

    def print_latex_table(self):


        v_ns =  ["EOS", "LK", "M1",   "M2", 'tcoll_gw', 'Mdisk3D', 'Mej_tot', 'Ye_ave', 'vel_inf_ave', 'theta_rms',
                 'Mej_tot', 'Ye_ave', 'vel_inf_ave', 'theta_rms']
        crits = ["",     "",   "",    "",    '',         '',        '_0',       '_0',       '_0',
                 '_0', '_0_b_w', '_0_b_w', '_0_b_w', '_0_b_w']
        precs = ["str", "str", "1.2", "1.2", "str",      ".2",     ".2",       ".2",       ".2",
                 ".2", ".2", ".2", ".2", ".0"]


        assert len(v_ns) == len(crits)
        assert len(precs) == len(v_ns)

        all_raws = []
        for ig, sim_group in enumerate(self.sim_groups):
            rows = []
            for i, sim in enumerate(sim_group):
                # 1 & 6 & 87837 & 787 \\
                row = ''
                j = 0
                for v_n, prec, crit in zip(v_ns, precs, crits):
                    print("\tPrinting {}".format(v_n))
                    # if prec != "str":
                    #     __val = self.get_value2(sim, v_n, crit=crit, prec=prec)
                    #     # if not np.isnan(__val):
                    #     #     val = "%{}f".format(prec) % __val
                    #     # else:
                    #     #     val = "N/A"
                    # else:
                    val = self.get_value2(sim, v_n, crit=crit, prec=prec)
                    row = row + val
                    if j != len(v_ns)-1: row = row + ' & '
                    j = j + 1
                    # if v_n != v_ns[-1]: row = row + ' & '
                row = row + ' \\\\'  # = \\
                rows.append(row)
            all_raws.append(rows)
        # -------------------------------------------
        print("\n")
        size = '{'
        head = ''
        i = 0
        for v_n, crit in zip(v_ns, crits):
            v_n = self.get_lbl(v_n, criterion=crit)
            size = size + 'c'
            head = head + '{}'.format(v_n)
            if v_n != v_ns[-1]: size = size + ' '
            if i != len(v_ns) - 1: head = head + ' & '
            i = i + 1
        size = size + '}'

        unit_bar = ''
        for i, v_n in enumerate(v_ns):
            if v_n in self.units_dic.keys():
                unit = self.units_dic[v_n]
            else:
                unit = v_n
            unit_bar = unit_bar + '{}'.format(unit)
            # if v_ns.index(v_n) != len(v_ns): unit_bar = unit_bar + ' & '
            if i != len(v_ns)-1: unit_bar = unit_bar + ' & '

        head = head + ' \\\\'  # = \\
        unit_bar = unit_bar + ' \\\\ '

        print('\\begin{table*}[t]')
        print('\\begin{center}')
        print('\\begin{tabular}' + '{}'.format(size))
        print('\\hline')
        print(head)
        print(unit_bar)
        print('\\hline\\hline')

        for ig, sim_group in enumerate(self.sim_groups):
            rows = all_raws[ig]
            for row in rows:
                print(row)
            print('\\hline')
        # rows = []
        # for i, sim in enumerate(sims):
        #
        #     # 1 & 6 & 87837 & 787 \\
        #     row = ''
        #     for v_n, prec, crit in zip(v_ns, precs, crits):
        #         print("\tPrinting {}".format(v_n))
        #         if prec != "str":
        #             val = "%{}f".format(prec) % self.get_value(sim, v_n, crit)
        #         else:
        #             val = self.get_value(sim, v_n)
        #         row = row + val
        #         if v_n != v_ns[-1]: row = row + ' & '
        #     row = row + ' \\\\'  # = \\
        #     rows.append(row)
            # print(row)


        print('\\hline')
        print('\\end{tabular}')
        print('\\end{center}')
        print('\\caption{I am your table! }')
        print('\\label{tbl:1}')
        print('\\end{table*}')

        exit(0)


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

    for sim1, mask1, sim2, mask2 in zip(sims, masks, sims2, masks2):

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
    sims_hr  = ["DD2_M13641364_M0_LK_HR_R04", "DD2_M15091235_M0_LK_HR", "", "LS220_M14691268_M0_LK_HR", "SFHo_M13641364_M0_LK_HR", "SFHo_M14521283_M0_LK_HR"]
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
    sims_hr  = ["", "DD2_M14971245_M0_HR", "LS220_M13641364_M0_HR", "LS220_M14691268_M0_HR", "SFHo_M13641364_M0_HR", "SFHo_M14521283_M0_HR"]
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

''' --- '''
if __name__ == '__main__':
    o_lf = COMPUTE_PAR("SLy4_M13641364_M0_LK_SR")
    # print(o_lf.get_outflow_data(0, "geo", "corr_vel_inf_theta.h5"))
    # print(o_lf.get_collated_data("dens_unbnd.norm1.asc"))
    # print(o_lf.get_gw_data("tmerger.dat"))

    print(o_lf.get_outflow_par(0, "geo", "Mej_tot"))
    print(o_lf.get_outflow_par(0, "geo", "Ye_ave"))
    print(o_lf.get_outflow_par(0, "geo", "vel_inf_ave"))
    print(o_lf.get_outflow_par(0, "geo", "s_ave"))
    print(o_lf.get_outflow_par(0, "geo", "theta_rms"))
    print(o_lf.get_disk_mass())
    print("---")
    print(o_lf.get_par("tmerg"))
    print(o_lf.get_par("Munb_tot"))
    print(o_lf.get_par("Munb_tot"))
    print(o_lf.get_par("Munb_bern_tot"))
    print(o_lf.get_par("tcoll_gw"))
