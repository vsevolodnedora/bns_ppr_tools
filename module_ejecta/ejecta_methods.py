"""
    descritpion
"""

from __future__ import division
import units as ut # for tmerg
from math import pi, log10, sqrt
import os.path
import copy
import h5py
import numpy as np
import os
import sys
import click
from argparse import ArgumentParser
from scipy.interpolate import RegularGridInterpolator

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

import multiprocessing as mp
from functools import partial

# from module_preanalysis.preanalysis import LOAD_ITTIME
from ejecta_formulas import FORMULAS
from uutils import Constants, Printcolor
from hist_bins import get_hist_bins_ej

class LOAD_OUTFLOW_SURFACE_H5:

    def __init__(self, fname):

        # LOAD_ITTIME.__init__(self, sim, pprdir=pprdir)

        self.fname = fname

        # self.list_detectors = [0, 1]

        self.list_v_ns = ["fluxdens", "w_lorentz", "eninf", "surface_element",
                          "alp", "rho", "vel[0]", "vel[1]", "vel[2]", "Y_e",
                          "press", "entropy", "temperature", "eps"]

        self.list_grid_v_ns = ["theta", "phi", 'iterations', 'times', "area"]

        self.list_v_ns += self.list_grid_v_ns

        self.grid_pars = ["radius", "ntheta", "nphi"]

        self.matrix_data = [np.empty(0,) for v in range(len(self.list_v_ns)+len(self.list_grid_v_ns))]


        self.matrix_grid_pars = {}

    # def update_v_n(self, new_v_n=None):
    #     if new_v_n != None:
    #         if not new_v_n in self.list_v_ns:
    #             self.list_v_ns.append(v_n)
    #
    #             self.matrix_data = [[np.empty(0, )
    #                                  for v in range(len(self.list_v_ns) + len(self.list_grid_v_ns))]
    #                                 for d in range(len(self.list_detectors))]


    def check_v_n(self, v_n):
        if not v_n in self.list_v_ns:
            raise NameError("v_n:{} not in the list of v_ns: {}"
                            .format(v_n, self.list_v_ns))

    def i_v_n(self, v_n):
        return int(self.list_v_ns.index(v_n))

    def load_h5_file(self):

        assert os.path.isfile(self.fname)
        print("\tLoading {}".format(self.fname))
        dfile = h5py.File(self.fname, "r")

        # attributes
        radius = float(dfile.attrs["radius"])
        ntheta = int(dfile.attrs["ntheta"])
        nphi   = int(dfile.attrs["nphi"])

        self.matrix_grid_pars["radius"] = radius
        self.matrix_grid_pars["ntheta"] = ntheta
        self.matrix_grid_pars["nphi"] = nphi

        for v_n in dfile:
            self.check_v_n(v_n)
            arr = np.array(dfile[v_n])
            self.matrix_data[self.i_v_n(v_n)] = arr

    def is_file_loaded(self):
        data = self.matrix_data[self.i_v_n(self.list_v_ns[0])]
        if len(data) == 0:
            self.load_h5_file()
        data = self.matrix_data[self.i_v_n(self.list_v_ns[0])]
        if len(data) == 0:
            raise ValueError("Error in loading/extracing data. Emtpy array")

    def get_full_arr(self, v_n):
        self.check_v_n(v_n)
        self.is_file_loaded()
        return self.matrix_data[self.i_v_n(v_n)]

    def get_grid_par(self, v_n):
        self.is_file_loaded()
        return self.matrix_grid_pars[v_n]


class COMPUTE_OUTFLOW_SURFACE_H5(LOAD_OUTFLOW_SURFACE_H5):

    def __init__(self, fname):

        LOAD_OUTFLOW_SURFACE_H5.__init__(self, fname=fname)

        self.list_comp_v_ns = ["enthalpy", "vel_inf", "vel_inf_bern", "vel", "logrho"]

        # self.list_v_ns = self.list_v_ns + self.list_comp_v_ns

        self.matrix_comp_data = [np.empty(0,) for v in self.list_comp_v_ns]

    def check_comp_v_n(self, v_n):
        if not v_n in self.list_comp_v_ns:
            raise NameError("v_n: {} is not in the list v_ns: {}"
                            .format(v_n, self.list_comp_v_ns))

    def i_comp_v_n(self, v_n):
        return int(self.list_comp_v_ns.index(v_n))

    def compute_arr(self, v_n):

        if v_n == "enthalpy":
            arr = FORMULAS.enthalpy(self.get_full_arr("eps"),
                                    self.get_full_arr("press"),
                                    self.get_full_arr("rho"))
        elif v_n == "vel_inf":
            arr = FORMULAS.vinf(self.get_full_arr("eninf"))
        elif v_n == "vel_inf_bern":
            # print("----------------------------------------")
            arr = FORMULAS.vinf_bern(self.get_full_arr("eninf"),
                                     self.get_full_comp_arr("enthalpy"))
        elif v_n == "vel":
            arr = FORMULAS.vel(self.get_full_arr("w_lorentz"))
        elif v_n == "logrho":
            arr = np.log10(self.get_full_arr("rho"))
        else:
            raise NameError("No computation method for v_n:{} is found"
                            .format(v_n))
        return arr


    def is_arr_computed(self, v_n):
        arr = self.matrix_comp_data[self.i_comp_v_n(v_n)]
        if len(arr) == 0:
            arr = self.compute_arr(v_n)
            self.matrix_comp_data[self.i_comp_v_n(v_n)] = arr
        if len(arr) == 0:
            raise ValueError("Computation of v_n:{} has failed. Array is emtpy"
                             .format(v_n))

    def get_full_comp_arr(self, v_n):
        self.check_comp_v_n(v_n)
        self.is_arr_computed(v_n)
        arr = self.matrix_comp_data[self.i_comp_v_n(v_n)]
        return arr

    def get_full_arr(self, v_n):
        if v_n in self.list_comp_v_ns:
            self.check_comp_v_n(v_n)
            self.is_arr_computed(v_n)
            arr = self.matrix_comp_data[self.i_comp_v_n(v_n)]
            return arr
        else:
            self.check_v_n(v_n)
            self.is_file_loaded()
            arr = self.matrix_data[self.i_v_n(v_n)]
            return arr


class ADD_MASK(COMPUTE_OUTFLOW_SURFACE_H5):

    def __init__(self, fname, add_mask=None):

        COMPUTE_OUTFLOW_SURFACE_H5.__init__(self, fname=fname)

        self.list_masks = ["geo", "geo_v06",
                           "bern", "bern_geoend", "Y_e04_geoend", "theta60_geoend",
                           "geo_entropy_above_10", "geo_entropy_below_10"]
        if add_mask != None and not add_mask in self.list_masks:
            self.list_masks.append(add_mask)


        # "Y_e04_geoend"
        self.mask_matrix = [np.zeros(0,) for i in range(len(self.list_masks))]

        self.set_min_eninf = 0.
        self.set_min_enthalpy = 1.0022

    def check_mask(self, mask):
        if not mask in self.list_masks:
            raise NameError("mask: {} is not in the list: {}"
                            .format(mask, self.list_masks))

    def i_mask(self, mask):
        return int(self.list_masks.index(mask))

    # ----------------------------------------------
    def __time_mask_end_geo(self, length=0.):

        fluxdens = self.get_full_arr("fluxdens")
        da = self.get_full_arr("surface_element")
        t = self.get_full_arr("times")
        dt = np.diff(t)
        dt = np.insert(dt, 0, 0)
        mask = self.get_mask("geo").astype(int)
        fluxdens = fluxdens * mask
        flux_arr = np.sum(np.sum(fluxdens * da, axis=1), axis=1)  # sum over theta and phi
        tot_mass = np.cumsum(flux_arr * dt)  # sum over time
        tot_flux = np.cumsum(flux_arr)  # sum over time
        # print("totmass:{}".format(tot_mass[-1]))
        fraction = 0.98
        i_t98mass = int(np.where(tot_mass >= fraction * tot_mass[-1])[0][0])
        # print(i_t98mass)
        # assert i_t98mass < len(t)

        if length > 0.:
            if length > t[-1]:
                raise ValueError("length:{} is > t[-1]:{} [ms]".format(length*Constants.time_constant,
                                                                  t[-1]*Constants.time_constant))
            if t[i_t98mass] + length > t[-1]:
                # because of bloody numerics it can > but just by a tiny bit. So I added this shit.
                if np.abs(t[i_t98mass] - length > t[-1]) < 10: # 10 is a rundomly chosen number
                    length = length - 10
                else:
                    raise ValueError("t[i_t98mass] + length > t[-1] : {} > {}"
                                     .format((t[i_t98mass] + length),
                                             t[-1]))

            i_mask = (t > t[i_t98mass]) & (t < t[i_t98mass] + length)
        else:
            i_mask = t > t[i_t98mass]
        # saving time at 98% mass for future use
        # fpath = Paths.ppr_sims + self.sim + '/outflow_{}/t98mass.dat'.format(det)
        # try: open(fpath, "w").write("{}\n".format(float(t[i_t98mass])))
        # except IOError: Printcolor.yellow("\tFailed to save t98mass.dat")
        # continuing with mask
        newmask = np.zeros(fluxdens.shape)
        for i in range(len(newmask[:, 0, 0])):
            newmask[i, :, :].fill(i_mask[i])
        return newmask.astype(bool)
    # ----------------------------------------------

    def compute_mask(self, mask):
        self.check_mask(mask)

        if mask == "geo":
            # 1 - if geodeisc is true
            einf = self.get_full_arr("eninf")
            res = (einf >= self.set_min_eninf)
            return res
        if mask == "geo_v06":
            # 1 - if geodeisc is true
            einf = self.get_full_arr("eninf")
            vinf = self.get_full_arr("vel_inf")
            res = (einf >= self.set_min_eninf) & (vinf >= 0.6)
            return res
        elif mask == "geo_entropy_below_10":
            einf = self.get_full_arr("eninf")
            res = (einf >= self.set_min_eninf)
            entropy = self.get_full_arr("entropy")
            mask_entropy = entropy < 10.
            return res & mask_entropy
        elif mask == "geo_entropy_above_10":
            einf = self.get_full_arr("eninf")
            res = (einf >= self.set_min_eninf)
            entropy = self.get_full_arr("entropy")
            mask_entropy = entropy > 10.
            return res & mask_entropy
        elif mask == "bern":
            # 1 - if Bernulli is true
            enthalpy = self.get_full_comp_arr("enthalpy")
            einf = self.get_full_arr("eninf")
            res = ((enthalpy * (einf + 1) - 1) > self.set_min_eninf) & (enthalpy >= self.set_min_enthalpy)
        elif mask == "bern_geoend":
            # 1 - data above 98% of GeoMass and if Bernoulli true and 0 if not
            mask2 = self.get_mask("bern")
            newmask = self.__time_mask_end_geo()

            res = newmask & mask2
        elif mask == "Y_e04_geoend":
            # 1 - data above Ye=0.4 and 0 - below
            ye = self.get_full_arr("Y_e")
            mask_ye = ye >= 0.4
            mask_bern = self.get_mask("bern")
            mask_geo_end = self.__time_mask_end_geo()
            return mask_ye & mask_bern & mask_geo_end
        elif mask == "theta60_geoend":
            # 1 - data above Ye=0.4 and 0 - below
            theta = self.get_full_arr("theta")
            # print((theta / np.pi * 180.).min(), (theta / np.pi * 180.).max())
            # exit(1)
            theta_ = 90 - (theta * 180 / np.pi)
            # print(theta_); #exit(1)
            theta_mask = (theta_ > 60.) | (theta_ < -60)
            # print(np.sum(theta_mask.astype(int)))
            # assert np.sum(theta_mask.astype(int)) > 0
            newmask = theta_mask[np.newaxis, : , :]

            # fluxdens = self.get_full_arr(det, "fluxdens")
            # newmask = np.zeros(fluxdens.shape)
            # for i in range(len(newmask[:, 0, 0])):
            #     newmask[i, :, :].fill(theta_mask)

            # print(newmask.shape)
            # exit(1)
            # mask_ye = ye >= 0.4
            mask_bern = self.get_mask("bern")
            # print(mask_bern.shape)
            # print(mask_bern.shape)
            mask_geo_end = self.__time_mask_end_geo()
            return newmask & mask_bern & mask_geo_end
        elif str(mask).__contains__("_tmax"):
            raise NameError(" mask with '_tmax' are not supported")
            #
            # # 1 - data below tmax and 0 - above
            # base_mask_name = str(str(mask).split("_tmax")[0])
            # base_mask = self.get_mask(det, base_mask_name)
            # #
            # tmax = float(str(mask).split("_tmax")[-1])
            # tmax = tmax / Constants.time_constant # Msun
            # # tmax loaded is postmerger tmax. Thus it need to be added to merger time
            # fpath = self.pprdir+"/waveforms/tmerger.dat"
            # try:
            #     tmerg = float(np.loadtxt(fpath, unpack=True)) # Msun
            #     Printcolor.yellow("\tWarning! using defauled M_Inf=2.748, R_GW=400.0 for retardet time")
            #     ret_time = PHYSICS.get_retarded_time(tmerg, M_Inf=2.748, R_GW=400.0)
            #     tmerg = ret_time
            #     # tmerg = ut.conv_time(ut.cactus, ut.cgs, ret_time)
            #     # tmerg = tmerg / (Constants.time_constant *1e-3)
            # except IOError:
            #     raise IOError("For the {} mask, the tmerger.dat is needed at {}"
            #                   .format(mask, fpath))
            # except:
            #     raise ValueError("failed to extract tmerg for outflow tmax mask analysis")
            #
            # t = self.get_full_arr(det, "times") # Msun
            # # tmax = tmax + tmerg       # Now tmax is absolute time (from the begniing ofthe simulation
            # print("t[-1]:{} tmax:{} tmerg:{} -> {}".format(t[-1]*Constants.time_constant*1e-3,
            #                                 tmax*Constants.time_constant*1e-3,
            #                                 tmerg*Constants.time_constant*1e-3,
            #                                 (tmax+tmerg)*Constants.time_constant*1e-3))
            # tmax = tmax + tmerg
            # if tmax > t[-1]:
            #     raise ValueError("tmax:{} for the mask is > t[-1]:{}".format(tmax*Constants.time_constant*1e-3,
            #                                                                  t[-1]*Constants.time_constant*1e-3))
            # if tmax < t[0]:
            #     raise ValueError("tmax:{} for the mask is < t[0]:{}".format(tmax * Constants.time_constant * 1e-3,
            #                                                                  t[0] * Constants.time_constant * 1e-3))
            # fluxdens = self.get_full_arr(det, "fluxdens")
            # i_mask = t < t[UTILS.find_nearest_index(t, tmax)]
            # newmask = np.zeros(fluxdens.shape)
            # for i in range(len(newmask[:, 0, 0])):
            #     newmask[i, :, :].fill(i_mask[i])

            # print(base_mask.shape,newmask.shape)

            # return base_mask & newmask.astype(bool)
        elif str(mask).__contains__("_length"):
            base_mask_name = str(str(mask).split("_length")[0])
            base_mask = self.get_mask(base_mask_name)
            delta_t = float(str(mask).split("_length")[-1])
            delta_t = (delta_t / 1e5) / (Constants.time_constant * 1e-3) # Msun
            t = self.get_full_arr("times")  # Msun
            print("\t t[0]: {}\n\t t[-1]: {}\n\t delta_t: {}\n\t mask: {}"
                  .format(t[0] * Constants.time_constant * 1e-3,
                          t[-1] * Constants.time_constant * 1e-3,
                          delta_t * Constants.time_constant * 1e-3,
                          mask))
            assert delta_t < t[-1]
            assert delta_t > t[0]
            mask2 = self.get_mask("bern")
            newmask = self.__time_mask_end_geo(length=delta_t)

            res = newmask & mask2

        else:
            raise NameError("No method found for computing mask:{}"
                            .format(mask))

        return res

    # ----------------------------------------------

    def is_mask_computed(self, mask):
        if len(self.mask_matrix[self.i_mask(mask)]) == 0:
            arr = self.compute_mask(mask)
            self.mask_matrix[self.i_mask(mask)] = arr

        if len(self.mask_matrix[self.i_mask(mask)]) == 0:
            raise ValueError("Failed to compute the mask: {}".format(mask))

    def get_mask(self, mask):
        self.check_mask(mask)
        self.is_mask_computed(mask)
        return self.mask_matrix[self.i_mask(mask)]


class EJECTA(ADD_MASK):

    def __init__(self, fname, skynetdir, add_mask=None):

        ADD_MASK.__init__(self, fname=fname, add_mask=add_mask)

        self.list_hist_v_ns = ["Y_e", "theta", "phi", "vel_inf", "entropy", "temperature", "logrho"]

        self.list_corr_v_ns = ["Y_e theta", "vel_inf theta", "Y_e vel_inf",
                               "logrho vel_inf", "logrho theta", "logrho Y_e"]

        self.list_ejecta_v_ns = [
                                    "tot_mass", "tot_flux",  "weights", "corr3d Y_e entropy tau",
                                ] +\
                                ["timecorr {}".format(v_n) for v_n in self.list_hist_v_ns] +\
                                ["hist {}".format(v_n) for v_n in self.list_hist_v_ns] +\
                                ["corr2d {}".format(v_n) for v_n in self.list_corr_v_ns] +\
                                ["mass_ave {}".format(v_n) for v_n in self.list_v_ns]

        self.matrix_ejecta = [[np.zeros(0,)
                                for k in range(len(self.list_ejecta_v_ns))]
                                for j in range(len(self.list_masks))]

        self.set_skynet_densmap_fpath = skynetdir + "densmap.h5"
        self.set_skyent_grid_fpath = skynetdir + "grid.h5"

    # ---

    def check_ej_v_n(self, v_n):
        if not v_n in self.list_ejecta_v_ns:
            raise NameError("module_ejecta v_n: {} is not in the list of module_ejecta v_ns {}"
                            .format(v_n, self.list_ejecta_v_ns))

    def i_ejv_n(self, v_n):
        return int(self.list_ejecta_v_ns.index(v_n))

    # --- methods for EJECTA arrays ----

    def get_cumulative_ejected_mass(self, mask):
        fluxdens = self.get_full_arr("fluxdens")
        da = self.get_full_arr("surface_element")
        t = self.get_full_arr("times")
        dt = np.diff(t)
        dt = np.insert(dt, 0, 0)
        mask = self.get_mask(mask).astype(int)
        fluxdens = fluxdens * mask
        flux_arr = np.sum(np.sum(fluxdens * da, axis=1), axis=1)  # sum over theta and phi
        tot_mass = np.cumsum(flux_arr * dt)  # sum over time
        tot_flux = np.cumsum(flux_arr)  # sum over time
        # print("totmass:{}".format(tot_mass[-1]))
        return (t * 0.004925794970773136 / 1e3, flux_arr, tot_mass) # time in [s]

    def get_weights(self, mask):

        dt = np.diff(self.get_full_arr("times"))
        dt = np.insert(dt, 0, 0)
        mask_arr = self.get_mask(mask).astype(int)
        weights = mask_arr * \
                  self.get_full_arr("fluxdens") * \
                  self.get_full_arr("surface_element") * \
                  dt[:, np.newaxis, np.newaxis]
        #
        if np.sum(weights) == 0.:
            _, _, mass = self.get_cumulative_ejected_mass(mask)
            print("Error. sum(weights) = 0. For mask:{} there is not mass (Total ej.mass is {})".format(mask,mass[-1]))
            raise ValueError("sum(weights) = 0. For mask:{} there is not mass (Total ej.mass is {})".format(mask,mass[-1]))
        #
        return weights

    def get_hist(self, mask, v_n, edge):

        times = self.get_full_arr("times")
        weights = np.array(self.get_ejecta_arr(mask, "weights"))
        data = np.array(self.get_full_arr(v_n))
        # if v_n == "rho":
        #     data = np.log10(data)
        historgram = np.zeros(len(edge) - 1)
        # tmp2 = []
        # print(data.shape, weights.shape, edge.shape)
        for i in range(len(times)):
            if np.array(data).ndim == 3: data_ = data[i, :, :].flatten()
            else: data_ = data.flatten()
            # print(data.min(), data.max())
            tmp, _ = np.histogram(data_, bins=edge, weights=weights[i, :, :].flatten())
            historgram += tmp
        middles = 0.5 * (edge[1:] + edge[:-1])
        assert len(historgram) == len(middles)
        if np.sum(historgram) == 0.:
            print("Error. Histogram weights.sum() = 0 ")
            raise ValueError("Error. Histogram weights.sum() = 0 ")
        return middles, historgram

        # res = np.vstack((middles, historgram))
        # return res

    def get_timecorr(self, mask, v_n, edge):

        historgrams = np.zeros(len(edge) - 1)

        times = self.get_full_arr("times")
        timeedges = np.linspace(times.min(), times.max(), 55)

        indexes = []
        for i_e, t_e in enumerate(timeedges[:-1]):
            i_indx = []
            for i, t in enumerate(times):
                if (t >= timeedges[i_e]) and (t<timeedges[i_e+1]):
                    i_indx = np.append(i_indx, int(i))
            indexes.append(i_indx)
        assert len(indexes) > 0
        #
        weights = np.array(self.get_ejecta_arr(mask, "weights"))
        data = np.array(self.get_full_arr(v_n))

        for i_ind, ind_list in enumerate(indexes):
            # print("{} {}/{}".format(i_ind,len(indexes), len(ind_list)))
            historgram = np.zeros(len(edge) - 1)
            for i in np.array(ind_list, dtype=int):
                if np.array(data).ndim == 3: data_ = data[i, :, :].flatten()
                else: data_ = data.flatten()
                tmp, _ = np.histogram(data_, bins=edge, weights=weights[i, :, :].flatten())
                historgram += tmp
            historgrams = np.vstack((historgrams, historgram))

        # print("done") #1min15
        # exit(1)

        bins = 0.5 * (edge[1:] + edge[:-1])

        # print("hist", historgrams.shape)
        # print("times", timeedges.shape)
        # print("edges", bins.shape)

        return bins, timeedges, historgrams

    @staticmethod
    def combine(x, y, xy, corner_val=None):
        '''creates a 2d array  1st raw    [0, 1:] -- x -- density     (log)
                               1st column [1:, 0] -- y -- lemperature (log)
                               Matrix     [1:,1:] -- xy --Opacity     (log)
           0th element in 1st raw (column) - can be used a corner value

        '''
        x = np.array(x)
        y = np.array(y)
        xy = np.array((xy))

        res = np.insert(xy, 0, x, axis=0)
        new_y = np.insert(y, 0, 0, axis=0)  # inserting a 0 to a first column of a
        res = np.insert(res, 0, new_y, axis=1)

        if corner_val != None:
            res[0, 0] = corner_val

        return res

    @staticmethod
    def combine3d(x, y, z, xyz, corner_val=None):
        '''creates a 2d array  1st raw    [0, 1:] -- x -- density     (log)
                               1st column [1:, 0] -- y -- lemperature (log)
                               Matrix     [1:,1:] -- xy --Opacity     (log)
           0th element in 1st raw (column) - can be used a corner value

        '''

        tmp = np.zeros((len(xyz[:, 0, 0])+1, len(xyz[0, :, 0])+1, len(xyz[0, 0, :])+1))
        tmp[1:, 1:, 1:] = xyz
        tmp[1:, 0, 0] = x
        tmp[0, 1:, 0] = y
        tmp[0, 0, 1:] = z
        return tmp

    def get_corr2d(self, mask, v_n1, v_n2, edge1, edge2):

        tuple_edges = tuple([edge1, edge2])
        correlation = np.zeros([len(edge) - 1 for edge in tuple_edges])
        times = self.get_full_arr("times")
        weights = self.get_ejecta_arr(mask, "weights")
        for i_t, t in enumerate(times):

            data1 = self.get_full_arr(v_n1)
            if data1.ndim == 3: data1_ = data1[i_t, :, :].flatten()
            else: data1_ = data1.flatten()

            data2 = self.get_full_arr(v_n2)
            if data2.ndim == 3: data2_ = data2[i_t, :, :].flatten()
            else: data2_ = data2.flatten()

            data = tuple([data1_, data2_])
            tmp, _ = np.histogramdd(data, bins=tuple_edges, weights=weights[i_t, :, :].flatten())
            correlation += tmp

        bins1 = 0.5 * (edge1[1:] + edge1[:-1])
        bins2 = 0.5 * (edge2[1:] + edge2[:-1])

        return bins1, bins2, correlation.T

    def get_corr3d(self, mask, v_n1, v_n2, v_n3, edge1, edge2, edge3):

        tuple_edges = tuple([edge1, edge2, edge3])
        correlation = np.zeros([len(edge) - 1 for edge in tuple_edges])
        times = self.get_full_arr("times")
        weights = self.get_ejecta_arr(mask, "weights")
        for i_t, t in enumerate(times):

            data1 = self.get_full_arr(v_n1)
            if data1.ndim == 3: data1_ = data1[i_t, :, :].flatten()
            else: data1_ = data1.flatten()
            #
            data2 = self.get_full_arr(v_n2)
            if data2.ndim == 3: data2_ = data2[i_t, :, :].flatten()
            else: data2_ = data2.flatten()
            #
            data3 = self.get_full_arr(v_n3)
            if data2.ndim == 3: data3_ = data3[i_t, :, :].flatten()
            else: data3_ = data3.flatten()
            #
            data = tuple([data1_, data2_, data3_])
            tmp, _ = np.histogramdd(data, bins=tuple_edges, weights=weights[i_t, :, :].flatten())
            correlation += tmp

        bins1 = 0.5 * (edge1[1:] + edge1[:-1])
        bins2 = 0.5 * (edge2[1:] + edge2[:-1])
        bins3 = 0.5 * (edge3[1:] + edge3[:-1])

        return bins1, bins2, bins3, correlation.T

    @staticmethod
    def get_edges_from_centers(bins):

        edges = np.array(0.5 * (bins[1:] + bins[:-1]))  # from edges to center of bins
        edges = np.insert(edges, 0, edges[0] - np.diff(edges)[0])
        edges = np.append(edges, edges[-1] + np.diff(edges)[-1])

        return edges

    def get_corr_ye_entr_tau(self, mask):

        densmap = h5py.File(self.set_skynet_densmap_fpath, "r")

        dmap_ye = np.array(densmap["Ye"])
        dmap_rho = np.log10(np.array(densmap["density"]))
        dmap_entr = np.array(densmap["entropy"])

        interpolator = RegularGridInterpolator((dmap_ye, dmap_entr), dmap_rho,
                                               method="linear", bounds_error=False)

        grid = h5py.File(self.set_skyent_grid_fpath,"r")
        grid_ye = np.array(grid["Ye"])
        grid_entr = np.array(grid["entropy"])
        grid_tau = np.array(grid["tau"])

        data_ye = self.get_full_arr("Y_e")
        data_entr = self.get_full_arr("entropy")
        data_rho = self.get_full_arr("rho") * 6.173937319029555e+17 # CGS
        data_vel = self.get_full_comp_arr("vel")

        lrho_b = [[np.zeros(len(data_ye[:, 0, 0]))
                   for i in range(len(data_ye[0, :, 0]))]
                  for j in range(len(data_ye[0, 0, :]))]
        for i_theta in range(len(data_ye[0, :, 0])):
            for i_phi in range(len(data_ye[0, 0, :])):
                data_ye_i = data_ye[:, i_theta, i_phi].flatten()
                data_entr_i = data_entr[:, i_theta, i_phi].flatten()

                data_ye_i[data_ye_i > grid_ye.max()] = grid_ye.max()
                data_entr_i[data_entr_i > grid_entr.max()] = grid_entr.max()
                data_ye_i[data_ye_i < grid_ye.min()] = grid_ye.min()
                data_entr_i[data_entr_i < grid_entr.min()] = grid_entr.min()

                A = np.zeros((len(data_ye_i), 2))
                A[:, 0] = data_ye_i
                A[:, 1] = data_entr_i
                lrho_b_i = interpolator(A)

                lrho_b[i_phi][i_theta] = lrho_b_i
                sys.stdout.flush()

        # from d3analysis import FORMULAS
        lrho_b = np.array(lrho_b, dtype=np.float).T
        data_tau = FORMULAS.get_tau(data_rho, data_vel, self.get_grid_par("radius"), lrho_b)

        weights = self.get_ejecta_arr(mask, "weights")
        edges_ye = self.get_edges_from_centers(grid_ye)
        edges_tau = self.get_edges_from_centers(grid_tau)
        edges_entr = self.get_edges_from_centers(grid_entr)
        edges = tuple([edges_ye, edges_entr, edges_tau])

        correlation = np.zeros([len(edge) - 1 for edge in edges])

        for i in range(len(weights[:, 0, 0])):
            data_ye_i = data_ye[i, :, :]
            data_entr_i = data_entr[i, : ,:]
            data_tau_i = data_tau[i, :, :]
            data = tuple([data_ye_i.flatten(), data_entr_i.flatten(), data_tau_i.flatten()])
            tmp, _ = np.histogramdd(data, bins=edges, weights=weights[i, :, :].flatten())
            correlation += tmp

        bins_ye = 0.5 * (edges_ye[1:] + edges_ye[:-1])
        bins_entr = 0.5 * (edges_entr[1:] + edges_entr[:-1])
        bins_tau = 0.5 * (edges_tau[1:] + edges_tau[:-1])

        if not (np.sum(correlation) > 0):
            print("Error. np.sum(correlation) = 0")
            raise ValueError("np.sum(correlation) = 0")

        if not (np.sum(correlation) <= np.sum(weights)):
            print("Error np.sum(correlation) > np.sum(weights)")
            raise ValueError("np.sum(correlation) <= np.sum(weights)")

        assert correlation.shape == (17, 17, 17)

        return bins_ye, bins_entr, bins_tau, correlation

    def get_mass_averaged(self, mask, v_n):

        dt = np.diff(self.get_full_arr("times"))
        dt = np.insert(dt, 0, 0)
        dt = dt[:, np.newaxis, np.newaxis]
        data = self.get_full_arr(v_n)

        mask = self.get_mask(mask)
        flux = self.get_full_arr("fluxdens") * mask
        mass_averages = np.zeros((len(flux[0, :, 0]), len(flux[0, 0, :])))
        total_flux = np.zeros((len(flux[0, :, 0]), len(flux[0, 0, :])))

        for i in range(len(flux[:, 0, 0])):

            total_flux += flux[i, :, :] * dt[i, :, :]

            if v_n == "fluxdens":
                mass_averages += flux[i, :, :] * dt[i, :, :]
            else:
                mass_averages += data[i, :, :] * flux[i, :, :] * dt[i, :, :]

        return np.array(mass_averages / total_flux)

    # ---------------------------------

    def compute_ejecta_arr(self, mask, v_n):

        # as Bernoulli criteria uses different vel_inf defention
        if v_n.__contains__("vel_inf") and mask.__contains__("bern"):
            v_n = v_n.replace("vel_inf", "vel_inf_bern")


        # ----------------------------------------
        if v_n in ["tot_mass", "tot_flux"]:
            t, flux, mass = self.get_cumulative_ejected_mass(mask)
            arr = np.vstack((t, flux, mass)).T

        elif v_n == "weights":
            arr = self.get_weights(mask)

        elif v_n.__contains__("hist "):
            v_n = str(v_n.split("hist ")[-1])
            edge = get_hist_bins_ej(v_n)
            middles, historgram = self.get_hist(mask, v_n, edge)
            arr = np.vstack((middles, historgram)).T

        elif v_n.__contains__("corr2d "):
            v_n1 = str(v_n.split(" ")[1])
            v_n2 = str(v_n.split(" ")[2])
            edge1 = get_hist_bins_ej(v_n1)
            edge2 = get_hist_bins_ej(v_n2)
            bins1, bins2, weights = self.get_corr2d(mask, v_n1, v_n2, edge1, edge2)
            arr = self.combine(bins1, bins2, weights) # y_arr [1:, 0] x_arr [0, 1:]

        elif v_n.__contains__("timecorr "):
            v_n = str(v_n.split(" ")[1])
            edge = get_hist_bins_ej(v_n)
            bins, binstime, weights = self.get_timecorr(mask, v_n, edge)
            return self.combine(binstime, bins, weights.T)

        elif v_n == "corr3d Y_e entropy tau":
            bins1, bins2, bins3, corr = self.get_corr_ye_entr_tau(mask)
            arr = self.combine3d(bins1, bins2, bins3, corr)

        elif v_n.__contains__("mass_ave "):
            v_n = str(v_n.split("mass_ave ")[-1])
            # print(v_n)
            arr = self.get_mass_averaged(mask, v_n)

        else:
            raise NameError("no method found for computing module_ejecta arr for mask:{} v_n:{}"
                            .format(mask, v_n))


        return arr

    # ---------------------------------

    def is_ejecta_arr_computed(self, mask, v_n):
        data = self.matrix_ejecta[self.i_mask(mask)][self.i_ejv_n(v_n)]
        if len(data) == 0:
            arr = self.compute_ejecta_arr(mask, v_n)
            self.matrix_ejecta[self.i_mask(mask)][self.i_ejv_n(v_n)] = arr

        data = self.matrix_ejecta[self.i_mask(mask)][self.i_ejv_n(v_n)]
        if len(data) == 0:
            raise ValueError("Failed to compute module_ejecta array for "
                             "mask:{} v_n:{}"
                             .format(mask, v_n))

    def get_ejecta_arr(self, mask, v_n):
        self.check_mask(mask)
        self.check_ej_v_n(v_n)
        self.is_ejecta_arr_computed(mask, v_n)
        data = self.matrix_ejecta[self.i_mask(mask)][self.i_ejv_n(v_n)]
        return data


class EJECTA_NUCLEO(EJECTA):

    def __init__(self, fname, skynetdir, add_mask=None):

        EJECTA.__init__(self, fname=fname, skynetdir=skynetdir, add_mask=add_mask)

        self._list_tab_nuc_v_ns = ["Y_final", "A", "Z"]
        self._list_sol_nuc_v_ns = ["Ysun", "Asun"]

        self.list_nucleo_v_ns = ["sim final", "solar final", "yields", "Ye", "mass"] \
                                + self._list_tab_nuc_v_ns + self._list_sol_nuc_v_ns

        self.matrix_ejecta_nucleo = [[np.zeros(0,)
                                     for i in range(len(self.list_nucleo_v_ns))]
                                     for j in range(len(self.list_masks))]

        self.set_table_solar_r_fpath = skynetdir + "solar_r.dat"
        self.set_tabulated_nuc_fpath = skynetdir + "tabulated_nucsyn.h5"

    # ---

    def check_nucleo_v_n(self, v_n):
        if not v_n in self.list_nucleo_v_ns:
            raise NameError("nucleo v_n: {} is not in the list:{}"
                            .format(v_n, self.list_nucleo_v_ns))

    def i_nuc_v_n(self, v_n):
        return int(self.list_nucleo_v_ns.index(v_n))

    # -------------------------------------------------

    def compute_nucleo_arr(self, mask, v_n):

        if v_n in self._list_tab_nuc_v_ns:
            assert os.path.isfile(self.set_tabulated_nuc_fpath)
            dfile = h5py.File(self.set_tabulated_nuc_fpath, "r")
            for v_n in self._list_tab_nuc_v_ns:
                arr = np.array(dfile[v_n])
                self.matrix_ejecta_nucleo[self.i_mask(mask)][self.i_nuc_v_n(v_n)] = arr

        elif v_n in ["Ye", "mass"]:
            data = self.get_ejecta_arr(mask, "corr3d Y_e entropy tau")
            Ye = data[1:, 0, 0]
            mass = data[1:, 1:, 1:]
            self.matrix_ejecta_nucleo[self.i_mask(mask)][self.i_nuc_v_n("Ye")] = Ye
            self.matrix_ejecta_nucleo[self.i_mask(mask)][self.i_nuc_v_n("mass")] = mass

        elif v_n in self._list_sol_nuc_v_ns:
            assert os.path.isfile(self.set_table_solar_r_fpath)
            Asun, Ysun = np.loadtxt(self.set_table_solar_r_fpath, unpack=True)
            self.matrix_ejecta_nucleo[self.i_mask(mask)][self.i_nuc_v_n("Asun")] = Asun
            self.matrix_ejecta_nucleo[self.i_mask(mask)][self.i_nuc_v_n("Ysun")] = Ysun

        elif v_n == "yields":
            mass = np.array(self.get_nucleo_arr(mask, "mass"))
            # Ye = np.array(self.get_nucleo_arr(det, mask, "Ye"))
            Ys = np.array(self.get_nucleo_arr(mask, "Y_final"))
            # As = np.array(self.get_nucleo_arr(det, mask, "A"))
            # Zs = np.array(self.get_nucleo_arr(det, mask, "Z"))

            yields = np.zeros(Ys.shape[-1])
            for i in range(yields.shape[0]):
                yields[i] = np.sum(mass[:, :, :] * Ys[:, :, :, i]) # Relative final abundances
            self.matrix_ejecta_nucleo[self.i_mask(mask)][self.i_nuc_v_n("yields")] = yields

        elif v_n == "sim final":
            yields = self.get_nucleo_arr(mask, "yields")
            A = self.get_nucleo_arr(mask, "A")
            Z = self.get_nucleo_arr(mask, "Z")
            # print(yields.shape)
            # print(A.shape)
            # print(Z.shape)
            arr = np.vstack((yields, A, Z)).T
            self.matrix_ejecta_nucleo[self.i_mask(mask)][self.i_nuc_v_n("sim final")] = arr

        elif v_n == "solar final":
            Asun = self.get_nucleo_arr(mask, "Asun")
            Ysun = self.get_nucleo_arr(mask, "Ysun")
            arr = np.vstack((Asun, Ysun)).T
            self.matrix_ejecta_nucleo[self.i_mask(mask)][self.i_nuc_v_n("solar final")] = arr
        else:
            raise NameError("no nucleo method found for v_n:{} mask:{}"
                            .format(v_n, mask))

    # -------------------------------------------------

    def is_nucleo_arr_computed(self, mask, v_n):
        data = self.matrix_ejecta_nucleo[self.i_mask(mask)][self.i_nuc_v_n(v_n)]
        if len(data) == 0:
            self.compute_nucleo_arr(mask, v_n)

        data = self.matrix_ejecta_nucleo[self.i_mask(mask)][self.i_nuc_v_n(v_n)]
        if len(data) == 0:
            raise ValueError("failed to compute nucleo arr for mask:{} v_n:{}"
                             .format(mask, v_n))

    def get_nucleo_arr(self, mask, v_n):
        self.check_mask(mask)
        self.check_nucleo_v_n(v_n)
        self.is_nucleo_arr_computed(mask, v_n)
        data = self.matrix_ejecta_nucleo[self.i_mask(mask)][self.i_nuc_v_n(v_n)]
        return data

    # def get_normalized_yeilds(self, det, mask, method="Asol=195"):
    #
    #     Ys = self.get_nucleo_arr(det, mask, "yields")
    #     As = self.get_nucleo_arr(det, mask, "As")
    #     Zs = self.get_nucleo_arr(det, mask, "Zs")
    #
    #     '''Sums all Ys for a given A (for all Z)'''
    #     Anrm = np.arange(As.max() + 1)
    #     Ynrm = np.zeros(int(As.max()) + 1)
    #     for i in range(Ynrm.shape[0]):  # changed xrange to range
    #         Ynrm[i] = Ys[As == i].sum()
    #
    #     if method == '':
    #         return Anrm, Ynrm
    #
    #     elif method == 'sum':
    #         ''' Normalizes to a sum of all A '''
    #         norm = Ynrm.sum()
    #         Ynrm /= norm
    #         return Anrm, Ynrm
    #
    #     elif method == "Asol=195":
    #         ''' Normalize to the solar abundances of a given element'''
    #         # a_sol = self.get_sol_data("Asun")
    #         # y_sol = self.get_sol_data("Ysun")
    #         a_sol = self.get_normalized_sol_data("Asun")
    #         y_sol = self.get_normalized_sol_data("Ysun")
    #
    #         element_a = int(method.split("=")[-1])
    #         if element_a not in a_sol: raise ValueError('Element: a:{} not in solar A\n{}'.format(element_a, a_sol))
    #         if element_a not in Anrm: raise ValueError('Element: a:{} not in a_arr\n{}'.format(element_a, Anrm))
    #
    #         delta = np.float(y_sol[np.where(a_sol == element_a)] / Ynrm[np.where(Anrm == element_a)])
    #         Ynrm *= delta
    #
    #         return Anrm, Ynrm
    #     else:
    #         raise NameError("Normalisation method '{}' for the simulation yields is not recognized. Use:{}"
    #                         .format(method, self.list_norm_methods))
    #
    # def get_nucleo_solar_yeilds(self, norm):


class EJECTA_NORMED_NUCLEO(EJECTA_NUCLEO):

    def __init__(self, fname, skynetdir, add_mask=None):

        EJECTA_NUCLEO.__init__(self, fname=fname, skynetdir=skynetdir, add_mask=add_mask)

        self.list_nucleo_norm_methods = [
            "sum", "Asol=195"
        ]

        self.matrix_normed_sim = [[np.zeros(0,)
                                  for z in range(len(self.list_nucleo_norm_methods))]
                                  for y in range(len(self.list_masks))]

        self.matrix_normed_sol = [np.zeros(0,) for z in range(len(self.list_nucleo_norm_methods))]

    # def update_mask(self, new_mask=None):
    #     if new_mask != None:
    #         if not new_mask in self.list_masks:
    #             self.list_masks.append(new_mask)
    #
    #             self.mask_matrix = [[np.zeros(0, )
    #                                  for i in range(len(self.list_masks))]
    #                                 for j in range(len(self.list_detectors))]
    #
    #             self.matrix_ejecta = [[[np.zeros(0, )
    #                                     for k in range(len(self.list_ejecta_v_ns))]
    #                                    for j in range(len(self.list_masks))]
    #                                   for i in range(len(self.list_detectors))]
    #
    #             self.matrix_ejecta_nucleo = [[[np.zeros(0, )
    #                                            for i in range(len(self.list_nucleo_v_ns))]
    #                                           for j in range(len(self.list_masks))]
    #                                          for k in range(len(self.list_detectors))]
    #
    #             self.matrix_normed_sim = [[[np.zeros(0, )
    #                                         for x in range(len(self.list_detectors))]
    #                                        for y in range(len(self.list_masks))]
    #                                       for z in range(len(self.list_nucleo_norm_methods))]
    # #

    def check_method(self, method):
        if not method in self.list_nucleo_norm_methods:
            raise NameError("method:{} not in the list of normalisation methods: {}"
                            .format(method, self.list_nucleo_norm_methods))

    def i_meth(self, method):
        return int(self.list_nucleo_norm_methods.index(method))

    def compute_normalized_sol_yields(self, method='sum'):

        As = self.get_nucleo_arr("geo", "Asun")
        Ys = self.get_nucleo_arr("geo", "Ysun")

        '''Sums all Ys for a given A (for all Z)'''
        Anrm = np.arange(As.max() + 1)
        Ynrm = np.zeros(int(As.max()) + 1)
        for i in range(Ynrm.shape[0]):  # changed xrange to range
            Ynrm[i] = Ys[As == i].sum()

        if method == 'sum':
            Ynrm /= np.sum(Ynrm)
            return Anrm, Ynrm
        else:
            raise NameError("Normalisation method '{}' for the solar is not recognized. Use:{}"
                            .format(method, self.list_nucleo_norm_methods))

    def is_sol_nucleo_yiled_normed(self, method):

        data = self.matrix_normed_sol[self.i_meth(method)]
        if len(data) == 0:
            a_sol, y_sol = self.compute_normalized_sol_yields(method)
            self.matrix_normed_sol[self.i_meth(method)] = np.vstack((a_sol, y_sol)).T

        data = self.matrix_normed_sol[self.i_meth(method)]
        if len(data) == 0:
            raise ValueError("failed to normalize simulations yeilds for: "
                             "method:{}".format(method))

    def compute_normalized_sim_yelds(self, mask, method):

        As = self.get_nucleo_arr(mask, "A")
        Ys = self.get_nucleo_arr(mask, "yields")

        '''Sums all Ys for a given A (for all Z)'''
        Anrm = np.arange(As.max() + 1)
        Ynrm = np.zeros(int(As.max()) + 1)
        for i in range(Ynrm.shape[0]):  # changed xrange to range
            Ynrm[i] = Ys[As == i].sum()

        if method == '':
            return Anrm, Ynrm

        elif method == 'sum':
            ''' Normalizes to a sum of all A '''
            norm = Ynrm.sum()
            Ynrm /= norm
            return Anrm, Ynrm

        elif method == "Asol=195":
            ''' Normalize to the solar abundances of a given element'''
            # a_sol = self.get_sol_data("Asun")
            # y_sol = self.get_sol_data("Ysun")

            tmp = self.get_nored_sol_abund("sum")
            a_sol, y_sol = tmp[:,0], tmp[:,1]

            element_a = int(method.split("=")[-1])
            if element_a not in a_sol: raise ValueError('Element: a:{} not in solar A\n{}'.format(element_a, a_sol))
            if element_a not in Anrm: raise ValueError('Element: a:{} not in a_arr\n{}'.format(element_a, Anrm))

            delta = np.float(y_sol[np.where(a_sol == element_a)] / Ynrm[np.where(Anrm == element_a)])
            Ynrm *= delta

            return Anrm, Ynrm
        else:
            raise NameError("Normalisation method '{}' for the simulation yields is not recognized. Use:{}"
                            .format(method, self.list_nucleo_norm_methods))

    def is_nucleo_yiled_normed(self, mask, method):

        if not mask in self.list_masks:
            raise NameError("mask:{} is not in the list:{}"
                            .format(mask, self.list_masks))

        # print(len(self.matrix_normed_sim))
        # print(self.matrix_normed_sim[self.i_mask(mask)])

        data = self.matrix_normed_sim[self.i_mask(mask)][self.i_meth(method)]
        if len(data) == 0:
            a_arr, y_arr = self.compute_normalized_sim_yelds(mask, method)
            data = np.vstack((a_arr, y_arr)).T
            self.matrix_normed_sim[self.i_mask(mask)][self.i_meth(method)] = data

        data = self.matrix_normed_sim[self.i_mask(mask)][self.i_meth(method)]
        if len(data) == 0:
            raise ValueError("failed to normalize simulations yeilds for: "
                             "mask:{} method:{}"
                             .format(mask, method))

    def get_normed_sim_abund(self, mask, method):

        self.check_mask(mask)
        self.check_method(method)
        self.is_nucleo_yiled_normed(mask, method)
        data = self.matrix_normed_sim[self.i_mask(mask)][self.i_meth(method)]
        return data

    def get_nored_sol_abund(self, method='sum'):

        self.check_method(method)
        self.is_sol_nucleo_yiled_normed(method)
        data = self.matrix_normed_sol[self.i_meth(method)]
        return data


class EJECTA_PARS(EJECTA_NORMED_NUCLEO):

    def __init__(self, fname, skynetdir, add_mask=None):

        EJECTA_NORMED_NUCLEO.__init__(self, fname=fname, skynetdir=skynetdir, add_mask=add_mask)

        self.list_ejecta_pars_v_n = [
            "Mej_tot", "Ye_ave", "s_ave", "vel_inf_ave",
            "vel_inf_bern_ave", "theta_rms", "E_kin_ave",
            "E_kin_bern_ave"]

        self.matrix_ejecta_pars = [[123456789.1
                                 for x in range(len(self.list_ejecta_pars_v_n))]
                                 for z in range(len(self.list_masks))]



        self.energy_constant = 1787.5521500932314

    # def update_mask(self, new_mask=None):
    #
    #     if new_mask != None:
    #         if not new_mask in self.list_masks:
    #             self.list_masks.append(new_mask)
    #
    #             self.mask_matrix = [[np.zeros(0, )
    #                                  for i in range(len(self.list_masks))]
    #                                 for j in range(len(self.list_detectors))]
    #
    #             self.matrix_ejecta = [[[np.zeros(0, )
    #                                     for k in range(len(self.list_ejecta_v_ns))]
    #                                    for j in range(len(self.list_masks))]
    #                                   for i in range(len(self.list_detectors))]
    #
    #             self.matrix_ejecta_nucleo = [[[np.zeros(0, )
    #                                            for i in range(len(self.list_nucleo_v_ns))]
    #                                           for j in range(len(self.list_masks))]
    #                                          for k in range(len(self.list_detectors))]
    #
    #             self.matrix_normed_sim = [[[np.zeros(0, )
    #                                         for x in range(len(self.list_detectors))]
    #                                        for y in range(len(self.list_masks))]
    #                                       for z in range(len(self.list_nucleo_norm_methods))]
    #
    #             self.matrix_ejecta_pars = [[[123456789.1
    #                                          for x in range(len(self.list_ejecta_pars_v_n))]
    #                                         for z in range(len(self.list_masks))]
    #                                        for y in range(len(self.list_detectors))]
    # #

    def check_ej_par_v_n(self, v_n):
        if not v_n in self.list_ejecta_pars_v_n:
            raise NameError("Parameter v_n: {} not in their list: {}"
                            .format(v_n, self.list_ejecta_pars_v_n))

    def i_ej_par(self, v_n):
        return int(self.list_ejecta_pars_v_n.index(v_n))

    # ----------------------------------------------

    @staticmethod
    def compute_ave_ye(mej, hist_ye):
        ye_ave = np.sum(hist_ye[:, 0] * hist_ye[:, 1]) / mej
        if ye_ave > 0.6: raise ValueError("Ye_ave > 0.6 ")
        value = np.float(ye_ave)
        return value

    @staticmethod
    def compute_ave_s(mej, hist_s):
        s_ave = np.sum(hist_s[:, 0] * hist_s[:, 1]) / mej
        value = np.float(s_ave)
        return value

    @staticmethod
    def compute_ave_vel_inf(mej, hist_vinf):
        vinf_ave = np.sum(hist_vinf[:, 0] * hist_vinf[:, 1]) / mej
        value = np.float(vinf_ave)
        return value

    @staticmethod
    def compute_ave_ekin(mej, hist_vinf):
        vinf_ave = EJECTA_PARS.compute_ave_vel_inf(mej, hist_vinf)
        E_kin_ave = np.sum(0.5 * vinf_ave ** 2 * hist_vinf[:, 1]) * Constants.energy_constant
        value = np.float(E_kin_ave)
        return value

    @staticmethod
    def compute_ave_theta_rms(hist_theta):
        theta, theta_M = hist_theta[:, 0], hist_theta[:, 1]
        # print(theta, theta_M)
        theta -= np.pi / 2.
        theta_rms = (180. / np.pi) * sqrt(np.sum(theta_M * theta ** 2) / np.sum(theta_M))
        value = np.float(theta_rms)
        return value

    # ----------------------------------------------

    def compute_ejecta_par(self, mask, v_n):

        # print("computing det:{} mask:{} v_n:{}".format(det, mask, v_n))

        if v_n == "Mej_tot":
            tarr_tot_flux_tot_mass = self.get_ejecta_arr(mask, "tot_mass")
            value = tarr_tot_flux_tot_mass[-1, 2]

        elif v_n == "Ye_ave":
            mej = self.get_ejecta_par(mask, "Mej_tot")
            hist_ye = self.get_ejecta_arr(mask, "hist Y_e")
            ye_ave = np.sum(hist_ye[:,0] * hist_ye[:,1]) / mej
            if ye_ave > 0.6: raise ValueError("Ye_ave > 0.6 "
                                              "mask:{} v_n:{}"
                                              .format(mask, v_n))
            value = np.float(ye_ave)

        elif v_n == "entropy_ave" or v_n == "s_ave":
            mej = self.get_ejecta_par(mask, "Mej_tot")
            hist_s = self.get_ejecta_arr(mask, "hist entropy")
            s_ave = np.sum(hist_s[:,0] * hist_s[:,1]) / mej
            value = np.float(s_ave)

        elif v_n == "vel_inf_ave":
            # if mask.__contains__("bern"):
            #     vel_v_n = "vel_inf_bern"
            # else:
            #     vel_v_n = "vel_inf"

            mej = self.get_ejecta_par(mask, "Mej_tot")
            hist_vinf = self.get_ejecta_arr(mask, "hist vel_inf")
            vinf_ave = np.sum(hist_vinf[:,0] * hist_vinf[:,1]) / mej
            value = np.float(vinf_ave)

        elif v_n == "E_kin_ave":
            # if v_n.__contains__("bern"):
            #     vel_v_n = "vel_inf_bern"
            # else:
            #     vel_v_n = "vel_inf"

            vinf_ave = self.get_ejecta_par(mask, "vel_inf_ave")
            hist_vinf = self.get_ejecta_arr(mask, "hist vel_inf")
            E_kin_ave = np.sum(0.5 * vinf_ave ** 2 * hist_vinf[:,1]) * self.energy_constant
            value = np.float(E_kin_ave)

        elif v_n == 'theta_rms':
            hist_theta = self.get_ejecta_arr(mask, "hist theta")
            theta, theta_M = hist_theta[:,0], hist_theta[:,1]
            theta -= pi / 2
            theta_rms = 180. / pi * sqrt(np.sum(theta_M * theta ** 2) / np.sum(theta_M))
            value = np.float(theta_rms)

        else:
            raise NameError("module_ejecta par v_n: {} (mask:{}) does not have a"
                            " method for computing".format(v_n, mask))
        return value

    # ----------------------------------------------

    def is_ej_par_computed(self, mask, v_n):

        data = self.matrix_ejecta_pars[self.i_mask(mask)][self.i_ej_par(v_n)]
        if data == 123456789.1:
            value = self.compute_ejecta_par(mask, v_n)
            self.matrix_ejecta_pars[self.i_mask(mask)][self.i_ej_par(v_n)] = value

        data = self.matrix_ejecta_pars[self.i_mask(mask)][self.i_ej_par(v_n)]
        if data == 123456789.1:
            raise ValueError("failed to compute module_ejecta par v_n:{} mask:{}"
                             .format(v_n, mask))

    def get_ejecta_par(self, mask, v_n):
        self.check_mask(mask)
        self.check_ej_par_v_n(v_n)
        self.is_ej_par_computed(mask, v_n)
        data = self.matrix_ejecta_pars[self.i_mask(mask)][self.i_ej_par(v_n)]
        return data

''' --- old versions with 'det' as well --- '''
"""
class LOAD_OUTFLOW_SURFACE_H5:

    def __init__(self, pprdir):

        # LOAD_ITTIME.__init__(self, sim, pprdir=pprdir)

        self.pprdir = pprdir

        self.list_detectors = [0, 1]

        self.list_v_ns = ["fluxdens", "w_lorentz", "eninf", "surface_element",
                          "alp", "rho", "vel[0]", "vel[1]", "vel[2]", "Y_e",
                          "press", "entropy", "temperature", "eps"]

        self.list_grid_v_ns = ["theta", "phi", 'iterations', 'times', "area"]

        self.list_v_ns += self.list_grid_v_ns

        self.grid_pars = ["radius", "ntheta", "nphi"]

        self.matrix_data = [[np.empty(0,)
                             for v in range(len(self.list_v_ns)+len(self.list_grid_v_ns))]
                             for d in range(len(self.list_detectors))]

        self.matrix_grid_pars = [{} for d in range(len(self.list_detectors))]

    def update_det(self, new_det=None):

        if new_det != None:
            if not new_det in self.list_detectors:
                self.list_detectors.append(new_det)

                self.matrix_data = [[np.empty(0, )
                                     for v in range(len(self.list_v_ns) + len(self.list_grid_v_ns))]
                                    for d in range(len(self.list_detectors))]

                self.matrix_grid_pars = [{} for d in range(len(self.list_detectors))]

    def update_grid_v_n(self, new_grid_v_n = None):
        if new_grid_v_n != None:
            if not new_grid_v_n in self.list_grid_v_ns:
                self.list_grid_v_ns.append(new_grid_v_n)
                self.matrix_data = [[np.empty(0, )
                                     for v in range(len(self.list_v_ns) + len(self.list_grid_v_ns))]
                                    for d in range(len(self.list_detectors))]

    # def update_v_n(self, new_v_n=None):
    #     if new_v_n != None:
    #         if not new_v_n in self.list_v_ns:
    #             self.list_v_ns.append(v_n)
    #
    #             self.matrix_data = [[np.empty(0, )
    #                                  for v in range(len(self.list_v_ns) + len(self.list_grid_v_ns))]
    #                                 for d in range(len(self.list_detectors))]


    def check_v_n(self, v_n):
        if not v_n in self.list_v_ns:
            raise NameError("v_n:{} not in the list of v_ns: {}"
                            .format(v_n, self.list_v_ns))

    def check_det(self, det):
        if not det in self.list_detectors:
            raise NameError("det: {} not in the list: {}"
                            .format(det, self.list_detectors))

    def i_v_n(self, v_n):
        return int(self.list_v_ns.index(v_n))

    def i_det(self, det):
        return int(self.list_detectors.index(det))

    def load_h5_file(self, det):

        fpath = self.pprdir + "outflow_surface_det_{:d}_fluxdens.h5".format(det)
        assert os.path.isfile(fpath)

        print("\tLoading {}".format("outflow_surface_det_{:d}_fluxdens.h5".format(det)))

        dfile = h5py.File(fpath, "r")

        # attributes
        radius = float(dfile.attrs["radius"])
        ntheta = int(dfile.attrs["ntheta"])
        nphi   = int(dfile.attrs["nphi"])

        self.matrix_grid_pars[self.i_det(det)]["radius"] = radius
        self.matrix_grid_pars[self.i_det(det)]["ntheta"] = ntheta
        self.matrix_grid_pars[self.i_det(det)]["nphi"] = nphi

        for v_n in dfile:
            self.check_v_n(v_n)
            arr = np.array(dfile[v_n])
            self.matrix_data[self.i_det(det)][self.i_v_n(v_n)] = arr

    def is_file_loaded(self, det):
        data = self.matrix_data[self.i_det(det)][self.i_v_n(self.list_v_ns[0])]
        if len(data) == 0:
            self.load_h5_file(det)
        data = self.matrix_data[self.i_det(det)][self.i_v_n(self.list_v_ns[0])]
        if len(data) == 0:
            raise ValueError("Error in loading/extracing data. Emtpy array")

    def get_full_arr(self, det, v_n):
        self.check_v_n(v_n)
        self.check_det(det)
        self.is_file_loaded(det)
        return self.matrix_data[self.i_det(det)][self.i_v_n(v_n)]

    def get_grid_par(self, det, v_n):
        self.check_det(det)
        self.is_file_loaded(det)
        return self.matrix_grid_pars[self.i_det(det)][v_n]


class COMPUTE_OUTFLOW_SURFACE_H5(LOAD_OUTFLOW_SURFACE_H5):

    def __init__(self, pprdir):

        LOAD_OUTFLOW_SURFACE_H5.__init__(self, pprdir=pprdir)

        self.list_comp_v_ns = ["enthalpy", "vel_inf", "vel_inf_bern", "vel"]

        # self.list_v_ns = self.list_v_ns + self.list_comp_v_ns

        self.matrix_comp_data = [[np.empty(0,)
                                  for v in self.list_comp_v_ns]
                                  for d in self.list_detectors]

    def update_comp_v_ns(self, new_v_n = None):
        if new_v_n != None:
            if not new_v_n in self.list_comp_v_ns:
                self.list_comp_v_ns.append(new_v_n)
                self.matrix_comp_data = [[np.empty(0, )
                                          for v in self.list_comp_v_ns]
                                         for d in self.list_detectors]

    def check_comp_v_n(self, v_n):
        if not v_n in self.list_comp_v_ns:
            raise NameError("v_n: {} is not in the list v_ns: {}"
                            .format(v_n, self.list_comp_v_ns))

    def i_comp_v_n(self, v_n):
        return int(self.list_comp_v_ns.index(v_n))

    def compute_arr(self, det, v_n):

        if v_n == "enthalpy":
            arr = FORMULAS.enthalpy(self.get_full_arr(det, "eps"),
                                    self.get_full_arr(det, "press"),
                                    self.get_full_arr(det, "rho"))
        elif v_n == "vel_inf":
            arr = FORMULAS.vinf(self.get_full_arr(det, "eninf"))
        elif v_n == "vel_inf_bern":
            # print("----------------------------------------")
            arr = FORMULAS.vinf_bern(self.get_full_arr(det, "eninf"),
                                     self.get_full_comp_arr(det, "enthalpy"))
        elif v_n == "vel":
            arr = FORMULAS.vel(self.get_full_arr(det, "w_lorentz"))
        else:
            raise NameError("No computation method for v_n:{} is found"
                            .format(v_n))
        return arr


    def is_arr_computed(self, det, v_n):
        arr = self.matrix_comp_data[self.i_det(det)][self.i_comp_v_n(v_n)]
        if len(arr) == 0:
            arr = self.compute_arr(det, v_n)
            self.matrix_comp_data[self.i_det(det)][self.i_comp_v_n(v_n)] = arr
        if len(arr) == 0:
            raise ValueError("Computation of v_n:{} has failed. Array is emtpy"
                             .format(v_n))

    def get_full_comp_arr(self, det, v_n):
        self.check_det(det)
        self.check_comp_v_n(v_n)
        self.is_arr_computed(det, v_n)
        arr = self.matrix_comp_data[self.i_det(det)][self.i_comp_v_n(v_n)]
        return arr

    def get_full_arr(self, det, v_n):
        self.check_det(det)
        if v_n in self.list_comp_v_ns:
            self.check_comp_v_n(v_n)
            self.is_arr_computed(det, v_n)
            arr = self.matrix_comp_data[self.i_det(det)][self.i_comp_v_n(v_n)]
            return arr
        else:
            self.check_v_n(v_n)
            self.is_file_loaded(det)
            arr = self.matrix_data[self.i_det(det)][self.i_v_n(v_n)]
            return arr


class ADD_MASK(COMPUTE_OUTFLOW_SURFACE_H5):

    def __init__(self, pprdir, add_mask=None):

        COMPUTE_OUTFLOW_SURFACE_H5.__init__(self, pprdir=pprdir)

        self.list_masks = ["geo", "bern", "bern_geoend", "Y_e04_geoend", "theta60_geoend",
                           "geo_entropy_above_10", "geo_entropy_below_10"]
        if add_mask != None and not add_mask in self.list_masks:
            self.list_masks.append(add_mask)


        # "Y_e04_geoend"
        self.mask_matrix = [[np.zeros(0,)
                            for i in range(len(self.list_masks))]
                            for j in range(len(self.list_detectors))]

        self.set_min_eninf = 0.
        self.set_min_enthalpy = 1.0022

    def update_mask(self, new_mask=None):
        if new_mask != None:
            if not new_mask in self.list_masks:
                self.list_masks.append(new_mask)

                self.mask_matrix = [[np.zeros(0, )
                                     for i in range(len(self.list_masks))]
                                    for j in range(len(self.list_detectors))]


    def check_mask(self, mask):
        if not mask in self.list_masks:
            raise NameError("mask: {} is not in the list: {}"
                            .format(mask, self.list_masks))

    def i_mask(self, mask):
        return int(self.list_masks.index(mask))

    # ----------------------------------------------
    def __time_mask_end_geo(self, det, length=0.):

        fluxdens = self.get_full_arr(det, "fluxdens")
        da = self.get_full_arr(det, "surface_element")
        t = self.get_full_arr(det, "times")
        dt = np.diff(t)
        dt = np.insert(dt, 0, 0)
        mask = self.get_mask(det, "geo").astype(int)
        fluxdens = fluxdens * mask
        flux_arr = np.sum(np.sum(fluxdens * da, axis=1), axis=1)  # sum over theta and phi
        tot_mass = np.cumsum(flux_arr * dt)  # sum over time
        tot_flux = np.cumsum(flux_arr)  # sum over time
        # print("totmass:{}".format(tot_mass[-1]))
        fraction = 0.98
        i_t98mass = int(np.where(tot_mass >= fraction * tot_mass[-1])[0][0])
        # print(i_t98mass)
        # assert i_t98mass < len(t)

        if length > 0.:
            if length > t[-1]:
                raise ValueError("length:{} is > t[-1]:{} [ms]".format(length*Constants.time_constant,
                                                                  t[-1]*Constants.time_constant))
            if t[i_t98mass] + length > t[-1]:
                # because of bloody numerics it can > but just by a tiny bit. So I added this shit.
                if np.abs(t[i_t98mass] - length > t[-1]) < 10: # 10 is a rundomly chosen number
                    length = length - 10
                else:
                    raise ValueError("t[i_t98mass] + length > t[-1] : {} > {}"
                                     .format((t[i_t98mass] + length),
                                             t[-1]))

            i_mask = (t > t[i_t98mass]) & (t < t[i_t98mass] + length)
        else:
            i_mask = t > t[i_t98mass]
        # saving time at 98% mass for future use
        # fpath = Paths.ppr_sims + self.sim + '/outflow_{}/t98mass.dat'.format(det)
        # try: open(fpath, "w").write("{}\n".format(float(t[i_t98mass])))
        # except IOError: Printcolor.yellow("\tFailed to save t98mass.dat")
        # continuing with mask
        newmask = np.zeros(fluxdens.shape)
        for i in range(len(newmask[:, 0, 0])):
            newmask[i, :, :].fill(i_mask[i])
        return newmask.astype(bool)
    # ----------------------------------------------

    def compute_mask(self, det, mask):
        self.check_mask(mask)

        if mask == "geo":
            # 1 - if geodeisc is true
            einf = self.get_full_arr(det, "eninf")
            res = (einf >= self.set_min_eninf)
        elif mask == "geo_entropy_below_10":
            einf = self.get_full_arr(det, "eninf")
            res = (einf >= self.set_min_eninf)
            entropy = self.get_full_arr(det, "entropy")
            mask_entropy = entropy < 10.
            return res & mask_entropy
        elif mask == "geo_entropy_above_10":
            einf = self.get_full_arr(det, "eninf")
            res = (einf >= self.set_min_eninf)
            entropy = self.get_full_arr(det, "entropy")
            mask_entropy = entropy > 10.
            return res & mask_entropy
        elif mask == "bern":
            # 1 - if Bernulli is true
            enthalpy = self.get_full_comp_arr(det, "enthalpy")
            einf = self.get_full_arr(det, "eninf")
            res = ((enthalpy * (einf + 1) - 1) > self.set_min_eninf) & (enthalpy >= self.set_min_enthalpy)
        elif mask == "bern_geoend":
            # 1 - data above 98% of GeoMass and if Bernoulli true and 0 if not
            mask2 = self.get_mask(det, "bern")
            newmask = self.__time_mask_end_geo(det)

            res = newmask & mask2
        elif mask == "Y_e04_geoend":
            # 1 - data above Ye=0.4 and 0 - below
            ye = self.get_full_arr(det, "Y_e")
            mask_ye = ye >= 0.4
            mask_bern = self.get_mask(det, "bern")
            mask_geo_end = self.__time_mask_end_geo(det)
            return mask_ye & mask_bern & mask_geo_end
        elif mask == "theta60_geoend":
            # 1 - data above Ye=0.4 and 0 - below
            theta = self.get_full_arr(det, "theta")
            # print((theta / np.pi * 180.).min(), (theta / np.pi * 180.).max())
            # exit(1)
            theta_ = 90 - (theta * 180 / np.pi)
            # print(theta_); #exit(1)
            theta_mask = (theta_ > 60.) | (theta_ < -60)
            # print(np.sum(theta_mask.astype(int)))
            # assert np.sum(theta_mask.astype(int)) > 0
            newmask = theta_mask[np.newaxis, : , :]

            # fluxdens = self.get_full_arr(det, "fluxdens")
            # newmask = np.zeros(fluxdens.shape)
            # for i in range(len(newmask[:, 0, 0])):
            #     newmask[i, :, :].fill(theta_mask)

            print(newmask.shape)
            # exit(1)
            # mask_ye = ye >= 0.4
            mask_bern = self.get_mask(det, "bern")
            print(mask_bern.shape)
            # print(mask_bern.shape)
            mask_geo_end = self.__time_mask_end_geo(det)
            return newmask & mask_bern & mask_geo_end
        elif str(mask).__contains__("_tmax"):
            raise NameError(" mask with '_tmax' are not supported")
            #
            # # 1 - data below tmax and 0 - above
            # base_mask_name = str(str(mask).split("_tmax")[0])
            # base_mask = self.get_mask(det, base_mask_name)
            # #
            # tmax = float(str(mask).split("_tmax")[-1])
            # tmax = tmax / Constants.time_constant # Msun
            # # tmax loaded is postmerger tmax. Thus it need to be added to merger time
            # fpath = self.pprdir+"/waveforms/tmerger.dat"
            # try:
            #     tmerg = float(np.loadtxt(fpath, unpack=True)) # Msun
            #     Printcolor.yellow("\tWarning! using defauled M_Inf=2.748, R_GW=400.0 for retardet time")
            #     ret_time = PHYSICS.get_retarded_time(tmerg, M_Inf=2.748, R_GW=400.0)
            #     tmerg = ret_time
            #     # tmerg = ut.conv_time(ut.cactus, ut.cgs, ret_time)
            #     # tmerg = tmerg / (Constants.time_constant *1e-3)
            # except IOError:
            #     raise IOError("For the {} mask, the tmerger.dat is needed at {}"
            #                   .format(mask, fpath))
            # except:
            #     raise ValueError("failed to extract tmerg for outflow tmax mask analysis")
            #
            # t = self.get_full_arr(det, "times") # Msun
            # # tmax = tmax + tmerg       # Now tmax is absolute time (from the begniing ofthe simulation
            # print("t[-1]:{} tmax:{} tmerg:{} -> {}".format(t[-1]*Constants.time_constant*1e-3,
            #                                 tmax*Constants.time_constant*1e-3,
            #                                 tmerg*Constants.time_constant*1e-3,
            #                                 (tmax+tmerg)*Constants.time_constant*1e-3))
            # tmax = tmax + tmerg
            # if tmax > t[-1]:
            #     raise ValueError("tmax:{} for the mask is > t[-1]:{}".format(tmax*Constants.time_constant*1e-3,
            #                                                                  t[-1]*Constants.time_constant*1e-3))
            # if tmax < t[0]:
            #     raise ValueError("tmax:{} for the mask is < t[0]:{}".format(tmax * Constants.time_constant * 1e-3,
            #                                                                  t[0] * Constants.time_constant * 1e-3))
            # fluxdens = self.get_full_arr(det, "fluxdens")
            # i_mask = t < t[UTILS.find_nearest_index(t, tmax)]
            # newmask = np.zeros(fluxdens.shape)
            # for i in range(len(newmask[:, 0, 0])):
            #     newmask[i, :, :].fill(i_mask[i])

            # print(base_mask.shape,newmask.shape)

            # return base_mask & newmask.astype(bool)
        elif str(mask).__contains__("_length"):
            base_mask_name = str(str(mask).split("_length")[0])
            base_mask = self.get_mask(det, base_mask_name)
            delta_t = float(str(mask).split("_length")[-1])
            delta_t = (delta_t / 1e5) / (Constants.time_constant * 1e-3) # Msun
            t = self.get_full_arr(det, "times")  # Msun
            print("\t t[0]: {}\n\t t[-1]: {}\n\t delta_t: {}\n\t mask: {}"
                  .format(t[0] * Constants.time_constant * 1e-3,
                          t[-1] * Constants.time_constant * 1e-3,
                          delta_t * Constants.time_constant * 1e-3,
                          mask))
            assert delta_t < t[-1]
            assert delta_t > t[0]
            mask2 = self.get_mask(det, "bern")
            newmask = self.__time_mask_end_geo(det, length=delta_t)

            res = newmask & mask2

        else:
            raise NameError("No method found for computing mask:{}"
                            .format(mask))

        return res

    # ----------------------------------------------

    def is_mask_computed(self, det, mask):
        if len(self.mask_matrix[self.i_det(det)][self.i_mask(mask)]) == 0:
            arr = self.compute_mask(det, mask)
            self.mask_matrix[self.i_det(det)][self.i_mask(mask)] = arr

        if len(self.mask_matrix[self.i_det(det)][self.i_mask(mask)]) == 0:
            raise ValueError("Failed to compute the mask: {} det: {}"
                             .format(mask, det))

    def get_mask(self, det, mask):
        self.check_mask(mask)
        self.check_det(det)
        self.is_mask_computed(det, mask)
        return self.mask_matrix[self.i_det(det)][self.i_mask(mask)]


class EJECTA(ADD_MASK):

    def __init__(self, pprdir, skynetdir, add_mask=None):

        ADD_MASK.__init__(self, pprdir=pprdir, add_mask=add_mask)

        self.list_hist_v_ns = ["Y_e", "theta", "phi", "vel_inf", "entropy", "temperature"]

        self.list_corr_v_ns = ["Y_e theta", "vel_inf theta", "Y_e vel_inf"]

        self.list_ejecta_v_ns = [
                                    "tot_mass", "tot_flux",  "weights", "corr3d Y_e entropy tau",
                                ] +\
                                ["timecorr {}".format(v_n) for v_n in self.list_hist_v_ns] +\
                                ["hist {}".format(v_n) for v_n in self.list_hist_v_ns] +\
                                ["corr2d {}".format(v_n) for v_n in self.list_corr_v_ns] +\
                                ["mass_ave {}".format(v_n) for v_n in self.list_v_ns]

        self.matrix_ejecta = [[[np.zeros(0,)
                                for k in range(len(self.list_ejecta_v_ns))]
                                for j in range(len(self.list_masks))]
                                for i in range(len(self.list_detectors))]

        self.set_skynet_densmap_fpath = skynetdir + "densmap.h5"
        self.set_skyent_grid_fpath = skynetdir + "grid.h5"

    def update_mask(self, new_mask=None):
        if new_mask != None:
            if not new_mask in self.list_masks:
                self.list_masks.append(new_mask)

                self.mask_matrix = [[np.zeros(0, )
                                     for i in range(len(self.list_masks))]
                                    for j in range(len(self.list_detectors))]

                self.matrix_ejecta = [[[np.zeros(0, )
                                        for k in range(len(self.list_ejecta_v_ns))]
                                       for j in range(len(self.list_masks))]
                                      for i in range(len(self.list_detectors))]

    # ---

    def check_ej_v_n(self, v_n):
        if not v_n in self.list_ejecta_v_ns:
            raise NameError("module_ejecta v_n: {} is not in the list of module_ejecta v_ns {}"
                            .format(v_n, self.list_ejecta_v_ns))

    def i_ejv_n(self, v_n):
        return int(self.list_ejecta_v_ns.index(v_n))

    # --- methods for EJECTA arrays ---

    def get_cumulative_ejected_mass(self, det, mask):
        fluxdens = self.get_full_arr(det, "fluxdens")
        da = self.get_full_arr(det, "surface_element")
        t = self.get_full_arr(det, "times")
        dt = np.diff(t)
        dt = np.insert(dt, 0, 0)
        mask = self.get_mask(det, mask).astype(int)
        fluxdens = fluxdens * mask
        flux_arr = np.sum(np.sum(fluxdens * da, axis=1), axis=1)  # sum over theta and phi
        tot_mass = np.cumsum(flux_arr * dt)  # sum over time
        tot_flux = np.cumsum(flux_arr)  # sum over time
        # print("totmass:{}".format(tot_mass[-1]))
        return t * 0.004925794970773136 / 1e3, flux_arr, tot_mass # time in [s]

    def get_weights(self, det, mask):

        dt = np.diff(self.get_full_arr(det, "times"))
        dt = np.insert(dt, 0, 0)
        mask = self.get_mask(det, mask).astype(int)
        weights = mask * self.get_full_arr(det, "fluxdens") * \
                  self.get_full_arr(det, "surface_element") * \
                  dt[:, np.newaxis, np.newaxis]
        #
        if np.sum(weights) == 0.:
            Printcolor.red("sum(weights) = 0. For det:{} mask:{} there is not mass"
                           .format(det, mask))
            raise ValueError("sum(weights) = 0. For det:{} mask:{} there is not mass"
                           .format(det, mask))
        #
        return weights

    def get_hist(self, det, mask, v_n, edge):

        times = self.get_full_arr(det, "times")
        weights = np.array(self.get_ejecta_arr(det, mask, "weights"))
        data = np.array(self.get_full_arr(det, v_n))
        if v_n == "rho": data = np.log10(data)
        historgram = np.zeros(len(edge) - 1)
        tmp2 = []
        # print(data.shape, weights.shape, edge.shape)
        for i in range(len(times)):
            if np.array(data).ndim == 3: data_ = data[i, :, :].flatten()
            else: data_ = data.flatten()
            # print(data.min(), data.max())
            tmp, _ = np.histogram(data_, bins=edge, weights=weights[i, :, :].flatten())
            historgram += tmp
        middles = 0.5 * (edge[1:] + edge[:-1])
        assert len(historgram) == len(middles)
        return middles, historgram

        # res = np.vstack((middles, historgram))
        # return res

    def get_timecorr(self, det, mask, v_n, edge):

        historgrams = np.zeros(len(edge) - 1)

        times = self.get_full_arr(det, "times")
        timeedges = np.linspace(times.min(), times.max(), 55)

        indexes = []
        for i_e, t_e in enumerate(timeedges[:-1]):
            i_indx = []
            for i, t in enumerate(times):
                if (t >= timeedges[i_e]) and (t<timeedges[i_e+1]):
                    i_indx = np.append(i_indx, int(i))
            indexes.append(i_indx)
        assert len(indexes) > 0
        #
        # print("indexes done")
        weights = np.array(self.get_ejecta_arr(det, mask, "weights"))
        data = np.array(self.get_full_arr(det, v_n))

        # print(weights[np.array(indexes[0], dtype=int), :, :].flatten())
        # exit(1)

        # historgram = []
        # for i_t, t in enumerate(times):
        #     print("{}".format(i_t))
        #     if np.array(data).ndim == 3:
        #         data_ = data[i_t, :, :].flatten()
        #     else:
        #         data_ = data.flatten()
        #     tmp, _ = np.histogram(data_, bins=edge, weights=weights[i_t, :, :].flatten())
        #     historgram = np.append(historgram, tmp)
        #
        # print("done") # 1min.15
        # exit(1)

        for i_ind, ind_list in enumerate(indexes):
            # print("{} {}/{}".format(i_ind,len(indexes), len(ind_list)))
            historgram = np.zeros(len(edge) - 1)
            for i in np.array(ind_list, dtype=int):
                if np.array(data).ndim == 3: data_ = data[i, :, :].flatten()
                else: data_ = data.flatten()
                tmp, _ = np.histogram(data_, bins=edge, weights=weights[i, :, :].flatten())
                historgram += tmp
            historgrams = np.vstack((historgrams, historgram))

        # print("done") #1min15
        # exit(1)

        bins = 0.5 * (edge[1:] + edge[:-1])

        print("hist", historgrams.shape)
        print("times", timeedges.shape)
        print("edges", bins.shape)

        return bins, timeedges, historgrams

    @staticmethod
    def combine(x, y, xy, corner_val=None):
        '''creates a 2d array  1st raw    [0, 1:] -- x -- density     (log)
                               1st column [1:, 0] -- y -- lemperature (log)
                               Matrix     [1:,1:] -- xy --Opacity     (log)
           0th element in 1st raw (column) - can be used a corner value

        '''
        x = np.array(x)
        y = np.array(y)
        xy = np.array((xy))

        res = np.insert(xy, 0, x, axis=0)
        new_y = np.insert(y, 0, 0, axis=0)  # inserting a 0 to a first column of a
        res = np.insert(res, 0, new_y, axis=1)

        if corner_val != None:
            res[0, 0] = corner_val

        return res

    @staticmethod
    def combine3d(x, y, z, xyz, corner_val=None):
        '''creates a 2d array  1st raw    [0, 1:] -- x -- density     (log)
                               1st column [1:, 0] -- y -- lemperature (log)
                               Matrix     [1:,1:] -- xy --Opacity     (log)
           0th element in 1st raw (column) - can be used a corner value

        '''

        print(xyz.shape, x.shape, y.shape, z.shape)

        tmp = np.zeros((len(xyz[:, 0, 0])+1, len(xyz[0, :, 0])+1, len(xyz[0, 0, :])+1))
        tmp[1:, 1:, 1:] = xyz
        tmp[1:, 0, 0] = x
        tmp[0, 1:, 0] = y
        tmp[0, 0, 1:] = z
        return tmp

    def get_corr2d(self, det, mask, v_n1, v_n2, edge1, edge2):

        tuple_edges = tuple([edge1, edge2])
        correlation = np.zeros([len(edge) - 1 for edge in tuple_edges])
        times = self.get_full_arr(det, "times")
        weights = self.get_ejecta_arr(det, mask, "weights")
        for i_t, t in enumerate(times):

            data1 = self.get_full_arr(det, v_n1)
            if data1.ndim == 3: data1_ = data1[i_t, :, :].flatten()
            else: data1_ = data1.flatten()

            data2 = self.get_full_arr(det, v_n2)
            if data2.ndim == 3: data2_ = data2[i_t, :, :].flatten()
            else: data2_ = data2.flatten()

            data = tuple([data1_, data2_])
            tmp, _ = np.histogramdd(data, bins=tuple_edges, weights=weights[i_t, :, :].flatten())
            correlation += tmp

        bins1 = 0.5 * (edge1[1:] + edge1[:-1])
        bins2 = 0.5 * (edge2[1:] + edge2[:-1])

        return bins1, bins2, correlation.T

    def get_corr3d(self, det, mask, v_n1, v_n2, v_n3, edge1, edge2, edge3):

        tuple_edges = tuple([edge1, edge2, edge3])
        correlation = np.zeros([len(edge) - 1 for edge in tuple_edges])
        times = self.get_full_arr(det, "times")
        weights = self.get_ejecta_arr(det, mask, "weights")
        for i_t, t in enumerate(times):

            data1 = self.get_full_arr(det, v_n1)
            if data1.ndim == 3: data1_ = data1[i_t, :, :].flatten()
            else: data1_ = data1.flatten()
            #
            data2 = self.get_full_arr(det, v_n2)
            if data2.ndim == 3: data2_ = data2[i_t, :, :].flatten()
            else: data2_ = data2.flatten()
            #
            data3 = self.get_full_arr(det, v_n3)
            if data2.ndim == 3: data3_ = data3[i_t, :, :].flatten()
            else: data3_ = data3.flatten()
            #
            data = tuple([data1_, data2_, data3_])
            tmp, _ = np.histogramdd(data, bins=tuple_edges, weights=weights[i_t, :, :].flatten())
            correlation += tmp

        bins1 = 0.5 * (edge1[1:] + edge1[:-1])
        bins2 = 0.5 * (edge2[1:] + edge2[:-1])
        bins3 = 0.5 * (edge3[1:] + edge3[:-1])

        return bins1, bins2, bins3, correlation.T

    @staticmethod
    def get_edges_from_centers(bins):

        # print(bins)

        edges = np.array(0.5 * (bins[1:] + bins[:-1]))  # from edges to center of bins
        edges = np.insert(edges, 0, edges[0] - np.diff(edges)[0])
        edges = np.append(edges, edges[-1] + np.diff(edges)[-1])

        # print(edges)
        # exit(1)

        return edges

    def get_corr_ye_entr_tau(self, det, mask):

        densmap = h5py.File(self.set_skynet_densmap_fpath, "r")

        dmap_ye = np.array(densmap["Ye"])
        dmap_rho = np.log10(np.array(densmap["density"]))
        dmap_entr = np.array(densmap["entropy"])

        from scipy.interpolate import RegularGridInterpolator
        interpolator = RegularGridInterpolator((dmap_ye, dmap_entr), dmap_rho,
                                               method="linear", bounds_error=False)

        grid = h5py.File(self.set_skyent_grid_fpath,"r")
        grid_ye = np.array(grid["Ye"])
        grid_entr = np.array(grid["entropy"])
        grid_tau = np.array(grid["tau"])

        data_ye = self.get_full_arr(det, "Y_e")
        data_entr = self.get_full_arr(det, "entropy")
        data_rho = self.get_full_arr(det, "rho") * 6.173937319029555e+17 # CGS
        data_vel = self.get_full_comp_arr(det, "vel")

        lrho_b = [[np.zeros(len(data_ye[:, 0, 0]))
                   for i in range(len(data_ye[0, :, 0]))]
                  for j in range(len(data_ye[0, 0, :]))]
        for i_theta in range(len(data_ye[0, :, 0])):
            for i_phi in range(len(data_ye[0, 0, :])):
                data_ye_i = data_ye[:, i_theta, i_phi].flatten()
                data_entr_i = data_entr[:, i_theta, i_phi].flatten()

                data_ye_i[data_ye_i > grid_ye.max()] = grid_ye.max()
                data_entr_i[data_entr_i > grid_entr.max()] = grid_entr.max()
                data_ye_i[data_ye_i < grid_ye.min()] = grid_ye.min()
                data_entr_i[data_entr_i < grid_entr.min()] = grid_entr.min()

                A = np.zeros((len(data_ye_i), 2))
                A[:, 0] = data_ye_i
                A[:, 1] = data_entr_i
                lrho_b_i = interpolator(A)

                lrho_b[i_phi][i_theta] = lrho_b_i
                sys.stdout.flush()

        # from d3analysis import FORMULAS
        lrho_b = np.array(lrho_b, dtype=np.float).T
        data_tau = FORMULAS.get_tau(data_rho, data_vel, self.get_grid_par(det, "radius"), lrho_b)

        weights = self.get_ejecta_arr(det, mask, "weights")
        edges_ye = self.get_edges_from_centers(grid_ye)
        edges_tau = self.get_edges_from_centers(grid_tau)
        edges_entr = self.get_edges_from_centers(grid_entr)
        edges = tuple([edges_ye, edges_entr, edges_tau])

        correlation = np.zeros([len(edge) - 1 for edge in edges])

        for i in range(len(weights[:, 0, 0])):
            data_ye_i = data_ye[i, :, :]
            data_entr_i = data_entr[i, : ,:]
            data_tau_i = data_tau[i, :, :]
            data = tuple([data_ye_i.flatten(), data_entr_i.flatten(), data_tau_i.flatten()])
            tmp, _ = np.histogramdd(data, bins=edges, weights=weights[i, :, :].flatten())
            # print(np.array(x).shape)
            # exit(1)
            correlation += tmp

        bins_ye = 0.5 * (edges_ye[1:] + edges_ye[:-1])
        bins_entr = 0.5 * (edges_entr[1:] + edges_entr[:-1])
        bins_tau = 0.5 * (edges_tau[1:] + edges_tau[:-1])

        assert np.sum(correlation) > 0
        assert np.sum(correlation) <= np.sum(weights)
        # print(correlation.shape)
        # print(grid_ye.shape)
        assert correlation.shape == (17, 17, 17)

        return bins_ye, bins_entr, bins_tau, correlation

    def get_mass_averaged(self, det, mask, v_n):

        dt = np.diff(self.get_full_arr(det, "times"))
        dt = np.insert(dt, 0, 0)
        dt = dt[:, np.newaxis, np.newaxis]
        data = self.get_full_arr(det, v_n)

        mask = self.get_mask(det, mask)
        flux = self.get_full_arr(det, "fluxdens") * mask
        mass_averages = np.zeros((len(flux[0, :, 0]), len(flux[0, 0, :])))
        total_flux = np.zeros((len(flux[0, :, 0]), len(flux[0, 0, :])))

        for i in range(len(flux[:, 0, 0])):

            total_flux += flux[i, :, :] * dt[i, :, :]

            if v_n == "fluxdens":
                mass_averages += flux[i, :, :] * dt[i, :, :]
            else:
                mass_averages += data[i, :, :] * flux[i, :, :] * dt[i, :, :]

        return np.array(mass_averages / total_flux)

    # ---------------------------------

    def compute_ejecta_arr(self, det, mask, v_n):

        # as Bernoulli criteria uses different vel_inf defention
        if v_n.__contains__("vel_inf") and mask.__contains__("bern"):
            # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            v_n = v_n.replace("vel_inf", "vel_inf_bern")
            # print(v_n)

        # ----------------------------------------
        if v_n in ["tot_mass", "tot_flux"]:
            t, flux, mass = self.get_cumulative_ejected_mass(det, mask)
            arr = np.vstack((t, flux, mass)).T

        elif v_n == "weights":
            arr = self.get_weights(det, mask)

        elif v_n.__contains__("hist "):
            v_n = str(v_n.split("hist ")[-1])
            edge = HISTOGRAM_EDGES.get_edge(v_n)
            middles, historgram = self.get_hist(det, mask, v_n, edge)
            arr = np.vstack((middles, historgram)).T

        elif v_n.__contains__("corr2d "):
            v_n1 = str(v_n.split(" ")[1])
            v_n2 = str(v_n.split(" ")[2])
            edge1 = HISTOGRAM_EDGES.get_edge(v_n1)
            edge2 = HISTOGRAM_EDGES.get_edge(v_n2)
            bins1, bins2, weights = self.get_corr2d(det, mask, v_n1, v_n2, edge1, edge2)
            arr = self.combine(bins1, bins2, weights) # y_arr [1:, 0] x_arr [0, 1:]

        elif v_n.__contains__("timecorr "):
            v_n = str(v_n.split(" ")[1])
            edge = HISTOGRAM_EDGES.get_edge(v_n)
            bins, binstime, weights = self.get_timecorr(det, mask, v_n, edge)
            return self.combine(binstime, bins, weights.T)

        elif v_n == "corr3d Y_e entropy tau":
            bins1, bins2, bins3, corr = self.get_corr_ye_entr_tau(det, mask)
            arr = self.combine3d(bins1, bins2, bins3, corr)

        elif v_n.__contains__("mass_ave "):
            v_n = str(v_n.split("mass_ave ")[-1])
            print(v_n)
            arr = self.get_mass_averaged(det, mask, v_n)

        else:
            raise NameError("no method found for computing module_ejecta arr for det:{} mask:{} v_n:{}"
                            .format(det, mask, v_n))


        return arr

    # ---------------------------------

    def is_ejecta_arr_computed(self, det, mask, v_n):
        data = self.matrix_ejecta[self.i_det(det)][self.i_mask(mask)][self.i_ejv_n(v_n)]
        if len(data) == 0:
            arr = self.compute_ejecta_arr(det, mask, v_n)
            self.matrix_ejecta[self.i_det(det)][self.i_mask(mask)][self.i_ejv_n(v_n)] = arr

        data = self.matrix_ejecta[self.i_det(det)][self.i_mask(mask)][self.i_ejv_n(v_n)]
        if len(data) == 0:
            raise ValueError("Failed to compute module_ejecta array for "
                             "det:{} mask:{} v_n:{}"
                             .format(det, mask, v_n))

    def get_ejecta_arr(self, det, mask, v_n):
        self.check_det(det)
        self.check_mask(mask)
        self.check_ej_v_n(v_n)
        self.is_ejecta_arr_computed(det, mask, v_n)
        data = self.matrix_ejecta[self.i_det(det)][self.i_mask(mask)][self.i_ejv_n(v_n)]
        return data


class EJECTA_NUCLEO(EJECTA):

    def __init__(self, pprdir, skynetdir, add_mask=None):

        EJECTA.__init__(self, pprdir=pprdir, skynetdir=skynetdir, add_mask=add_mask)

        self._list_tab_nuc_v_ns = ["Y_final", "A", "Z"]
        self._list_sol_nuc_v_ns = ["Ysun", "Asun"]

        self.list_nucleo_v_ns = ["sim final", "solar final", "yields", "Ye", "mass"] \
                                + self._list_tab_nuc_v_ns + self._list_sol_nuc_v_ns

        self.matrix_ejecta_nucleo = [[[np.zeros(0,)
                                     for i in range(len(self.list_nucleo_v_ns))]
                                     for j in range(len(self.list_masks))]
                                     for k in range(len(self.list_detectors))]

        self.set_table_solar_r_fpath = skynetdir + "solar_r.dat"
        self.set_tabulated_nuc_fpath = skynetdir + "tabulated_nucsyn.h5"

    def update_mask(self, new_mask=None):
        if new_mask != None:
            if not new_mask in self.list_masks:
                self.list_masks.append(new_mask)

                self.mask_matrix = [[np.zeros(0, )
                                     for i in range(len(self.list_masks))]
                                    for j in range(len(self.list_detectors))]

                self.matrix_ejecta = [[[np.zeros(0, )
                                        for k in range(len(self.list_ejecta_v_ns))]
                                       for j in range(len(self.list_masks))]
                                      for i in range(len(self.list_detectors))]

                self.matrix_ejecta_nucleo = [[[np.zeros(0, )
                                               for i in range(len(self.list_nucleo_v_ns))]
                                              for j in range(len(self.list_masks))]
                                             for k in range(len(self.list_detectors))]
    # ---

    def check_nucleo_v_n(self, v_n):
        if not v_n in self.list_nucleo_v_ns:
            raise NameError("nucleo v_n: {} is not in the list:{}"
                            .format(v_n, self.list_nucleo_v_ns))

    def i_nuc_v_n(self, v_n):
        return int(self.list_nucleo_v_ns.index(v_n))

    # -------------------------------------------------

    def compute_nucleo_arr(self, det, mask, v_n):

        if v_n in self._list_tab_nuc_v_ns:
            assert os.path.isfile(self.set_tabulated_nuc_fpath)
            dfile = h5py.File(self.set_tabulated_nuc_fpath, "r")
            for v_n in self._list_tab_nuc_v_ns:
                arr = np.array(dfile[v_n])
                self.matrix_ejecta_nucleo[self.i_det(det)][self.i_mask(mask)][self.i_nuc_v_n(v_n)] = arr

        elif v_n in ["Ye", "mass"]:
            data = self.get_ejecta_arr(det, mask, "corr3d Y_e entropy tau")
            Ye = data[1:, 0, 0]
            mass = data[1:, 1:, 1:]
            self.matrix_ejecta_nucleo[self.i_det(det)][self.i_mask(mask)][self.i_nuc_v_n("Ye")] = Ye
            self.matrix_ejecta_nucleo[self.i_det(det)][self.i_mask(mask)][self.i_nuc_v_n("mass")] = mass

        elif v_n in self._list_sol_nuc_v_ns:
            assert os.path.isfile(self.set_table_solar_r_fpath)
            Asun, Ysun = np.loadtxt(self.set_table_solar_r_fpath, unpack=True)
            self.matrix_ejecta_nucleo[self.i_det(det)][self.i_mask(mask)][self.i_nuc_v_n("Asun")] = Asun
            self.matrix_ejecta_nucleo[self.i_det(det)][self.i_mask(mask)][self.i_nuc_v_n("Ysun")] = Ysun

        elif v_n == "yields":
            mass = np.array(self.get_nucleo_arr(det, mask, "mass"))
            # Ye = np.array(self.get_nucleo_arr(det, mask, "Ye"))
            Ys = np.array(self.get_nucleo_arr(det, mask, "Y_final"))
            # As = np.array(self.get_nucleo_arr(det, mask, "A"))
            # Zs = np.array(self.get_nucleo_arr(det, mask, "Z"))

            yields = np.zeros(Ys.shape[-1])
            for i in range(yields.shape[0]):
                yields[i] = np.sum(mass[:, :, :] * Ys[:, :, :, i]) # Relative final abundances
            self.matrix_ejecta_nucleo[self.i_det(det)][self.i_mask(mask)][self.i_nuc_v_n("yields")] = yields

        elif v_n == "sim final":
            yields = self.get_nucleo_arr(det, mask, "yields")
            A = self.get_nucleo_arr(det, mask, "A")
            Z = self.get_nucleo_arr(det, mask, "Z")
            # print(yields.shape)
            # print(A.shape)
            # print(Z.shape)
            arr = np.vstack((yields, A, Z)).T
            self.matrix_ejecta_nucleo[self.i_det(det)][self.i_mask(mask)][self.i_nuc_v_n("sim final")] = arr

        elif v_n == "solar final":
            Asun = self.get_nucleo_arr(det, mask, "Asun")
            Ysun = self.get_nucleo_arr(det, mask, "Ysun")
            arr = np.vstack((Asun, Ysun)).T
            self.matrix_ejecta_nucleo[self.i_det(det)][self.i_mask(mask)][self.i_nuc_v_n("solar final")] = arr
        else:
            raise NameError("no nucleo method found for v_n:{} det:{} mask:{}"
                            .format(v_n, det, mask))

    # -------------------------------------------------

    def is_nucleo_arr_computed(self, det, mask, v_n):
        data = self.matrix_ejecta_nucleo[self.i_det(det)][self.i_mask(mask)][self.i_nuc_v_n(v_n)]
        if len(data) == 0:
            self.compute_nucleo_arr(det, mask, v_n)

        data = self.matrix_ejecta_nucleo[self.i_det(det)][self.i_mask(mask)][self.i_nuc_v_n(v_n)]
        if len(data) == 0:
            raise ValueError("failed to compute nucleo arr for det:{} mask:{} v_n:{}"
                             .format(det, mask, v_n))

    def get_nucleo_arr(self, det, mask, v_n):
        self.check_det(det)
        self.check_mask(mask)
        self.check_nucleo_v_n(v_n)
        self.is_nucleo_arr_computed(det, mask, v_n)
        data = self.matrix_ejecta_nucleo[self.i_det(det)][self.i_mask(mask)][self.i_nuc_v_n(v_n)]
        return data

    # def get_normalized_yeilds(self, det, mask, method="Asol=195"):
    #
    #     Ys = self.get_nucleo_arr(det, mask, "yields")
    #     As = self.get_nucleo_arr(det, mask, "As")
    #     Zs = self.get_nucleo_arr(det, mask, "Zs")
    #
    #     '''Sums all Ys for a given A (for all Z)'''
    #     Anrm = np.arange(As.max() + 1)
    #     Ynrm = np.zeros(int(As.max()) + 1)
    #     for i in range(Ynrm.shape[0]):  # changed xrange to range
    #         Ynrm[i] = Ys[As == i].sum()
    #
    #     if method == '':
    #         return Anrm, Ynrm
    #
    #     elif method == 'sum':
    #         ''' Normalizes to a sum of all A '''
    #         norm = Ynrm.sum()
    #         Ynrm /= norm
    #         return Anrm, Ynrm
    #
    #     elif method == "Asol=195":
    #         ''' Normalize to the solar abundances of a given element'''
    #         # a_sol = self.get_sol_data("Asun")
    #         # y_sol = self.get_sol_data("Ysun")
    #         a_sol = self.get_normalized_sol_data("Asun")
    #         y_sol = self.get_normalized_sol_data("Ysun")
    #
    #         element_a = int(method.split("=")[-1])
    #         if element_a not in a_sol: raise ValueError('Element: a:{} not in solar A\n{}'.format(element_a, a_sol))
    #         if element_a not in Anrm: raise ValueError('Element: a:{} not in a_arr\n{}'.format(element_a, Anrm))
    #
    #         delta = np.float(y_sol[np.where(a_sol == element_a)] / Ynrm[np.where(Anrm == element_a)])
    #         Ynrm *= delta
    #
    #         return Anrm, Ynrm
    #     else:
    #         raise NameError("Normalisation method '{}' for the simulation yields is not recognized. Use:{}"
    #                         .format(method, self.list_norm_methods))
    #
    # def get_nucleo_solar_yeilds(self, norm):


class EJECTA_NORMED_NUCLEO(EJECTA_NUCLEO):

    def __init__(self, pprdir, skynetdir, add_mask=None):

        EJECTA_NUCLEO.__init__(self, pprdir=pprdir, skynetdir=skynetdir, add_mask=add_mask)

        self.list_nucleo_norm_methods = [
            "sum", "Asol=195"
        ]

        self.matrix_normed_sim = [[[np.zeros(0,)
                                     for x in range(len(self.list_detectors))]
                                     for y in range(len(self.list_masks))]
                                     for z in range(len(self.list_nucleo_norm_methods))]

        self.matrix_normed_sol = [np.zeros(0,) for z in range(len(self.list_nucleo_norm_methods))]

    def update_mask(self, new_mask=None):
        if new_mask != None:
            if not new_mask in self.list_masks:
                self.list_masks.append(new_mask)

                self.mask_matrix = [[np.zeros(0, )
                                     for i in range(len(self.list_masks))]
                                    for j in range(len(self.list_detectors))]

                self.matrix_ejecta = [[[np.zeros(0, )
                                        for k in range(len(self.list_ejecta_v_ns))]
                                       for j in range(len(self.list_masks))]
                                      for i in range(len(self.list_detectors))]

                self.matrix_ejecta_nucleo = [[[np.zeros(0, )
                                               for i in range(len(self.list_nucleo_v_ns))]
                                              for j in range(len(self.list_masks))]
                                             for k in range(len(self.list_detectors))]

                self.matrix_normed_sim = [[[np.zeros(0, )
                                            for x in range(len(self.list_detectors))]
                                           for y in range(len(self.list_masks))]
                                          for z in range(len(self.list_nucleo_norm_methods))]
    #

    def check_method(self, method):
        if not method in self.list_nucleo_norm_methods:
            raise NameError("method:{} not in the list of normalisation methods: {}"
                            .format(method, self.list_nucleo_norm_methods))

    def i_meth(self, method):
        return int(self.list_nucleo_norm_methods.index(method))

    def compute_normalized_sol_yields(self, method='sum'):

        As = self.get_nucleo_arr(0, "geo", "Asun")
        Ys = self.get_nucleo_arr(0, "geo", "Ysun")

        '''Sums all Ys for a given A (for all Z)'''
        Anrm = np.arange(As.max() + 1)
        Ynrm = np.zeros(int(As.max()) + 1)
        for i in range(Ynrm.shape[0]):  # changed xrange to range
            Ynrm[i] = Ys[As == i].sum()

        if method == 'sum':
            Ynrm /= np.sum(Ynrm)
            return Anrm, Ynrm
        else:
            raise NameError("Normalisation method '{}' for the solar is not recognized. Use:{}"
                            .format(method, self.list_nucleo_norm_methods))

    def is_sol_nucleo_yiled_normed(self, method):

        data = self.matrix_normed_sol[self.i_meth(method)]
        if len(data) == 0:
            a_sol, y_sol = self.compute_normalized_sol_yields(method)
            self.matrix_normed_sol[self.i_meth(method)] = np.vstack((a_sol, y_sol)).T

        data = self.matrix_normed_sol[self.i_meth(method)]
        if len(data) == 0:
            raise ValueError("failed to normalize simulations yeilds for: "
                             "method:{}".format(method))

    def compute_normalized_sim_yelds(self, det, mask, method):

        As = self.get_nucleo_arr(det, mask, "A")
        Ys = self.get_nucleo_arr(det, mask, "yields")

        '''Sums all Ys for a given A (for all Z)'''
        Anrm = np.arange(As.max() + 1)
        Ynrm = np.zeros(int(As.max()) + 1)
        for i in range(Ynrm.shape[0]):  # changed xrange to range
            Ynrm[i] = Ys[As == i].sum()

        if method == '':
            return Anrm, Ynrm

        elif method == 'sum':
            ''' Normalizes to a sum of all A '''
            norm = Ynrm.sum()
            Ynrm /= norm
            return Anrm, Ynrm

        elif method == "Asol=195":
            ''' Normalize to the solar abundances of a given element'''
            # a_sol = self.get_sol_data("Asun")
            # y_sol = self.get_sol_data("Ysun")

            tmp = self.get_nored_sol_abund("sum")
            a_sol, y_sol = tmp[:,0], tmp[:,1]

            element_a = int(method.split("=")[-1])
            if element_a not in a_sol: raise ValueError('Element: a:{} not in solar A\n{}'.format(element_a, a_sol))
            if element_a not in Anrm: raise ValueError('Element: a:{} not in a_arr\n{}'.format(element_a, Anrm))

            delta = np.float(y_sol[np.where(a_sol == element_a)] / Ynrm[np.where(Anrm == element_a)])
            Ynrm *= delta

            return Anrm, Ynrm
        else:
            raise NameError("Normalisation method '{}' for the simulation yields is not recognized. Use:{}"
                            .format(method, self.list_nucleo_norm_methods))

    def is_nucleo_yiled_normed(self, det, mask, method):

        data = self.matrix_normed_sim[self.i_det(det)][self.i_mask(mask)][self.i_meth(method)]
        if len(data) == 0:
            a_arr, y_arr = self.compute_normalized_sim_yelds(det, mask, method)
            data = np.vstack((a_arr, y_arr)).T
            self.matrix_normed_sim[self.i_det(det)][self.i_mask(mask)][self.i_meth(method)] = data

        data = self.matrix_normed_sim[self.i_det(det)][self.i_mask(mask)][self.i_meth(method)]
        if len(data) == 0:
            raise ValueError("failed to normalize simulations yeilds for: "
                             "det:{} mask:{} method:{}"
                             .format(det, mask, method))

    def get_normed_sim_abund(self, det, mask, method):

        self.check_det(det)
        self.check_mask(mask)
        self.check_method(method)
        self.is_nucleo_yiled_normed(det, mask, method)
        data = self.matrix_normed_sim[self.i_det(det)][self.i_mask(mask)][self.i_meth(method)]
        return data

    def get_nored_sol_abund(self, method='sum'):

        self.check_method(method)
        self.is_sol_nucleo_yiled_normed(method)
        data = self.matrix_normed_sol[self.i_meth(method)]
        return data


class EJECTA_PARS(EJECTA_NORMED_NUCLEO):

    def __init__(self, pprdir, skynetdir, add_mask=None):

        EJECTA_NORMED_NUCLEO.__init__(self, pprdir=pprdir, skynetdir=skynetdir, add_mask=add_mask)

        self.list_ejecta_pars_v_n = [
            "Mej_tot", "Ye_ave", "s_ave", "vel_inf_ave",
            "vel_inf_bern_ave", "theta_rms", "E_kin_ave",
            "E_kin_bern_ave"]

        self.matrix_ejecta_pars = [[[123456789.1
                                 for x in range(len(self.list_ejecta_pars_v_n))]
                                 for z in range(len(self.list_masks))]
                                 for y in range(len(self.list_detectors))]



        self.energy_constant = 1787.5521500932314

    def update_mask(self, new_mask=None):

        if new_mask != None:
            if not new_mask in self.list_masks:
                self.list_masks.append(new_mask)

                self.mask_matrix = [[np.zeros(0, )
                                     for i in range(len(self.list_masks))]
                                    for j in range(len(self.list_detectors))]

                self.matrix_ejecta = [[[np.zeros(0, )
                                        for k in range(len(self.list_ejecta_v_ns))]
                                       for j in range(len(self.list_masks))]
                                      for i in range(len(self.list_detectors))]

                self.matrix_ejecta_nucleo = [[[np.zeros(0, )
                                               for i in range(len(self.list_nucleo_v_ns))]
                                              for j in range(len(self.list_masks))]
                                             for k in range(len(self.list_detectors))]

                self.matrix_normed_sim = [[[np.zeros(0, )
                                            for x in range(len(self.list_detectors))]
                                           for y in range(len(self.list_masks))]
                                          for z in range(len(self.list_nucleo_norm_methods))]

                self.matrix_ejecta_pars = [[[123456789.1
                                             for x in range(len(self.list_ejecta_pars_v_n))]
                                            for z in range(len(self.list_masks))]
                                           for y in range(len(self.list_detectors))]
    #

    def check_ej_par_v_n(self, v_n):
        if not v_n in self.list_ejecta_pars_v_n:
            raise NameError("Parameter v_n: {} not in their list: {}"
                            .format(v_n, self.list_ejecta_pars_v_n))

    def i_ej_par(self, v_n):
        return int(self.list_ejecta_pars_v_n.index(v_n))

    # ----------------------------------------------

    @staticmethod
    def compute_ave_ye(mej, hist_ye):
        ye_ave = np.sum(hist_ye[:, 0] * hist_ye[:, 1]) / mej
        if ye_ave > 0.6: raise ValueError("Ye_ave > 0.6 ")
        value = np.float(ye_ave)
        return value

    @staticmethod
    def compute_ave_s(mej, hist_s):
        s_ave = np.sum(hist_s[:, 0] * hist_s[:, 1]) / mej
        value = np.float(s_ave)
        return value

    @staticmethod
    def compute_ave_vel_inf(mej, hist_vinf):
        vinf_ave = np.sum(hist_vinf[:, 0] * hist_vinf[:, 1]) / mej
        value = np.float(vinf_ave)
        return value

    @staticmethod
    def compute_ave_ekin(mej, hist_vinf):
        vinf_ave = EJECTA_PARS.compute_ave_vel_inf(mej, hist_vinf)
        E_kin_ave = np.sum(0.5 * vinf_ave ** 2 * hist_vinf[:, 1]) * Constants.energy_constant
        value = np.float(E_kin_ave)
        return value

    @staticmethod
    def compute_ave_theta_rms(hist_theta):
        theta, theta_M = hist_theta[:, 0], hist_theta[:, 1]
        # print(theta, theta_M)
        theta -= np.pi / 2.
        theta_rms = (180. / np.pi) * sqrt(np.sum(theta_M * theta ** 2) / np.sum(theta_M))
        value = np.float(theta_rms)
        return value

    # ----------------------------------------------

    def compute_ejecta_par(self, det, mask, v_n):

        # print("computing det:{} mask:{} v_n:{}".format(det, mask, v_n))

        if v_n == "Mej_tot":
            tarr_tot_flux_tot_mass = self.get_ejecta_arr(det, mask, "tot_mass")
            value = tarr_tot_flux_tot_mass[-1, 2]

        elif v_n == "Ye_ave":
            mej = self.get_ejecta_par(det, mask, "Mej_tot")
            hist_ye = self.get_ejecta_arr(det, mask, "hist Y_e")
            ye_ave = np.sum(hist_ye[:,0] * hist_ye[:,1]) / mej
            if ye_ave > 0.6: raise ValueError("Ye_ave > 0.6 "
                                              "det:{} mask:{} v_n:{}"
                                              .format(det,mask, v_n))
            value = np.float(ye_ave)

        elif v_n == "entropy_ave" or v_n == "s_ave":
            mej = self.get_ejecta_par(det, mask, "Mej_tot")
            hist_s = self.get_ejecta_arr(det, mask, "hist entropy")
            s_ave = np.sum(hist_s[:,0] * hist_s[:,1]) / mej
            value = np.float(s_ave)

        elif v_n == "vel_inf_ave":
            # if mask.__contains__("bern"):
            #     vel_v_n = "vel_inf_bern"
            # else:
            #     vel_v_n = "vel_inf"

            mej = self.get_ejecta_par(det, mask, "Mej_tot")
            hist_vinf = self.get_ejecta_arr(det, mask, "hist vel_inf")
            vinf_ave = np.sum(hist_vinf[:,0] * hist_vinf[:,1]) / mej
            value = np.float(vinf_ave)

        elif v_n == "E_kin_ave":
            # if v_n.__contains__("bern"):
            #     vel_v_n = "vel_inf_bern"
            # else:
            #     vel_v_n = "vel_inf"

            vinf_ave = self.get_ejecta_par(det, mask, "vel_inf_ave")
            hist_vinf = self.get_ejecta_arr(det, mask, "hist vel_inf")
            E_kin_ave = np.sum(0.5 * vinf_ave ** 2 * hist_vinf[:,1]) * self.energy_constant
            value = np.float(E_kin_ave)

        elif v_n == 'theta_rms':
            hist_theta = self.get_ejecta_arr(det, mask, "hist theta")
            theta, theta_M = hist_theta[:,0], hist_theta[:,1]
            theta -= pi / 2
            theta_rms = 180. / pi * sqrt(np.sum(theta_M * theta ** 2) / np.sum(theta_M))
            value = np.float(theta_rms)

        else:
            raise NameError("module_ejecta par v_n: {} (det:{}, mask:{}) does not have a"
                            " method for computing".format(v_n, det, mask))
        return value

    # ----------------------------------------------

    def is_ej_par_computed(self, det, mask, v_n):

        data = self.matrix_ejecta_pars[self.i_det(det)][self.i_mask(mask)][self.i_ej_par(v_n)]
        if data == 123456789.1:
            value = self.compute_ejecta_par(det, mask, v_n)
            self.matrix_ejecta_pars[self.i_det(det)][self.i_mask(mask)][self.i_ej_par(v_n)] = value

        data = self.matrix_ejecta_pars[self.i_det(det)][self.i_mask(mask)][self.i_ej_par(v_n)]
        if data == 123456789.1:
            raise ValueError("failed to compute module_ejecta par v_n:{} det:{} mask:{}"
                             .format(v_n, det, mask))

    def get_ejecta_par(self, det, mask, v_n):
        self.check_mask(mask)
        self.check_ej_par_v_n(v_n)
        self.check_det(det)
        self.is_ej_par_computed(det, mask, v_n)
        data = self.matrix_ejecta_pars[self.i_det(det)][self.i_mask(mask)][self.i_ej_par(v_n)]
        return data

"""