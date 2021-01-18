"""

"""
from __future__ import division
import os.path
import numpy as np
import click
import h5py
from math import log10
import sys
from argparse import ArgumentParser
import time
from scidata.carpet.interp import Interpolator

from profile_grids import (CARTESIAN_GRID, CYLINDRICAL_GRID, POLAR_GRID)

from profile_formulas import FORMULAS

# from module_preanalysis.preanalysis import LOAD_ITTIME

from uutils import Printcolor

import paths as Paths

# from profile import __masks__

__masks__ = ["disk", "remnant"]

class LOAD_PROFILE_SLICE:

    def __init__(self, flist, itlist, timesteplist):

        # LOAD_ITTIME.__init__(self, sim, pprdir=pprdir)

        assert len(flist) == len(itlist)
        assert len(itlist) == len(timesteplist)

        self.set_max_nlevels = 8

        # self.sim = sim

        # self.set_rootdir = resdir

        self.list_files = flist

        self.list_iterations = itlist#Paths.get_list_iterations_from_res_3d(resdir)
        # isprof, itprof, tprof = self.get_ittime("profiles", "")
        # self.times = interpoate_time_form_it(self.list_iterations, Paths.gw170817+sim+'/')
        self.times = timesteplist
        # for it in self.list_iterations:
        #     self.times.append(self.get_time_for_it(it, output="profiles", d1d2d3prof="prof")) # prof
        # self.times = np.array(self.times)

        self.list_attrs_v_n = ["delta", "extent", "iteration", "origin", "reflevel", "time"]

        self.list_prof_v_ns = ["x", "y", "z", "rho", "w_lorentz", "vol", "press", "eps", "lapse", "velx", "vely", "velz",
                          "gxx", "gxy", "gxz", "gyy", "gyz", "gzz", "betax", "betay", "betaz", 'temp', 'Ye', "entr"] + \
                              ["u_0", "density",  "enthalpy", "vphi", "vr", "dens_unb_geo", "dens_unb_bern",
                          "dens_unb_garch", "ang_mom", "ang_mom_flux", "theta", "r", "phi"] + \
                              ["Q_eff_nua", "Q_eff_nue", "Q_eff_nux", "R_eff_nua", "R_eff_nue", "R_eff_nux",
                          "optd_0_nua", "optd_0_nue", "optd_0_nux", "optd_1_nua", "optd_1_nue", "optd_1_nux"] + \
                              ["rl_mask"] + \
                          ['abs_energy', 'abs_nua', 'abs_nue', 'abs_number', 'eave_nua', 'eave_nue',
                                'eave_nux', 'E_nua', 'E_nue', 'E_nux', 'flux_fac', 'ndens_nua', 'ndens_nue',
                                'ndens_nux', 'N_nua', 'N_nue', 'N_nux']
        self.list_planes = ["xy", "xz", "yz"]

        self.list_nlevels = [0 for i in range(len(self.list_iterations))]

        self.prof_data_matrix = [[[np.zeros(0, )
                                    for v_n in range(len(self.list_prof_v_ns))]
                                  for x in range(self.set_max_nlevels)]  # Here 2 * () as for correlation 2 v_ns are aneeded
                                 for y in range(len(self.list_iterations))]

        self.prof_attr_matrix = [[[np.zeros(0, )
                                    for v_n in range(len(self.list_prof_v_ns))]
                                  for x in range(self.set_max_nlevels)]  # Here 2 * () as for correlation 2 v_ns are aneeded
                                 for y in range(len(self.list_iterations))]

    def check_it(self, it):
        if not it in self.list_iterations:
            raise NameError("it:{} not in the list of iterations\n{}"
                            .format(it, self.list_iterations))

    def check_v_n(self, v_n):
        if not v_n in self.list_prof_v_ns:
            raise NameError("v_n:{} not in list of list_v_ns\n{}"
                            .format(v_n, self.list_prof_v_ns))

    def check_attrs_v_n(self, v_n):
        if not v_n in self.list_attrs_v_n:
            raise NameError("v_n:{} not in the list of list_attrs_v_ns\n{}"
                            .format(v_n, self.list_attrs_v_n))

    def i_it(self, it):
        self.check_it(it)
        return int(self.list_iterations.index(it))

    def i_prof_v_n(self, v_n):
        self.check_v_n(v_n)
        return int(self.list_prof_v_ns.index(v_n))

    def i_attr_v_n(self, v_n):
        return int(self.list_attrs_v_n.index(v_n))

    # ---

    def loaded_extract(self, it):

        # path = self.set_rootdir + str(it) + '/'
        # fname = "profile" + '.' + plane + ".h5"
        # fpath = path + fname

        idx = self.list_iterations.index(it)
        fpath = self.list_files[idx]

        if not os.path.isfile(fpath):
            raise IOError("file: {} not found".format(fpath))

        try:
            dfile = h5py.File(fpath, "r")
        except IOError:
            raise IOError("unable to open {}".format(fpath))

        nlevels = 0
        for key in dfile.keys():
            if key.__contains__("reflevel="):
                nlevels+=1

        # print("it:{} nlevels:{}".format(it, nlevels))

        nlevels = nlevels
        self.list_nlevels[self.i_it(it)] = nlevels

        for rl in np.arange(start=0, stop=nlevels, step=1):
            # datasets
            group = dfile["reflevel=%d" % rl]
            missing_v_ns = []
            # extracting data
            for v_n in self.list_prof_v_ns:
                if v_n in group.keys():
                    data = np.array(group[v_n])
                else:
                    missing_v_ns.append(v_n)
                    data = np.zeros(0,)
                self.prof_data_matrix[self.i_it(it)][rl][self.i_prof_v_n(v_n)] = data
            missing_attrs = []
            # extracting attributes
            for v_n in self.list_attrs_v_n:
                if v_n in group.attrs.keys():
                    attr = group.attrs[v_n]
                else:
                    missing_attrs.append(v_n)
                    attr = 0.
                self.prof_attr_matrix[self.i_it(it)][rl][self.i_attr_v_n(v_n)] = attr
            # checks
            if len(missing_v_ns) > 0:
                print("\tmissing data from \n\t{}".format(fpath, missing_v_ns))
            if len(missing_attrs) > 0:
                print("\tmissing attr from \n\t{}".format(fpath, missing_attrs))


        dfile.close()

    # ---

    def is_data_loaded_extracted(self, it, rl, v_n):
        data = self.prof_data_matrix[self.i_it(it)][rl][self.i_prof_v_n(v_n)]
        if len(data) == 0:
            self.loaded_extract(it)

        data = self.prof_data_matrix[self.i_it(it)][rl][self.i_prof_v_n(v_n)]
        if len(data) == 0:
            raise NameError("Failed to extract the data. it:{} rl:{} v_n:{}"
                             .format(it, rl, v_n))

    def is_data_loaded(self, it):
        rl = int(self.list_nlevels[self.i_it(it)])
        if rl == 0:
            self.loaded_extract(it)
        #

    def get_nlevels(self, it):
        self.is_data_loaded(it)
        return int(self.list_nlevels[self.i_it(it)])

    def get_data(self, it, rl, v_n):
        self.check_v_n(v_n)
        self.check_it(it)

        self.is_data_loaded_extracted(it, rl, v_n)
        return self.prof_data_matrix[self.i_it(it)][rl][self.i_prof_v_n(v_n)]

    def is_attr_loaded_extracted(self, it, rl, v_n):
        data = self.prof_attr_matrix[self.i_it(it)][rl][self.i_attr_v_n(v_n)]
        if len(data) == 0:
            self.loaded_extract(it)

        data = self.prof_attr_matrix[self.i_it(it)][rl][self.i_attr_v_n(v_n)]
        if len(data) == 0:
            raise NameError("failed tp extract attr. it:{} rl:{} v_n:{}".format(it, rl, v_n))

    def get_attr(self, it, rl, v_n):
        self.check_attrs_v_n(v_n)
        self.check_it(it)

        self.is_attr_loaded_extracted(it, rl, v_n)
        return self.prof_attr_matrix[self.i_it(it)][rl][self.i_attr_v_n(v_n)]


class COMPUTE_STORE_SLICE(LOAD_PROFILE_SLICE):

    def __init__(self, flist, itlist, timesteplist):

        LOAD_PROFILE_SLICE.__init__(self, flist=flist, itlist=itlist, timesteplist=timesteplist)

        self.list_comp_v_ns = ["hu_0", "Q_eff_nua_over_density", "abs_energy_over_density"]

        self.list_all_v_ns = self.list_prof_v_ns + self.list_comp_v_ns

        self.data_matrix = [[[np.zeros(0,)
                             for y in range(len(self.list_all_v_ns))]
                             for x in range(self.set_max_nlevels)]
                             for i in range(len(self.list_iterations))]

    def check_comp_v_n(self, v_n):
        if v_n not in self.list_all_v_ns:
            raise NameError("v_n:{} not in the v_n list \n{}"
                            .format(v_n, self.list_all_v_ns))

    def i_v_n(self, v_n):
        return int(self.list_all_v_ns.index(v_n))

    def set_data(self, it, rl, v_n, arr):
        self.data_matrix[self.i_it(it)][rl][self.i_v_n(v_n)] = arr

    def extract_data(self, it, rl, v_n):
        data = self.get_data(it, rl, v_n)
        self.data_matrix[self.i_it(it)][rl][self.i_v_n(v_n)] = data

    # --- #

    def compute_data(self, it, rl, v_n):

        if v_n == "Q_eff_nua_over_density":
            arr = FORMULAS.q_eff_nua_over_density(self.get_comp_data(it, rl, "Q_eff_nua"),
                                                  self.get_comp_data(it, rl, "density"))
        elif v_n == "abs_energy_over_density":
            arr = FORMULAS.abs_energy_over_density(self.get_comp_data(it, rl, "abs_energy"),
                                                  self.get_comp_data(it, rl, "density"))
        elif v_n == "hu_0":
            arr = FORMULAS.hu_0(self.get_comp_data(it, rl, "enthalpy"),
                                self.get_comp_data(it, rl, "u_0"))
        else:
            raise NameError("No method found for v_n:{} rl:{} it:{} Add entry to 'compute()'".format(v_n, rl, it))

        self.data_matrix[self.i_it(it)][rl][self.i_v_n(v_n)] = arr

    # --- #

    def is_available(self, it, rl, v_n):

        data = self.data_matrix[self.i_it(it)][rl][self.i_v_n(v_n)]
        if len(data) == 0:
            if v_n in self.list_prof_v_ns:
                self.extract_data(it, rl, v_n)
            elif v_n in self.list_comp_v_ns:
                self.compute_data(it, rl, v_n)
            else:
                raise NameError("v_n is not recognized: '{}' [COMPUTE STORE]".format(v_n))

    def get_comp_data(self, it, rl, v_n):
        self.check_it(it)
        self.check_comp_v_n(v_n)
        self.is_available(it, rl, v_n)

        return self.data_matrix[self.i_it(it)][rl][self.i_v_n(v_n)]


class ADD_MASK_SLICE(COMPUTE_STORE_SLICE):

    def __init__(self, flist, itlist, timesteplist):

        COMPUTE_STORE_SLICE.__init__(self, flist=flist, itlist=itlist, timesteplist=timesteplist)

        self.list_mask_v_ns = __masks__

        self.disk_mask_setup = {'rm_rl': True,  # REMOVE previouse ref. level from the next
                                'rho': [6.e4 / 6.176e+17, 1.e13 / 6.176e+17],  # REMOVE atmo and NS
                                'lapse': [0.15, 1.]}  # remove apparent horizon

        self.mask_matrix = [[[[np.zeros(0,)
                             for y in range(len(self.list_mask_v_ns))]
                             for p in range(len(self.list_planes))]
                             for x in range(self.set_max_nlevels)]
                             for i in range(len(self.list_iterations))]

    def check_mask_v_n(self, v_n):
        if not v_n in self.list_mask_v_ns:
            raise NameError("mask: {} is not recognized. Use:\n\t{}"
                            .format(v_n, self.list_mask_v_ns))

    def i_mask(self, v_n):
        return int(self.list_mask_v_ns.index(v_n))

    # -----------------------

    def compute_mask(self, it, rl, mask_v_n):

        if mask_v_n == "None":
            rho = self.get_comp_data(it, rl, "rho")
            arr = np.ones(rho.shape)
        elif mask_v_n == "rl":
            arr = self.get_comp_data(it, rl, "rl_mask")
        elif mask_v_n == "disk":
            rl_arr = self.get_comp_data(it, rl, "rl_mask")
            disk_mask_setup = self.disk_mask_setup
            for v_n in disk_mask_setup.keys()[1:]:
                arr = self.get_comp_data(it, rl, v_n)
                val1, val2 = disk_mask_setup[v_n][0], disk_mask_setup[v_n][1]
                tmp = np.ones(arr.shape)
                tmp[(arr<val1)&(arr>val2)] = 0
                rl_arr * tmp
            arr = rl_arr
        elif mask_v_n == "rl_Ye04":
            rl_mask = self.get_mask(it, rl, "rl")
            ye_mask = self.get_comp_data(it, rl, "Ye")
            ye_mask[ye_mask < 0.4] = 0
            ye_mask[ye_mask >= 0.4] = 1
            arr = rl_mask * ye_mask
        elif mask_v_n == "rl_theta60":
            rl_mask = self.get_mask(it, rl, "rl")
            theta = self.get_comp_data(it, rl, "theta")
            theta = 90 - (theta * 180 / np.pi)
            # print(theta); exit(1)
            theta = np.nan_to_num(theta)
            # print("{}: min:{} max:{} shape:{}".format("theta", theta.min(), theta.max(), theta.shape));
            # exit(1)
            theta[theta < 60.] = 0
            theta[theta >= 60.] = 1
            # print(theta)
            arr = rl_mask * theta
        elif mask_v_n == "rl_hu0":
            rl_mask = self.get_mask(it, rl, "rl")
            hu0 = self.get_comp_data(it, rl, "hu_0") * -1. # -1.6 -0.6
            hu0[hu0 < 1.] = 0.
            hu0[hu0 >= 1.] = 1
            arr = rl_mask * hu0
        else:
            raise NameError("No method set for mask: {}".format(mask_v_n))
        return arr

    # -----------------------

    def is_mask_computed(self, it, rl, mask_v_n):

        arr = self.mask_matrix[self.i_it(it)][rl][self.i_mask(mask_v_n)]
        if len(arr) == 0:
            arr = self.compute_mask(it, rl, mask_v_n)

        self.mask_matrix[self.i_it(it)][rl][self.i_mask(mask_v_n)] = arr

    def get_mask(self, it, rl, mask_v_n):
        #
        self.check_it(it)
        self.check_mask_v_n(mask_v_n)
        self.is_mask_computed(it, rl,mask_v_n)
        #
        arr = self.mask_matrix[self.i_it(it)][rl][self.i_mask(mask_v_n)]
        return arr


class MAINMETHODS_STORE_SLICE(ADD_MASK_SLICE):

    def __init__(self, flist, itlist, timesteplist):

        ADD_MASK_SLICE.__init__(self, flist=flist, itlist=itlist, timesteplist=timesteplist)

        # correlation tasks

        self.corr_task_dic_q_eff_nua_ye = [
            {"v_n": "Q_eff_nua", "edges": 10.0 ** np.linspace(-15., -10., 500)},
            {"v_n": "Ye", "edges": np.linspace(0, 0.5, 500)}
        ]

        self.corr_task_dic_q_eff_nua_dens_unb_bern = [
            {"v_n": "Q_eff_nua", "edges": 10.0 ** np.linspace(-15., -10., 500)},
            {"v_n": "dens_unb_bern", "edges": 10.0 ** np.linspace(-12., -6., 500)}
        ]

        self.corr_task_dic_q_eff_nua_over_D_theta = [
            {"v_n": "Q_eff_nua_over_density", "edges": 10.0 ** np.linspace(-10., -2., 500)},
            {"v_n": "theta", "edges": np.linspace(0., np.pi / 2., 500)}
        ]

        self.corr_task_dic_q_eff_nua_over_D_Ye = [
            {"v_n": "Q_eff_nua_over_density", "edges": 10.0 ** np.linspace(-10., -2., 500)},
            {"v_n": "Ye", "edges": np.linspace(0., 0.5, 500)}
        ]

        self.corr_task_dic_q_eff_nua_hu_0 = [
            {"v_n": "Q_eff_nua", "edges": 10.0 ** np.linspace(-15., -10., 500)},
            {"v_n": "hu_0", "edges": np.linspace(-1.2, -0.8, 500)}
        ]

        self.corr_task_dic_q_eff_nua_u_0 = [
            {"v_n": "Q_eff_nua", "edges": 10.0 ** np.linspace(-15., -10., 500)},
            {"v_n": "u_0", "edges": np.linspace(-1.2, -0.8, 500)}
        ]

        self.corr_task_dic_q_eff_nua_over_D_hu_0 = [
            {"v_n": "Q_eff_nua_over_density", "edges": 10.0 ** np.linspace(-10., -2., 500)},
            {"v_n": "hu_0", "edges": np.linspace(-1.2, -0.8, 500)}
        ]

        self.corr_task_dic_velz_ye = [
            {"v_n": "velz", "edges": np.linspace(-1., 1., 500)},  # in c
            {"v_n": "Ye", "edges": np.linspace(0, 0.5, 500)}
        ]

    def get_edges(self, it, corr_task_dic):

        dic = dict(corr_task_dic)

        if "edges" in dic.keys():
            return dic["edges"]

        # if "points" in dic.keys() and "scale" in dic.keys():
        #     min_, max_ = self.get_min_max(it, dic["v_n"])
        #     if "min" in dic.keys(): min_ = dic["min"]
        #     if "max" in dic.keys(): max_ = dic["max"]
        #     print("\tv_n: {} is in ({}->{}) range"
        #           .format(dic["v_n"], min_, max_))
        #     if dic["scale"] == "log":
        #         if min_ <= 0: raise ValueError("for Logscale min cannot be < 0. "
        #                                        "found: {}".format(min_))
        #         if max_ <= 0:raise ValueError("for Logscale max cannot be < 0. "
        #                                        "found: {}".format(max_))
        #         edges = 10.0 ** np.linspace(np.log10(min_), np.log10(max_), dic["points"])
        #
        #     elif dic["scale"] == "linear":
        #         edges = np.linspace(min_, max_, dic["points"])
        #     else:
        #         raise NameError("Unrecoginzed scale: {}".format(dic["scale"]))
        #     return edges

        raise NameError("specify 'points' or 'edges' in the setup dic for {}".format(dic['v_n']))

    def get_correlation(self, it, list_corr_task_dic, mask_v_n, multiplier=2.):

        edges = []
        for setup_dictionary in list_corr_task_dic:
            edges.append(self.get_edges(it, setup_dictionary))
        edges = tuple(edges)
        #
        correlation = np.zeros([len(edge) - 1 for edge in edges])
        #
        nlevels = self.get_nlevels(it)
        assert nlevels > 0
        for rl in range(nlevels):
            data = []
            # ye_mask = self.get_comp_data(it, rl, plane, "Ye")
            # ye_mask[ye_mask < 0.4] = 0
            # ye_mask[ye_mask >= 0.4] = 1
            mask = self.get_mask(it, rl, mask_v_n)
            dens = self.get_comp_data(it, rl, "density")
            weights = ((dens * mask) * np.prod(self.get_attr(it, rl, "delta")) * multiplier)
            print("rl:{} weights:{}".format(rl, weights.shape))
            for corr_dic in list_corr_task_dic:
                tmp = self.get_comp_data(it, rl, corr_dic["v_n"])
                # print("\tdata:{} | {} min:{} max:{} "
                #       .format(tmp.shape, corr_dic["v_n"], tmp.min(), tmp.max()))
                data.append(tmp.flatten())
            data = tuple(data)

            #
            #
            # mask = self.get_data(it, rl, plane, "rl_mask")
            # print("mask", mask.shape)
            # dens = self.get_data(it, rl, plane, "density")
            # print("dens", dens.shape)
            # dens_ = dens * mask
            # print("dens[mask]", dens_.shape)
            # weights = dens * np.prod(self.get_attr(it, rl, plane, "delta")) * multiplier
            # print(weights.shape),
            # weights = weights.flatten()
            # print(weights.shape)
            # # print("rl:{} mass:{} masked:{}".format(rl, np.sum(weights), np.sum(weights[mask])))
            # # weights = weights[mask]
            # for corr_dic in list_corr_task_dic:
            #     v_n = corr_dic["v_n"]
            #     data_ = self.get_data(it, rl, plane, v_n)
            #     # print(data_.shape)
            #     data.append(data_.flatten())
            #     print("data: {} {}".format(data_.shape, data[-1].shape))
            #     # if v_n == "Q_eff_nua":
            #     #     data[-1] = data[-1][3:-3, 3:-3]
            #     print("\t{} min:{} max:{} ".format(v_n, data[-1].min(), data[-1].max()))
            # data = tuple(data)
            try:
                tmp, _ = np.histogramdd(data, bins=edges, weights=weights.flatten())
            except ValueError:
                tmp = np.zeros([len(edge)-1 for edge in edges])
                Printcolor.red("ValueError it:{} rl:{} ".format(it, rl))
            correlation += tmp

        if np.sum(correlation) == 0:
            # print("\t")
            raise ValueError("sum(corr) = 0")

        return edges, correlation


""" old methods with planes also

class LOAD_PROFILE_XYXZ:

    def __init__(self, sim, pprdir, resdir):

        # LOAD_ITTIME.__init__(self, sim, pprdir=pprdir)


        self.set_max_nlevels = 8

        self.sim = sim

        self.set_rootdir = resdir

        self.list_iterations = Paths.get_list_iterations_from_res_3d(resdir)
        # isprof, itprof, tprof = self.get_ittime("profiles", "")
        # self.times = interpoate_time_form_it(self.list_iterations, Paths.gw170817+sim+'/')
        self.times = []
        for it in self.list_iterations:
            self.times.append(self.get_time_for_it(it, output="profiles", d1d2d3prof="prof")) # prof
        self.times = np.array(self.times)

        self.list_attrs_v_n = ["delta", "extent", "iteration", "origin", "reflevel", "time"]

        self.list_prof_v_ns = ["x", "y", "z", "rho", "w_lorentz", "vol", "press", "eps", "lapse", "velx", "vely", "velz",
                          "gxx", "gxy", "gxz", "gyy", "gyz", "gzz", "betax", "betay", "betaz", 'temp', 'Ye', "entr"] + \
                              ["u_0", "density",  "enthalpy", "vphi", "vr", "dens_unb_geo", "dens_unb_bern",
                          "dens_unb_garch", "ang_mom", "ang_mom_flux", "theta", "r", "phi"] + \
                              ["Q_eff_nua", "Q_eff_nue", "Q_eff_nux", "R_eff_nua", "R_eff_nue", "R_eff_nux",
                          "optd_0_nua", "optd_0_nue", "optd_0_nux", "optd_1_nua", "optd_1_nue", "optd_1_nux"] + \
                              ["rl_mask"] + \
                          ['abs_energy', 'abs_nua', 'abs_nue', 'abs_number', 'eave_nua', 'eave_nue',
                                'eave_nux', 'E_nua', 'E_nue', 'E_nux', 'flux_fac', 'ndens_nua', 'ndens_nue',
                                'ndens_nux', 'N_nua', 'N_nue', 'N_nux']
        self.list_planes = ["xy", "xz", "yz"]

        self.list_nlevels = [[0 for p in range(len(self.list_planes))]
                                for i in range(len(self.list_iterations))]

        self.prof_data_matrix = [[[[np.zeros(0, )
                                    for v_n in range(len(self.list_prof_v_ns))]
                                   for p in range(len(self.list_planes))]
                                  for x in range(self.set_max_nlevels)]  # Here 2 * () as for correlation 2 v_ns are aneeded
                                 for y in range(len(self.list_iterations))]

        self.prof_attr_matrix = [[[[np.zeros(0, )
                                    for v_n in range(len(self.list_prof_v_ns))]
                                   for p in range(len(self.list_planes))]
                                  for x in range(self.set_max_nlevels)]  # Here 2 * () as for correlation 2 v_ns are aneeded
                                 for y in range(len(self.list_iterations))]

    def check_it(self, it):
        if not it in self.list_iterations:
            raise NameError("it:{} not in the list of iterations\n{}"
                            .format(it, self.list_iterations))

    def check_v_n(self, v_n):
        if not v_n in self.list_prof_v_ns:
            raise NameError("v_n:{} not in list of list_v_ns\n{}"
                            .format(v_n, self.list_prof_v_ns))

    def check_attrs_v_n(self, v_n):
        if not v_n in self.list_attrs_v_n:
            raise NameError("v_n:{} not in the list of list_attrs_v_ns\n{}"
                            .format(v_n, self.list_attrs_v_n))

    def check_plane(self, plane):
        if plane not in self.list_planes:
            raise NameError("plane:{} not in the plane_list (in the class)\n{}"
                            .format(plane, self.list_planes))

    def i_it(self, it):
        self.check_it(it)
        return int(self.list_iterations.index(it))

    def i_plane(self, plane):
        self.check_plane(plane)
        return int(self.list_planes.index(plane))

    def i_prof_v_n(self, v_n):
        self.check_v_n(v_n)
        return int(self.list_prof_v_ns.index(v_n))

    def i_attr_v_n(self, v_n):
        return int(self.list_attrs_v_n.index(v_n))

    # ---

    def loaded_extract(self, it, plane):

        path = self.set_rootdir + str(it) + '/'
        fname = "module_profile" + '.' + plane + ".h5"
        fpath = path + fname

        if not os.path.isfile(fpath):
            raise IOError("file: {} not found".format(fpath))

        try:
            dfile = h5py.File(fpath, "r")
        except IOError:
            raise IOError("unable to open {}".format(fpath))

        nlevels = 0
        for key in dfile.keys():
            if key.__contains__("reflevel="):
                nlevels+=1

        # print("it:{} nlevels:{}".format(it, nlevels))

        nlevels = nlevels
        self.list_nlevels[self.i_it(it)][self.i_plane(plane)] = nlevels

        for rl in np.arange(start=0, stop=nlevels, step=1):
            # datasets
            group = dfile["reflevel=%d" % rl]
            missing_v_ns = []
            # extracting data
            for v_n in self.list_prof_v_ns:
                if v_n in group.keys():
                    data = np.array(group[v_n])
                else:
                    missing_v_ns.append(v_n)
                    data = np.zeros(0,)
                self.prof_data_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_prof_v_n(v_n)] = data
            missing_attrs = []
            # extracting attributes
            for v_n in self.list_attrs_v_n:
                if v_n in group.attrs.keys():
                    attr = group.attrs[v_n]
                else:
                    missing_attrs.append(v_n)
                    attr = 0.
                self.prof_attr_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_attr_v_n(v_n)] = attr
            # checks
            if len(missing_v_ns) > 0:
                print("\tmissing data from {}/profile_{}.h5\n\t{}".format(it, plane, missing_v_ns))
            if len(missing_attrs) > 0:
                print("\tmissing attr from {}/profile_{}.h5\n\t{}".format(it, plane, missing_attrs))


        dfile.close()

    # ---

    def is_data_loaded_extracted(self, it, rl, plane, v_n):
        data = self.prof_data_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_prof_v_n(v_n)]
        if len(data) == 0:
            self.loaded_extract(it, plane)

        data = self.prof_data_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_prof_v_n(v_n)]
        if len(data) == 0:
            raise NameError("Failed to extract the data. it:{} rl:{} plane:{} v_n:{}"
                             .format(it, rl, plane, v_n))

    def is_data_loaded(self, it, plane):
        rl = int(self.list_nlevels[self.i_it(it)][self.i_plane(plane)])
        if rl == 0:
            self.loaded_extract(it, plane)
        #

    def get_nlevels(self, it, plane):
        self.is_data_loaded(it, plane)
        return int(self.list_nlevels[self.i_it(it)][self.i_plane(plane)])

    def get_data(self, it, rl, plane, v_n):
        self.check_v_n(v_n)
        self.check_it(it)
        self.check_plane(plane)

        self.is_data_loaded_extracted(it, rl, plane, v_n)
        return self.prof_data_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_prof_v_n(v_n)]

    def is_attr_loaded_extracted(self, it, rl, plane, v_n):
        data = self.prof_attr_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_attr_v_n(v_n)]
        if len(data) == 0:
            self.loaded_extract(it, plane)

        data = self.prof_attr_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_attr_v_n(v_n)]
        if len(data) == 0:
            raise NameError("failed tp extract attr. it:{} rl:{} plane:{} v_n:{}"
                            .format(it, rl, plane, v_n))

    def get_attr(self, it, rl, plane, v_n):
        self.check_attrs_v_n(v_n)
        self.check_it(it)
        self.check_plane(plane)

        self.is_attr_loaded_extracted(it, rl, plane, v_n)
        return self.prof_attr_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_attr_v_n(v_n)]


class COMPUTE_STORE_XYXZ(LOAD_PROFILE_XYXZ):

    def __init__(self, sim, pprdir, resdir):

        LOAD_PROFILE_XYXZ.__init__(self, sim, pprdir=pprdir, resdir=resdir)

        self.list_comp_v_ns = ["hu_0", "Q_eff_nua_over_density", "abs_energy_over_density"]

        self.list_all_v_ns = self.list_prof_v_ns + self.list_comp_v_ns

        self.data_matrix = [[[[np.zeros(0,)
                             for y in range(len(self.list_all_v_ns))]
                             for p in range(len(self.list_planes))]
                             for x in range(self.set_max_nlevels)]
                             for i in range(len(self.list_iterations))]

    def check_comp_v_n(self, v_n):
        if v_n not in self.list_all_v_ns:
            raise NameError("v_n:{} not in the v_n list \n{}"
                            .format(v_n, self.list_all_v_ns))

    def i_v_n(self, v_n):
        return int(self.list_all_v_ns.index(v_n))

    def set_data(self, it, rl, plane, v_n, arr):
        self.data_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_v_n(v_n)] = arr

    def extract_data(self, it, rl, plane, v_n):
        data = self.get_data(it, rl, plane, v_n)
        self.data_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_v_n(v_n)] = data

    # --- #

    def compute_data(self, it, rl, plane, v_n):

        if v_n == "Q_eff_nua_over_density":
            arr = FORMULAS.q_eff_nua_over_density(self.get_comp_data(it, rl, plane, "Q_eff_nua"),
                                                  self.get_comp_data(it, rl, plane, "density"))
        elif v_n == "abs_energy_over_density":
            arr = FORMULAS.abs_energy_over_density(self.get_comp_data(it, rl, plane, "abs_energy"),
                                                  self.get_comp_data(it, rl, plane, "density"))
        elif v_n == "hu_0":
            arr = FORMULAS.hu_0(self.get_comp_data(it, rl, plane, "enthalpy"),
                                self.get_comp_data(it, rl, plane, "u_0"))
        else:
            raise NameError("No method found for v_n:{} plane:{} rl:{} it:{} Add entry to 'compute()'"
                            .format(v_n, plane, rl, it))

        self.data_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_v_n(v_n)] = arr

    # --- #

    def is_available(self, it, rl, plane, v_n):

        data = self.data_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_v_n(v_n)]
        if len(data) == 0:
            if v_n in self.list_prof_v_ns:
                self.extract_data(it, rl, plane, v_n)
            elif v_n in self.list_comp_v_ns:
                self.compute_data(it, rl, plane, v_n)
            else:
                raise NameError("v_n is not recognized: '{}' [COMPUTE STORE]".format(v_n))

    def get_comp_data(self, it, rl, plane, v_n):
        self.check_it(it)
        self.check_plane(plane)
        self.check_comp_v_n(v_n)
        self.is_available(it, rl, plane, v_n)

        return self.data_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_v_n(v_n)]


class ADD_MASK_XYXZ(COMPUTE_STORE_XYXZ):

    def __init__(self, sim, pprdir, resdir):

        COMPUTE_STORE_XYXZ.__init__(self, sim, pprdir=pprdir, resdir=resdir)

        self.list_mask_v_ns = __masks__

        self.disk_mask_setup = {'rm_rl': True,  # REMOVE previouse ref. level from the next
                                'rho': [6.e4 / 6.176e+17, 1.e13 / 6.176e+17],  # REMOVE atmo and NS
                                'lapse': [0.15, 1.]}  # remove apparent horizon

        self.mask_matrix = [[[[np.zeros(0,)
                             for y in range(len(self.list_mask_v_ns))]
                             for p in range(len(self.list_planes))]
                             for x in range(self.set_max_nlevels)]
                             for i in range(len(self.list_iterations))]

    def check_mask_v_n(self, v_n):
        if not v_n in self.list_mask_v_ns:
            raise NameError("mask: {} is not recognized. Use:\n\t{}"
                            .format(v_n, self.list_mask_v_ns))

    def i_mask(self, v_n):
        return int(self.list_mask_v_ns.index(v_n))

    # -----------------------

    def compute_mask(self, it, rl, plane, mask_v_n):

        if mask_v_n == "None":
            rho = self.get_comp_data(it, rl, plane, "rho")
            arr = np.ones(rho.shape)
        elif mask_v_n == "rl":
            arr = self.get_comp_data(it, rl, plane, "rl_mask")
        elif mask_v_n == "disk":
            rl_arr = self.get_comp_data(it, rl, plane, "rl_mask")
            disk_mask_setup = self.disk_mask_setup
            for v_n in disk_mask_setup.keys()[1:]:
                arr = self.get_comp_data(it, rl, plane, v_n)
                val1, val2 = disk_mask_setup[v_n][0], disk_mask_setup[v_n][1]
                tmp = np.ones(arr.shape)
                tmp[(arr<val1)&(arr>val2)] = 0
                rl_arr * tmp
            arr = rl_arr
        elif mask_v_n == "rl_Ye04":
            rl_mask = self.get_mask(it, rl, plane, "rl")
            ye_mask = self.get_comp_data(it, rl, plane, "Ye")
            ye_mask[ye_mask < 0.4] = 0
            ye_mask[ye_mask >= 0.4] = 1
            arr = rl_mask * ye_mask
        elif mask_v_n == "rl_theta60":
            rl_mask = self.get_mask(it, rl, plane, "rl")
            theta = self.get_comp_data(it, rl, plane, "theta")
            theta = 90 - (theta * 180 / np.pi)
            # print(theta); exit(1)
            theta = np.nan_to_num(theta)
            # print("{}: min:{} max:{} shape:{}".format("theta", theta.min(), theta.max(), theta.shape));
            # exit(1)
            theta[theta < 60.] = 0
            theta[theta >= 60.] = 1
            # print(theta)
            arr = rl_mask * theta
        elif mask_v_n == "rl_hu0":
            rl_mask = self.get_mask(it, rl, plane, "rl")
            hu0 = self.get_comp_data(it, rl, plane, "hu_0") * -1. # -1.6 -0.6
            hu0[hu0 < 1.] = 0.
            hu0[hu0 >= 1.] = 1
            arr = rl_mask * hu0
        else:
            raise NameError("No method set for mask: {}".format(mask_v_n))
        return arr

    # -----------------------

    def is_mask_computed(self, it, rl, plane, mask_v_n):

        arr = self.mask_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_mask(mask_v_n)]
        if len(arr) == 0:
            arr = self.compute_mask(it, rl, plane, mask_v_n)

        self.mask_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_mask(mask_v_n)] = arr

    def get_mask(self, it, rl, plane, mask_v_n):
        #
        self.check_it(it)
        self.check_plane(plane)
        self.check_mask_v_n(mask_v_n)
        self.is_mask_computed(it, rl, plane, mask_v_n)
        #
        arr = self.mask_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_mask(mask_v_n)]
        return arr


class MAINMETHODS_STORE_XYXZ(ADD_MASK_XYXZ):

    def __init__(self, sim, pprdir, resdir):

        ADD_MASK_XYXZ.__init__(self, sim, pprdir=pprdir, resdir=resdir)

        # correlation tasks

        self.corr_task_dic_q_eff_nua_ye = [
            {"v_n": "Q_eff_nua", "edges": 10.0 ** np.linspace(-15., -10., 500)},
            {"v_n": "Ye", "edges": np.linspace(0, 0.5, 500)}
        ]

        self.corr_task_dic_q_eff_nua_dens_unb_bern = [
            {"v_n": "Q_eff_nua", "edges": 10.0 ** np.linspace(-15., -10., 500)},
            {"v_n": "dens_unb_bern", "edges": 10.0 ** np.linspace(-12., -6., 500)}
        ]

        self.corr_task_dic_q_eff_nua_over_D_theta = [
            {"v_n": "Q_eff_nua_over_density", "edges": 10.0 ** np.linspace(-10., -2., 500)},
            {"v_n": "theta", "edges": np.linspace(0., np.pi / 2., 500)}
        ]

        self.corr_task_dic_q_eff_nua_over_D_Ye = [
            {"v_n": "Q_eff_nua_over_density", "edges": 10.0 ** np.linspace(-10., -2., 500)},
            {"v_n": "Ye", "edges": np.linspace(0., 0.5, 500)}
        ]

        self.corr_task_dic_q_eff_nua_hu_0 = [
            {"v_n": "Q_eff_nua", "edges": 10.0 ** np.linspace(-15., -10., 500)},
            {"v_n": "hu_0", "edges": np.linspace(-1.2, -0.8, 500)}
        ]

        self.corr_task_dic_q_eff_nua_u_0 = [
            {"v_n": "Q_eff_nua", "edges": 10.0 ** np.linspace(-15., -10., 500)},
            {"v_n": "u_0", "edges": np.linspace(-1.2, -0.8, 500)}
        ]

        self.corr_task_dic_q_eff_nua_over_D_hu_0 = [
            {"v_n": "Q_eff_nua_over_density", "edges": 10.0 ** np.linspace(-10., -2., 500)},
            {"v_n": "hu_0", "edges": np.linspace(-1.2, -0.8, 500)}
        ]

        self.corr_task_dic_velz_ye = [
            {"v_n": "velz", "edges": np.linspace(-1., 1., 500)},  # in c
            {"v_n": "Ye", "edges": np.linspace(0, 0.5, 500)}
        ]

    def get_edges(self, it, corr_task_dic):

        dic = dict(corr_task_dic)

        if "edges" in dic.keys():
            return dic["edges"]

        # if "points" in dic.keys() and "scale" in dic.keys():
        #     min_, max_ = self.get_min_max(it, dic["v_n"])
        #     if "min" in dic.keys(): min_ = dic["min"]
        #     if "max" in dic.keys(): max_ = dic["max"]
        #     print("\tv_n: {} is in ({}->{}) range"
        #           .format(dic["v_n"], min_, max_))
        #     if dic["scale"] == "log":
        #         if min_ <= 0: raise ValueError("for Logscale min cannot be < 0. "
        #                                        "found: {}".format(min_))
        #         if max_ <= 0:raise ValueError("for Logscale max cannot be < 0. "
        #                                        "found: {}".format(max_))
        #         edges = 10.0 ** np.linspace(np.log10(min_), np.log10(max_), dic["points"])
        #
        #     elif dic["scale"] == "linear":
        #         edges = np.linspace(min_, max_, dic["points"])
        #     else:
        #         raise NameError("Unrecoginzed scale: {}".format(dic["scale"]))
        #     return edges

        raise NameError("specify 'points' or 'edges' in the setup dic for {}".format(dic['v_n']))

    def get_correlation(self, it, plane, list_corr_task_dic, mask_v_n, multiplier=2.):

        edges = []
        for setup_dictionary in list_corr_task_dic:
            edges.append(self.get_edges(it, setup_dictionary))
        edges = tuple(edges)
        #
        correlation = np.zeros([len(edge) - 1 for edge in edges])
        #
        nlevels = self.get_nlevels(it, plane)
        assert nlevels > 0
        for rl in range(nlevels):
            data = []
            # ye_mask = self.get_comp_data(it, rl, plane, "Ye")
            # ye_mask[ye_mask < 0.4] = 0
            # ye_mask[ye_mask >= 0.4] = 1
            mask = self.get_mask(it, rl, plane, mask_v_n)
            dens = self.get_comp_data(it, rl, plane, "density")
            weights = ((dens * mask) * np.prod(self.get_attr(it, rl, plane, "delta")) * multiplier)
            print("rl:{} weights:{}".format(rl, weights.shape))
            for corr_dic in list_corr_task_dic:
                tmp = self.get_comp_data(it, rl, plane, corr_dic["v_n"])
                # print("\tdata:{} | {} min:{} max:{} "
                #       .format(tmp.shape, corr_dic["v_n"], tmp.min(), tmp.max()))
                data.append(tmp.flatten())
            data = tuple(data)

            #
            #
            # mask = self.get_data(it, rl, plane, "rl_mask")
            # print("mask", mask.shape)
            # dens = self.get_data(it, rl, plane, "density")
            # print("dens", dens.shape)
            # dens_ = dens * mask
            # print("dens[mask]", dens_.shape)
            # weights = dens * np.prod(self.get_attr(it, rl, plane, "delta")) * multiplier
            # print(weights.shape),
            # weights = weights.flatten()
            # print(weights.shape)
            # # print("rl:{} mass:{} masked:{}".format(rl, np.sum(weights), np.sum(weights[mask])))
            # # weights = weights[mask]
            # for corr_dic in list_corr_task_dic:
            #     v_n = corr_dic["v_n"]
            #     data_ = self.get_data(it, rl, plane, v_n)
            #     # print(data_.shape)
            #     data.append(data_.flatten())
            #     print("data: {} {}".format(data_.shape, data[-1].shape))
            #     # if v_n == "Q_eff_nua":
            #     #     data[-1] = data[-1][3:-3, 3:-3]
            #     print("\t{} min:{} max:{} ".format(v_n, data[-1].min(), data[-1].max()))
            # data = tuple(data)
            try:
                tmp, _ = np.histogramdd(data, bins=edges, weights=weights.flatten())
            except ValueError:
                tmp = np.zeros([len(edge)-1 for edge in edges])
                Printcolor.red("ValueError it:{} rl:{} plane:{}".format(it, rl, plane))
            correlation += tmp

        if np.sum(correlation) == 0:
            # print("\t")
            raise ValueError("sum(corr) = 0")

        return edges, correlation
"""