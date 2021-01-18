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


# from module_preanalysis.preanalysis import LOAD_ITTIME

from uutils import Printcolor, Tools

import paths as Paths

from profile_methods import (get_time_for_it, get_it_for_time)


class LOAD_RES_CORR:

    def __init__(self, dirlist, itlist, timesteplist):

        # LOAD_ITTIME.__init__(self, sim, pprdir=pprdir)

        self.set_corr_fname_intro = "corr_"

        # self.set_rootdir = resdir

        # self.sim = sim
        self.list_dirs = list(dirlist)

        self.list_iterations = itlist#Paths.get_list_iterations_from_res_3d(resdir)
        # self.times = interpoate_time_form_it(self.list_iterations, Paths.gw170817+sim+'/')
        self.times = []
        # for it in self.list_iterations:
        #     self.times.append(get_time_for_it(it, ))
        self.times = np.array(timesteplist)

        self.list_corr_v_ns = ["temp", "Ye", "rho", "theta", "r", "phi", "entr",
                               "ang_mom", "ang_mom_flux", "dens_unb_bern",
                               "inv_ang_mom_flux", 'vr', 'velz', 'vely', 'velx',
                               "Q_eff_nua", "Q_eff_nua_over_density", "hu_0"
                               ]

        self.corr_matrix = [[np.zeros(0,)
                             for x in range(2 * len(self.list_corr_v_ns) + 2)] # Here 2 * () as for correlation 2 v_ns are aneeded
                             for y in range(len(self.list_iterations))]

    def check_it(self, it):
        if not it in self.list_iterations:
            raise NameError("it:{} not in the list of iterations\n{}"
                            .format(it, self.list_iterations))

    def check_v_n(self, v_n):
        if not v_n in self.list_corr_v_ns:
            raise NameError("v_n:{} not in list of corr_v_ns\n{}"
                            .format(v_n, self.list_corr_v_ns))

    def i_v_n(self, v_n_x, v_n_y):
        self.check_v_n(v_n_x)
        self.check_v_n(v_n_y)
        idx1 = int(self.list_corr_v_ns.index(v_n_x))
        idx2 = int(self.list_corr_v_ns.index(v_n_y))
        # shift = len(self.list_corr_v_ns)
        return int(idx1 + idx2)

    def i_it(self, it):
        self.check_it(it)
        return int(self.list_iterations.index(it))

    def get_corr_fpath(self, it, v_n):
        self.check_it(it)
        fname = "/corr_" + v_n + ".h5"
        idx = self.list_dirs.index(it)
        fpath = self.list_dirs[idx] + fname
        if not os.path.isfile(fpath):
            raise IOError("Correlation file not found:\n{}".format(fpath))
        return fpath

    # ---

    def load_corr_file(self, it, v_n_x, v_n_y):

        v_n_x = str(v_n_x)
        v_n_y = str(v_n_y)

        self.check_v_n(v_n_x)
        self.check_v_n(v_n_y)

        idx = self.list_iterations.index(it)
        fdir = self.list_dirs[idx]

        # check if the direct file exists or the inverse
        fpath_direct = fdir + self.set_corr_fname_intro + v_n_x + '_' + v_n_y + ".h5"
        fpath_inverse = fdir + self.set_corr_fname_intro + v_n_y + '_' + v_n_x + ".h5"
        if os.path.isfile(fpath_direct):
            fpath = fpath_direct
        elif os.path.isfile(fpath_inverse):
            fpath = fpath_inverse
        else:
            print("IOError file not found :\n{}\n nor \n{}".format(fpath_direct, fpath_inverse))
            raise IOError("Correlation files not found:\n{}\n nor \n{}".format(fpath_direct, fpath_inverse))

        # check if the data inside is in the right format
        dfile = h5py.File(fpath, "r")
        v_ns_in_data = []
        for v_n_ in dfile:
            v_ns_in_data.append(v_n_)
        if not "mass" in v_ns_in_data:
            raise NameError("mass is not found in file:{}".format(fpath))
        if len(v_ns_in_data) > 3:
            raise NameError("More than 3 datasets found in corr file: {}".format(fpath))

        # extract edges and convert them into center of bins
        edge_x = np.array(dfile[v_n_x])
        edge_y = np.array(dfile[v_n_y])
        arr_x = 0.5 * (edge_x[1:] + edge_x[:-1])  # from edges to center of bins
        arr_y = 0.5 * (edge_y[1:] + edge_y[:-1])

        # extract mass (weights)
        if fpath == fpath_direct:
            mass = np.array(dfile["mass"]).T
        else:
            mass = np.array(dfile["mass"])

        # create a 2D table of the data (convenient format)
        result = Tools.combine(arr_x, arr_y, mass)
        self.corr_matrix[self.i_it(it)][self.i_v_n(v_n_x, v_n_y)] = result

        # fpath = Paths.ppr_sims + self.sim + "/res_3d/" + str(it) + "/corr_" + v_n_x + '_' + v_n_y + ".h5"
        #
        # if os.path.isfile(fpath):
        #     dfile = h5py.File(fpath, "r")
        #     v_ns_in_data = []
        #     for v_n_ in dfile:
        #         v_ns_in_data.append(v_n_)
        #     if not "mass" in v_ns_in_data:
        #         raise NameError("mass is not found in file:{}".format(fpath))
        #     if len(v_ns_in_data) > 3:
        #         raise NameError("More than 3 datasets found in corr file: {}".format(fpath))
        #     edge_x = np.array(dfile[v_n_x])
        #     edge_y = np.array(dfile[v_n_y])
        #     mass = np.array(dfile["mass"]).T
        #
        # if not os.path.isfile(fpath):
        #     print("Correlation file not found:\n{}".format(fpath))
        #     fpath_in = Paths.ppr_sims + self.sim + "/res_3d/" + str(it) + "/corr_" + v_n_y + '_' + v_n_x + ".h5"
        #     print("Loading inverse file:\n{}".format(fpath_in))
        #
        #     dfile = h5py.File(fpath_in, "r")
        #     v_ns_in_data = []
        #     for v_n_ in dfile:
        #         v_ns_in_data.append(v_n_)
        #     if not "mass" in v_ns_in_data:
        #         raise NameError("mass is not found in file:{}".format(fpath))
        #     if len(v_ns_in_data) > 3:
        #         raise NameError("More than 3 datasets found in corr file: {}".format(fpath))
        #     edge_x = np.array(dfile[v_n_x])
        #     edge_y = np.array(dfile[v_n_y])
        #     mass = np.array(dfile["mass"]).T
        #
        #     if not os.path.isfile(fpath_in):
        #         raise IOError("Correlation files not found:\n{}\n or \n{}".format(fpath, fpath_in))
        #
        #
        #
        # dfile = h5py.File(fpath, "r")
        #
        # v_ns_in_data = []
        # for v_n_ in dfile:
        #     v_ns_in_data.append(v_n_)
        #
        # if not "mass" in v_ns_in_data:
        #     raise NameError("mass is not found in file:{}".format(fpath))
        #
        # if len(v_ns_in_data) > 3:
        #     raise NameError("More than 3 datasets found in corr file: {}".format(fpath))
        #
        # v_ns_in_data.remove("mass")
        #
        # # for v_n__ in v_ns_in_data:
        # #     if not v_n__ in v_n:
        # #         raise NameError("in_data_v_n: {} is not in corr name v_n: {}"
        # #                         .format(v_n__, v_n))
        #
        #
        # # part1 = v_n.split(v_ns_in_data[0])
        # # part2 = v_n.split(v_ns_in_data[1])
        # # if v_ns_in_data[0] + '_' == part1[0]:
        # #     v_n1 = v_ns_in_data[0]
        # #     v_n2 = v_ns_in_data[1]
        # # elif '_' + v_ns_in_data[0] == part1[1]:
        # #     v_n1 = v_ns_in_data[1]
        # #     v_n2 = v_ns_in_data[0]
        # # elif v_ns_in_data[1] + '_' == part1[0]:
        # #     v_n1 = v_ns_in_data[1]
        # #     v_n2 = v_ns_in_data[0]
        # # elif '_' + v_ns_in_data[1] == part1[1]:
        # #     v_n1 = v_ns_in_data[0]
        # #     v_n2 = v_ns_in_data[1]
        # # else:
        # #     print("v_n: {}".format(v_n))
        # #     print("v_n_in_data: {}".format(v_ns_in_data))
        # #     print("v_n.split({}): {}".format(v_ns_in_data[0], part1))
        # #     print("v_n.split({}): {}".format(v_ns_in_data[1], part2))
        # #     print("v_ns_in_data[0]: {}".format(v_ns_in_data[0]))
        # #     print("v_ns_in_data[1]: {}".format(v_ns_in_data[1]))
        # #     raise NameError("Get simpler for f*ck sake...")
        # #
        # # print("v_n1: {}".format(v_n1))
        # # print("v_n2: {}".format(v_n2))
        # edge_x = np.array(dfile[v_n_x])
        # edge_y = np.array(dfile[v_n_y])
        # mass = np.array(dfile["mass"]).T
        #
        # arr_x = 0.5 * (edge_x[1:] + edge_x[:-1]) # from edges to center of bins
        # arr_y = 0.5 * (edge_y[1:] + edge_y[:-1])
        #
        # result = combine(arr_x, arr_y, mass)
        #
        # self.corr_matrix[self.i_it(it)][self.i_v_n(v_n_x, v_n_y)] = result

    # ---

    def is_corr_loaded(self, it, v_n_x, v_n_y):

        if len(self.corr_matrix[self.i_it(it)]) < self.i_v_n(v_n_x, v_n_y):
            raise ValueError("{} < {}".format(len(self.corr_matrix[self.i_it(it)]), self.i_v_n(v_n_x, v_n_y)))

        corr = self.corr_matrix[self.i_it(it)][self.i_v_n(v_n_x, v_n_y)]
        if len(corr) == 0:
            self.load_corr_file(it, v_n_x, v_n_y)
        else:
            Printcolor.yellow("Warning. Rewriting loaded data: v_n_x:{} v_n_y:{}, it:{}"
                              .format(v_n_x, v_n_y, it))

    def get_res_corr(self, it, v_n_x, v_n_y):
        self.check_v_n(v_n_x)
        self.check_v_n(v_n_y)
        self.check_it(it)
        self.is_corr_loaded(it, v_n_x, v_n_y)
        return self.corr_matrix[self.i_it(it)][self.i_v_n(v_n_x, v_n_y)]

    def get_time(self, it):
        self.check_it(it)
        return self.times[self.list_iterations.index(it)]

    def get_it(self, t):
        if t < self.times.min():
            raise ValueError("t:{} below the range: [{}, {}]"
                             .format(t, self.times.min(), self.times.max()))
        if t > self.times.max():
            raise ValueError("t:{} above the range: [{}, {}]"
                             .format(t, self.times.min(), self.times.max()))

        idx = Tools.find_nearest_index(self.times, t)
        return self.list_iterations[idx]

    def load_corr3d(self, it, v_n_x, v_n_y, v_n_z):

        v_n_x = str(v_n_x)
        v_n_y = str(v_n_y)
        v_n_z = str(v_n_z)

        self.check_v_n(v_n_x)
        self.check_v_n(v_n_y)
        self.check_v_n(v_n_z)

        idx = self.list_iterations.index(it)
        fdir = self.list_dirs[idx]

        fpath_direct = fdir + "/corr_" + v_n_x + '_' + v_n_y + '_' + v_n_z + ".h5"
        if not os.path.isfile(fpath_direct):
            raise IOError("Correlation files not found:\n{}".format(fpath_direct))

        dfile = h5py.File(fpath_direct, "r")

        edge_x = np.array(dfile[v_n_x])
        edge_y = np.array(dfile[v_n_y])
        edge_z = np.array(dfile[v_n_z])
        arr_x = 0.5 * (edge_x[1:] + edge_x[:-1])  # from edges to center of bins
        arr_y = 0.5 * (edge_y[1:] + edge_y[:-1])
        arr_z = 0.5 * (edge_z[1:] + edge_z[:-1])

        mass = np.array(dfile["mass"])

        print("arr_x.shape {}".format(arr_x.shape))
        print("arr_y.shape {}".format(arr_y.shape))
        print("arr_z.shape {}".format(arr_z.shape))
        print("mass.shape {}".format(mass.shape))

        return arr_x, arr_y, arr_z, mass

        # exit(1)


class LOAD_DENSITY_MODES:

    def __init__(self, fpath):

        # self.sim = sim
        #
        # self.set_rootdir = resdir

        self.gen_set = {
            'maximum_modes': 50,
            'fname' :  fpath,
            'int_phi': 'int_phi', # 1D array ( C_m )
            'int_phi_r': 'int_phi_r', # 2D array (1D for every iteration ( C_m(r) )
            'xcs': 'xc', # 1D array
            'ycs': 'yc', # 1D array
            'rs': 'rs', # 2D array (1D for every iteration)
            'times': 'times',
            'iterations':'iterations'
        }

        self.n_of_modes_max = 50
        self.list_data_v_ns = ["int_phi", "int_phi_r"]
        self.list_grid_v_ns = ["r_cyl", "times", "iterations", "xc", "yc", "rs"]

        self.data_dm_matrix = [[np.zeros(0,)
                              for k in range(len(self.list_data_v_ns))]
                              for z in range(self.n_of_modes_max)]

        self.grid_matrix = [np.zeros(0,)
                              for k in range(len(self.list_grid_v_ns))]

        self.list_modes = []#range(50)

    def check_data_v_n(self, v_n):
        if not v_n in self.list_data_v_ns:
            raise NameError("v_n: {} not in data list:\n{}"
                            .format(v_n, self.list_data_v_ns))

    def check_grid_v_n(self, v_n):
        if not v_n in self.list_grid_v_ns:
            raise NameError("v_n: {} not in grid list:\n{}"
                            .format(v_n,  self.list_grid_v_ns))

    def i_v_n(self, v_n):
        if v_n in self.list_data_v_ns:
            return int(self.list_data_v_ns.index(v_n))
        else:
            return int(self.list_grid_v_ns.index(v_n))

    def check_mode(self, mode):
        if len(self.list_modes) == 0:
            raise ValueError("list of modes was not loaded before data extraction")
        if not mode in self.list_modes:
            raise ValueError("mode: {} available modes: {}"
                             .format(mode, self.list_modes))

    def i_mode(self, mode):
        if len(self.list_modes) == 0:
            raise ValueError("list of modes was not loaded before data extraction")
        return int(self.list_modes.index(mode))

    # ---

    def load_density_modes(self):
        #
        if not os.path.isfile(self.gen_set['fname']):
            raise IOError("{} not found".format(self.gen_set['fname']))
        dfile = h5py.File(self.gen_set['fname'], "r")
        list_modes = []
        # setting list of density modes in the file
        for v_n in dfile:
            if str(v_n).__contains__("m="):
                mode = int(v_n.split("m=")[-1])
                list_modes.append(mode)
        self.list_modes = list_modes
        if len(self.list_modes) > self.n_of_modes_max - 1:
            raise ValueError("too many modes {} \n (>{}) in the file:{}"
                             .format(self.list_modes, self.n_of_modes_max, self.gen_set['fname']))
        # extracting data
        for v_n in dfile:
            if str(v_n).__contains__("m="):
                mode = int(v_n.split("m=")[-1])
                group = dfile[v_n]
                for v_n_ in group:
                    if str(v_n_) in self.list_data_v_ns:
                        self.data_dm_matrix[self.i_mode(mode)][self.i_v_n(v_n_)] = np.array(group[v_n_])
                    else:
                        raise NameError("{} group has a v_n: {} that is not in the data list:\n{}"
                                        .format(v_n, v_n_, self.list_data_v_ns))
            # extracting grid data, for overall
            else:
                if v_n in self.list_grid_v_ns:
                    self.grid_matrix[self.i_v_n(v_n)] = np.array(dfile[v_n])
                else:
                    NameError("dfile v_n: {} not in list of grid v_ns\n{}"
                                    .format(v_n, self.list_grid_v_ns))
        dfile.close()
        print("  modes: {}".format(self.list_modes))

    # ---

    def is_loaded(self, mode, v_n):

        if len(self.list_modes) == 0:
            self.load_density_modes()
        elif len(self.data_dm_matrix[self.i_mode(mode)][self.i_v_n(v_n)]) == 0:
            self.load_density_modes()

    def get_grid(self, v_n):

        if len(self.list_modes) == 0:
            self.load_density_modes()
        self.check_grid_v_n(v_n)
        self.is_loaded(self.list_modes[0], self.list_grid_v_ns[0])

        return self.grid_matrix[self.i_v_n(v_n)]

    def get_data(self, mode, v_n):

        self.check_data_v_n(v_n)
        if len(self.list_modes) == 0:
            self.load_density_modes()

        self.is_loaded(mode, v_n)

        return self.data_dm_matrix[self.i_mode(mode)][self.i_v_n(v_n)]

    #

    def get_grid_for_it(self, it, v_n):
        iterations = list(self.get_grid("iterations"))
        data =self.get_grid(v_n)
        return data[iterations.index(it)]

    def get_data_for_it(self, it, mode, v_n):
        iteration = list(self.get_grid("iterations"))
        data = self.get_data(mode, v_n)
        return data[iteration.index(it)]


''' NOT UPDATED & might not bee needed '''
"""
class LOAD_INT_DATA(LOAD_ITTIME):

    def __init__(self, sim, grid_object, pprdir, resdir):
        print("Warning. LOAD_INT_DATA is using only the '.grid_type' and '.list_int_grid_v_ns'\n"
              " It does not use the grid itself. Instead it loads the 'grid_type'_grid.h5 file")

        LOAD_ITTIME.__init__(self, sim, pprdir=pprdir)

        self.sim = sim

        self.set_rootdir = resdir

        self.list_iterations = list(Paths.get_list_iterations_from_res_3d(resdir))
        self.times = []
        for it in self.list_iterations:
            self.times.append(self.get_time_for_it(it, "profiles", d1d2d3prof="prof"))
        self.times = np.array(self.times)

        # GRID
        self.grid_type = grid_object.grid_type
        self.list_grid_v_ns = grid_object.list_int_grid_v_ns
        # self.list_grid_v_ns = ["x_cyl", "y_cyl", "z_cyl",
        #                       "r_cyl", "phi_cyl",
        #                       "dr_cyl", "dphi_cyl", "dz_cyl"]

        # for overriding the search for a grid.h5 in every iteration folder
        self.flag_force_unique_grid = False

        self.it_for_unique_grid = \
            Paths.get_it_from_itdir(
                Paths.find_itdir_with_grid(sim, "{}_grid.h5".format(grid_object.grid_type))
            )

        self.grid_data_matrix = [[np.zeros(0)
                                 for x in range(len(self.list_grid_v_ns))]
                                 for y in range(len(self.list_iterations))]


        self.list_of_v_ns = ["ang_mom", "ang_mom_flux", "density", "dens_unb_geo",
                             "dens_unb_bern","rho", "temp", "Ye", "lapse", "vr"]

        self.data_int_matrix = [[np.zeros(0)
                                 for x in range(len(self.list_of_v_ns))]
                                for y in range(len(self.list_iterations))]

    def check_grid_v_n(self, v_n):
        if not v_n in self.list_grid_v_ns:
            raise NameError("v_n:{} not in list of grid v_ns\n{}"
                            .format(v_n, self.list_grid_v_ns))

    def check_data_v_n(self, v_n):
        if not v_n in self.list_of_v_ns:
            raise NameError("v_n:{} not in list of data v_ns\n{}"
                            .format(v_n, self.list_of_v_ns))

    def check_it(self, it):
        if not it in self.list_iterations:
            raise NameError("it:{} not in the list of iterations \n{}"
                            .format(it,
                                    # self.list_iterations[find_nearest_index(np.array(self.list_iterations), it)],
                                    self.list_iterations))

    def i_data_v_n(self, v_n):
        self.check_data_v_n(v_n)
        return int(self.list_of_v_ns.index(v_n))

    def i_grid_v_n(self, v_n):
        self.check_grid_v_n(v_n)
        return int(self.list_grid_v_ns.index(v_n))

    def i_it(self, it):
        self.check_it(it)
        return int(self.list_iterations.index(it))

    def load_grid(self, it):

        path = Paths.ppr_sims + self.sim + '/' + self.set_rootdir + str(int(it)) + '/'
        fname = path + self.grid_type + '_grid.h5'
        if not os.path.isfile(fname):
            raise IOError("file: {} not found".format(fname))
        print("\tloading grid: {}".format(fname))
        grid_file = h5py.File(fname, "r")
        # print(grid_file)
        for v_n in self.list_grid_v_ns:
            if v_n not in grid_file:
                raise NameError("Loaded grid file {} does not have v_n:{} Expected only:\n{}"
                                .format(fname, v_n, self.list_grid_v_ns))

            grid_data = np.array(grid_file[v_n], dtype=np.float)
            self.grid_data_matrix[self.i_it(it)][self.i_grid_v_n(v_n)] = grid_data

    def load_data(self, it, v_n):

        path = Paths.ppr_sims + self.sim + '/' + self.set_rootdir + str(int(it)) + '/'
        fname = path + self.grid_type + '_' + v_n + ".h5"
        if not os.path.isfile(fname):
            raise IOError("file: {} not found".format(fname))
        data_file = h5py.File(fname, "r")
        if len(data_file) > 1:
            raise IOError("More than one v_n is found in data_file: {}".format(fname))
        if len(data_file) == 0:
            raise IOError("No datasets found in data_file: {}".format(fname))

        for v_n_ in data_file:
            if v_n_ != v_n:
                raise NameError("required v_n:{} not the same as the one in datafile:{}"
                                .format(v_n, v_n_))

        data = np.array(data_file[v_n], dtype=np.float)

        # print("loaded data ")
        # print(data)

        self.data_int_matrix[self.i_it(it)][self.i_data_v_n(v_n)] = data

    def is_grid_loaded(self, it):

        # if true it will only checks one it (and one grid) regardless of what it is called
        if self.flag_force_unique_grid and self.it_for_unique_grid != None:
            it = self.it_for_unique_grid

        grid_arr = self.grid_data_matrix[self.i_it(it)][self.i_grid_v_n(self.list_grid_v_ns[0])]
        # print(grid_arr);
        # exit(1)
        if len(grid_arr) == 0:
            self.load_grid(it)

    def is_data_loaded(self, it, v_n):
        data = self.data_int_matrix[self.i_it(it)][self.i_data_v_n(v_n)]
        # print(data); exit(1)
        if len(data) == 0:
            self.load_data(it, v_n)

    def get_grid_data(self, it, v_n):
        self.check_it(it)
        self.check_grid_v_n(v_n)

        self.is_grid_loaded(it)

        if self.flag_force_unique_grid and self.it_for_unique_grid != None:
            return self.grid_data_matrix[self.i_it(self.it_for_unique_grid)][self.i_grid_v_n(v_n)]
        else:
            return self.grid_data_matrix[self.i_it(it)][self.i_grid_v_n(v_n)]

    def get_int_data(self, it, v_n):
        self.check_it(it)
        self.check_data_v_n(v_n)

        self.is_data_loaded(it, v_n)

        return np.array(self.data_int_matrix[self.i_it(it)][self.i_data_v_n(v_n)])

    def get_it(self, t):
        if t < self.times.min():
            raise ValueError("t:{} below the range: [{}, {}]"
                             .format(t, self.times.min(), self.times.max()))
        if t > self.times.max():
            raise ValueError("t:{} above the range: [{}, {}]"
                             .format(t, self.times.min(), self.times.max()))

        idx = Tools.find_nearest_index(self.times, t)
        return self.list_iterations[idx]

    def get_time(self, it):
        self.check_it(it)
        return self.times[self.list_iterations.index(it)]


class ADD_METHODS_FOR_INT_DATA(LOAD_INT_DATA):

    def __init__(self, sim, grid_object):

        LOAD_INT_DATA.__init__(self, sim, grid_object)

    def ingeg_over_z(self, it, z3d_arr):
        dz = self.get_grid_data(it, "dz_cyl")
        return 2 * np.sum(z3d_arr * dz, axis=(2))

    def fill_pho0_and_phi2pi(self, phi1d_arr, z2d_arr):
        # adding phi = 360 point *copy of phi = 358(
        phi1d_arr = np.append(phi1d_arr, 2 * np.pi)
        z2d_arr = np.vstack((z2d_arr.T, z2d_arr[:, -1])).T
        # adding phi == 0 point (copy of phi=1)
        phi1d_arr = np.insert(phi1d_arr, 0, 0)
        z2d_arr = np.vstack((z2d_arr[:, 0], z2d_arr.T)).T
        return phi1d_arr, z2d_arr

    def get_modified_2d_data(self, it, v_n_x, v_n_y, v_n_z, mod):

        x_arr = self.get_grid_data(it, v_n_y)
        y_arr = self.get_grid_data(it, v_n_x)
        z_arr = self.get_int_data(it, v_n_z)

        if mod == 'xy slice':
            return np.array(x_arr[:, 0, 0]), np.array(y_arr[0, :, 0]), np.array(z_arr[:, :, 0]),

        elif mod == 'integ_over_z':
            return  np.array(x_arr[:, 0, 0]),np.array(y_arr[0, :, 0]), self.ingeg_over_z(it, z_arr)

        elif mod == 'integ_over_z fill_phi':
            y_arr, z_arr = self.fill_pho0_and_phi2pi(np.array(y_arr[0, :, 0]),
                                                     self.ingeg_over_z(it, z_arr))
            print(x_arr[:, 0, 0].shape, y_arr.shape, z_arr.shape)
            return np.array(x_arr[:, 0, 0]), y_arr, z_arr

        elif mod == 'integ_over_z fill_phi *r':
            r2d_arr = np.array(x_arr[:, :, 0])
            phi_arr = np.array(y_arr[0, :, 0])
            z2d_arr = self.ingeg_over_z(it, z_arr)

            rz2d = r2d_arr * z2d_arr
            phi_arr, rz2d = self.fill_pho0_and_phi2pi(phi_arr, rz2d)

            return np.array(x_arr[:, 0, 0]), phi_arr, rz2d

        elif mod == 'integ_over_z fill_phi *r log':

            r2d_arr = np.array(x_arr[:, :, 0])
            phi_arr = np.array(y_arr[0, :, 0])
            z2d_arr = self.ingeg_over_z(it, z_arr)

            rz2d = r2d_arr * z2d_arr
            phi_arr, rz2d = self.fill_pho0_and_phi2pi(phi_arr, rz2d)

            return np.array(x_arr[:, 0, 0]), phi_arr, np.log10(rz2d)

        elif mod == 'integ_over_z fill_phi -ave(r)':

            r2d_arr = np.array(x_arr[:, :, 0])
            phi_arr = np.array(y_arr[0, :, 0])
            z2d_arr = self.ingeg_over_z(it, z_arr)

            for i in range(len(x_arr[:, 0, 0])):
                z2d_arr[i, :] = z2d_arr[i, :] - (np.sum(z2d_arr[i, :]) / len(z2d_arr[i, :]))

            phi_arr, rz2d = self.fill_pho0_and_phi2pi(phi_arr, z2d_arr)

            return np.array(x_arr[:, 0, 0]), phi_arr, rz2d

        else:
            raise NameError("Unknown 'mod' parameter:{} ".format(mod))


# old class not used in the pipeline
class COMPUTE_STORE_DESITYMODES(LOAD_INT_DATA):

    def __init__(self, sim, grid_object):

        LOAD_INT_DATA.__init__(self, sim, grid_object)
        #
        self.gen_set = {
            'v_n': 'density',
            'v_n_r': 'r_cyl',
            'v_n_dr': 'dr_cyl',
            'v_n_phi': 'phi_cyl',
            'v_n_dphi': 'dphi_cyl',
            'v_n_dz': 'dz_cyl',
            'iterations': 'all',
            'do_norm': True,
            'm_to_norm': 0,
            'outfname': 'density_modes_int_lapse15.h5',
            'outdir': Paths.ppr_sims + sim + '/' + self.set_rootdir,
            'lapse_mask': 0.15
        }
        #
        self.list_modes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.list_dm_v_ns = ["int_phi", "int_phi_r"]
        self.data_dm_matrix = [[[np.zeros(0,)
                              for k in range(len(self.list_dm_v_ns))]
                             for z in range(len(self.list_modes))]
                            for k in range(len(self.list_iterations))]

    def check_dm_v_n(self, v_n):
        if v_n not in self.list_dm_v_ns:
            raise NameError("v_n: {} not in the list of Density Modes v_ns\n{}"
                            .format(v_n, self.list_dm_v_ns))

    def check_mode(self, mode):
        if not int(mode) in self.list_modes:
            raise NameError("mode:{} not in the list of modes\n{}"
                            .format(mode, self.list_modes))

    def i_mode(self, mode):
        self.check_mode(mode)
        return int(self.list_modes.index(mode))

    def i_dm_v_n(self, v_n):
        self.check_dm_v_n(v_n)
        return int(self.list_dm_v_ns.index(v_n))

    # ---

    def compute_density_mode_old(self, it, mode):

        # getting grid
        r_cyl = self.get_grid_data(it, self.gen_set["v_n_r"])
        dr_cyl = self.get_grid_data(it, self.gen_set["v_n_dr"])
        phi_cyl = self.get_grid_data(it, self.gen_set["v_n_phi"])
        dphi_cyl = self.get_grid_data(it, self.gen_set["v_n_dphi"])
        dz_cyl = self.get_grid_data(it, self.gen_set["v_n_dz"])

        # getting data
        density = self.get_int_data(it, self.gen_set["v_n"])

        if self.gen_set["lapse_mask"] != None:
            lapse =  self.get_int_data(it, "lapse")
            density[lapse < float(self.gen_set["lapse_mask"])] = 0

        # print(density.shape, phi_cyl.shape, r_cyl.shape, dr_cyl.shape)
        # print(dr_cyl[:, :, 0])

        m_int_phi, m_int_phi_r = \
            PHYSICS.get_dens_decomp_3d(density, r_cyl, phi_cyl, dphi_cyl, dr_cyl, dz_cyl, m=mode)

        if self.gen_set["do_norm"]:
            # print("norming")
            m_int_phi_norm, m_int_phi_r_norm = \
                PHYSICS.get_dens_decomp_3d(density, r_cyl, phi_cyl, dphi_cyl, dr_cyl, dz_cyl, m=int(self.gen_set["m_to_norm"]))
            m_int_phi /= m_int_phi_norm
            m_int_phi_r /= m_int_phi_r_norm

        self.data_dm_matrix[self.i_it(it)][self.i_mode(mode)][self.i_dm_v_n("int_phi")] = \
            m_int_phi
        self.data_dm_matrix[self.i_it(it)][self.i_mode(mode)][self.i_dm_v_n("int_phi_r")] = \
            np.array([m_int_phi_r])

    def compute_density_mode(self, it, mode):

        # getting grid
        r_cyl = self.get_grid_data(it, self.gen_set["v_n_r"])
        dr_cyl = self.get_grid_data(it, self.gen_set["v_n_dr"])
        phi_cyl = self.get_grid_data(it, self.gen_set["v_n_phi"])
        dphi_cyl = self.get_grid_data(it, self.gen_set["v_n_dphi"])
        dz_cyl = self.get_grid_data(it, self.gen_set["v_n_dz"])

        # getting data
        density = self.get_int_data(it, self.gen_set["v_n"])

        if self.gen_set["lapse_mask"] != None:
            lapse =  self.get_int_data(it, "lapse")
            density[lapse < float(self.gen_set["lapse_mask"])] = 0

        # print(density.shape, phi_cyl.shape, r_cyl.shape, dr_cyl.shape)
        # print(dr_cyl[:, :, 0])

        m_int_phi, m_int_phi_r = \
            PHYSICS.get_dens_decomp_3d(density, r_cyl, phi_cyl, dphi_cyl, dr_cyl, dz_cyl, m=mode)

        if self.gen_set["do_norm"]:
            # print("norming")
            m_int_phi_norm, m_int_phi_r_norm = \
                PHYSICS.get_dens_decomp_3d(density, r_cyl, phi_cyl, dphi_cyl, dr_cyl, dz_cyl, m=int(self.gen_set["m_to_norm"]))
            m_int_phi /= m_int_phi_norm
            m_int_phi_r /= m_int_phi_r_norm

        self.data_dm_matrix[self.i_it(it)][self.i_mode(mode)][self.i_dm_v_n("int_phi")] = \
            m_int_phi
        self.data_dm_matrix[self.i_it(it)][self.i_mode(mode)][self.i_dm_v_n("int_phi_r")] = \
            np.array([m_int_phi_r])

    # ---

    def is_computed(self, it, mode, v_n):

        if len(self.data_dm_matrix[self.i_it(it)][self.i_mode(mode)][self.i_dm_v_n(v_n)]) == 0:
            self.compute_density_mode(it, mode)

    def get_density_mode(self, it, mode, v_n):
        self.check_it(it)
        self.check_mode(mode)
        self.check_dm_v_n(v_n)
        self.is_computed(it, mode, v_n)
        return self.data_dm_matrix[self.i_it(it)][self.i_mode(mode)][self.i_dm_v_n(v_n)]
"""