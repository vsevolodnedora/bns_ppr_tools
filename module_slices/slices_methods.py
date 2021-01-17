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
from scidata.utils import locate
import scidata.carpet.hdf5 as h5
# from scidata.carpet.interp import Interpolator


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
# import h5py
# import csv
# import os
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
from uutils import Tools, Printcolor
from module_preanalysis.it_time import LOAD_ITTIME
# from plotting_methods import PLOT_MANY_TASKS


class LOAD_STORE_DATASETS(LOAD_ITTIME):
    """
    Allows easy access to a scidata datasets of 2d data of a given simulation,
    loading them only if they are needed, and storing them for future accesses.

    Assumes that the simulation data is stired in /output-xxxx/data/ folders, where
    'xxxx' stands for the number.
    To know, which iterations are stored in what output-xxxx it first loads an ascii
    file <'file_for_it'> which should be present in all output-xxxx and have a
    structure with columns: 1:it 2:time (other columns do not matter)

    The 2d datasets, located in output-xxxx, expected to be named like:
    'rho.xy.h5'. The list of possible variables, (like 'rho') is set in
    <'self.list_v_ns'>. The list of possible planes (like 'xy') in set in
    <'self.list_planes'>.

    Logic:
        Every time when the dataset is requested via <'get_dataset(it, plane, v_n)'>,
        the class do:
            1. Checks what output-xxxx contains iteration 'it'
            2. Checks if the dataset for this output, plane and variable name 'v_n'
                has already been loaded and is present in the storage
                <'self.dataset_matrix[]'>
                If so: it will return the dataset from the storage.
                If not: it will load the required dataset and add it to the storage
                for future uses.

    """

    list_neut_v_ns = ["Q_eff_nua", "Q_eff_nue", "Q_eff_nux", "R_eff_nua", "R_eff_nue", "R_eff_nux",
                      "optd_0_nua", "optd_0_nux", "optd_0_nue", "optd_1_nua", "optd_1_nux", "optd_1_nue"]

    def __init__(self, sim, indir, pprdir):

        LOAD_ITTIME.__init__(self, sim, pprdir=pprdir)

        self.sim = sim
        self.nlevels = 7
        self.gen_set = {'nlevels':7,
                        'sim': sim,
                        'file_for_it': 'H.norm2.asc',
                        'iterations':0,
                        'indir': indir,
                        'outdir': pprdir + '/res_2d/'
                        }

        # self.output_it_map = {}

        self.list_outputs = self.get_list_outputs()
        _, self.iterations, self.times = \
            self.get_ittime(output="overall", d1d2d3prof="d2")
        # print(self.iterations[0], self.iterations[-1]); exit(1)

        # self.output_it_map, self.it_time = \
        #     set_it_output_map(Paths.gw170817+self.sim+'/')

        # self.iterations = np.array(self.it_time[:, 0], dtype=int)
        # self.times =  np.array(self.it_time[:, 1], dtype=float)


        self.list_v_ns = ['rho', 'Y_e', 'temperature', 's_phi', 'entropy', 'dens_unbnd'] + self.list_neut_v_ns
        self.list_planes=['xy', 'xz', 'xy']

        self.set_use_new_output_if_duplicated = False

        self.dataset_matrix = [[[0
                                  for z in range(len(self.list_v_ns))]
                                  for k in range(len(self.list_planes))]
                                  for s in range(len(self.list_outputs))]

    # def set_it_output_map(self):
    #     """
    #     Loads set of files that have '1:it 2:time ...' structure to get a map
    #     of what output-xxxx contains what iteration (and time)
    #     """
    #     print('-' * 25 + 'LOADING it list ({})'
    #           .format(self.gen_set['file_for_it']) + '-' * 25)
    #     print("\t loading from: {}".format(self.gen_set['indir']))
    #     files = locate(self.gen_set['file_for_it'], root=self.gen_set["indir"], followlinks=True)
    #     # remove folders like 'collated'
    #     selected = []
    #     for file in files:
    #         if file.__contains__('output-'):
    #             selected.append(file)
    #     # for overall count of number of iterations and files
    #     it_time = np.zeros(2)
    #     for file in selected:
    #         o_name = file.split('/')
    #         o_dir = ''
    #         for o_part in o_name:
    #             if o_part.__contains__('output-'):
    #                 o_dir = o_part
    #         if o_dir == '':
    #             raise NameError("Did not find output-xxxx in {}".format(o_name))
    #         it_time_i = np.loadtxt(file, usecols=(0,1))
    #         self.output_it_map[o_dir] = it_time_i
    #         it_time = np.vstack((it_time, it_time_i))
    #     it_time = np.delete(it_time, 0, 0)
    #     print('outputs:{} iterations:{} [{}->{}]'.format(len(selected),
    #                                                      len(it_time[:,0]),
    #                                                      int(it_time[:,0].min()),
    #                                                      int(it_time[:,0].max())))
    #     print('-' * 30 + '------DONE-----' + '-' * 30)
    #
    #     self.output_it_map, it_time = set_it_output_map(Paths.gw170817+self.sim+'/')
    #
    #     return it_time

    def check_v_n(self, v_n):
        if v_n not in self.list_v_ns:
            raise NameError("v_n:{} not in the v_n list (in the class)\n{}".format(v_n, self.list_v_ns))

    def check_plane(self, plane):
        if plane not in self.list_planes:
            raise NameError("plane:{} not in the plane_list (in the class)\n{}".format(plane, self.list_planes))

    def i_v_n(self, v_n):
        self.check_v_n(v_n)
        return int(self.list_v_ns.index(v_n))
    #
    def i_plane(self, plane):
        self.check_plane(plane)
        return int(self.list_planes.index(plane))

    def load_dataset(self, o_dir, plane, v_n):
        fname = v_n + '.' + plane + '.h5'
        files = locate(fname, root=self.gen_set['indir'] + o_dir +'/', followlinks=False)
        print("\t Loading: {} plane:{} v_n:{} dataset ({} files)"
              .format(o_dir, plane, v_n, len(files)))
        if len(files) > 1:
            raise ValueError("More than 1 file ({}) found. \nFile:{} location:{}"
                             "\nFiles: {}"
                             .format(len(files), fname, self.gen_set['indir'] + o_dir +'/', files))
        if len(files) == 0:
            raise IOError("NO fils found for {}. \nlocation:{}"
                             .format(fname, self.gen_set['indir'] + o_dir +'/'))
        dset = h5.dataset(files)
        # grid = dset.get_grid(iteration=it)
        # print("grid.origin: {}".format(grid.origin))
        # print("grid.dim   : {}".format(grid.dim))
        # print("grid.coordinates(): {}".format([ [np.array(coord).min(), np.array(coord).max()] for coord in grid.coordinates()]))
        # print("grid.levels: {}".format([level for level in grid.levels]))
        # print("grid.extent: {}".format(grid.extent))

        # exit(1)
        # print("\t loading it:{} plane:{} v_n:{} dset:{}"
        #       .format(o_dir, plane, v_n, dset))
        dset.get_grid().mesh()
        # dset.get_grid_data()
        self.dataset_matrix[self.i_output(o_dir)][self.i_plane(plane)][self.i_v_n(v_n)] = dset

    def i_output(self, o_dir):
        if o_dir not in self.list_outputs:
            raise NameError("plane:{} not in the plane_list (in the class)\n{}"
                            .format(o_dir, self.list_outputs))

        return int(self.list_outputs.index(o_dir))

    def is_dataset_loaded(self, o_dir, plane, v_n):
        if isinstance(self.dataset_matrix[self.i_output(o_dir)][self.i_plane(plane)][self.i_v_n(v_n)], int):
            self.load_dataset(o_dir, plane, v_n)

    # def it_to_output_dir(self, it):
    #     req_output_data_dir = []
    #     for output_data_dir in self.list_outputs:
    #         if int(it) in np.array(self.output_it_map[output_data_dir], dtype=int)[:, 0]:
    #             req_output_data_dir.append(output_data_dir)
    #
    #     if len(req_output_data_dir) > 1:
    #         if self.set_use_new_output_if_duplicated:
    #             print("Warning: it:{} is found in multiple outputs:{}"
    #                          .format(it, req_output_data_dir))
    #             return req_output_data_dir[0]
    #
    #         raise ValueError("it:{} is found in multiple outputs:{}\n"
    #                          "to overwrite, set 'set_use_new_output_if_duplicated=True' "
    #                          .format(it, req_output_data_dir))
    #     elif len(req_output_data_dir) == 0:
    #         raise ValueError("it:{} not found in a output_it_map:\n{}\n"
    #                              .format(it, self.output_it_map.keys()))
    #     else:
    #         return req_output_data_dir[0]

    def get_dataset(self, it, plane, v_n):
        # o_dir = self.it_to_output_dir(it)
        output = self.get_output_for_it(it)
        self.is_dataset_loaded(output, plane, v_n)
        dset = self.dataset_matrix[self.i_output(output)][self.i_plane(plane)][self.i_v_n(v_n)]
        if not it in dset.iterations:
            it__ = int(dset.iterations[Tools.find_nearest_index(np.array(dset.iterations), it)])
            raise ValueError("Iteration it:{} (located in {}) \n"
                             "not in the dataset list. Closest:{} Full list:\n{}"
                             .format(it, output, it__, dset.iterations))
        return dset

    def del_dataset(self, it, plane, v_n):
        # o_dir = self.it_to_output_dir(it)
        output = self.get_output_for_it(it)
        self.dataset_matrix[self.i_output(output)][self.i_plane(plane)][self.i_v_n(v_n)] = 0

    # def get_time(self, it):
    #
    #     time = self.it_time[np.where(self.it_time[:,0] == it), 1]
    #     time = [item for sublist in time for item in sublist]
    #     print(time)
    #     if len(time) == 2:
    #         Printcolor.yellow("for it:{} more than one timestep found {}"
    #                          .format(it, time))
    #         if time[0] == time[1]:
    #             return float(time[0]) * time_constant / 1000
    #         else:
    #             raise ValueError("for it:{} more than one timestep found {}"
    #                              .format(it, time))
    #     if len(time) == 0:
    #         raise ValueError("for it:{} no timesteps found"
    #                          .format(it))
    #     return float(time[0]) * time_constant / 1000

    def load_all(self, plane, v_n):

        print('-' * 25 + 'LOADING ALL DATASETS ({})'
              .format(self.gen_set['file_for_it']) + '-' * 25)
        Printcolor.yellow("Warning: loading all {} datasets "
                          "is a slow process".format(len(self.list_outputs)))
        for o_dir in self.list_outputs:
            try:
                self.is_dataset_loaded(o_dir, plane, v_n)
            except ValueError:
                Printcolor.red("Failed to load o_dir:{} plane:{} v_n:{}"
                               .format(o_dir, plane, v_n))

        self.set_all_it_times_from_outputs(plane, v_n)

        print('-' * 30 + '------DONE-----' + '-' * 30)

    def get_all_iterations_times(self, plane, v_n):

        iterations = []
        times = []
        for output in self.list_outputs:
            if isinstance(self.dataset_matrix[self.i_output(output)][self.i_plane(plane)][self.i_v_n(v_n)], int):
                raise ValueError("Not all datasets are loaded. Missing: {}".format(output))
            dset = self.dataset_matrix[self.i_output(output)][self.i_plane(plane)][self.i_v_n(v_n)]
            # iterations.append(dset.iterations)
            for it in dset.iterations:
                iterations.append(it)
                time = dset.get_time(it) * 0.004925794970773136 / 1000
                times.append(time)
                # print("it:{}, time:{}".format(it, time))

        assert len(list(set(iterations))) == len(list(set(times)))

        iterations = np.sort(list(set(iterations)))
        times = np.sort(list(set(times)))

        return iterations, times

    def set_all_it_times_from_outputs(self, plane, v_n):

        self.iterations, self.times = self.get_all_iterations_times(plane, v_n)
        print('\tIterations [{}->{}] and times [{:.3f}->{:.3f}] have been reset.'
              .format(self.iterations[0], self.iterations[-1],
                      self.times[0], self.times[-1]))

        # return list(set([item for sublist in iterations for item in sublist])), \
        #        list(set([item for sublist in iterations for item in sublist]))

    # def get_all_timesteps(self, plane, v_n):
    #
    #     iterations = self.get_all_iterations(plane, v_n)
    #     times = []
    #     for iteration in iterations:
    #         times.append(self.get_time(iteration))
    #     return times


class EXTRACT_STORE_DATA(LOAD_STORE_DATASETS):
    """
    blablabla
    """

    def __init__(self, sim, indir, pprdir):

        LOAD_STORE_DATASETS.__init__(self, sim, indir=indir, pprdir=pprdir)

        # self.gen_set = {'nlevels': 7,
        #                 'file_for_it': 'H.norm2.asc',
        #                 'iterations': 0,
        #                 'indir': Paths.gw170817 + sim + '/',
        #                 'outdir': Paths.ppr_sims + sim + '/2d/'}

        # self.list_v_ns   = ['rho', 'Y_e', 'temperature', 's_phi', 'entropy', 'dens_unbnd']
        # self.list_planes = ['xy', 'xz']
        self.v_n_map = {
            'rho':          "HYDROBASE::rho",
            'Y_e':          "HYDROBASE::Y_e",
            's_phi':        "BNSANALYSIS::s_phi",
            'temperature':  "HYDROBASE::temperature",
            'entropy':      "HYDROBASE::entropy",
            'dens_unbnd':   "BNSANALYSIS::dens_unbnd",
            'R_eff_nua':    "THC_LEAKAGEBASE::R_eff_nua",
            'R_eff_nue':    "THC_LEAKAGEBASE::R_eff_nue",
            'R_eff_nux':    "THC_LEAKAGEBASE::R_eff_nux",
            'Q_eff_nua':    "THC_LEAKAGEBASE::Q_eff_nua",
            'Q_eff_nue':    "THC_LEAKAGEBASE::Q_eff_nue",
            'Q_eff_nux':    "THC_LEAKAGEBASE::Q_eff_nux",
            'optd_0_nua':   "THC_LEAKAGEBASE::optd_0_nua",
            'optd_0_nue':   "THC_LEAKAGEBASE::optd_0_nue",
            'optd_0_nux':   "THC_LEAKAGEBASE::optd_0_nux",
            'optd_1_nua':   "THC_LEAKAGEBASE::optd_1_nua",
            'optd_1_nue':   "THC_LEAKAGEBASE::optd_1_nue",
            'optd_1_nux':   "THC_LEAKAGEBASE::optd_1_nux",
        }


        # self.output_it_map = {}
        # self.it_time = self.set_it_output_map()

        self.data_matrix = [[[np.zeros(0,)
                            for z in range(len(self.list_v_ns))]
                            for k in range(len(self.list_planes))]
                            for s in range(len(self.iterations))]

        self.grid_matrix = [[[0
                            for z in range(len(self.list_v_ns))]
                            for k in range(len(self.list_planes))]
                            for s in range(len(self.iterations))]

    def check_it(self, it):
        if not int(it) in np.array(self.iterations, dtype=int):
            it_ = int(self.iterations[Tools.find_nearest_index(self.iterations, it), 0])
            raise NameError("it:{} not in the list on iterations: Closest one: {}"
                            .format(it, it_))

        idx = np.where(np.array(self.iterations, dtype=int) == int(it))

        if len(idx) == 0:
            raise ValueError("For it:{} NO it are found in the it_time[:,0]".format(it))

        if len(idx) > 1:
            raise ValueError("For it:{} multiple it are found in the it_time[:,0]".format(it))

        # print("it:{} idx:{}".format(it, idx))

    def i_it(self, it):
        self.check_it(it)
        idx = list(self.iterations).index(int(it))
        return idx
        #
        #
        #
        #
        # self.check_it(it)
        # idx = np.array(np.where(np.array(self.it_time[:,0], dtype=int) == int(it)), dtype=int)
        # idx = list(set([item for sublist in idx for item in sublist]))
        # print("it:{} idx:{}, type:{} len:{}".format(it, idx, type(idx), len(idx)))
        # return int(idx)

    # ---------- GRID

    def extract_grid(self, it, plane, v_n):
        print("\t extracting grid it:{} plane:{} v_n:{}".format(it, plane, v_n))
        dset = self.get_dataset(it, plane, v_n)
        self.grid_matrix[self.i_it(it)][self.i_plane(plane)][self.i_v_n(v_n)] = \
            dset.get_grid(iteration=it)

        # exit(0)

    def is_grid_extracted(self, it, plane, v_n):

        if isinstance(self.grid_matrix[self.i_it(it)][self.i_plane(plane)][self.i_v_n(v_n)], int):
            self.extract_grid(it, plane, v_n)

    def get_grid(self, it, plane, v_n):

        self.check_plane(plane)
        self.check_v_n(v_n)
        self.is_grid_extracted(it, plane, v_n)

        return self.grid_matrix[self.i_it(it)][self.i_plane(plane)][self.i_v_n(v_n)]

    def del_grid(self, it, plane, v_n):

        self.check_plane(plane)
        self.check_v_n(v_n)

        self.grid_matrix[self.i_it(it)][self.i_plane(plane)][self.i_v_n(v_n)] = 0

    # ---------- DATA

    def extract(self, it, plane, v_n):

        print("\t extracting it:{} plane:{} v_n:{}".format(it, plane, v_n))
        dset = self.get_dataset(it, plane, v_n)
        try:
            grid = self.get_grid(it, plane, v_n)
            data = dset.get_grid_data(grid, iteration=it, variable=self.v_n_map[v_n])
            self.data_matrix[self.i_it(it)][self.i_plane(plane)][self.i_v_n(v_n)] = data

        except KeyError:
            raise KeyError("Wrong Key. Data not found. dset contains:{} attmeped:{} {} {}"
                           .format(dset.metadata[0], self.v_n_map[v_n], plane, it))

    def is_data_extracted(self, it, plane, v_n):

        if len(self.data_matrix[self.i_it(it)][self.i_plane(plane)][self.i_v_n(v_n)]) == 0:
            self.extract(it, plane, v_n)

    def get_data(self, it, plane, v_n):
        self.check_plane(plane)
        self.check_v_n(v_n)

        self.is_data_extracted(it, plane, v_n)

        return self.data_matrix[self.i_it(it)][self.i_plane(plane)][self.i_v_n(v_n)]

    def del_data(self, it, plane, v_n):

        self.check_plane(plane)
        self.check_v_n(v_n)

        self.data_matrix[self.i_it(it)][self.i_plane(plane)][self.i_v_n(v_n)] = np.zeros(0,)

    # ----------- TIME

    # def get_time(self, it):
    #     self.check_it(it)
    #     return float(self.it_time[np.where(self.it_time[:,0] == it), 1][0]) * time_constant / 1000



    # def get_time_(self, it):
    #     return self.get_time__(it)


class EXTRACT_FOR_RL(EXTRACT_STORE_DATA):

    def __init__(self, sim, indir, pprdir):
        EXTRACT_STORE_DATA.__init__(self, sim, indir=indir, pprdir=pprdir)

        self.list_grid_v_ns = ["x", "y", "z", "delta"]


        self.extracted_grid_matrix = [[[[np.zeros(0,)
                            for z in range(len(self.list_grid_v_ns))]
                            for j in range(self.nlevels)]
                            for k in range(len(self.list_planes))]
                            for s in range(len(self.iterations))]

        self.extracted_data_matrix = [[[[np.zeros(0,)
                            for z in range(len(self.list_v_ns))]
                            for j in range(self.nlevels)]
                            for k in range(len(self.list_planes))]
                            for s in range(len(self.iterations))]

        self.default_v_n = "rho"

    def check_rl(self, rl):
        if rl < 0:
            raise ValueError("Unphysical rl:{} ".format(rl))
        if rl < 0 or rl > self.nlevels:
            raise ValueError("rl is not in limits: {}"
                             .format(rl, self.nlevels))

    def i_rl(self, rl):
        return int(rl)

    def check_grid_v_n(self, v_n):
        if not v_n in self.list_grid_v_ns:
            raise NameError("v_n:{} not in list_grid_v_ns:{}"
                            .format(v_n, self.list_grid_v_ns))

    def i_gr_v_n(self, v_n):
        return int(self.list_grid_v_ns.index(v_n))

    def __extract_grid_data_rl(self, it, plane, rl, grid):

        coords = grid.coordinates()
        mesh = list(np.meshgrid(coords[self.i_rl(rl)], indexing='ij'))
        points = np.column_stack([x.flatten() for x in mesh])
        xyz = []
        delta = grid.levels[self.i_rl(rl)].delta
        # for x in mesh:
        #     xyz.append(x)
        #     print("x: {} ".format(x.shape))
        # print(len(coords))
        # # print(mesh)
        # print(len(mesh))
        # print(len(xyz))
        # print(points.shape)

        self.extracted_grid_matrix[self.i_it(it)][self.i_plane(plane)][self.i_rl(rl)][self.i_gr_v_n("delta")] = delta
        if plane == "xy":
            x, y = grid.mesh()[rl]
            # print(x.shape)
            # print(y.shape)
            self.extracted_grid_matrix[self.i_it(it)][self.i_plane(plane)][self.i_rl(rl)][self.i_gr_v_n("x")] = x
            self.extracted_grid_matrix[self.i_it(it)][self.i_plane(plane)][self.i_rl(rl)][self.i_gr_v_n("y")] = y
        elif plane == "xz":
            x, z = grid.mesh()[rl]
            # print(x.shape)
            # print(z.shape)
            self.extracted_grid_matrix[self.i_it(it)][self.i_plane(plane)][self.i_rl(rl)][self.i_gr_v_n("x")] = x
            self.extracted_grid_matrix[self.i_it(it)][self.i_plane(plane)][self.i_rl(rl)][self.i_gr_v_n("z")] = z
        else:
            raise NameError("Plane: {} is not recognized")


    def extract_grid_data_rl(self, it, plane, rl, v_n):

        self.check_grid_v_n(v_n)

        for data_v_n in self.list_v_ns:
            # scrolling through all possible v_ns, looking for the one loaded
            if not isinstance(self.grid_matrix[self.i_it(it)][self.i_plane(plane)][self.i_v_n(data_v_n)], int):
                grid = self.get_grid(it, plane, data_v_n)
                self.__extract_grid_data_rl(it, plane, rl, grid)
                return 0
        print("\tNo pre-loaded data found. Loading default ({}))".format(self.default_v_n))
        grid = self.get_grid(it, plane, self.default_v_n)
        self.__extract_grid_data_rl(it, plane, rl, grid)
        return 0


    def is_grid_extracted_for_rl(self, it, plane, rl, v_n):

        arr = self.extracted_grid_matrix[self.i_it(it)][self.i_plane(plane)][self.i_rl(rl)][self.i_gr_v_n(v_n)]
        if len(arr) == 0:
            self.extract_grid_data_rl(it, plane, rl, v_n)

    def get_grid_v_n_rl(self, it, plane, rl, v_n):

        self.check_grid_v_n(v_n)
        self.check_plane(plane)
        self.check_it(it)
        self.check_rl(rl)

        self.is_grid_extracted_for_rl(it, plane, rl, v_n)

        data = self.extracted_grid_matrix[self.i_it(it)][self.i_plane(plane)][self.i_rl(rl)][self.i_gr_v_n(v_n)]
        return data



    def extract_data_rl(self, it, plane, rl, v_n):

        data = self.get_data(it, plane, v_n)
        data = np.array(data[self.i_rl(rl)])

        self.extracted_data_matrix[self.i_it(it)][self.i_plane(plane)][self.i_rl(rl)][self.i_v_n(v_n)] = data

    def is_data_extracted_for_rl(self, it, plane,rl, v_n):

        arr = self.extracted_data_matrix[self.i_it(it)][self.i_plane(plane)][self.i_rl(rl)][self.i_v_n(v_n)]
        if len(arr) == 0:
            self.extract_data_rl(it, plane, rl, v_n)

    def get_data_rl(self, it, plane, rl, v_n):

        self.check_v_n(v_n)
        self.check_plane(plane)
        self.check_it(it)
        self.check_rl(rl)

        self.is_data_extracted_for_rl(it, plane, rl, v_n)

        data = self.extracted_data_matrix[self.i_it(it)][self.i_plane(plane)][self.i_rl(rl)][self.i_v_n(v_n)]
        return data


class COMPUTE_STORE(EXTRACT_FOR_RL):

    def __init__(self, sim, indir, pprdir):
        EXTRACT_FOR_RL.__init__(self, sim, indir=indir, pprdir=pprdir)

    def get_rho_modes_for_rl(self, rl=6, mmax=8):
        import numexpr as ne

        iterations = self.iterations  # apply limits on it
        #
        times = []
        modes = [[] for m in range(mmax + 1)]
        xcs = []
        ycs = []
        #
        for idx, it in enumerate(iterations):
            print("\tprocessing iteration: {}/{}".format(idx, len(iterations)))
            # get z=0 slice
            # lapse = o_slice.get_data_rl(it, "xy", rl, "lapse")
            rho = self.get_data_rl(it, "xy", rl, "rho")
            # o_slice.get_data_rl(it, "xy", rl, "vol")
            # w_lorentz = o_slice.get_data_rl(it, "xy", rl, "w_lorentz")

            delta = self.get_grid_v_n_rl(it, "xy", rl, "delta")#[:-1]
            #
            dxyz = np.prod(delta)
            x = self.get_grid_v_n_rl(it, "xy", rl, "x")
            y = self.get_grid_v_n_rl(it, "xy", rl, "y")
            # z = self.get_grid_v_n_rl(it, "xy", rl, "z")
            # x = x[:, :, 0]
            # y = y[:, :, 0]

            # apply mask to cut off the horizon
            # rho[lapse < 0.15] = 0

            # Exclude region outside refinement levels
            idx = np.isnan(rho)
            rho[idx] = 0.0
            # vol[idx] = 0.0
            # w_lorentz[idx] = 0.0

            # Compute center of mass
            # modes[0].append(dxyz * ne.evaluate("sum(rho * w_lorentz * vol)"))
            # Ix = dxyz * ne.evaluate("sum(rho * w_lorentz * vol * x)")
            # Iy = dxyz * ne.evaluate("sum(rho * w_lorentz * vol * y)")
            # xc = Ix / modes[0][-1]
            # yc = Iy / modes[0][-1]
            # phi = ne.evaluate("arctan2(y - yc, x - xc)")
            #
            modes[0].append(dxyz * ne.evaluate("sum(rho)"))
            Ix = dxyz * ne.evaluate("sum(rho * x)")
            Iy = dxyz * ne.evaluate("sum(rho * y)")
            xc = Ix / modes[0][-1]
            yc = Iy / modes[0][-1]
            phi = ne.evaluate("arctan2(y - yc, x - xc)")


            # phi = ne.evaluate("arctan2(y, x)")

            xcs.append(xc)
            ycs.append(yc)

            # Extract modes
            times.append(self.get_time_for_it(it, "overall", d1d2d3prof="d2"))
            for m in range(1, mmax + 1):
                # modes[m].append(dxyz * ne.evaluate("sum(rho * w_lorentz * vol * exp(-1j * m * phi))"))
                modes[m].append(dxyz * ne.evaluate("sum(rho * exp(-1j * m * phi))"))

        return times, iterations, xcs, ycs, modes

    def density_with_radis(self):

        v_n = "rho"
        rl = 0
        it = self.iterations[0]
        #
        x = self.get_grid_v_n_rl(it, "xy", rl, "x")
        y = self.get_grid_v_n_rl(it, "xy", rl, "y")
        r = np.sqrt(x**2 + y**2)
        rho = self.get_data_rl(it, "xy", rl, "rho")