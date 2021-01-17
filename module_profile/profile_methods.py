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

from scipy import interpolate

def get_time_for_it(it, list_times, list_iterations):
    # assert not (it > list_iterations[-1])
    # assert not (it < list_iterations[0])
    return interpolate.interp1d(list_iterations, list_times, kind="linear", fill_value="extrapolate")(it)

def get_it_for_time(time, list_times, list_iterations):
    assert not (time > list_times[-1])
    assert not (time < list_times[0])
    return interpolate.interp1d(list_times, list_iterations, kind="linear", fill_value="extrapolate")(time)


class LOAD_PROFILE:

    """
        Loads module_profile.h5 and extract grid object using scidata
    """

    def __init__(self, flist, itlist, timesteplist, symmetry=None):

        assert len(flist) == len(itlist)
        assert len(flist) == len(timesteplist)

        self.enforce_xy_grid = False

        self.symmetry = symmetry

        self.list_files = flist

        self.list_iterations = list(itlist)

        self.list_times = timesteplist

        self.list_prof_v_ns = [
                             "rho", "w_lorentz", "vol",  # basic
                             "press", "entr", "eps", "lapse",    # basic + lapse
                             "velx", "vely", "velz",     # velocities
                             "gxx", "gxy", "gxz", "gyy", "gyz", "gzz",  # metric
                             "betax", "betay", "betaz",  # shift components
                             'temp', 'Ye']

        self.list_grid_v_ns = ["x", "y", "z", "delta", "extent", "origin"]

        # self.nlevels = 7
        self.set_max_nlevels = 7 # Note that setting to 8 creats an error!
        self.list_nlevels = [0 for it in range(len(self.list_iterations))]

        # storage

        self.dfile_matrix = [0 for it in range(len(self.list_iterations))]

        self.grid_matrix = [0 for it in range(len(self.list_iterations))]

        self.grid_data_matrix = [[[np.zeros(0,)
                                  for v_n in range(len(self.list_grid_v_ns))]
                                  for rl in range(self.set_max_nlevels)]
                                  for it in range(len(self.list_iterations))]

    def update_storage_lists(self, new_iterations=np.zeros(0,), new_times=np.zeros(0,)):
        """
        In case iteration/times are updated -- call
        :return:
        """
        if len(new_iterations) > 0 or len(new_times) > 0:
            assert len(new_iterations) == len(new_times)
            self.list_iterations = list(new_iterations)
            self.list_times = np.array(new_times)
        #
        self.dfile_matrix = [0 for it in range(len(self.list_iterations))]
        self.grid_matrix = [0 for it in range(len(self.list_iterations))]
        self.grid_data_matrix = [[[np.zeros(0,)
                                  for v_n in range(len(self.list_grid_v_ns))]
                                  for rl in range(self.set_max_nlevels)]
                                  for it in range(len(self.list_iterations))]

    def check_prof_v_n(self, v_n):
        if not v_n in self.list_prof_v_ns:
            raise NameError("v_n:{} not in list of module_profile v_ns:{}"
                            .format(v_n, self.list_prof_v_ns))

    def check_it(self, it):
        if not int(it) in self.list_iterations:
            raise NameError("it:{} not in list of iterations:{}"
                            .format(it, self.list_iterations))

    def i_it(self, it):
        return int(self.list_iterations.index(it))

    def check_grid_v_n(self, v_n):
        if not v_n in self.list_grid_v_ns:
            raise NameError("v_n:{} not in list_grid_v_ns"
                            .format(v_n, self.list_grid_v_ns))

    def i_grid_v_n(self, v_n):
        return int(self.list_grid_v_ns.index(v_n))

    # ---

    def load_dfile(self, it):

        idx = self.list_iterations.index(it)
        fname = self.list_files[idx]
        if not os.path.isfile(fname):
            raise IOError("Expected file:{} NOT found".format(fname))
        try:
            dfile = h5py.File(fname, "r")
        except IOError:
            raise IOError("Cannot open file: {}".format(fname))
        reflevels = 0
        for key in dfile.keys():
            if key.__contains__("reflevel="):
                reflevels += 1
        # print("it:{} len(dfile.keys():{} dfile.keys():{} | {}".format(it, len(dfile.keys()), dfile.keys(), reflevels))

        if reflevels > self.set_max_nlevels:
            print ("Warning. Profile {} contains {} reflevels. Rejecting those above {}"
                   .format(it, reflevels, self.set_max_nlevels))

        self.list_nlevels[self.i_it(it)] = reflevels if reflevels < self.set_max_nlevels else self.set_max_nlevels
        self.dfile_matrix[self.i_it(it)] = dfile

    def is_dfile_loaded(self, it):
        if isinstance(self.dfile_matrix[self.i_it(it)], int): # dfile! not grid_matrix
            self.load_dfile(it)

    def get_profile_dfile(self, it):
        self.check_it(it)
        self.is_dfile_loaded(it)
        return self.dfile_matrix[self.i_it(it)]

        # self.symmetry = symmetry
        # self.nlevels = 7
        # self.module_profile = fname
        # self.dfile = h5py.File(fname, "r")
        # group_0 = self.dfile["reflevel={}".format(0)]
        # self.time = group_0.attrs["time"] * 0.004925794970773136 * 1e-3 # [sec]
        # self.iteration = group_0.attrs["iteration"]
        # print("\t\t symmetry: {}".format(self.symmetry))
        # print("\t\t time: {}".format(self.time))
        # print("\t\t iteration: {}".format(self.iteration))
        # self.grid = self.read_carpet_grid(self.dfile)
        #
        # # print("grid: {}".format(self.grid))
        #
        #
        #
        # if self.symmetry == "pi" and not str(self.module_profile).__contains__("_PI"):
        #     raise NameError("module_profile {} does not seem to have a pi symmetry. Check"
        #                     .format(self.module_profile))

    def get_nlevels(self, it):
        self.check_it(it)
        self.is_dfile_loaded(it)
        return int(self.list_nlevels[self.i_it(it)])

    # ---

    def get_group(self, it, rl):
        self.check_it(it)
        dfile = self.get_profile_dfile(it)
        return dfile["reflevel={}".format(int(rl))]

    def get_prof_time(self, it):
        group0 = self.get_group(it, 0)
        time = group0.attrs["time"] * 0.004925794970773136 * 1e-3  # [sec]
        return time

    # ---

    def read_carpet_grid(self, it):
        import scidata.carpet.grid as grid
        L = []
        dfile = self.get_profile_dfile(it)

        nlevels = self.get_nlevels(it)
        if self.enforce_xy_grid:
            for il in range(nlevels):
                gname = "reflevel={}".format(il)
                group = dfile[gname]
                level = grid.basegrid()
                level.delta = np.array(group.attrs["delta"])[:-1]
                # print(level.delta); exit(1)
                # print("delta: {} ".format(np.array(group.attrs["delta"]))); exit(1)
                level.dim = 2

                level.time = group.attrs["time"]
                # level.timestep = group.attrs["timestep"]
                level.directions = range(2)
                level.iorigin = np.array([0, 0], dtype=np.int32)

                # print("origin {} ".format(np.array(group.attrs["extent"][0::2])))
                if self.symmetry == 'pi':
                    origin = np.array(group.attrs["extent"][0::2])
                    origin[0] = origin[1] # x = y extend
                elif self.symmetry is None:
                    origin = np.array(group.attrs["extent"][0::2])
                    # print(origin)
                else:
                    raise NameError("symmetry is not recognized in a parfile. Set None or pi. Given:{}"
                                    .format(self.symmetry))
                level.origin = origin[:-1] # [-1044. -1044.   -20.]
                # print("sym: {} origin {} ".format(self.symmetry, origin)); exit()

                # level.n = np.array(group["rho"].shape, dtype=np.int32)
                level.n = np.array(self.get_prof_arr(it, il, 'rho').shape, dtype=np.int32)

                level.rlevel = il
                L.append(level)
        else:
            for il in range(nlevels):
                gname = "reflevel={}".format(il)
                group = dfile[gname]
                level = grid.basegrid()
                level.delta = np.array(group.attrs["delta"])
                # print("delta: {} ".format(np.array(group.attrs["delta"]))); exit(1)
                level.dim = 3
                level.time = group.attrs["time"]
                # level.timestep = group.attrs["timestep"]
                level.directions = range(3)
                level.iorigin = np.array([0, 0, 0], dtype=np.int32)

                # print("origin {} ".format(np.array(group.attrs["extent"][0::2])))
                if self.symmetry == 'pi':
                    origin = np.array(group.attrs["extent"][0::2])
                    origin[0] = origin[1] # x = y extend
                elif self.symmetry is None:
                    origin = np.array(group.attrs["extent"][0::2])
                else:
                    raise NameError("symmetry is not recognized in a parfile. Set None or pi. Given:{}"
                                    .format(self.symmetry))
                level.origin = origin
                # print("sym: {} origin {} ".format(self.symmetry, origin)); exit()

                # level.n = np.array(group["rho"].shape, dtype=np.int32)
                level.n = np.array(self.get_prof_arr(it, il, 'rho').shape, dtype=np.int32)
                level.rlevel = il
                L.append(level)

        self.grid_matrix[self.i_it(it)] = \
            grid.grid(sorted(L, key=lambda x: x.rlevel))

    def is_grid_extracted(self, it):
        if isinstance(self.grid_matrix[self.i_it(it)], int):
            self.read_carpet_grid(it)

    def get_grid(self, it):
        self.check_it(it)
        self.is_grid_extracted(it)
        return self.grid_matrix[self.i_it(it)]

    # ---

    def extract_prof_grid_data(self, it, rl):
        if self.enforce_xy_grid:
            grid = self.get_grid(it)
            x, y = grid.mesh()[rl]
            delta = grid[rl].delta
            extent = self.get_group(it, rl).attrs["extent"]
            origin = grid[rl].origin
            self.grid_data_matrix[self.i_it(it)][rl][self.i_grid_v_n("x")] = x
            self.grid_data_matrix[self.i_it(it)][rl][self.i_grid_v_n("y")] = y
            self.grid_data_matrix[self.i_it(it)][rl][self.i_grid_v_n("delta")] = delta
            self.grid_data_matrix[self.i_it(it)][rl][self.i_grid_v_n("extent")] = extent
            self.grid_data_matrix[self.i_it(it)][rl][self.i_grid_v_n("origin")] = origin
        else:
            grid = self.get_grid(it)
            x, y, z = grid.mesh()[rl]
            delta = grid[rl].delta
            extent = self.get_group(it, rl).attrs["extent"]
            origin = grid[rl].origin
            self.grid_data_matrix[self.i_it(it)][rl][self.i_grid_v_n("x")] = x
            self.grid_data_matrix[self.i_it(it)][rl][self.i_grid_v_n("y")] = y
            self.grid_data_matrix[self.i_it(it)][rl][self.i_grid_v_n("z")] = z
            self.grid_data_matrix[self.i_it(it)][rl][self.i_grid_v_n("delta")] = delta
            self.grid_data_matrix[self.i_it(it)][rl][self.i_grid_v_n("extent")] = extent
            self.grid_data_matrix[self.i_it(it)][rl][self.i_grid_v_n("origin")] = origin

    def is_grid_data_extracted(self, it, rl):
        if len(self.grid_data_matrix[self.i_it(it)][rl][self.i_grid_v_n("x")]) == 0:
            self.extract_prof_grid_data(it, rl)

    def get_grid_data(self, it, rl, v_n):
        self.check_it(it)
        self.check_grid_v_n(v_n)
        self.is_grid_data_extracted(it, rl)
        return self.grid_data_matrix[self.i_it(it)][rl][self.i_grid_v_n(v_n)]

    # ---

    def get_prof_arr(self, it, rl, v_n):
        self.check_it(it)
        self.check_prof_v_n(v_n)

        group = self.get_group(it, rl)# self.dfile["reflevel={}".format(rl)]

        try:
            if self.enforce_xy_grid:
                arr = np.array(group[v_n])[:, :, 0]
                if self.symmetry == 'pi':
                    # print("rl: {} x:({}):[{:.1f},{:.1f}] y:({}):[{:.1f},{:.1f}] z:({}):[{:.1f},{:.1f}]"
                    #       .format(rl, arr.shape, arr[0, 0, 0], arr[-1, 0, 0],
                    #               arr.shape, arr[0, 0, 0], arr[0, -1, 0],
                    #               arr.shape, arr[0, 0, 0], arr[0, 0, -1]))

                    ### removing ghosts x[-2] x[-1] | x[0] x[1] x[2], to attach the x[-1] ... x[2] x[1] there
                    arr = np.delete(arr, 0, axis=0)
                    arr = np.delete(arr, 0, axis=0)
                    arr = np.delete(arr, 0, axis=0)

                    ## flipping the array  to get the following: Consider for section of xy plane:
                    ##   y>0  empy | [1]            y>0   [2][::-1] | [1]
                    ##   y<0  empy | [2]     ->     y<0   [1][::-1] | [2]
                    ##        x<0    x>0                       x<0    x>0
                    ## This fills the grid from -x[-1] to x[-1], reproduing Pi symmetry.
                    arr_n = arr[::-1, ::-1]
                    arr = np.concatenate((arr_n, arr), axis=0)

                    # print("rl: {} x:({}):[{:.1f},{:.1f}] y:({}):[{:.1f},{:.1f}] z:({}):[{:.1f},{:.1f}]"
                    #       .format(rl, arr.shape, arr[0, 0, 0], arr[-1, 0, 0],
                    #               arr.shape, arr[0, 0, 0], arr[0, -1, 0],
                    #               arr.shape, arr[0, 0, 0], arr[0, 0, -1]))
            else:
                arr = np.array(group[v_n])
                if self.symmetry == 'pi':

                    # print("rl: {} x:({}):[{:.1f},{:.1f}] y:({}):[{:.1f},{:.1f}] z:({}):[{:.1f},{:.1f}]"
                    #       .format(rl, arr.shape, arr[0, 0, 0], arr[-1, 0, 0],
                    #               arr.shape, arr[0, 0, 0], arr[0, -1, 0],
                    #               arr.shape, arr[0, 0, 0], arr[0, 0, -1]))

                    ### removing ghosts x[-2] x[-1] | x[0] x[1] x[2], to attach the x[-1] ... x[2] x[1] there
                    arr = np.delete(arr, 0, axis=0)
                    arr = np.delete(arr, 0, axis=0)
                    arr = np.delete(arr, 0, axis=0)

                    ## flipping the array  to get the following: Consider for section of xy plane:
                    ##   y>0  empy | [1]            y>0   [2][::-1] | [1]
                    ##   y<0  empy | [2]     ->     y<0   [1][::-1] | [2]
                    ##        x<0    x>0                       x<0    x>0
                    ## This fills the grid from -x[-1] to x[-1], reproduing Pi symmetry.
                    arr_n = arr[::-1, ::-1, :]
                    arr = np.concatenate((arr_n, arr), axis=0)

                    # print("rl: {} x:({}):[{:.1f},{:.1f}] y:({}):[{:.1f},{:.1f}] z:({}):[{:.1f},{:.1f}]"
                    #       .format(rl, arr.shape, arr[0, 0, 0], arr[-1, 0, 0],
                    #               arr.shape, arr[0, 0, 0], arr[0, -1, 0],
                    #               arr.shape, arr[0, 0, 0], arr[0, 0, -1]))
        except:
            print('\nAvailable Parameters:')
            print(list(v_n_aval for v_n_aval in group))
            print('\n')
            raise ValueError('Error extracting v_n:{} from module_profile for it:{} rl:{}'.format(v_n, it, rl))
        return arr

    # def __delete__(self, instance):
    #
    #     instance.dfile_matrix = [0
    #                               for it in range(len(self.list_iterations))]
    #     instance.grid_matrix = [0
    #                               for it in range(len(self.list_iterations))]
    #     instance.grid_data_matrix = [[[np.zeros(0,)
    #                               for v_n in range(len(self.list_grid_v_ns))]
    #                               for rl in range(7)]
    #                               for it in range(len(self.list_iterations))]


class COMPUTE_STORE(LOAD_PROFILE):

    def __init__(self, flist, itlist, timesteplist, symmetry=None):

        LOAD_PROFILE.__init__(self, flist=flist, itlist=itlist, timesteplist=timesteplist, symmetry=symmetry)

        self.list_comp_v_ns = [
            "density", "vup", "metric", "shift",
            "enthalpy", "shvel", "u_0", "hu_0",
            "vlow", "vphi", "vr",
            "dens_unb_geo", "dens_unb_bern", "dens_unb_garch",
            "ang_mom", "ang_mom_flux",
            "theta", "r", "phi" # assumes cylindircal coordinates. r = x^2 + y^2
        ]

        self.list_all_v_ns = self.list_prof_v_ns + \
                             self.list_grid_v_ns + \
                             self.list_comp_v_ns

        self.data_matrix = [[[np.zeros(0,)
                             for y in range(len(self.list_all_v_ns))]
                             for x in range(self.set_max_nlevels)]
                             for i in range(len(self.list_iterations))]

    def check_v_n(self, v_n):
        if v_n not in self.list_all_v_ns:
            raise NameError("v_n:{} not in the v_n list \n{}"
                            .format(v_n, self.list_all_v_ns))

    def i_v_n(self, v_n):
        self.check_v_n(v_n)
        return int(self.list_all_v_ns.index(v_n))

    def set_data(self, it, rl, v_n, arr):
        self.data_matrix[self.i_it(it)][rl][self.i_v_n(v_n)] = arr

    def extract_data(self, it, rl, v_n):
        data = self.get_prof_arr(it, rl, v_n)
        self.data_matrix[self.i_it(it)][rl][self.i_v_n(v_n)] = data

    def extract_grid_data(self, it, rl, v_n):
        if v_n in ["x", "y", "z"]:
            self.data_matrix[self.i_it(it)][rl][self.i_v_n("x")] = self.get_grid_data(it, rl, "x")
            self.data_matrix[self.i_it(it)][rl][self.i_v_n("y")] = self.get_grid_data(it, rl, "y")
            self.data_matrix[self.i_it(it)][rl][self.i_v_n("z")] = self.get_grid_data(it, rl, "z")
        elif v_n == "delta":
            self.data_matrix[self.i_it(it)][rl][self.i_v_n("delta")] = self.get_grid_data(it, rl, "delta")
        else:
            raise NameError("Grid variable {} not recognized".format(v_n))

    # --- #

    def compute_data(self, it, rl, v_n):

        if v_n == 'density':
            arr = FORMULAS.density(self.get_comp_data(it, rl, "rho"),
                                   self.get_comp_data(it, rl, "w_lorentz"),
                                   self.get_comp_data(it, rl, "vol"))

        elif v_n == 'vup':
            arr = FORMULAS.vup(self.get_comp_data(it, rl, "velx"),
                               self.get_comp_data(it, rl, "vely"),
                               self.get_comp_data(it, rl, "velz"))

        elif v_n == 'metric':  # gxx, gxy, gxz, gyy, gyz, gzz
            arr = FORMULAS.metric(self.get_comp_data(it, rl, "gxx"),
                                  self.get_comp_data(it, rl, "gxy"),
                                  self.get_comp_data(it, rl, "gxz"),
                                  self.get_comp_data(it, rl, "gyy"),
                                  self.get_comp_data(it, rl, "gyz"),
                                  self.get_comp_data(it, rl, "gzz"))

        elif v_n == 'shift':
            arr = FORMULAS.shift(self.get_comp_data(it, rl, "betax"),
                                 self.get_comp_data(it, rl, "betay"),
                                 self.get_comp_data(it, rl, "betaz"))

        elif v_n == 'enthalpy':
            arr = FORMULAS.enthalpy(self.get_comp_data(it, rl, "eps"),
                                    self.get_comp_data(it, rl, "press"),
                                    self.get_comp_data(it, rl, "rho"))

        elif v_n == 'shvel':
            arr = FORMULAS.shvel(self.get_comp_data(it, rl, "shift"),
                                 self.get_comp_data(it, rl, "vlow"))

        elif v_n == 'u_0':
            arr = FORMULAS.u_0(self.get_comp_data(it, rl, "w_lorentz"),
                               self.get_comp_data(it, rl, "shvel"),  # not input
                               self.get_comp_data(it, rl, "lapse"))

        elif v_n == 'hu_0':
            arr = FORMULAS.hu_0(self.get_comp_data(it, rl, "enthalpy"),
                                self.get_comp_data(it, rl, "u_0"))

        elif v_n == 'vlow':
            arr = FORMULAS.vlow(self.get_comp_data(it, rl, "metric"),
                                self.get_comp_data(it, rl, "vup"))

        elif v_n == 'vphi':
            arr = FORMULAS.vphi(self.get_comp_data(it, rl, "x"),
                                self.get_comp_data(it, rl, "y"),
                                self.get_comp_data(it, rl, "vlow"))

        elif v_n == 'vr':
            arr = FORMULAS.vr(self.get_comp_data(it, rl, "x"),
                              self.get_comp_data(it, rl, "y"),
                              self.get_comp_data(it, rl, "r"),
                              self.get_comp_data(it, rl, "vup"))

        elif v_n == "r":
            arr = FORMULAS.r(self.get_comp_data(it, rl, "x"),
                             self.get_comp_data(it, rl, "y"))

        elif v_n == "phi":
            arr = FORMULAS.phi(self.get_comp_data(it, rl, "x"),
                               self.get_comp_data(it, rl, "y"))

        elif v_n == 'theta':
            arr = FORMULAS.theta(self.get_comp_data(it, rl, "r"),
                                 self.get_comp_data(it, rl, "z"))

        elif v_n == 'ang_mom':
            arr = FORMULAS.ang_mom(self.get_comp_data(it, rl, "rho"),
                                   self.get_comp_data(it, rl, "eps"),
                                   self.get_comp_data(it, rl, "press"),
                                   self.get_comp_data(it, rl, "w_lorentz"),
                                   self.get_comp_data(it, rl, "vol"),
                                   self.get_comp_data(it, rl, "vphi"))

        elif v_n == 'ang_mom_flux':
            arr = FORMULAS.ang_mom_flux(self.get_comp_data(it, rl, "ang_mom"),
                                        self.get_comp_data(it, rl, "lapse"),
                                        self.get_comp_data(it, rl, "vr"))

        elif v_n == 'dens_unb_geo':
            arr = FORMULAS.dens_unb_geo(self.get_comp_data(it, rl, "u_0"),
                                        self.get_comp_data(it, rl, "rho"),
                                        self.get_comp_data(it, rl, "w_lorentz"),
                                        self.get_comp_data(it, rl, "vol"))

        elif v_n == 'dens_unb_bern':
            arr = FORMULAS.dens_unb_bern(self.get_comp_data(it, rl, "enthalpy"),
                                         self.get_comp_data(it, rl, "u_0"),
                                         self.get_comp_data(it, rl, "rho"),
                                         self.get_comp_data(it, rl, "w_lorentz"),
                                         self.get_comp_data(it, rl, "vol"))

        elif v_n == 'dens_unb_garch':
            arr = FORMULAS.dens_unb_garch(self.get_comp_data(it, rl, "enthalpy"),
                                          self.get_comp_data(it, rl, "u_0"),
                                          self.get_comp_data(it, rl, "lapse"),
                                          self.get_comp_data(it, rl, "press"),
                                          self.get_comp_data(it, rl, "rho"),
                                          self.get_comp_data(it, rl, "w_lorentz"),
                                          self.get_comp_data(it, rl, "vol"))

        else:
            raise NameError("No method found for v_n:{} rl:{} it:{} Add entry to 'compute()'"
                            .format(v_n, rl, it))

        self.data_matrix[self.i_it(it)][rl][self.i_v_n(v_n)] = arr

    # --- #

    def is_available(self, it, rl, v_n):
        self.check_it(it)
        self.check_v_n(v_n)
        data = self.data_matrix[self.i_it(it)][rl][self.i_v_n(v_n)]
        if len(data) == 0:
            if v_n in self.list_prof_v_ns:
                self.extract_data(it, rl, v_n)
            elif v_n in self.list_grid_v_ns:
                self.extract_grid_data(it, rl, v_n)
            elif v_n in self.list_comp_v_ns:
                self.compute_data(it, rl, v_n)
            else:
                raise NameError("v_n is not recognized: '{}' [COMPUTE STORE]".format(v_n))

    def get_comp_data(self, it, rl, v_n):
        self.check_it(it)
        self.check_v_n(v_n)
        self.is_available(it, rl, v_n)

        return self.data_matrix[self.i_it(it)][rl][self.i_v_n(v_n)]

    # def __delete__(self, instance):
    #     instance.dfile.close()
    #     instance.data_matrix = [[np.zeros(0, )
    #                          for x in range(self.nlevels)]
    #                         for y in range(len(self.list_all_v_ns))]


class MASK_STORE(COMPUTE_STORE):

    disk_mask_setup = {'rm_rl': True,  # REMOVE previouse ref. level from the next
                       'rho': [6.e4 / 6.176e+17, 1.e13 / 6.176e+17],  # REMOVE atmo and NS
                       'lapse': [0.15, 1.]}  # remove apparent horizon

    remnant_mask_setup = {'rm_rl': True,
                          'rho': [1.e13 / 6.176e+17, 1.e30],
                          'lapse': [0.15, 1.]}

    def __init__(self, flist, itlist, timesteplist, symmetry=None):
        COMPUTE_STORE.__init__(self, flist=flist, itlist=itlist, timesteplist=timesteplist, symmetry=symmetry)

        # self.mask_setup = {'rm_rl': True,  # REMOVE previouse ref. level from the next
        #                    'rho': [6.e4 / 6.176e+17, 1.e13 / 6.176e+17],  # REMOVE atmo and NS
        #                    'lapse': [0.15, 1.]} # remove apparent horizon

        # self.disk_mask_setup = {'rm_rl': True,  # REMOVE previouse ref. level from the next
        #                    'rho': [6.e4 / 6.176e+17, 1.e13 / 6.176e+17],  # REMOVE atmo and NS
        #                    'lapse': [0.15, 1.]} # remove apparent horizon

        self.list_mask_names = ["disk", "remnant", "rl_xy", "rl_xz", "rl"]

        self.mask_matrix = [[[np.ones(0, dtype=bool)
                              for i in range(len(self.list_mask_names))]
                              for x in range(self.set_max_nlevels)]
                              for y in range(len(self.list_iterations))]

        self._list_mask_v_n = ["x", "y", "z"]

    def i_mask_v_n(self, v_n):
        return int(self.list_mask_names.index(v_n))

    def check_mask_name(self, v_n):
        if not v_n in self.list_mask_names:
            raise NameError("mask name:{} is not recognized. \nAvailable: {}"
                            .format(v_n, self.list_mask_names))

    # ---

    def compute_mask(self, it, name="disk"):

        if name == "rl":
            #
            nlevels = self.get_nlevels(it)
            mask_setup = self.disk_mask_setup
            nlevelist = np.arange(nlevels, 0, -1) - 1
            x = []
            y = []
            z = []
            for ii, rl in enumerate(nlevelist):
                x.append(self.get_grid_data(it, rl, "x")[3:-3, 3:-3, 3:-3])
                y.append(self.get_grid_data(it, rl, "y")[3:-3, 3:-3, 3:-3])
                z.append(self.get_grid_data(it, rl, "z")[3:-3, 3:-3, 3:-3])
                mask = np.ones(x[ii].shape, dtype=bool)
                if ii > 0 and mask_setup["rm_rl"]:
                    x_ = (x[ii][:, :, :] <= x[ii - 1][:, 0, 0].max()) & (
                            x[ii][:, :, :] >= x[ii - 1][:, 0, 0].min())
                    y_ = (y[ii][:, :, :] <= y[ii - 1][0, :, 0].max()) & (
                            y[ii][:, :, :] >= y[ii - 1][0, :, 0].min())
                    z_ = (z[ii][:, :, :] <= z[ii - 1][0, 0, :].max()) & (
                            z[ii][:, :, :] >= z[ii - 1][0, 0, :].min())
                    mask = mask & np.invert((x_ & y_ & z_))

                self.mask_matrix[self.i_it(it)][rl][self.i_mask_v_n(name)] = mask
        elif name == "rl_xy":
            nlevels = self.get_nlevels(it)
            nlevelist = np.arange(nlevels, 0, -1) - 1
            x = []
            y = []
            for ii, rl in enumerate(nlevelist):
                __z = self.get_grid_data(it, rl, "z")
                iz0 = np.argmin(np.abs(__z[0, 0, :]))
                # print( abs(__z[0, 0, iz0]))
                # assert abs(__z[0, 0, iz0]) < 1e-10
                x.append(self.get_grid_data(it, rl, "x")[3:-3, 3:-3, iz0])
                y.append(self.get_grid_data(it, rl, "y")[3:-3, 3:-3, iz0])
                mask = np.ones(x[ii].shape, dtype=bool)
                if ii > 0:
                    x_ = (x[ii][:, :] <= x[ii - 1][:, 0].max()) & (x[ii][:, :] >= x[ii - 1][:, 0].min())
                    y_ = (y[ii][:, :] <= y[ii - 1][0, :].max()) & (y[ii][:, :] >= y[ii - 1][0, :].min())
                    mask = mask & np.invert((x_ & y_))
                #
                self.mask_matrix[self.i_it(it)][rl][self.i_mask_v_n(name)] = mask
        # elif name == "rl_xz":

        elif name == "rl_xz":
            nlevels = self.get_nlevels(it)
            nlevelist = np.arange(nlevels, 0, -1) - 1
            x = []
            z = []
            for ii, rl in enumerate(nlevelist):
                x.append(self.get_grid_data(it, rl, "x")[3:-3, 3:-3, 3:-3])
                __y = self.get_grid_data(it, rl, "y")[3:-3, 3:-3, 3:-3]
                z.append(self.get_grid_data(it, rl, "z")[3:-3, 3:-3, 3:-3])

                mask = np.ones(x[ii][:, 0, :].shape, dtype=bool)

                if ii > 0:
                    # if y=0 slice is right at the 0 ->
                    iy0 = np.argmin(np.abs(__y[0, :, 0]))
                    if abs(__y[0, iy0, 0]) < 1e-15:
                        x_ = (x[ii][:, iy0, :] <= x[ii - 1][:, iy0, 0].max()) & (x[ii][:, iy0, :] >= x[ii - 1][:, iy0, 0].min())
                        z_ = (z[ii][:, iy0, :] <= z[ii - 1][0, iy0, :].max()) & (z[ii][:, iy0, :] >= z[ii - 1][0, iy0, :].min())
                        mask = mask & np.invert((x_ & z_))
                    else:
                        # if y = 0 Does not exists, only y = -0.1 and y 0.1 ->
                        if __y[0, iy0, 0] > 0:
                            iy0 -= 1
                        x_ = (x[ii][:, iy0, :] <= x[ii - 1][:, iy0, 0].max()) & (x[ii][:, iy0, :] >= x[ii - 1][:, iy0, 0].min())
                        z_ = (z[ii][:, iy0, :] <= z[ii - 1][0, iy0, :].max()) & (z[ii][:, iy0, :] >= z[ii - 1][0, iy0, :].min())
                        mask_ = np.invert((x_ & z_))
                        #
                        iy0 = iy0 + 1
                        #
                        x_ = (x[ii][:, iy0, :] <= x[ii - 1][:, iy0, 0].max()) & (x[ii][:, iy0, :] >= x[ii - 1][:, iy0, 0].min())
                        z_ = (z[ii][:, iy0, :] <= z[ii - 1][0, iy0, :].max()) & (z[ii][:, iy0, :] >= z[ii - 1][0, iy0, :].min())
                        mask__ = np.invert((x_ & z_))
                        #
                        mask = mask & mask_ #& mask__
                self.mask_matrix[self.i_it(it)][rl][self.i_mask_v_n(name)] = mask

                # print(abs(__y[0, iy0, 0]))
                # assert abs(__y[0, iy0, 0]) < 1e-10
                # x.append(self.get_grid_data(it, rl, "x")[3:-3, iy0, 3:-3])
                # z.append(self.get_grid_data(it, rl, "z")[3:-3, iy0, 3:-3])
                # mask = np.ones(x[ii].shape, dtype=bool)
                # if ii > 0:
                #     x_ = (x[ii][:, :] <= x[ii - 1][:, 0].max()) & (x[ii][:, :] >= x[ii - 1][:, 0].min())
                #     z_ = (z[ii][:, :] <= z[ii - 1][0, :].max()) & (z[ii][:, :] >= z[ii - 1][0, :].min())
                #     mask = mask & np.invert((x_ & z_))
                # #
                # self.mask_matrix[self.i_it(it)][rl][self.i_mask_v_n(name)] = mask
        elif name == "disk":
            #
            mask_setup = self.disk_mask_setup
            nlevels = self.get_nlevels(it)
            nlevelist = np.arange(nlevels, 0, -1) - 1

            x = []
            y = []
            z = []
            for ii, rl in enumerate(nlevelist):
                x.append(self.get_grid_data(it, rl, "x")[3:-3, 3:-3, 3:-3])
                y.append(self.get_grid_data(it, rl, "y")[3:-3, 3:-3, 3:-3])
                z.append(self.get_grid_data(it, rl, "z")[3:-3, 3:-3, 3:-3])
                mask = np.ones(x[ii].shape, dtype=bool)
                if ii > 0 and mask_setup["rm_rl"]:
                    x_ = (x[ii][:, :, :] <= x[ii - 1][:, 0, 0].max()) & (x[ii][:, :, :] >= x[ii - 1][:, 0, 0].min())
                    y_ = (y[ii][:, :, :] <= y[ii - 1][0, :, 0].max()) & (y[ii][:, :, :] >= y[ii - 1][0, :, 0].min())
                    z_ = (z[ii][:, :, :] <= z[ii - 1][0, 0, :].max()) & (z[ii][:, :, :] >= z[ii - 1][0, 0, :].min())
                    mask = mask & np.invert((x_ & y_ & z_))

                for v_n in mask_setup.keys()[1:]:
                    self.check_v_n(v_n)
                    if len(mask_setup[v_n]) != 2:
                        raise NameError("Error. 2 values are required to set a limit. Give {} for {}"
                                        .format(mask_setup[v_n], v_n))
                    arr_1 = self.get_comp_data(it, rl, v_n)[3:-3, 3:-3, 3:-3]
                    min_val = float(mask_setup[v_n][0])
                    max_val = float(mask_setup[v_n][1])
                    if isinstance(min_val, str):
                        if min_val == "min": min_val = arr_1.min()
                        elif min_val == "max": min_val = arr_1.max()
                        else:
                            raise NameError("unrecognized min_val:{} for mask:{}"
                                            .format(min_val, name))
                    else:
                        min_val = float(mask_setup[v_n][0])
                    #
                    if isinstance(max_val, str):
                        if max_val == "min": max_val = arr_1.min()
                        elif max_val == "max": max_val = arr_1.max()
                        else:
                            raise NameError("unrecognized max_val:{} for mask:{}"
                                            .format(max_val, name))
                    else:
                        max_val = float(mask_setup[v_n][1])
                    mask_i = (arr_1 > min_val) & (arr_1 < max_val)
                    mask = mask & mask_i
                    del arr_1
                    del mask_i

                self.mask_matrix[self.i_it(it)][rl][self.i_mask_v_n(name)] = mask


        elif name == "remnant":
            #
            mask_setup = self.remnant_mask_setup
            nlevels = self.get_nlevels(it)
            nlevelist = np.arange(nlevels, 0, -1) - 1
            x = []
            y = []
            z = []
            for ii, rl in enumerate(nlevelist):
                x.append(self.get_grid_data(it, rl, "x")[3:-3, 3:-3, 3:-3])
                y.append(self.get_grid_data(it, rl, "y")[3:-3, 3:-3, 3:-3])
                z.append(self.get_grid_data(it, rl, "z")[3:-3, 3:-3, 3:-3])
                mask = np.ones(x[ii].shape, dtype=bool)
                if ii > 0 and mask_setup["rm_rl"]:
                    x_ = (x[ii][:, :, :] <= x[ii - 1][:, 0, 0].max()) & (
                            x[ii][:, :, :] >= x[ii - 1][:, 0, 0].min())
                    y_ = (y[ii][:, :, :] <= y[ii - 1][0, :, 0].max()) & (
                            y[ii][:, :, :] >= y[ii - 1][0, :, 0].min())
                    z_ = (z[ii][:, :, :] <= z[ii - 1][0, 0, :].max()) & (
                            z[ii][:, :, :] >= z[ii - 1][0, 0, :].min())
                    mask = mask & np.invert((x_ & y_ & z_))
                #
                for v_n in mask_setup.keys()[1:]:
                    self.check_v_n(v_n)
                    if len(mask_setup[v_n]) != 2:
                        raise NameError("Error. 2 values are required to set a limit. Give {} for {}"
                                        .format(mask_setup[v_n], v_n))
                    arr_1 = self.get_comp_data(it, rl, v_n)[3:-3, 3:-3, 3:-3]
                    min_val = mask_setup[v_n][0]
                    max_val = mask_setup[v_n][1]
                    if isinstance(min_val, str):
                        if min_val == "min": min_val = arr_1.min()
                        elif min_val == "max": min_val = arr_1.max()
                        else:
                            raise NameError("unrecognized min_val:{} for mask:{}"
                                            .format(min_val, name))
                    else:
                        min_val = float(mask_setup[v_n][0])
                    #
                    if isinstance(max_val, str):
                        if max_val == "min": max_val = arr_1.min()
                        elif max_val == "max": max_val = arr_1.max()
                        else:
                            raise NameError("unrecognized max_val:{} for mask:{}"
                                            .format(max_val, name))
                    else:
                        max_val = float(mask_setup[v_n][1])
                    #
                    mask_i = (arr_1 > min_val) & (arr_1 <= max_val)
                    mask = mask & mask_i
                    del arr_1
                    del mask_i
                #
                self.mask_matrix[self.i_it(it)][rl][self.i_mask_v_n(name)] = mask
        else:
            NameError("No method found to compute mask: {} ".format(name))

    # ---

    def is_mask_available(self, it, rl, v_n="disk"):
        mask = self.mask_matrix[self.i_it(it)][rl][self.i_mask_v_n(v_n)]
        if len(mask) == 0:
            self.compute_mask(it, v_n)

    def get_mask(self, it, rl, v_n="disk"):
        self.check_it(it)
        self.is_mask_available(it, rl, v_n)
        mask = self.mask_matrix[self.i_it(it)][rl][self.i_mask_v_n(v_n)]
        return mask

    def get_masked_data(self, it, rl, v_n, mask_v_n="disk"):
        self.check_v_n(v_n)
        self.check_it(it)
        self.check_mask_name(mask_v_n)
        self.is_available(it, rl, v_n)
        self.is_mask_available(it, rl, mask_v_n)
        data = np.array(self.get_comp_data(it, rl, v_n))[3:-3, 3:-3, 3:-3]
        mask = self.mask_matrix[self.i_it(it)][rl][self.i_mask_v_n(mask_v_n)]
        return data[mask]

    # def __delete__(self, instance):
    #     instance.dfile.close()
    #     instance.data_matrix = [[np.zeros(0, )
    #                              for x in range(self.nlevels)]
    #                              for y in range(len(self.list_all_v_ns))]
    #     instance.mask_matrix = [np.ones(0, dtype=bool) for x in range(self.nlevels)]


class MAINMETHODS_STORE(MASK_STORE):

    def __init__(self, flist, itlist, timesteplist, symmetry=None):

        MASK_STORE.__init__(self, flist=flist, itlist=itlist, timesteplist=timesteplist, symmetry=symmetry)

        # "v_n": "temp", "edges": np.array()

        # "v_n": "temp", "points: number, "scale": "log", (and "min":number, "max":number)

        rho_const = 6.176269145886162e+17

        self.corr_task_dic_hu_0_ang_mom = [
            {"v_n": "hu_0", "edges": np.linspace(-1.2, -0.8, 500)},
            {"v_n": "ang_mom", "points": 500, "scale": "log", "min":1e-9} # find min, max yourself
        ]

        self.corr_task_dic_hu_0_ang_mom_flux = [
            {"v_n": "hu_0", "edges": np.linspace(-1.2, -0.8, 500)},
            {"v_n": "ang_mom_flux", "points": 300, "scale": "log", "min":1e-12},  # not in CGS :^
        ]

        self.corr_task_dic_hu_0_ye = [
            {"v_n": "hu_0", "edges": np.linspace(-1.2, -0.8, 500)},
            {"v_n": "Ye", "edges": np.linspace(0, 0.5, 500)},  # not in CGS :^
        ]

        self.corr_task_dic_hu_0_temp = [
            {"v_n": "hu_0", "edges": np.linspace(-1.2, -0.8, 500)},
            {"v_n": "temp", "edges": 10.0 ** np.linspace(-2, 2, 300)},
        ]

        self.corr_task_dic_hu_0_entr = [
            {"v_n": "hu_0", "edges": np.linspace(-1.2, -0.8, 500)},
            {"v_n": "entr", "edges": np.linspace(0., 200., 500)}
        ]

        self.corr_task_dic_r_phi = [
            {"v_n": "r", "edges": np.linspace(0, 50, 500)},
            {"v_n": "phi", "edges": np.linspace(-np.pi, np.pi, 500)},
        ]

        self.corr_task_dic_r_ye = [
            # {"v_n": "rho",  "edges": 10.0 ** np.linspace(4.0, 16.0, 500) / rho_const},  # not in CGS :^
            {"v_n": "r", "edges": np.linspace(0, 100, 500)},
            {"v_n": "Ye", "edges": np.linspace(0, 0.5, 500)}
        ]

        self.corr_task_dic_rho_r = [
            {"v_n": "rho", "edges": 10.0 ** np.linspace(4.0, 13.0, 500) / rho_const},  # not in CGS :^
            {"v_n": "r", "edges": np.linspace(0, 100, 500)}
        ]

        self.corr_task_dic_rho_ye = [
            # {"v_n": "temp", "edges": 10.0 ** np.linspace(-2, 2, 300)},
            {"v_n": "rho",  "edges": 10.0 ** np.linspace(4.0, 13.0, 500) / rho_const},  # not in CGS :^
            {"v_n": "Ye",   "edges": np.linspace(0, 0.5, 500)}
        ]

        self.corr_task_dic_ye_entr = [
            # {"v_n": "temp", "edges": 10.0 ** np.linspace(-2, 2, 300)},
            {"v_n": "Ye", "edges": np.linspace(0, 0.5, 500)},
            {"v_n": "entr", "edges": np.linspace(0., 100., 500)}
        ]

        self.corr_task_dic_temp_ye = [
            # {"v_n": "rho",  "edges": 10.0 ** np.linspace(4.0, 16.0, 500) / rho_const},  # not in CGS :^
            {"v_n": "temp", "edges": 10.0 ** np.linspace(-2, 2, 300)},
            {"v_n": "Ye",   "edges": np.linspace(0, 0.5, 500)}
        ]

        self.corr_task_dic_velz_ye = [
            # {"v_n": "rho",  "edges": 10.0 ** np.linspace(4.0, 16.0, 500) / rho_const},  # not in CGS :^
            {"v_n": "velz", "edges": np.linspace(-1., 1., 500)},
            {"v_n": "Ye",   "edges": np.linspace(0, 0.5, 500)}
        ]

        self.corr_task_dic_rho_temp = [
            # {"v_n": "temp", "edges": 10.0 ** np.linspace(-2, 2, 300)},
            {"v_n": "rho", "edges": 10.0 ** np.linspace(4.0, 13.0, 500) / rho_const},  # not in CGS :^
            {"v_n": "temp", "edges": 10.0 ** np.linspace(-2, 2, 300)},
        ]

        self.corr_task_dic_rho_theta = [
            {"v_n": "rho", "edges": 10.0 ** np.linspace(4.0, 13.0, 500) / rho_const},  # not in CGS :^
            {"v_n": "theta", "edges": np.linspace(0, 0.5*np.pi, 500)}
        ]

        self.corr_task_dic_velz_theta = [
            {"v_n": "velz", "edges": np.linspace(-1., 1., 500)},  # not in CGS :^
            {"v_n": "theta", "edges": np.linspace(0, 0.5*np.pi, 500)}
        ]

        self.corr_task_dic_theta_dens_unb_bern = [
            {"v_n": "theta", "edges": np.linspace(0, 0.5 * np.pi, 500)},
            {"v_n": "dens_unb_bern", "edges": 10.0 ** np.linspace(-12., -6., 500)}  # not in CGS :^
        ]

        self.corr_task_dic_rho_ang_mom = [
            {"v_n": "rho", "edges": 10.0 ** np.linspace(4.0, 13.0, 500) / rho_const},  # not in CGS :^
            {"v_n": "ang_mom", "points": 500, "scale": "log", "min":1e-9} # find min, max yourself
        ]

        self.corr_task_dic_ye_dens_unb_bern = [
            {"v_n": "Ye",            "edges": np.linspace(0, 0.5, 500)}, #"edges": np.linspace(-1., 1., 500)},  # in c
            {"v_n": "dens_unb_bern", "edges": 10.0 ** np.linspace(-12., -6., 500)}
        ]

        self.corr_task_dic_rho_ang_mom_flux = [
            {"v_n": "rho", "edges": 10.0 ** np.linspace(4.0, 13.0, 500) / rho_const},  # not in CGS :^
            {"v_n": "ang_mom_flux", "points": 500, "scale": "log", "min":1e-12}
        ]

        self.corr_task_dic_rho_dens_unb_bern = [
            {"v_n": "rho", "edges": 10.0 ** np.linspace(4.0, 13.0, 500) / rho_const},  # not in CGS :^
            {"v_n": "dens_unb_bern", "edges": 10.0 ** np.linspace(-12., -6., 500)}
        ]

        self.corr_task_dic_velz_dens_unb_bern = [
            {"v_n": "velz", "points": 500, "scale": "linear"}, #"edges": np.linspace(-1., 1., 500)},  # in c
            {"v_n": "dens_unb_bern", "edges": 10.0 ** np.linspace(-12., -6., 500)}
        ]

        self.corr_task_dic_ang_mom_flux_theta = [
            {"v_n": "ang_mom_flux", "points": 300, "scale": "log", "min":1e-12},  # not in CGS :^
            {"v_n": "theta", "edges": np.linspace(0, 0.5*np.pi, 500)}
        ]

        self.corr_task_dic_ang_mom_flux_dens_unb_bern = [
            {"v_n": "ang_mom_flux", "points": 500, "scale": "log", "min":1e-12},  # not in CGS :^
            {"v_n": "dens_unb_bern", "edges": 10.0 ** np.linspace(-12., -6., 500)}
        ]

        self.corr_task_dic_inv_ang_mom_flux_dens_unb_bern = [
            {"v_n": "inv_ang_mom_flux", "points": 500, "scale": "log", "min":1e-12},  # not in CGS :^
            {"v_n": "dens_unb_bern", "edges": 10.0 ** np.linspace(-12., -6., 500)}
        ]

        # -- 3D

        self.corr_task_dic_r_phi_ang_mom_flux = [
            {"v_n": "r", "edges": np.linspace(0, 100, 50)},
            {"v_n": "phi", "edges": np.linspace(-np.pi, np.pi, 500)},
            {"v_n": "ang_mom_flux", "points": 500, "scale": "log", "min": 1e-12}
        ]

        # hist [d - disk, r - remnant

        self.hist_task_dic_entropy_d ={"v_n": "entr", "edges": np.linspace(0., 200., 500)}
        self.hist_task_dic_entropy_r = {"v_n": "entr", "edges": np.linspace(0., 25., 300)}
        self.hist_task_dic_r =      {"v_n": "r", "edges": np.linspace(0., 200., 500)}
        self.hist_task_dic_theta =  {"v_n": "theta", "edges": np.linspace(0., np.pi / 2., 500)}
        self.hist_task_dic_ye =     {"v_n": "Ye",   "edges": np.linspace(0., 0.5, 500)}
        self.hist_task_dic_temp =   {"v_n": "temp", "edges": 10.0 ** np.linspace(-2., 2., 300)}
        self.hist_task_dic_velz =   {"v_n": "velz", "edges": np.linspace(-1., 1., 500)}
        self.hist_task_dic_rho_d =    {"v_n": "rho", "edges": 10.0 ** np.linspace(4.0, 13.0, 500) / rho_const}
        self.hist_task_dic_rho_r =    {"v_n": "rho", "edges": 10.0 ** np.linspace(10.0, 17.0, 500) / rho_const}
        self.hist_task_dens_unb_bern = {"v_n": "dens_unb_bern", "edges": 10.0 ** np.linspace(-12., -6., 500)}
        self.hist_task_pressure =   {"v_n": "press", "edges": 10.0 ** np.linspace(-13., 5., 300)}

    def get_min_max(self, it, v_n):
        self.check_it(it)
        # self.check_v_n(v_n)
        min_, max_ = [], []
        nlevels = self.get_nlevels(it)
        for rl in range(nlevels):
            # print("rl:{}".format(rl))s
            if v_n == 'inv_ang_mom_flux':
                v_n = 'ang_mom_flux'
                data = -1. * self.get_masked_data(it, rl, v_n)
            else:
                data = self.get_masked_data(it, rl, v_n)
            if len(data) == 0:
                raise ValueError("len(data)=0 for it:{} rl:{} v_n:{}"
                                 .format(it, rl, v_n))

            min_.append(data.min())
            max_.append(data.max())
        min_ = np.array(min_)
        max_ = np.array(max_)
        return min_.min(), max_.max()
            # print("rl:{} min:{} max:{}".format(rl, data.min(), data.max()))

    def get_edges(self, it, corr_task_dic):

        dic = dict(corr_task_dic)

        if "edges" in dic.keys():
            return dic["edges"]

        if "points" in dic.keys() and "scale" in dic.keys():
            min_, max_ = self.get_min_max(it, dic["v_n"])
            if "min" in dic.keys(): min_ = dic["min"]
            if "max" in dic.keys(): max_ = dic["max"]
            print("\tv_n: {} is in ({}->{}) range"
                  .format(dic["v_n"], min_, max_))
            if dic["scale"] == "log":
                if min_ <= 0: raise ValueError("for Logscale min cannot be < 0. "
                                               "found: {}".format(min_))
                if max_ <= 0:raise ValueError("for Logscale max cannot be < 0. "
                                               "found: {}".format(max_))
                edges = 10.0 ** np.linspace(np.log10(min_), np.log10(max_), dic["points"])

            elif dic["scale"] == "linear":
                edges = np.linspace(min_, max_, dic["points"])
            else:
                raise NameError("Unrecoginzed scale: {}".format(dic["scale"]))
            return edges

        raise NameError("specify 'points' or 'edges' in the setup dic for {}".format(dic['v_n']))

    # ----------------------

    def get_total_mass(self, it, multiplier=2., mask_v_n="disk"):
        #
        self.check_it(it)
        self.check_mask_name(mask_v_n)
        mass = 0.
        nlevels = self.get_nlevels(it)
        for rl in range(nlevels):
            density = np.array(self.get_masked_data(it, rl, "density", mask_v_n))
            delta = self.get_grid_data(it, rl, "delta")
            mass += float(multiplier * np.sum(density) * np.prod(delta))
            # print rl
        # assert mass > 0.
        return mass

    def get_histogram(self, it, hist_task_dic, mask, multiplier=2.):

        v_n = hist_task_dic["v_n"]
        edge = self.get_edges(it, hist_task_dic)
        # print(edge); exit(1)
        histogram = np.zeros(len(edge) - 1)
        _edge = []
        nlevels = self.get_nlevels(it)
        for rl in range(nlevels):
            weights = self.get_masked_data(it, rl, "density", mask).flatten() * \
                      np.prod(self.get_grid_data(it, rl, "delta")) * multiplier
            data = self.get_masked_data(it, rl, v_n, mask)
            tmp1, _ = np.histogram(data, bins=edge, weights=weights)
            histogram = histogram + tmp1
        # print(len(histogram), len(_edge), len(edge))
        # assert len(histogram) == len(edge)# 0.5 * (edge_x[1:] + edge_x[:-1])
        outarr = np.vstack((0.5*(edge[1:]+edge[:-1]), histogram)).T
        return outarr

    def get_correlation(self, it, list_corr_task_dic, mask, multiplier=2.):

        edges = []
        for setup_dictionary in list_corr_task_dic:
            edges.append(self.get_edges(it, setup_dictionary))
        edges = tuple(edges)
        #
        correlation = np.zeros([len(edge) - 1 for edge in edges])
        #
        nlevels = self.get_nlevels(it)
        for rl in range(nlevels):
            data = []
            weights = self.get_masked_data(it, rl, "density", mask).flatten() * \
                      np.prod(self.get_grid_data(it, rl, "delta")) * multiplier
            for corr_dic in list_corr_task_dic:
                v_n = corr_dic["v_n"]
                if v_n == 'inv_ang_mom_flux':
                    v_n = 'ang_mom_flux'
                    data.append(-1. * self.get_masked_data(it, rl, v_n, mask).flatten())
                else:
                    data.append(self.get_masked_data(it, rl, v_n, mask).flatten())
            data = tuple(data)
            tmp, _ = np.histogramdd(data, bins=edges, weights=weights)
            correlation += tmp

        assert np.sum(correlation) > 0

        return edges, correlation

    def make_save_prof_slice(self, it, plane, v_ns, outfname):

        self.check_it(it)
        for v_n in v_ns:
            self.check_v_n(v_n)
        if not plane in ["xy", "xz", "yz"]:
            raise NameError("Plane:{} is not recognized".format(plane))

        outfile = h5py.File(outfname, "w")
        nlevels = self.get_nlevels(it)
        for rl in np.arange(start=0, stop=nlevels, step=1):
            gname = "reflevel=%d" % rl
            delta = self.get_grid_data(it, rl, "delta")
            extent = self.get_grid_data(it, rl, "extent")
            origin = self.get_grid_data(it, rl, "origin")
            # [ x y z ]
            if plane == 'xy':
                delta = np.delete(np.array(delta), -1, axis=0)
                origin = np.delete(np.array(origin), -1, axis=0)
            elif plane == 'xz':
                delta = np.delete(np.array(delta), 1, axis=0)
                origin = np.delete(np.array(origin), 1, axis=0)
            elif plane == 'yz':
                delta = np.delete(np.array(delta), 0, axis=0)
                origin = np.delete(np.array(origin), 0, axis=0)

            time = self.get_prof_time(it)

            # print("creating: {}".format(gname))
            outfile.create_group(gname)
            outfile[gname].attrs.create("delta", delta)  # grid[rl].delta)
            outfile[gname].attrs.create("extent", extent)  # grid[rl].extent())
            outfile[gname].attrs.create("origin", origin)  # grid[rl].extent())
            outfile[gname].attrs.create("iteration", int(it))  # iteration)
            outfile[gname].attrs.create("reflevel", rl)
            outfile[gname].attrs.create("time", time)  # dset.get_time(iteration))
            # saving masks
            mask = self.get_mask(it, rl, "rl".format(plane))
            # outfile[gname].create_dataset("rl_mask", data=np.array(mask, dtype=np.int))
            # print(np.array(mask, dtype=np.int)); exit(1)
            if plane == 'xy':
                mask = mask[:, :, 0]
            elif plane == 'xz':
                y = self.get_comp_data(it, rl, "y")
                iy0 = np.argmin(np.abs(y[0, :, 0]))
                mask = mask[:, iy0, :]
            elif plane == 'yz':
                x = self.get_comp_data(it, rl, "x")
                ix0 = np.argmin(np.abs(x[:, 0, 0]))
                mask = mask[ix0, :, :]
            outfile[gname].create_dataset("rl_mask", data=np.array(mask, dtype=np.int))
            #
            for v_n in v_ns:
                data = self.get_comp_data(it, rl, v_n)[3:-3, 3:-3, 3:-3]
                # print("{} {} {}".format(it, rl, v_n))
                if plane == 'xy':
                    data = data[:, :, 0]
                elif plane == 'xz':
                    # wierd stuff from david's script extract_slice.py
                    y = self.get_comp_data(it, rl, "y")
                    iy0 = np.argmin(np.abs(y[0, :, 0]))
                    if abs(y[0, iy0, 0]) < 1e-15:
                        _i_ = iy0
                        data = data[:, iy0, :]
                    else:
                        if y[0, iy0, 0] > 0:
                            iy0 -= 1
                        _i_ = iy0
                        data = 0.5 * (data[:, iy0, :] + data[:, iy0 + 1, :])
                elif plane == 'yz':
                    # wierd stuff from david's script extract_slice.py
                    x = self.get_comp_data(it, rl, "x")
                    ix0 = np.argmin(np.abs(x[:, 0, 0]))
                    if abs(x[ix0, 0, 0]) < 1e-15:
                        _i_ = ix0
                        data = data[ix0, :, :]
                    else:
                        if x[ix0, 0, 0] > 0:
                            ix0 -= 1
                        _i_ = ix0
                        data = 0.5 * (data[ix0, :, :] + data[ix0 + 1, :, :])

                # print(mask.shape, data.shape)
                assert mask.shape == data.shape
                outfile[gname].create_dataset(v_n, data=np.array(data, dtype=np.float32))

        outfile.close()

    def get_dens_modes_for_rl_old(self, rl=6, mmax = 8, rho_dens="dens"):

        import numexpr as ne

        if rho_dens == "rho":
            pass
        elif rho_dens == "dens":
            pass
        else:
            Printcolor.red("Wrong name rho_dens:{} for density modes".format(rho_dens))
            raise NameError("Wrong name rho_dens:{} for density modes".format(rho_dens))
        #
        iterations = self.list_iterations  # apply limits on it
        #
        times = []
        modes = [[] for m in range(mmax + 1)]
        xcs = []
        ycs = []
        #

        for idx, it in enumerate(iterations):
            print("\tcomputing {} modes, it: {}/{}".format(rho_dens, idx, len(iterations)))

            delta = self.get_grid_data(it, rl, "delta")[:-1]
            # print(delta); exit(0)
            dxyz = np.prod(delta)
            x = self.get_grid_data(it, rl, 'x')
            y = self.get_grid_data(it, rl, 'y')
            z = self.get_grid_data(it, rl, 'z')
            x = x[:, :, 0]
            y = y[:, :, 0]

            # get z=0 slice
            rho = self.get_prof_arr(it, rl, "rho")[:, :, 0]

            # Exclude region outside refinement levels
            idx = np.isnan(rho)
            rho[idx] = 0.0
            #
            if rho_dens == "dens":
                lapse = self.get_prof_arr(it, rl, "lapse")[:, :, 0]
                vol = self.get_prof_arr(it, rl, "vol")[:, :, 0]
                w_lorentz = self.get_prof_arr(it, rl, "w_lorentz")[:, :, 0]
                # Exclude region outside refinement levels
                vol[idx] = 0.0
                w_lorentz[idx] = 0.0
                # apply mask to cut off the horizon
                rho[lapse < 0.15] = 0

            # Compute center of mass
            # modes[0].append(dxyz * ne.evaluate("sum(rho * w_lorentz * vol)"))
            modes[0].append(dxyz * ne.evaluate("sum(rho)"))
            if rho_dens == "dens":
                Ix = dxyz * ne.evaluate("sum(rho * w_lorentz * vol * x)")
                Iy = dxyz * ne.evaluate("sum(rho * w_lorentz * vol * y)")
            else:
                Ix = dxyz * ne.evaluate("sum(rho * x)")
                Iy = dxyz * ne.evaluate("sum(rho * y)")
            xc = Ix / modes[0][-1]
            yc = Iy / modes[0][-1]
            phi = ne.evaluate("arctan2(y - yc, x - xc)")

            # phi = ne.evaluate("arctan2(y, x)")

            xcs.append(xc)
            ycs.append(yc)

            # Extract modes
            times.append(get_time_for_it(it, self.list_times, self.list_iterations))

            for m in range(1, mmax + 1):
                if rho_dens == "dens":
                    modes[m].append(dxyz * ne.evaluate("sum(rho * w_lorentz * vol * exp(-1j * m * phi))"))
                else:
                    modes[m].append(dxyz * ne.evaluate("sum(rho * exp(-1j * m * phi))"))
                #

        return times, iterations, xcs, ycs, modes

    def get_dens_modes_for_rl(self, rl=6, mmax = 8, nshells=50):


        import numexpr as ne

        iterations = self.list_iterations  # apply limits on it
        #
        times = []
        int_modes = [[0. for it in iterations] for m in range(mmax + 1)]
        r_modes = [[[] for it in iterations] for m in range(mmax + 1)]
        rbins = [np.zeros(0,) for it in iterations]
        xcs = []
        ycs = []
        #
        for i_it, it in enumerate(iterations):
            print("\tcomputing modes for iteration {}/{}".format(i_it+1, len(iterations)))
            # collecting data
            t = get_time_for_it(it, self.list_times, self.list_iterations)
            times.append(t)
            #
            delta = self.get_grid_data(it, rl, "delta")[:-1] # for XY plane
            dxyz = np.prod(delta) # for integration
            x = self.get_grid_data(it, rl, 'x')[:, :, 0] # grid on the Plane YX
            y = self.get_grid_data(it, rl, 'y')[:, :, 0]
            #
            rho = self.get_prof_arr(it, rl, "rho")[:, :, 0] # rest-mass density
            lapse = self.get_prof_arr(it, rl, "lapse")[:, :, 0] # mor BH exclusion
            vol = self.get_prof_arr(it, rl, "vol")[:, :, 0] # for Dens
            w_lorentz = self.get_prof_arr(it, rl, "w_lorentz")[:, :, 0] # for Dens
            #
            idx = np.isnan(rho) # out_of_grid masking
            rho[idx] = 0.0 # applying mask for out of grid level
            rho[lapse < 0.15] = 0 # applying mask for "apparent horizon"
            #
            dens = ne.evaluate("rho * w_lorentz * vol") # conserved density (?)
            #
            mode0 = dxyz * ne.evaluate("sum(dens)") # total mass of the BH
            int_modes[0][i_it] = mode0
            #
            Ix = dxyz * ne.evaluate("sum(dens * x)") # computing inertia center
            Iy = dxyz * ne.evaluate("sum(dens * y)")
            xc = Ix / mode0 # computing center of mass
            yc = Iy / mode0
            xcs.append(xc)
            ycs.append(yc)
            #
            phi = ne.evaluate("arctan2(y - yc, x - xc)") # shifting coordinates for center of mass
            r = ne.evaluate("sqrt(x**2 + y**2)")
            #
            for m in range(1, mmax + 1):
                _mode = dxyz * np.sum(dens * np.exp(-1j * m * phi)) # = C_m that is not f(r)
                int_modes[m][i_it] = _mode # np.complex128 number
            #
            shells = np.linspace(r.min(), r.max(), nshells) # for C_m that is a f(r)
            rbins[i_it] =  0.5 * (shells[:-1] + shells[1:]) # middle of every shell

            for i_shell, inner, outer in zip(range(nshells), shells[:-1], shells[1:]):
                mask = ((r > inner) & (r <= outer)) # to render False all points outside of the i_shell
                for m in range(0, mmax + 1):
                    _mode = dxyz * np.sum(dens[mask] * np.exp(-1j * m * phi[mask])) # complex128 numer
                    r_modes[m][i_it].append(_mode)

        return times, iterations, xcs, ycs, int_modes, rbins, r_modes

        # # int_modes = [modes][iterations] -> number
        # # r_modes = [modes][iteratiopns] -> array for every 'r'
        # # plot(radii[it], rmodes[mode][it])
        #
        # r_res = []
        # for m in range(mmax + 1):
        #     for i_it, it in enumerate(iterations):
        #         for m in range(mmax + 1):
        #
        #
        #
        # for m in range(mmax + 1):
        #     combined = np.zeros(len(iterations))
        #     for ir in range(nshells):
        #         combined = np.vstack((combined, r_modes[m][:][ir]))
        #     combined = np.delete(combined, 0, 0)
        #
        # for m in range(mmax + 1):
        #     combined = np.zeros(len(iterations))
        #     for ir in range(nshells):
        #         combined = np.vstack((combined, r_modes[m][:][ir]))
        #     combined = np.delete(combined, 0, 0)
        #     r_res.append(combined)
        #
        # return times, iterations, xcs, ycs, int_modes, rs, mmodes
        # #
        # exit(1)
        #
        #
        # # times = []
        # # modes = [[] for m in range(mmax + 1)]
        # # mmodes = [[[] for p in range(nshells)] for m in range(mmax + 1)]
        # # xcs = []
        # # ycs = []
        # #
        #
        # for idx, it in enumerate(iterations):
        #     print("\tcomputing {} modes, it: {}/{}".format(rho_dens, idx, len(iterations)))
        #     #
        #     delta = self.get_grid_data(it, rl, "delta")[:-1]
        #     # print(delta)
        #     dxyz = np.prod(delta)
        #     # print(dxyz); exit(0)
        #     x = self.get_grid_data(it, rl, 'x')
        #     y = self.get_grid_data(it, rl, 'y')
        #     z = self.get_grid_data(it, rl, 'z')
        #     x = x[:, :, 0]
        #     y = y[:, :, 0]
        #
        #     # get z=0 slice
        #     rho = self.get_prof_arr(it, rl, "rho")[:, :, 0]
        #
        #     # Exclude region outside refinement levels
        #     idx = np.isnan(rho)
        #     rho[idx] = 0.0
        #     #
        #     lapse = self.get_prof_arr(it, rl, "lapse")[:, :, 0]
        #     vol = self.get_prof_arr(it, rl, "vol")[:, :, 0]
        #     w_lorentz = self.get_prof_arr(it, rl, "w_lorentz")[:, :, 0]
        #     # Exclude region outside refinement levels
        #     vol[idx] = 0.0
        #     w_lorentz[idx] = 0.0
        #     # apply mask to cut off the horizon
        #     rho[lapse < 0.15] = 0
        #
        #     dens = ne.evaluate("rho * w_lorentz * vol")
        #
        #     # Compute center of mass
        #     print(idx)
        #     int_modes[0][idx] = dxyz * ne.evaluate("sum(dens)")
        #     # modes[0].append(dxyz * ne.evaluate("sum(rho)"))
        #     Ix = dxyz * ne.evaluate("sum(dens * x)")
        #     Iy = dxyz * ne.evaluate("sum(dens * y)")
        #     xc = Ix / int_modes[0][-1]
        #     yc = Iy / int_modes[0][-1]
        #     phi = ne.evaluate("arctan2(y - yc, x - xc)")
        #     r = ne.evaluate("sqrt(x**2 + y**2)")
        #     print(r.max(), r.min())
        #     # phi = ne.evaluate("arctan2(y, x)")
        #     xcs.append(xc)
        #     ycs.append(yc)
        #
        #     times.append(self.get_time_for_it(it, d1d2d3prof="prof"))
        #     print('\tm:'),
        #     for m in range(1, mmax + 1):
        #         print(m),
        #         # _mode1 = dxyz * ne.evaluate("sum(rho * w_lorentz * vol * exp(-1j * m * phi))")
        #         _mode = dxyz * np.sum(dens * np.exp(-1j * m * phi))
        #         # print(_mode2)
        #         # exit(1)
        #         int_modes[m][idx] = _mode
        #
        #     #
        #     print('r:'),
        #     shells = np.linspace(r.min(), r.max(), nshells)
        #     for i_shell, inner, outer in zip(range(nshells), shells[:-1], shells[1:]):
        #         print(i_shell),
        #         for m in range(0, mmax + 1):
        #             mask = ((r > inner) & (r <= outer))
        #             # _mode1 = dxyz * ne.evaluate("sum(rho * w_lorentz * vol * exp(-1j * m * phi))")
        #             _mode = dxyz * np.sum(dens[mask] * np.exp(-1j * m * phi[mask]))
        #             # print(_mode1, _mode2)
        #             # exit(1)
        #             r_modes[m][idx].append(_mode)
        #
        #     rs = 0.5 * (shells[:-1] + shells[1:])
        #     # print(len(rs), len(mmodes))
        #     # assert len(rs) == len(mmodes)
        #     print('done')
        #     # exit(0)
        #         #
        #
        #
        #     # r_modes = np.vstack((r_modes[][][:]))
        #
        #     # for i_shell in range(shells):
        #
        #
        # return times, iterations, xcs, ycs, modes, rs, mmodes

    # def __delete__(self, instance):
    #     # instance.dfile.close()
    #     instance.data_matrix = [[np.zeros(0, )
    #                              for x in range(self.nlevels)]
    #                              for y in range(len(self.list_all_v_ns))]
    #     instance.mask_matrix = [np.ones(0, dtype=bool) for x in range(self.nlevels)]

    def delete_for_it(self, it, except_v_ns, rm_masks=True, rm_comp=True, rm_prof=True):
        self.check_it(it)
        nlevels = self.get_nlevels(it)
        # clean up mask array
        if rm_masks:
            for v_n in self.list_mask_names:
                for rl in range(nlevels):
                    # print("it:{} rl:{} v_n:{} [all len(rls):{}]".format(it, rl, v_n, nlevels))
                    self.mask_matrix[self.i_it(it)][rl][self.i_mask_v_n(v_n)] = np.ones(0, dtype=bool)
        # clean up data
        if rm_comp:
            for v_n in self.list_all_v_ns:
                if v_n not in except_v_ns:
                    self.check_v_n(v_n)
                    for rl in range(nlevels):
                        self.data_matrix[self.i_it(it)][rl][self.i_v_n(v_n)] = np.zeros(0, )

        # clean up the initial data
        if rm_prof:
            self.dfile_matrix[self.i_it(it)] = 0
            self.grid_matrix[self.i_it(it)] = 0
            for v_n in self.list_grid_v_ns:
                if not v_n in except_v_ns:
                    for rl in range(nlevels):
                        self.grid_data_matrix[self.i_it(it)][rl][self.i_grid_v_n(v_n)] = np.zeros(0,)


class INTERPOLATE_STORE(MAINMETHODS_STORE):

    def __init__(self, grid_object, flist, itlist, timesteplist, symmetry=None):
        """
            fname - of the module_profile

            sim - name of the simulation (for directory searching)

            grid_object -
                object of the class with the interpolated grid. Must contain:

                list(list_grid_v_ns) that comtains the list of variable names of new grid,
                    for examply x_cyl ... z_cyl, r_cyl ... z_cyl, dr_cyl ... dz_cyl
                get_xi() function that returns array of the type
                    return np.column_stack([self.x_cyl_3d.flatten(),
                                self.y_cyl_3d.flatten(),
                                self.z_cyl_3d.flatten()])
                get_shape() function that returns the shape of the new grid such as
                    example: self.x_cyl_3d.shape
                get_int_grid(v_n) fucntion that returns the array of the new grid
                    for variable v_n. For ecample for v_n = "r_cyl"

        :param fname:
        :param sim:
        :param grid_object:
        """

        MAINMETHODS_STORE.__init__(self, flist=flist, itlist=itlist, timesteplist=timesteplist, symmetry=symmetry)

        self.new_grid = grid_object

        self.list_int_grid_v_ns = grid_object.list_int_grid_v_ns
        self.list_int_v_ns = self.list_prof_v_ns + \
                             self.list_comp_v_ns + \
                             self.list_grid_v_ns

        self.int_data_matrix = [[np.zeros(0,)
                                for y in range(len(self.list_int_v_ns))]
                                for x in range(len(self.list_iterations))]

    def check_int_v_n(self, v_n):
        if v_n not in self.list_int_v_ns:
            raise NameError("v_n: '{}' not in the v_n list \n{}"
                            .format(v_n, self.list_int_v_ns))

    def i_int_v_n(self, v_n):
        self.check_int_v_n(v_n)
        return int(self.list_int_v_ns.index(v_n))

    def do_append_grid_var(self, it, v_n):
        self.int_data_matrix[self.i_it(it)][self.i_int_v_n(v_n)] = \
            self.new_grid.get_int_grid(v_n)

    # ---

    def do_interpolate(self, it, v_n):

        tmp = []
        nlevels = self.get_nlevels(it)
        for rl in range(nlevels):
            data = self.get_comp_data(it, rl, v_n)
            if self.new_grid.grid_type == "pol":
                tmp.append(data)
            else:
                tmp.append(data)

        xi = self.new_grid.get_xi()
        shape = self.new_grid.get_shape()

        # print(xi.shape)

        print("\t\tInterpolating: it:{} v_n:{} -> {} grid"
              .format(it, v_n, self.new_grid.grid_type))
        # carpet_grid = self.get_grid(it)
        if self.enforce_xy_grid:
            carpet_grid = self.get_grid(it)
        else:
            carpet_grid = self.get_grid(it)
        # exit(1)
        F = Interpolator(carpet_grid, tmp, interp=1)
        arr = F(xi).reshape(shape)

        self.int_data_matrix[self.i_it(it)][self.i_int_v_n(v_n)] = arr

    # ----

    def is_data_interpolated(self, it, v_n):

        if len(self.int_data_matrix[self.i_it(it)][self.i_int_v_n(v_n)]) == 0:
            if v_n in self.list_int_grid_v_ns:
                self.do_append_grid_var(it, v_n)
            else:
                self.do_interpolate(it, v_n)

    def get_int(self, it, v_n):
        self.check_it(it)
        self.check_int_v_n(v_n)
        self.is_data_interpolated(it, v_n)
        return self.int_data_matrix[self.i_it(it)][self.i_int_v_n(v_n)]


class INTMETHODS_STORE(INTERPOLATE_STORE):

    """
        interpolates the data for any variable onto one of the
        grids: cyindrical, spherical, cartesian (see classes above)
    """

    def __init__(self, grid_object, flist, itlist, timesteplist, symmetry=None):

        INTERPOLATE_STORE.__init__(self, grid_object=grid_object, flist=flist, itlist=itlist,
                                   timesteplist=timesteplist, symmetry=symmetry)

    def save_new_grid(self, it, outdir):
        self.check_it(it)

        grid_type = self.new_grid.grid_info['type']

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        path = outdir + str(it) + '/'

        if os.path.isfile(path + grid_type + '_grid.h5'):
            os.remove(path + grid_type + '_grid.h5')

        outfile = h5py.File(path + grid_type + '_grid.h5', "w")

        if not os.path.exists(path):
            os.makedirs(path)

        # print("Saving grid...")
        for v_n in self.list_int_grid_v_ns:
            outfile.create_dataset(v_n, data=self.new_grid.get_int_grid(v_n))
        outfile.close()

    def save_int_v_n(self, it, v_n, outdir, overwrite=False):

        self.check_it(it)

        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        path = outdir + str(it) + '/'
        if not os.path.isdir(path):
            os.mkdir(path)

        grid_type = self.new_grid.grid_type

        fname = path + grid_type + '_' + v_n + '.h5'

        if os.path.isfile(fname):
            if overwrite:
                print("File: {} already exists -- overwriting".format(fname))
                os.remove(fname)
                outfile = h5py.File(fname, "w")
                outfile.create_dataset(v_n, data=self.get_int(it, v_n))
                outfile.close()
            else:
                print("File: {} already exists -- skipping".format(fname))
        else:
            outfile = h5py.File(fname, "w")
            outfile.create_dataset(v_n, data=self.get_int(it, v_n))
            outfile.close()

    def save_vtk_file(self, it, v_n_s, outdir, overwrite=False, private_dir="vtk"):

        # This requires PyEVTK to be insalled. You can get it with:
        # $ hg clone https://bitbucket.org/pauloh/pyevtk PyEVTK

        self.check_it(it)

        try:
            from evtk.hl import gridToVTK
        except ImportError:
            raise ImportError("Error importing gridToVTK. Is evtk installed? \n"
                              "If not, do: hg clone https://bitbucket.org/pauloh/pyevtk PyEVTK ")

        if self.new_grid.grid_type != "cart":
            raise AttributeError("only 'cart' grid is supported")

        path = outdir + str(it) + '/'
        if not os.path.isdir(path):
            os.mkdir(path)
        if private_dir != None and private_dir != '':
            path = path + private_dir + '/'
        if not os.path.isdir(path):
            os.mkdir(path)
        fname = "iter_" + str(it).zfill(10)
        fpath = path + fname

        if os.path.isfile(fpath) and not overwrite:
            print("Skipping it:{} ".format(it))
        else:

            xf = self.new_grid.get_int_grid("xf")
            yf = self.new_grid.get_int_grid("yf")
            zf = self.new_grid.get_int_grid("zf")

            celldata = {}
            for v_n in v_n_s:
                celldata[str(v_n)] = self.get_int(it, v_n)
            gridToVTK(fpath, xf, yf, zf, cellData=celldata)

    def compute_density_modes(self, rl=3, mmode=8, masklapse = 0.15):
        """
        :param rl:
        :param mmode:
        :return:
        """
        import numexpr as ne

        iterations = self.list_iterations

        times = []
        modes_r = [[] for m in range(mmode + 1)]
        modes = [[] for m in range(mmode + 1)]
        rcs = []
        phics = []

        for idx, it in enumerate(iterations):
            print("\tcomputing density modes, it: {}/{}".format(idx, len(iterations)))
            # getting grid
            r_pol = self.new_grid.get_int_grid("r_pol")
            dr_pol = self.new_grid.get_int_grid("dr_pol")
            phi_pol = self.new_grid.get_int_grid("phi_pol")
            dphi_pol = self.new_grid.get_int_grid("dphi_pol")

            # r_cyl = self.new_grid.get_int_grid("r_cyl")
            # dr_cyl = self.new_grid.get_int_grid('dr_cyl')
            # phi_cyl = self.new_grid.get_int_grid('phi_cyl')
            # dphi_cyl = self.new_grid.get_int_grid('dphi_cyl')
            # dz_cyl = self.new_grid.get_int_grid('dz_cyl')
            # # getting data
            # print(r_cyl.shape)
            # print(dr_cyl.shape)
            # print(phi_cyl.shape)
            # print(dphi_cyl.shape)
            # print(dz_cyl.shape)
            #

            drdphi = dr_pol * dphi_pol
            # print(drdphi); exit(1)

            density = self.get_int(it, "density")

            idx = np.isnan(density)
            density[idx] = 0.0

            # print(density.shape)
            # exit(1)
            if masklapse != None and masklapse > 0.:
                lapse = self.get_int(it, "lapse")
                density[lapse < masklapse] = 0
            #
            modes[0].append(drdphi * ne.evaluate("sum(density)"))
            Ir = drdphi * ne.evaluate("sum(density * r_pol)")
            Iphi = drdphi * ne.evaluate("sum(density * phi_pol)")
            rc = Ir / modes[0][-1]
            phic = Iphi / modes[0][-1]
            r_pol = r_pol - rc
            phi_pol = phi_pol - phic

            for m in range(1, mmode + 1):
                _mode = np.sum(density * np.exp(1j * m * phi_pol) * dphi_pol, axis=1)
                modes_r[m].append(_mode)
                # print("len(r_pol):{} len(modes_r[m]):{} {}".format(len(r_pol), len(_mode), _mode.shape)); exit(1)
                modes[m].append(np.sum(modes_r[m] * dr_pol[:, 0]))
                # _modes = drdphi * ne.evaluate("sum(density * exp(-1j * m * phi_pol))")
                # modes[m].append(drdphi * ne.evaluate("sum(density * exp(-1j * m * phi_pol))"))
                # print(modes[m])# , _modes); exit(1)
                # exit(1)
            return times, iterations, rcs, phics, modes, r_pol, modes_r

            # m_int_phi, m_int_phi_r = \
            #     PHYSICS.get_dens_decomp_2d(density, phi_pol, dphi_pol, dr_pol, m=mode)

