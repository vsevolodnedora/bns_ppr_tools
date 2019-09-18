# ///////////////////////////////////////////////////////////////////////////////
# // disk: analysis of the disk in a THC run from the 3D carpet data
# // Copyright (C) 2019, Vsevolod Nedora <vsevolod.nedora@uni-jena.de>
# //
# // This program is free software: you can redistribute it and/or modify
# // it under the terms of the GNU General Public License as published by
# // the Free Software Foundation, either version 3 of the License, or
# // (at your option) any later version.
# //
# // This program is distributed in the hope that it will be useful,
# // but WITHOUT ANY WARRANTY; without even the implied warranty of
# // MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# // GNU General Public License for more details.
# //
# // You should have received a copy of the GNU General Public License
# // along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ///////////////////////////////////////////////////////////////////////////////
# // This package contains utilities for
# // . Parsing 3D profile output of the carpet and hydro thorns
# // . Produce histograms-correlation maps and other properties
# //   of the disk for a cross analysis of multiple variables,
# //   including, density, angular momentum, its flux, density unbound.
# // . Produce total mass of the disk, interpolated 2D slices
# //
# // Usage
# // . Set the setup() according to required tasks to do
# // . python disk /path/to/profile.h5
# ///////////////////////////////////////////////////////////////////////////////

from __future__ import division
from sys import path
path.append('modules/')
from _curses import raw
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import scivis.units as ut # for tmerg
import statsmodels.formula.api as smf
import scipy.optimize as opt
from math import pi, sqrt
import matplotlib as mpl
from glob import glob
import pandas as pd
import numpy as np
import itertools
import os.path
import cPickle
import time
import copy
import click
import h5py
import csv
import os
import gc
# from visit_utils import *
from scidata.utils import locate
import scidata.carpet.hdf5 as h5
import scidata.xgraph as xg
from scidata.carpet.interp import Interpolator
import scivis.data.carpet2d
from scipy import interpolate
cmap = plt.get_cmap("viridis")
# from sklearn.linear_model import LinearRegression-
from scipy.optimize import fmin
from matplotlib.ticker import AutoMinorLocator, FixedLocator, NullFormatter, \
    MultipleLocator
from matplotlib.colors import LogNorm, Normalize

from general import *
from lists import *
from filework import *
from units import time_constant, volume_constant, energy_constant

from math import pi, log10
import time
from utils import *
from preanalysis import LOAD_ITTIME

""" --- --- SETUP --- --- """

def setup():
    rho_const = 6.176269145886162e+17 #

    tasks_for_rl = { # neutron star in GEO units 1.6191004634e-5
        "mask": {'rm_rl': True, 'rho': [6.e4 / rho_const, 1.e13 / rho_const], 'lapse': [0.15, 1.]},  # rho in cgs

        # "task1::correlation": [
        #     {"v_n": "rho",  "edges": 10.0 ** np.linspace(4.0, 16.0, 500) / rho_const},  # not in CGS :^
        #     {"v_n": "temp", "edges": 10.0 ** np.linspace(-2, 2, 300)},
        #     {"v_n": "Ye",   "edges": np.linspace(0, 0.5, 300)}
        # ],

        # "task::mass": {}, # computes total mass

        # "task2::correlation": [
        #     {"v_n": "ang_mom_flux",  "edges": 10.0 ** np.linspace(-12., -7, 500)},  # not in CGS :^
        #     {"v_n": "dens_unb_bern", "edges": 10.0 ** np.linspace(-9., -7., 500)}
        # ]

        # "task3::correlation": [
        #     {"v_n": "ang_mom_flux", "edges": 10.0 ** np.linspace(-12., -5, 500)},  # not in CGS :^
        #     {"v_n": "Ye", "edges": np.linspace(0.01, 0.5, 500)}
        # ]

        "task3::correlation": [
            {"v_n": "density", "edges": 10.0 ** np.linspace(-12., -5, 500)},  # not in CGS :^
            {"v_n": "theta", "edges": np.linspace(0, 3.2, 500)}
        ]


    }

    tasks_for_int = {
        "grid": {'type': 'cyl', 'n_r': 150, 'n_phi': 150, 'n_z': 100},
        "save": {"grid_v_ns":["phi_cyl", "r_cyl", "z_cyl",
                              "dphi_cyl", "dr_cyl", "dz_cyl"],
                 "v_ns": ['ang_mom', 'ang_mom_flux', 'density', 'dens_unb_geo',
                          'dens_unb_bern','rho', 'temp', 'Ye']}
    }

    general_settings = {}
    general_settings['indataformat'] = 'profile' # profile output-xxxx
    general_settings['indatadir'] = './'
    general_settings['outdir'] = './postprocess/'
    general_settings['figdir'] = './postprocess/'
    general_settings['nlevels'] = 7

    return general_settings, tasks_for_rl, tasks_for_int

""" --- --- OTHERS --- --- """

class CYLINDRICAL_GRID:
    """
        Creates a stretched cylindrical grid and allows
        to interpolate any data from carpet grid onto it.
        Stretched means, that the grid consists of 2 parts:
            1) linear distribution in terms of radius (0-15)
            2) logarithmic dist. in terms of radius (15-512)
        Class stores the grid information in its own variables
        that can be accessed directly or through
        `get_new_grid(v_n)` method

        Requirements:
            > dictionary grid_info{} that describes the grid:
            > class `carpet_grid` from scidata
        Usage:
            to access the new grid mesh arrays use:
                get_new_grid(v_n)
            to do the interpolation of arr, use
                get_int_arr(arr)
    """

    def __init__(self, grid_info=None):

        self.grid_info = {'type': 'cyl', 'n_r': 150, 'n_phi': 150, 'n_z': 100}



        self.grid_type = self.grid_info['type']

        # self.carpet_grid = carpet_grid

        self.list_int_grid_v_ns = ["x_cyl", "y_cyl", "z_cyl",
                                  "r_cyl", "phi_cyl",
                                  "dr_cyl", "dphi_cyl", "dz_cyl"]

        print('-' * 25 + 'INITIALIZING CYLINDRICAL GRID' + '-' * 25)

        phi_cyl, r_cyl, z_cyl, \
        self.dphi_cyl_3d, self.dr_cyl_3d, self.dz_cyl_3d = self.get_phi_r_z_grid()

        self.r_cyl_3d, self.phi_cyl_3d, self.z_cyl_3d \
            = np.meshgrid(r_cyl, phi_cyl, z_cyl, indexing='ij')
        self.x_cyl_3d = self.r_cyl_3d * np.cos(self.phi_cyl_3d)
        self.y_cyl_3d = self.r_cyl_3d * np.sin(self.phi_cyl_3d)

        print("\t GRID: [phi:r:z] = [{}:{}:{}]".format(len(phi_cyl), len(r_cyl), len(z_cyl)))

        print("\t GRID: [x_sph_3d:  ({},{})] {} pints".format(self.x_cyl_3d.min(), self.x_cyl_3d.max(), len(self.x_cyl_3d[:,0,0])))
        print("\t GRID: [y_sph_3d:  ({},{})] {} pints".format(self.y_cyl_3d.min(), self.y_cyl_3d.max(), len(self.y_cyl_3d[0,:,0])))
        print("\t GRID: [z_sph_3d:  ({},{})] {} pints".format(self.z_cyl_3d.min(), self.z_cyl_3d.max(), len(self.z_cyl_3d[0,0,:])))

        print('-' * 30 + '------DONE-----' + '-' * 30)
        print('\n')

    # cylindrical grid
    @staticmethod
    def make_stretched_grid(x0, x1, x2, nlin, nlog):
        assert x1 > 0
        assert x2 > 0
        x_lin_f = np.linspace(x0, x1, nlin)
        x_log_f = 10.0 ** np.linspace(log10(x1), log10(x2), nlog)
        return np.concatenate((x_lin_f, x_log_f))

    def get_phi_r_z_grid(self):

        # extracting grid info
        n_r = self.grid_info["n_r"]
        n_phi = self.grid_info["n_phi"]
        n_z = self.grid_info["n_z"]

        # constracting the grid
        r_cyl_f = self.make_stretched_grid(0., 15., 512., n_r, n_phi)
        z_cyl_f = self.make_stretched_grid(0., 15., 512., n_r, n_phi)
        phi_cyl_f = np.linspace(0, 2 * np.pi, n_phi)

        # edges -> bins (cells)
        r_cyl = 0.5 * (r_cyl_f[1:] + r_cyl_f[:-1])
        z_cyl = 0.5 * (z_cyl_f[1:] + z_cyl_f[:-1])
        phi_cyl = 0.5 * (phi_cyl_f[1:] + phi_cyl_f[:-1])

        # 1D grind -> 3D grid (to mimic the r, z, phi structure)
        dr_cyl = np.diff(r_cyl_f)[:, np.newaxis, np.newaxis]
        dphi_cyl = np.diff(phi_cyl_f)[np.newaxis, :, np.newaxis]
        dz_cyl = np.diff(z_cyl_f)[np.newaxis, np.newaxis, :]

        return phi_cyl, r_cyl, z_cyl, dphi_cyl, dr_cyl, dz_cyl

    # generic methods to be present in all INTERPOLATION CLASSES
    # def get_int_arr(self, arr_3d):
    #
    #     # if not self.x_cyl_3d.shape == arr_3d.shape:
    #     #     raise ValueError("Passed for interpolation 3d array has wrong shape:\n"
    #     #                      "{} Expected {}".format(arr_3d.shape, self.x_cyl_3d.shape))
    #     xi = np.column_stack([self.x_cyl_3d.flatten(),
    #                           self.y_cyl_3d.flatten(),
    #                           self.z_cyl_3d.flatten()])
    #     F = Interpolator(self.carpet_grid, arr_3d, interp=1)
    #     res_arr_3d = F(xi).reshape(self.x_cyl_3d.shape)
    #     return res_arr_3d

    def get_xi(self):
        return np.column_stack([self.x_cyl_3d.flatten(),
                                self.y_cyl_3d.flatten(),
                                self.z_cyl_3d.flatten()])

    def get_shape(self):
        return self.x_cyl_3d.shape

    def get_int_grid(self, v_n):

        if v_n == "x_cyl":
            return self.x_cyl_3d
        elif v_n == "y_cyl":
            return self.y_cyl_3d
        elif v_n == "z_cyl":
            return self.z_cyl_3d
        elif v_n == "r_cyl":
            return self.r_cyl_3d
        elif v_n == "phi_cyl":
            return self.phi_cyl_3d
        elif v_n == "dr_cyl":
            return self.dr_cyl_3d
        elif v_n == "dphi_cyl":
            return self.dphi_cyl_3d
        elif v_n == "dz_cyl":
            return self.dz_cyl_3d
        else:
            raise NameError("v_n: {} not recogized in grid. Available:{}"
                            .format(v_n, self.list_int_grid_v_ns))

    def save_grid(self, sim):

        grid_type = self.grid_type

        path = Paths.ppr_sims + sim + "/res_3d/"
        outfile = h5py.File(path + self.grid_type + '_grid.h5', "w")

        if not os.path.exists(path):
            os.makedirs(path)

        # print("Saving grid...")
        for v_n in self.list_int_grid_v_ns:
            outfile.create_dataset(v_n, data=self.get_int_grid(v_n))
        outfile.close()


class SPHERICAL_GRID:
    """
        Creates a stretched cylindrical grid and allows
        to interpolate any data from carpet grid onto it.
        Stretched means, that the grid consists of 2 parts:
            1) linear distribution in terms of radius (0-15)
            2) logarithmic dist. in terms of radius (15-512)
        Class stores the grid information in its own variables
        that can be accessed directly or through
        `get_new_grid(v_n)` method

        Requirements:
            > dictionary grid_info{} that describes the grid:
            > class `carpet_grid` from scidata
        Usage:
            to access the new grid mesh arrays use:
                get_new_grid(v_n)
            to do the interpolation of arr, use
                get_int_arr(arr)
    """

    def __init__(self):

        self.grid_info = {'type': 'sph', 'n_r': 200, 'n_phi': 200, 'n_theta': 150}

        self.grid_type = self.grid_info['type']

        # self.carpet_grid = carpet_grid

        self.list_int_grid_v_ns = ["x_sph", "y_sph", "z_sph",
                                   "r_sph", "phi_sph", "theta_sph",
                                   "dr_sph", "dphi_sph", "dtheta_sph"]

        print('-' * 25 + 'INITIALIZING SPHERICAL GRID' + '-' * 25)

        phi_sph, r_sph, theta_sph, \
        self.dphi_sph_3d, self.dr_sph_3d, self.dtheta_sph_3d = self.get_phi_r_theta_grid()

        self.r_sph_3d, self.phi_sph_3d, self.theta_sph_3d \
            = np.meshgrid(r_sph, phi_sph, theta_sph, indexing='ij')
        self.x_sph_3d = self.r_sph_3d * np.cos(self.phi_sph_3d) * np.sin(self.theta_sph_3d)
        self.y_sph_3d = self.r_sph_3d * np.sin(self.phi_sph_3d) * np.sin(self.theta_sph_3d)
        self.z_sph_3d = self.r_sph_3d * np.cos(self.theta_sph_3d)

        print("\t GRID: [phi_sph:   ({},{})] {} pints".format(phi_sph[0], phi_sph[-1], len(phi_sph)))
        print("\t GRID: [r_sph:     ({},{})] {} pints".format(r_sph[0], r_sph[-1], len(r_sph), len(r_sph)))
        print("\t GRID: [theta_sph: ({},{})] {} pints".format(theta_sph[0], theta_sph[-1], len(theta_sph)))
        print('   --- --- ---   ')
        print("\t GRID: [x_sph_3d:  ({},{})] {} pints".format(self.x_sph_3d.min(), self.x_sph_3d.max(), len(self.x_sph_3d[:,0,0])))
        print("\t GRID: [y_sph_3d:  ({},{})] {} pints".format(self.y_sph_3d.min(), self.y_sph_3d.max(), len(self.y_sph_3d[0,:,0])))
        print("\t GRID: [z_sph_3d:  ({},{})] {} pints".format(self.z_sph_3d.min(), self.z_sph_3d.max(), len(self.z_sph_3d[0,0,:])))
        # print('   --- --- ---   ')
        # print("\t GRID: [x_sph_3d:  ({},{})] {} pints".format(self.x_sph_3d[0,0,0], self.x_sph_3d[0,-1,0], len(self.x_sph_3d[0,:,0])))
        # print("\t GRID: [y_sph_3d:  ({},{})] {} pints".format(self.y_sph_3d[0,0,0], self.y_sph_3d[-1,0,0], len(self.y_sph_3d[:,0,0])))
        # print("\t GRID: [z_sph_3d:  ({},{})] {} pints".format(self.z_sph_3d[0,0,0], self.z_sph_3d[0,0,-1], len(self.z_sph_3d[0,0,:])))

        print("\t GRID: [phi:r:theta] = [{}:{}:{}]".format(len(phi_sph), len(r_sph), len(theta_sph)))

        print('-' * 30 + '--------DONE-------' + '-' * 30)
        print('\n')

    # cylindrical grid
    @staticmethod
    def make_stretched_grid(x0, x1, x2, nlin, nlog):
        assert x1 > 0
        assert x2 > 0
        x_lin_f = np.linspace(x0, x1, nlin)
        x_log_f = 10.0 ** np.linspace(log10(x1), log10(x2), nlog)
        return np.concatenate((x_lin_f, x_log_f))

    def get_phi_r_theta_grid(self):

        # extracting grid info
        n_r = self.grid_info["n_r"]
        n_phi = self.grid_info["n_phi"]
        n_theta = self.grid_info["n_theta"]

        # constracting the grid
        r_sph_f = self.make_stretched_grid(0., 15., 512., n_r, n_phi)
        # z_cyl_f = self.make_stretched_grid(0., 15., 512., n_r, n_phi)
        phi_sph_f = np.linspace(0, 2 * np.pi, n_phi)
        theta_sph_f = np.linspace(-np.pi / 2, np.pi / 2, n_theta)

        # edges -> bins (cells)
        r_sph = 0.5 * (r_sph_f[1:] + r_sph_f[:-1])
        # z_cyl = 0.5 * (z_cyl_f[1:] + z_cyl_f[:-1])
        phi_sph = 0.5 * (phi_sph_f[1:] + phi_sph_f[:-1])
        theta_sph = 0.5 * (theta_sph_f[1:] + theta_sph_f[:-1])

        # 1D grind -> 3D grid (to mimic the r, z, phi structure)
        dr_sph = np.diff(r_sph_f)[:, np.newaxis, np.newaxis]
        dphi_sph = np.diff(phi_sph_f)[np.newaxis, :, np.newaxis]
        # dz_cyl = np.diff(z_cyl_f)[np.newaxis, np.newaxis, :]
        dtheta_sph = np.diff(theta_sph_f)[np.newaxis, np.newaxis, :]

        return phi_sph, r_sph, theta_sph, dphi_sph, dr_sph, dtheta_sph

    # generic methods to be present in all INTERPOLATION CLASSES
    # def get_int_arr(self, arr_3d):
    #
    #     # if not self.x_cyl_3d.shape == arr_3d.shape:
    #     #     raise ValueError("Passed for interpolation 3d array has wrong shape:\n"
    #     #                      "{} Expected {}".format(arr_3d.shape, self.x_cyl_3d.shape))
    #     xi = np.column_stack([self.x_cyl_3d.flatten(),
    #                           self.y_cyl_3d.flatten(),
    #                           self.z_cyl_3d.flatten()])
    #     F = Interpolator(self.carpet_grid, arr_3d, interp=1)
    #     res_arr_3d = F(xi).reshape(self.x_cyl_3d.shape)
    #     return res_arr_3d

    def get_xi(self):
        return np.column_stack([self.x_sph_3d.flatten(),
                                self.y_sph_3d.flatten(),
                                self.z_sph_3d.flatten()])

    def get_shape(self):
        return self.x_sph_3d.shape

    def get_int_grid(self, v_n):

        if v_n == "x_sph":
            return self.x_sph_3d
        elif v_n == "y_sph":
            return self.y_sph_3d
        elif v_n == "z_sph":
            return self.z_sph_3d
        elif v_n == "r_sph":
            return self.r_sph_3d
        elif v_n == "phi_sph":
            return self.phi_sph_3d
        elif v_n == "theta_sph":
            return self.theta_sph_3d
        elif v_n == "dr_sph":
            return self.dr_sph_3d
        elif v_n == "dphi_sph":
            return self.dphi_sph_3d
        elif v_n == "dtheta_sph":
            return self.dtheta_sph_3d
        else:
            raise NameError("v_n: {} not recogized in grid. Available:{}"
                            .format(v_n, self.list_int_grid_v_ns))

    def save_grid(self, sim):

        grid_type = self.grid_type

        path = Paths.ppr_sims + sim + "/res_3d/"
        outfile = h5py.File(path + str(self.grid_type) + "_grid.h5", "w")

        if not os.path.exists(path):
            os.makedirs(path)

        # print("Saving grid...")
        for v_n in self.list_int_grid_v_ns:
            outfile.create_dataset(v_n, data=self.get_int_grid(v_n))
        outfile.close()


class CARTESIAN_GRID:
    """
    Courtasy of David Radice,
    modified by Vsevolod Nedora
    """
    def __init__(self):

        self.grid_type = "cart" # cartesian stretched grid

        self.gen_set = {
            "reflecting_xy": True,  # Apply reflection symmetry across the xy-plane
            "xmin": -100.0,         # Include region with x >= xmin
            "xmax": 100.0,          # Include region with x <= xmax
            "xix": 0.2,             # Stretch factor for the grid in the x-direction
            "nlinx": 80,            # Number of grid points in the linear portion of the x-grid
            "nlogx": 160,            # Number of grid points in the log portion of the x-grid
            "ymin": -100,           # Include region with y >= ymin
            "ymax": 100,            # Include region with y <= ymax
            "xiy": 0.2,
            "nliny": 80,            # Number of grid points in the linear portion of the y-grid
            "nlogy": 160,            # Number of grid points in the log portion of the y-grid
            "zmin": -10.0,           # Include region with z >= zmin
            "zmax": 10.0,            # Include region with z <= zmax
            "xiz": 0.2,             # Stretch factor for the grid in the z-direction
            "nlinz": 80,            # Number of grid points in the linear portion of the z-grid
            "nlogz": 160,            # Number of grid points in the log portion of the z-grid
        }

        self.list_int_grid_v_ns = ["xc", "yc", "zc",
                                   "xf", "yf", "zf",
                                   "dx", "dy", "dz",
                                   "xi"]

        self.grid_matric = [np.zeros(0) for o in range(len(self.list_int_grid_v_ns))]

        # do make grid
        self.make_grid()

    def check_v_n(self, v_n):
        if not v_n in self.list_int_grid_v_ns:
            raise NameError("v_n: {} not in list of gric v_ns: {}"
                            .format(v_n, self.list_int_grid_v_ns))

    def i_v_n(self, v_n):
        return int(self.list_int_grid_v_ns.index(v_n))

    @staticmethod
    def make_stretched_grid(xmin, xmax, xi, nlin, nlog):
        dx = xi / nlin
        x_lin = np.arange(0, xi, dx)
        x_log = 10.0 ** np.linspace(np.log10(xi), 0.0, nlog // 2)
        x_grid = np.concatenate((x_lin, x_log))
        x_grid *= (xmax - xmin) / 2.
        x_ave = (xmax + xmin) / 2.
        return np.concatenate(((x_ave - x_grid)[::-1][:-1], x_grid + x_ave))

    def make_grid(self):

        print("Generating interpolation grid..."),
        start_t = time.time()
        xf = self.make_stretched_grid(self.gen_set["xmin"], self.gen_set["xmax"], self.gen_set["xix"],
                                      self.gen_set["nlinx"], self.gen_set["nlogx"])
        yf = self.make_stretched_grid(self.gen_set["ymin"], self.gen_set["ymax"], self.gen_set["xiy"],
                                      self.gen_set["nliny"], self.gen_set["nlogy"])
        zf = self.make_stretched_grid(self.gen_set["zmin"], self.gen_set["zmax"], self.gen_set["xiz"],
                                      self.gen_set["nlinz"], self.gen_set["nlogz"])

        xc = 0.5 * (xf[:-1] + xf[1:])  # center op every cell
        yc = 0.5 * (yf[:-1] + yf[1:])
        zc = 0.5 * (zf[:-1] + zf[1:])

        dx = np.diff(xf)[:, np.newaxis, np.newaxis]
        dy = np.diff(yf)[np.newaxis, :, np.newaxis]
        dz = np.diff(zf)[np.newaxis, np.newaxis, :]

        xc, yc, zc = np.meshgrid(xc, yc, zc, indexing='xy')
        if self.gen_set["reflecting_xy"]:
            xi = np.column_stack([xc.flatten(), yc.flatten(), np.abs(zc).flatten()])
        else:
            xi = np.column_stack([xc.flatten(), yc.flatten(), zc.flatten()])  #

        self.grid_matric[self.i_v_n("xi")] = xi
        self.grid_matric[self.i_v_n("xc")] = xc
        self.grid_matric[self.i_v_n("yc")] = yc
        self.grid_matric[self.i_v_n("zc")] = zc
        self.grid_matric[self.i_v_n("xf")] = xf
        self.grid_matric[self.i_v_n("yf")] = yf
        self.grid_matric[self.i_v_n("zf")] = zf
        self.grid_matric[self.i_v_n("dx")] = dx
        self.grid_matric[self.i_v_n("dy")] = dy
        self.grid_matric[self.i_v_n("dz")] = dz

        print("done! (%.2f sec)" % (time.time() - start_t))

    def get_int_grid(self, v_n):
        self.check_v_n(v_n)
        return self.grid_matric[self.i_v_n(v_n)]

    def get_xi(self):
        return self.grid_matric[self.i_v_n("xi")]

    def get_shape(self):
        return np.array(self.grid_matric[self.i_v_n("xc")]).shape

    def save_grid(self, sim):

        path = Paths.ppr_sims + sim + "/res_3d/"
        outfile = h5py.File(path + self.grid_type + '_grid.h5', "w")

        if not os.path.exists(path):
            os.makedirs(path)

        # print("Saving grid...")
        for v_n in self.list_int_grid_v_ns:
            outfile.create_dataset(v_n, data=self.get_int_grid(v_n))
        outfile.close()


class SAVE_RESULT:

    def __init__(self):
        pass

    @staticmethod
    def correlation(it, v_ns, edges, corr, outdir):


        name = outdir + str(int(it)) + '_corr_'
        for v_n in v_ns:
            name += v_n
            if v_n != v_ns[-1]:
                name += '_'
        name += '.h5'
        print('-' * 30 + 'SAVING CORRELATION' + '-' * 30)
        print(' ' * 30 + '{}'.format(name) + ' ' * 30)
        outfile = h5py.File(name, "w")
        for v_n, edge in zip(v_ns, edges):
            outfile.create_dataset(v_n, data=edge)
        outfile.create_dataset("mass", data=corr)
        print('-' * 30 + '------DONE-----' + '-' * 30)
        print('\n')

        return name

""" --- --- DATA PROCESSING --- --- """


class LOAD_PROFILE(LOAD_ITTIME):

    def __init__(self, sim, symmetry=None):

        LOAD_ITTIME.__init__(self, sim)

        self.symmetry = symmetry

        self.profpath = Paths.gw170817 + sim + '/' + "profiles/3d/"

        isprofs, itprofs, timeprofs = \
            self.get_ittime("profiles", "prof")
        if not isprofs:
            is3ddata, it3d, t3d = self.get_ittime("overall", d1d2d3prof="d3")
            if is3ddata:
                raise IOError("ittime.h5 says there are no profiles, while there is 3D data for times:\n{}"
                              "\n Extract profiles before proceeding"
                              .format(t3d))
            else:
                raise IOError("ittime.h5 says there ae no profiles, and no 3D data found.")

        self.list_iterations = list(itprofs)
        self.list_times = timeprofs

        self.list_prof_v_ns = [
                             "rho", "w_lorentz", "vol",  # basic
                             "press", "eps", "lapse",    # basic + lapse
                             "velx", "vely", "velz",     # velocities
                             "gxx", "gxy", "gxz", "gyy", "gyz", "gzz",  # metric
                             "betax", "betay", "betaz",  # shift components
                             'temp', 'Ye']

        self.list_grid_v_ns = ["x", "y", "z", "delta", "extent", "origin"]

        self.nlevels = 7

        self.dfile_matrix = [0
                             for it in range(len(self.list_iterations))]

        self.grid_matrix = [0
                             for it in range(len(self.list_iterations))]

        self.grid_data_matrix = [[[np.zeros(0,)
                                  for v_n in range(len(self.list_grid_v_ns))]
                                  for rl in range(self.nlevels)]
                                  for it in range(len(self.list_iterations))]

    def check_prof_v_n(self, v_n):
        if not v_n in self.list_prof_v_ns:
            raise NameError("v_n:{} not in list of profile v_ns:{}"
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
        fpath = self.profpath + str(it) + ".h5"
        if not os.path.isfile(fpath):
            raise IOError("Expected file:{} NOT found"
                          .format(fpath))
        dfile = h5py.File(fpath, "r")
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
        # self.profile = fname
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
        # if self.symmetry == "pi" and not str(self.profile).__contains__("_PI"):
        #     raise NameError("profile {} does not seem to have a pi symmetry. Check"
        #                     .format(self.profile))

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
        for il in range(self.nlevels):
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
            elif self.symmetry == None:
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
            raise ValueError('Error extracting v_n:{} from profile for it:{} rl:{}'.format(v_n, it, rl))
        return arr

    def __delete__(self, instance):

        instance.dfile_matrix = [0
                                  for it in range(len(self.list_iterations))]
        instance.grid_matrix = [0
                                  for it in range(len(self.list_iterations))]
        instance.grid_data_matrix = [[[np.zeros(0,)
                                  for v_n in range(len(self.list_grid_v_ns))]
                                  for rl in range(self.nlevels)]
                                  for it in range(len(self.list_iterations))]


class COMPUTE_STORE(LOAD_PROFILE):

    def __init__(self, sim, symmetry=None):

        LOAD_PROFILE.__init__(self, sim, symmetry)

        self.list_comp_v_ns = [
            "density", "vup", "metric", "shift",
            "enthalpy", "shvel", "u_0",
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
                             for x in range(self.nlevels)]
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

    def __delete__(self, instance):
        instance.dfile.close()
        instance.data_matrix = [[np.zeros(0, )
                             for x in range(self.nlevels)]
                            for y in range(len(self.list_all_v_ns))]


class MASK_STORE(COMPUTE_STORE):

    def __init__(self, sim, symmetry=None):
        COMPUTE_STORE.__init__(self, sim, symmetry)


        self.mask_setup = {'rm_rl': True,  # REMOVE previouse ref. level from the next
                           'rho': [6.e4 / 6.176e+17, 1.e13 / 6.176e+17],  # REMOVE atmo and NS
                           'lapse': [0.15, 1.]} # remove apparent horizon

        self.mask_matrix = [[np.ones(0, dtype=bool)
                            for x in range(self.nlevels)]
                            for y in range(len(self.list_iterations))]


        self.list_mask_v_n = ["x", "y", "z"]

    def compute_mask(self, it):

        nlevelist = np.arange(self.nlevels, 0, -1) - 1

        x = []
        y = []
        z = []

        for ii, rl in enumerate(nlevelist):
            x.append(self.get_grid_data(it, rl, "x")[3:-3, 3:-3, 3:-3])
            y.append(self.get_grid_data(it, rl, "y")[3:-3, 3:-3, 3:-3])
            z.append(self.get_grid_data(it, rl, "z")[3:-3, 3:-3, 3:-3])
            mask = np.ones(x[ii].shape, dtype=bool)
            if ii > 0 and self.mask_setup["rm_rl"]:
                x_ = (x[ii][:, :, :] <= x[ii - 1][:, 0, 0].max()) & (
                        x[ii][:, :, :] >= x[ii - 1][:, 0, 0].min())
                y_ = (y[ii][:, :, :] <= y[ii - 1][0, :, 0].max()) & (
                        y[ii][:, :, :] >= y[ii - 1][0, :, 0].min())
                z_ = (z[ii][:, :, :] <= z[ii - 1][0, 0, :].max()) & (
                        z[ii][:, :, :] >= z[ii - 1][0, 0, :].min())
                mask = mask & np.invert((x_ & y_ & z_))

            for v_n in self.mask_setup.keys()[1:]:
                self.check_v_n(v_n)
                if len(self.mask_setup[v_n]) != 2:
                    raise NameError("Error. 2 values are required to set a limit. Give {} for {}"
                                    .format(self.mask_setup[v_n], v_n))
                arr_1 = self.get_comp_data(it, rl, v_n)[3:-3, 3:-3, 3:-3]
                min_val = float(self.mask_setup[v_n][0])
                max_val = float(self.mask_setup[v_n][1])
                mask_i = (arr_1 > min_val) & (arr_1 < max_val)
                mask = mask & mask_i
                del arr_1
                del mask_i

            self.mask_matrix[self.i_it(it)][rl] = mask

    def is_mask_available(self, it, rl):
        mask = self.mask_matrix[self.i_it(it)][rl]
        if len(mask) == 0:
            self.compute_mask(it)

    def get_masked_data(self, it, rl, v_n):
        self.check_v_n(v_n)
        self.check_it(it)
        self.is_available(it, rl, v_n)
        self.is_mask_available(it, rl)
        data = np.array(self.get_comp_data(it, rl, v_n))[3:-3, 3:-3, 3:-3]
        mask = self.mask_matrix[self.i_it(it)][rl]
        return data[mask]

    def __delete__(self, instance):
        instance.dfile.close()
        instance.data_matrix = [[np.zeros(0, )
                                 for x in range(self.nlevels)]
                                 for y in range(len(self.list_all_v_ns))]
        instance.mask_matrix = [np.ones(0, dtype=bool) for x in range(self.nlevels)]


class MAINMETHODS_STORE(MASK_STORE):

    def __init__(self, sim, symmetry=None):

        MASK_STORE.__init__(self, sim, symmetry)

        self.sim = sim

        # "v_n": "temp", "edges": np.array()
        ''''''
        # "v_n": "temp", "points: number, "scale": "log", (and "min":number, "max":number)

        rho_const = 6.176269145886162e+17

        self.corr_task_dic_r_phi = [
            {"v_n": "r", "edges": np.linspace(0, 50, 500)},
            {"v_n": "phi", "edges": np.linspace(-np.pi, np.pi, 500)},
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
            {"v_n": "Ye",   "edges": np.linspace(0, 0.5, 500)}, #"edges": np.linspace(-1., 1., 500)},  # in c
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

        # hist

        self.hist_task_dic_r = {"v_n": "r", "edges": np.linspace(10, 200, 500)}
        self.hist_task_dic_theta = {"v_n": "theta", "edges": np.linspace(0, 2*np.pi, 500)}
        self.hist_task_dic_ye = {"v_n": "Ye",   "edges": np.linspace(0, 0.5, 500)}


    @staticmethod
    def if_comput_if_save(path, fname, fpath, sim, save=True, overwrite=False):

        do_compute = False
        do_save = False

        if not os.path.isdir(Paths.ppr_sims + sim + "/res_3d/") and save:
            os.mkdir(Paths.ppr_sims + sim + "/res_3d/")
            do_compute = True
            do_save = True
        elif not os.path.isdir(Paths.ppr_sims + sim + "/res_3d/") and not save:
            do_compute = True
            do_save = False

        if not os.path.isdir(path) and save:
            os.mkdir(path)
            do_compute = True
            do_save = True
        elif not os.path.isdir(path) and not save:
            do_compute = True
            do_save = False

        if not os.path.isfile(fpath) and save:
            do_compute = True
            do_save = True
        elif not os.path.isfile(fpath) and not save:
            do_compute = True
            do_save = False

        if os.path.isfile(fpath) and save and overwrite:
            os.remove(fpath)
            print("\tFile: {} for already exist. Overwriting."
                  .format(fname))
            do_compute = True
            do_save = True
        elif os.path.isfile(fpath) and save and not overwrite:
            print("\tFile: {} for already exist. Skipping."
                  .format(fname))
            do_compute = False
            do_save = False

        elif os.path.isfile(fpath) and not save:
            do_compute = True
            do_save = False

        return do_compute, do_save

    def get_total_mass_old(self, it, multiplier=2., fname="disk_mass.txt", save=False, overwrite=False):

        path = Paths.ppr_sims + self.sim + "/res_3d/" + str(it) + '/'

        fpath = path + fname

        do_compute, do_save = self.if_comput_if_save(path, fname, fpath, self.sim, save, overwrite)

        if do_compute:

            self.check_it(it)
            mass = 0.
            for rl in range(self.nlevels):
                density = np.array(self.get_masked_data(it, rl, "density"))
                delta = self.get_grid_data(it, rl, "delta")
                mass += float(multiplier * np.sum(density) * np.prod(delta))
            # print("\tit:{} mass:{:3f}Msun".format(it, mass))

            if do_save:
                if not os.path.exists(path):
                    os.makedirs(path)
                np.savetxt(path + fname, np.array([mass]), fmt='%.5f')

            return do_compute, mass
        else:
            return do_compute, 0.

    def get_min_max(self, it, v_n):
        self.check_it(it)
        # self.check_v_n(v_n)
        min_, max_ = [], []
        for rl in range(self.nlevels):

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

    def get_correlation_old(self, it, corr_task_dic, multiplier=2., save=False, overwrite=False):

        v_ns = []
        path = Paths.ppr_sims + self.sim + "/res_3d/" + str(it) + '/'
        for setup_dictionary in corr_task_dic:
            v_ns.append(setup_dictionary["v_n"])
        fname = "corr_".format(it)
        for v_n in v_ns:
            fname += v_n
            if v_n != v_ns[-1]:
                fname += '_'
        fname += '.h5'
        fpath = path + fname

        do_compute, do_save = self.if_comput_if_save(path, fname, fpath, self.sim, save, overwrite)

        # ---

        if do_compute:

            edges = []
            for setup_dictionary in corr_task_dic:
                edges.append(self.get_edges(it, setup_dictionary))
            edges = tuple(edges)

            correlation = np.zeros([len(edge) - 1 for edge in edges])

            for rl in range(self.nlevels):
                data = []
                weights = self.get_masked_data(it, rl, "density").flatten() * \
                          np.prod(self.get_grid_data(it, rl, "delta")) * multiplier
                for i_vn, v_n in enumerate(v_ns):

                    if v_n == 'inv_ang_mom_flux':
                        v_n = 'ang_mom_flux'
                        data.append(-1. * self.get_masked_data(it, rl, v_n).flatten())
                    else:
                        data.append(self.get_masked_data(it, rl, v_n).flatten())


                data = tuple(data)
                tmp, _ = np.histogramdd(data, bins=edges, weights=weights)
                correlation += tmp

            if do_save:

                if not os.path.exists(Paths.ppr_sims + self.sim + "/res_3d/"):
                    os.makedirs(Paths.ppr_sims + self.sim + "/res_3d/")

                if not os.path.exists(path):
                    os.makedirs(path)

                outfile = h5py.File(path + fname, "w")
                for v_n, edge in zip(v_ns, edges):
                    outfile.create_dataset(v_n, data=edge)
                outfile.create_dataset("mass", data=correlation)
                outfile.close()
        else:
            correlation = np.zeros(0,)

        return do_compute, correlation

    def get_slice_old(self, it, plane, v_ns, save=True, overwrite=False, description=None):

        self.check_it(it)
        for v_n in v_ns:
            self.check_v_n(v_n)
        if not plane in ["xy", "xz", "yz"]:
            raise NameError("Plane:{} is not recognized".format(plane))


        path = Paths.ppr_sims + self.sim + "/res_3d/" + str(it) + '/'
        fname = "profile" + '.' + plane + ".h5"
        fpath = path + fname

        do_compute, do_save = self.if_comput_if_save(path, fname, fpath, self.sim, save, overwrite)

        if do_compute and do_save:
            outfile = h5py.File(path + fname, "w")
            if description is not None:
                outfile.create_dataset("description", data=np.string_(description))
            for rl in np.arange(start=0, stop=self.nlevels, step=1):
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

                for v_n in v_ns:
                    data = self.get_comp_data(it, rl, v_n)
                    # print("{} {} {}".format(it, rl, v_n))
                    if plane == 'xy':
                        data = data[:, :, 0]
                    elif plane == 'xz':
                        # wierd stuff from david's script extract_slice.py
                        y = self.get_comp_data(it, rl, "y")
                        iy0 = np.argmin(np.abs(y[0, :, 0]))
                        if abs(y[0, iy0, 0]) < 1e-15:
                            _i_ = iy0
                            data = data[:,iy0,:]
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
                            data = data[ix0,:,:]
                        else:
                            if x[ix0, 0, 0] > 0:
                                ix0 -= 1
                            _i_ = ix0
                            data = 0.5 * (data[ix0, :, :] + data[ix0+1, :, :])
                    outfile[gname].create_dataset(v_n, data=np.array(data, dtype=np.float32))

            outfile.close()

    def get_dens_modes_for_rl_old(self, rl=6, mmax = 8, tmin=None, tmax=None, fname="density_modes_lap15.h5",
                                  save=True, overwrite=True):

        import numexpr as ne

        path = Paths.ppr_sims + self.sim + "/res_3d/"
        if not os.path.isdir(path):
            os.mkdir(path)
        fpath = path + fname

        iterations = self.list_iterations # apply limits on it
        times = self.list_times

        if tmax != None and tmin != None:
            assert tmin > tmax
        if tmin != None:
            assert tmin > times.min()
            iterations = np.array(iterations[times > tmin], dtype=int)
            times = times[times > tmin]
        if tmax != None:
            assert tmax < times.max()
            iterations = np.array(iterations[times < tmax], dtype=int)
            times = times[times < tmax]

        do_compute, do_save = self.if_comput_if_save(path, fname, fpath, self.sim, save, overwrite)

        if do_save and do_compute:

            times = []
            modes = [[] for m in range(mmax + 1)]
            xcs = []
            ycs = []

            for idx, it in enumerate(iterations):
                print("\tprocessing iteration: {}/{}".format(idx, len(iterations)))
                # get z=0 slice
                lapse = self.get_prof_arr(it, rl, "lapse")[:, :, 0]
                rho = self.get_prof_arr(it, rl, "rho")[:, :, 0]
                vol = self.get_prof_arr(it, rl, "vol")[:, :, 0]
                w_lorentz = self.get_prof_arr(it, rl, "w_lorentz")[:, :, 0]

                delta = self.get_grid_data(it, rl, "delta")[:-1]
                # print(delta); exit(0)
                dxyz = np.prod(delta)
                x = self.get_grid_data(it, rl, 'x')
                y = self.get_grid_data(it, rl, 'y')
                z = self.get_grid_data(it, rl, 'z')
                x = x[:, :, 0]
                y = y[:, :, 0]

                # apply mask to cut off the horizon
                rho[lapse < 0.15] = 0

                # Exclude region outside refinement levels
                idx = np.isnan(rho)
                rho[idx] = 0.0
                vol[idx] = 0.0
                w_lorentz[idx] = 0.0

                # Compute center of mass
                modes[0].append(dxyz * ne.evaluate("sum(rho * w_lorentz * vol)"))
                Ix = dxyz * ne.evaluate("sum(rho * w_lorentz * vol * x)")
                Iy = dxyz * ne.evaluate("sum(rho * w_lorentz * vol * y)")
                xc = Ix / modes[0][-1]
                yc = Iy / modes[0][-1]
                phi = ne.evaluate("arctan2(y - yc, x - xc)")

                # phi = ne.evaluate("arctan2(y, x)")

                xcs.append(xc)
                ycs.append(yc)

                # Extract modes
                times.append(self.get_time_for_it(it, d1d2d3prof="prof"))
                for m in range(1, mmax + 1):
                    modes[m].append(dxyz * ne.evaluate("sum(rho * w_lorentz * vol * exp(-1j * m * phi))"))

            dfile = h5py.File(fpath, "w")
            dfile.create_dataset("times", data=times)
            dfile.create_dataset("iterations", data=iterations)
            dfile.create_dataset("xc", data=xcs)
            dfile.create_dataset("yc", data=ycs)
            for m in range(mmax + 1):
                group = dfile.create_group("m=%d" % m)
                group["int_phi"] = np.zeros(0, )
                group["int_phi_r"] = np.array(modes[m]).flatten()
                # dfile.create_dataset(("m=%d" % m), data=modes[m])
            dfile.close()
        else:
            return 0

    # ----------------------

    def get_disk_mass(self, it, multiplier=2.):

        self.check_it(it)
        mass = 0.
        for rl in range(self.nlevels):
            density = np.array(self.get_masked_data(it, rl, "density"))
            delta = self.get_grid_data(it, rl, "delta")
            mass += float(multiplier * np.sum(density) * np.prod(delta))
        assert mass > 0.
        return mass

    def get_histogram(self, it, hist_task_dic, multiplier=2.):

        v_n = hist_task_dic["v_n"]
        edge = self.get_edges(it, hist_task_dic)
        histogram = np.zeros(len(edge) - 1)
        _edge = []
        for rl in range(self.nlevels):
            weights = self.get_masked_data(it, rl, "density").flatten() * \
                      np.prod(self.get_grid_data(it, rl, "delta")) * multiplier
            data = self.get_masked_data(it, rl, v_n)
            tmp1, _edge = np.histogram(data, bins=edge, weights=weights)
            histogram+=tmp1
        # print(len(histogram), len(_edge), len(edge))
        # assert len(histogram) == len(edge)
        outarr = np.vstack((0.5*(edge[1:]+edge[:1]), histogram)).T
        return outarr

    def get_correlation(self, it, list_corr_task_dic, multiplier=2.):

        edges = []
        for setup_dictionary in list_corr_task_dic:
            edges.append(self.get_edges(it, setup_dictionary))
        edges = tuple(edges)

        correlation = np.zeros([len(edge) - 1 for edge in edges])

        for rl in range(self.nlevels):
            data = []
            weights = self.get_masked_data(it, rl, "density").flatten() * \
                      np.prod(self.get_grid_data(it, rl, "delta")) * multiplier
            for corr_dic in list_corr_task_dic:
                v_n = corr_dic["v_n"]
                if v_n == 'inv_ang_mom_flux':
                    v_n = 'ang_mom_flux'
                    data.append(-1. * self.get_masked_data(it, rl, v_n).flatten())
                else:
                    data.append(self.get_masked_data(it, rl, v_n).flatten())
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

        for rl in np.arange(start=0, stop=self.nlevels, step=1):
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

            for v_n in v_ns:
                data = self.get_comp_data(it, rl, v_n)
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
                outfile[gname].create_dataset(v_n, data=np.array(data, dtype=np.float32))

        outfile.close()

    def get_dens_modes_for_rl(self, rl=6, mmax = 8):

        import numexpr as ne

        iterations = self.list_iterations # apply limits on it

        times = []
        modes = [[] for m in range(mmax + 1)]
        xcs = []
        ycs = []

        for idx, it in enumerate(iterations):
            print("\tprocessing iteration: {}/{}".format(idx, len(iterations)))
            # get z=0 slice
            lapse = self.get_prof_arr(it, rl, "lapse")[:, :, 0]
            rho = self.get_prof_arr(it, rl, "rho")[:, :, 0]
            vol = self.get_prof_arr(it, rl, "vol")[:, :, 0]
            w_lorentz = self.get_prof_arr(it, rl, "w_lorentz")[:, :, 0]

            delta = self.get_grid_data(it, rl, "delta")[:-1]
            # print(delta); exit(0)
            dxyz = np.prod(delta)
            x = self.get_grid_data(it, rl, 'x')
            y = self.get_grid_data(it, rl, 'y')
            z = self.get_grid_data(it, rl, 'z')
            x = x[:, :, 0]
            y = y[:, :, 0]

            # apply mask to cut off the horizon
            rho[lapse < 0.15] = 0

            # Exclude region outside refinement levels
            idx = np.isnan(rho)
            rho[idx] = 0.0
            vol[idx] = 0.0
            w_lorentz[idx] = 0.0

            # Compute center of mass
            modes[0].append(dxyz * ne.evaluate("sum(rho * w_lorentz * vol)"))
            Ix = dxyz * ne.evaluate("sum(rho * w_lorentz * vol * x)")
            Iy = dxyz * ne.evaluate("sum(rho * w_lorentz * vol * y)")
            xc = Ix / modes[0][-1]
            yc = Iy / modes[0][-1]
            phi = ne.evaluate("arctan2(y - yc, x - xc)")

            # phi = ne.evaluate("arctan2(y, x)")

            xcs.append(xc)
            ycs.append(yc)

            # Extract modes
            times.append(self.get_time_for_it(it, d1d2d3prof="prof"))
            for m in range(1, mmax + 1):
                modes[m].append(dxyz * ne.evaluate("sum(rho * w_lorentz * vol * exp(-1j * m * phi))"))


        return times, iterations, xcs, ycs, modes

    def __delete__(self, instance):
        # instance.dfile.close()
        instance.data_matrix = [[np.zeros(0, )
                                 for x in range(self.nlevels)]
                                 for y in range(len(self.list_all_v_ns))]
        instance.mask_matrix = [np.ones(0, dtype=bool) for x in range(self.nlevels)]


class INTERPOLATE_STORE(MAINMETHODS_STORE):

    def __init__(self, sim, grid_object, symmetry=None):
        """
            fname - of the profile

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

        MAINMETHODS_STORE.__init__(self, sim, symmetry)

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

    def do_interpolate(self, it, v_n):

        tmp = []
        for rl in range(self.nlevels):
            data = self.get_comp_data(it, rl, v_n)
            tmp.append(data)

        xi = self.new_grid.get_xi()
        shape = self.new_grid.get_shape()

        print("\t\tInterpolating: it:{} v_n:{} -> {} grid"
              .format(it, v_n, self.new_grid.grid_type))
        carpet_grid = self.get_grid(it)
        F = Interpolator(carpet_grid, tmp, interp=1)
        arr = F(xi).reshape(shape)

        self.int_data_matrix[self.i_it(it)][self.i_int_v_n(v_n)] = arr

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

    def __init__(self, sim, grid_object, symmetry=None):

        INTERPOLATE_STORE.__init__(self, sim, grid_object, symmetry)

    def save_new_grid(self, it):
        self.check_it(it)

        grid_type = self.new_grid.grid_info['type']

        if not os.path.exists(Paths.ppr_sims + self.sim + "/res_3d/"):
            os.makedirs(Paths.ppr_sims + self.sim + "/res_3d/")

        path = Paths.ppr_sims + self.sim + "/res_3d/" + str(it) + '/'

        if os.path.isfile(path + grid_type + '_grid.h5'):
            os.remove(path + grid_type + '_grid.h5')

        outfile = h5py.File(path + grid_type + '_grid.h5', "w")

        if not os.path.exists(path):
            os.makedirs(path)

        # print("Saving grid...")
        for v_n in self.list_int_grid_v_ns:
            outfile.create_dataset(v_n, data=self.new_grid.get_int_grid(v_n))
        outfile.close()

    def save_int_v_n(self, it, v_n, overwrite=False):

        self.check_it(it)

        path = Paths.ppr_sims + self.sim + "/res_3d/"
        if not os.path.isdir(path):
            os.mkdir(path)

        path = Paths.ppr_sims + self.sim + "/res_3d/" + str(it) + '/'
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

    def save_vtk_file(self, it, v_n_s, overwrite=False, private_dir="vtk"):

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

        path = Paths.ppr_sims + self.sim + "/res_3d/" + str(it) + '/'
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


""" --- --- DATA PROCESSING OLD --- --- """
'''
class LOAD_PROFILE:

    def __init__(self, fname ,symmetry=None):

        self.symmetry = symmetry
        self.nlevels = 7
        self.profile = fname
        self.dfile = h5py.File(fname, "r")
        group_0 = self.dfile["reflevel={}".format(0)]
        self.time = group_0.attrs["time"] * 0.004925794970773136 * 1e-3 # [sec]
        self.iteration = group_0.attrs["iteration"]
        print("\t\t symmetry: {}".format(self.symmetry))
        print("\t\t time: {}".format(self.time))
        print("\t\t iteration: {}".format(self.iteration))
        self.grid = self.read_carpet_grid(self.dfile)

        # print("grid: {}".format(self.grid))

        self.list_prof_v_ns = [
                             "rho", "w_lorentz", "vol",  # basic
                             "press", "eps", "lapse",    # basic + lapse
                             "velx", "vely", "velz",     # velocities
                             "gxx", "gxy", "gxz", "gyy", "gyz", "gzz",  # metric
                             "betax", "betay", "betaz",  # shift components
                             'temp', 'Ye']

        self.list_grid_v_ns = ["x", "y", "z", "delta"]

        if self.symmetry == "pi" and not str(self.profile).__contains__("_PI"):
            raise NameError("profile {} does not seem to have a pi symmetry. Check"
                            .format(self.profile))

    def read_carpet_grid(self, dfile):
        import scidata.carpet.grid as grid
        L = []
        for il in range(self.nlevels):
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
            elif self.symmetry == None:
                origin = np.array(group.attrs["extent"][0::2])
            else:
                raise NameError("symmetry is not recognized in a parfile. Set None or pi. Given:{}"
                                .format(self.symmetry))
            level.origin = origin
            # print("sym: {} origin {} ".format(self.symmetry, origin)); exit()

            # level.n = np.array(group["rho"].shape, dtype=np.int32)
            level.n = np.array(self.get_prof_arr(il, 'rho').shape, dtype=np.int32)
            level.rlevel = il
            L.append(level)
        return grid.grid(sorted(L, key=lambda x: x.rlevel))

    def get_prof_arr(self, rl, v_n):
        group = self.dfile["reflevel={}".format(rl)]
        try:
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
            raise ValueError('v_n:{} not in file:{}'.format(v_n, self.profile))
        return arr

    def get_prof_delta(self, rl):
        return self.grid[rl].delta

    def get_prof_x_y_z(self, rl):
        x, y, z = self.grid.mesh()[rl]

        # print("rl: {} x:({}):[{:.1f},{:.1f}] y:({}):[{:.1f},{:.1f}] z:({}):[{:.1f},{:.1f}]"
        #       .format(rl, x.shape, x[0, 0, 0], x[-1, 0, 0],
        #               y.shape, y[0, 0, 0], y[0, -1, 0],
        #               z.shape, z[0, 0, 0], z[0, 0, -1]))

        # if self.symmetry == 'pi':
        #     x = np.delete(x, 0, axis=0)
        #     x = np.delete(x, 0, axis=0)
        #     x = np.delete(x, 0, axis=0)
        #
        #     x_n = x * -1
        #     x_n = x_n[::-1, :, :]
        #     x = np.dstack((x_n.T, x.T)).T
        #
        #
        #     y = np.delete(y, 0, axis=0)
        #     y = np.delete(y, 0, axis=0)
        #     y = np.delete(y, 0, axis=0)
        #
        #     y_n = y
        #     # y_n = y_n[::-1, :, :]
        #     y = np.dstack((y_n.T, y.T)).T
        #
        #     z = np.delete(z, 0, axis=0)
        #     z = np.delete(z, 0, axis=0)
        #     z = np.delete(z, 0, axis=0)
        #
        #     z_n = z
        #     # z_n = z_n[::-1, :, :]
        #     z = np.dstack((z_n.T, z.T)).T
        #
        #
        #
        #     print("rl: {} x:({}):[{:.1f},{:.1f}] y:({}):[{:.1f},{:.1f}] z:({}):[{:.1f},{:.1f}]"
        #           .format(rl, x.shape, x[0, 0, 0], x[-1, 0, 0],
        #                   y.shape, y[0, 0, 0], y[0, -1, 0],
        #                   z.shape, z[0, 0, 0], z[0, 0, -1]))

        # exit(1)

        return x, y, z

    def __delete__(self, instance):
        instance.dfile.close()

class COMPUTE_STORE(LOAD_PROFILE):

    def __init__(self, fname, symmetry=None):
        LOAD_PROFILE.__init__(self, fname, symmetry)

        self.list_comp_v_ns = [
            "density", "vup", "metric", "shift",
            "enthalpy", "shvel", "u_0",
            "vlow", "vphi", "vr",
            "dens_unb_geo", "dens_unb_bern", "dens_unb_garch",
            "ang_mom", "ang_mom_flux",
            "theta", "r", "phi" # assumes cylindircal coordinates. r = x^2 + y^2
        ]

        self.list_all_v_ns = self.list_prof_v_ns + \
                             self.list_grid_v_ns + \
                             self.list_comp_v_ns

        self.data_matrix = [[np.zeros(0,)
                             for x in range(self.nlevels)]
                             for y in range(len(self.list_all_v_ns))]

    def check_v_n(self, v_n):
        if v_n not in self.list_all_v_ns:
            raise NameError("v_n:{} not in the v_n list \n{}"
                            .format(v_n, self.list_all_v_ns))

    def i_v_n(self, v_n):
        self.check_v_n(v_n)
        return int(self.list_all_v_ns.index(v_n))

    def set_data(self, rl, v_n, arr):
        self.data_matrix[self.i_v_n(v_n)][rl] = arr

    def extract_data(self, rl, v_n):
        data = self.get_prof_arr(rl, v_n)
        self.data_matrix[self.i_v_n(v_n)][rl] = data

    def extract_grid_data(self, rl, v_n):
        if v_n in ["x", "y", "z"]:
            x, y, z = self.get_prof_x_y_z(rl)
            self.data_matrix[self.i_v_n("x")][rl] = x
            self.data_matrix[self.i_v_n("y")][rl] = y
            self.data_matrix[self.i_v_n("z")][rl] = z
        elif v_n == "delta":
            delta = self.get_prof_delta(rl)
            self.data_matrix[self.i_v_n("x")][rl] = delta
        else:
            raise NameError("Grid variable {} not recognized".format(v_n))

    def compute_data(self, rl, v_n):

        if v_n == 'density':
            arr = FORMULAS.density(self.get_comp_data(rl, "rho"),
                                   self.get_comp_data(rl, "w_lorentz"),
                                   self.get_comp_data(rl, "vol"))

        elif v_n == 'vup':
            arr = FORMULAS.vup(self.get_comp_data(rl, "velx"),
                               self.get_comp_data(rl, "vely"),
                               self.get_comp_data(rl, "velz"))

        elif v_n == 'metric':  # gxx, gxy, gxz, gyy, gyz, gzz
            arr = FORMULAS.metric(self.get_comp_data(rl, "gxx"),
                                  self.get_comp_data(rl, "gxy"),
                                  self.get_comp_data(rl, "gxz"),
                                  self.get_comp_data(rl, "gyy"),
                                  self.get_comp_data(rl, "gyz"),
                                  self.get_comp_data(rl, "gzz"))

        elif v_n == 'shift':
            arr = FORMULAS.shift(self.get_comp_data(rl, "betax"),
                                 self.get_comp_data(rl, "betay"),
                                 self.get_comp_data(rl, "betaz"))

        elif v_n == 'enthalpy':
            arr = FORMULAS.enthalpy(self.get_comp_data(rl, "eps"),
                                    self.get_comp_data(rl, "press"),
                                    self.get_comp_data(rl, "rho"))

        elif v_n == 'shvel':
            arr = FORMULAS.shvel(self.get_comp_data(rl, "shift"),
                                 self.get_comp_data(rl, "vlow"))

        elif v_n == 'u_0':
            arr = FORMULAS.u_0(self.get_comp_data(rl, "w_lorentz"),
                               self.get_comp_data(rl, "shvel"),  # not input
                               self.get_comp_data(rl, "lapse"))

        elif v_n == 'vlow':
            arr = FORMULAS.vlow(self.get_comp_data(rl, "metric"),
                                self.get_comp_data(rl, "vup"))

        elif v_n == 'vphi':
            arr = FORMULAS.vphi(self.get_comp_data(rl, "x"),
                                self.get_comp_data(rl, "y"),
                                self.get_comp_data(rl, "vlow"))

        elif v_n == 'vr':
            arr = FORMULAS.vr(self.get_comp_data(rl, "x"),
                              self.get_comp_data(rl, "y"),
                              self.get_comp_data(rl, "r"),
                              self.get_comp_data(rl, "vup"))

        elif v_n == "r":
            arr = FORMULAS.r(self.get_comp_data(rl, "x"),
                             self.get_comp_data(rl, "y"))

        elif v_n == "phi":
            arr = FORMULAS.phi(self.get_comp_data(rl, "x"),
                               self.get_comp_data(rl, "y"))

        elif v_n == 'theta':
            arr = FORMULAS.theta(self.get_comp_data(rl, "r"),
                                 self.get_comp_data(rl, "z"))

        elif v_n == 'ang_mom':
            arr = FORMULAS.ang_mom(self.get_comp_data(rl, "rho"),
                                   self.get_comp_data(rl, "eps"),
                                   self.get_comp_data(rl, "press"),
                                   self.get_comp_data(rl, "w_lorentz"),
                                   self.get_comp_data(rl, "vol"),
                                   self.get_comp_data(rl, "vphi"))

        elif v_n == 'ang_mom_flux':
            arr = FORMULAS.ang_mom_flux(self.get_comp_data(rl, "ang_mom"),
                                        self.get_comp_data(rl, "lapse"),
                                        self.get_comp_data(rl, "vr"))

        elif v_n == 'dens_unb_geo':
            arr = FORMULAS.dens_unb_geo(self.get_comp_data(rl, "u_0"),
                                        self.get_comp_data(rl, "rho"),
                                        self.get_comp_data(rl, "w_lorentz"),
                                        self.get_comp_data(rl, "vol"))

        elif v_n == 'dens_unb_bern':
            arr = FORMULAS.dens_unb_bern(self.get_comp_data(rl, "enthalpy"),
                                         self.get_comp_data(rl, "u_0"),
                                         self.get_comp_data(rl, "rho"),
                                         self.get_comp_data(rl, "w_lorentz"),
                                         self.get_comp_data(rl, "vol"))

        elif v_n == 'dens_unb_garch':
            arr = FORMULAS.dens_unb_garch(self.get_comp_data(rl, "enthalpy"),
                                          self.get_comp_data(rl, "u_0"),
                                          self.get_comp_data(rl, "lapse"),
                                          self.get_comp_data(rl, "press"),
                                          self.get_comp_data(rl, "rho"),
                                          self.get_comp_data(rl, "w_lorentz"),
                                          self.get_comp_data(rl, "vol"))

        else:
            raise NameError("No method found for v_n:{} rl:{} Add entry to 'compute()'"
                            .format(v_n, rl))

        self.data_matrix[self.i_v_n(v_n)][rl] = arr

    def is_available(self, rl, v_n):
        self.check_v_n(v_n)
        data = self.data_matrix[self.i_v_n(v_n)][rl]
        if len(data) == 0:
            if v_n in self.list_prof_v_ns:
                self.extract_data(rl, v_n)
            elif v_n in self.list_grid_v_ns:
                self.extract_grid_data(rl, v_n)
            elif v_n in self.list_comp_v_ns:
                self.compute_data(rl, v_n)
            else:
                raise NameError("v_n is not recognized: '{}' [COMPUTE STORE]".format(v_n))

    def get_comp_data(self, rl, v_n):
        self.check_v_n(v_n)
        self.is_available(rl, v_n)

        return self.data_matrix[self.i_v_n(v_n)][rl]

    def __delete__(self, instance):
        instance.dfile.close()
        instance.data_matrix = [[np.zeros(0, )
                             for x in range(self.nlevels)]
                            for y in range(len(self.list_all_v_ns))]

class MASK_STORE(COMPUTE_STORE):

    def __init__(self, fname, symmetry=None):
        COMPUTE_STORE.__init__(self, fname, symmetry)

        rho_const = 6.176269145886162e+17
        self.mask_setup = {'rm_rl': True,  # REMOVE previouse ref. level from the next
                           'rho': [6.e4 / rho_const, 1.e13 / rho_const],  # REMOVE atmo and NS
                           'lapse': [0.15, 1.]} # remove apparent horizon

        self.mask_matrix = [np.ones(0, dtype=bool) for x in range(self.nlevels)]

        self.list_mask_v_n = ["x", "y", "z"]


    def compute_mask(self):

        nlevelist = np.arange(self.nlevels, 0, -1) - 1

        x = []
        y = []
        z = []

        for ii, rl in enumerate(nlevelist):
            x.append(self.get_comp_data(rl, "x")[3:-3, 3:-3, 3:-3])
            y.append(self.get_comp_data(rl, "y")[3:-3, 3:-3, 3:-3])
            z.append(self.get_comp_data(rl, "z")[3:-3, 3:-3, 3:-3])
            mask = np.ones(x[ii].shape, dtype=bool)
            if ii > 0 and self.mask_setup["rm_rl"]:
                x_ = (x[ii][:, :, :] <= x[ii - 1][:, 0, 0].max()) & (
                        x[ii][:, :, :] >= x[ii - 1][:, 0, 0].min())
                y_ = (y[ii][:, :, :] <= y[ii - 1][0, :, 0].max()) & (
                        y[ii][:, :, :] >= y[ii - 1][0, :, 0].min())
                z_ = (z[ii][:, :, :] <= z[ii - 1][0, 0, :].max()) & (
                        z[ii][:, :, :] >= z[ii - 1][0, 0, :].min())
                mask = mask & np.invert((x_ & y_ & z_))

            for v_n in self.mask_setup.keys()[1:]:
                self.check_v_n(v_n)
                if len(self.mask_setup[v_n]) != 2:
                    raise NameError("Error. 2 values are required to set a limit. Give {} for {}"
                                    .format(self.mask_setup[v_n], v_n))
                arr_1 = self.get_comp_data(rl, v_n)[3:-3, 3:-3, 3:-3]
                min_val = float(self.mask_setup[v_n][0])
                max_val = float(self.mask_setup[v_n][1])
                mask_i = (arr_1 > min_val) & (arr_1 < max_val)
                mask = mask & mask_i
                del arr_1
                del mask_i

            self.mask_matrix[rl] = mask

    def is_mask_available(self, rl):
        mask = self.mask_matrix[rl]
        if len(mask) == 0:
            self.compute_mask()

    def get_masked_data(self, rl, v_n):
        self.check_v_n(v_n)
        self.is_available(rl, v_n)
        self.is_mask_available(rl)
        data = np.array(self.get_comp_data(rl, v_n))[3:-3, 3:-3, 3:-3]
        mask = self.mask_matrix[rl]
        return data[mask]

    def __delete__(self, instance):
        instance.dfile.close()
        instance.data_matrix = [[np.zeros(0, )
                                 for x in range(self.nlevels)]
                                 for y in range(len(self.list_all_v_ns))]
        instance.mask_matrix = [np.ones(0, dtype=bool) for x in range(self.nlevels)]

class MAINMETHODS_STORE(MASK_STORE):

    def __init__(self, fname, sim, symmetry=None):

        MASK_STORE.__init__(self, fname, symmetry)

        self.sim = sim

        # "v_n": "temp", "edges": np.array()
        ''''''
        # "v_n": "temp", "points: number, "scale": "log", (and "min":number, "max":number)

        rho_const = 6.176269145886162e+17
        self.corr_task_dic_temp_ye = [
            # {"v_n": "rho",  "edges": 10.0 ** np.linspace(4.0, 16.0, 500) / rho_const},  # not in CGS :^
            {"v_n": "temp", "edges": 10.0 ** np.linspace(-2, 2, 300)},
            {"v_n": "Ye",   "edges": np.linspace(0, 0.5, 300)}
        ]

        self.corr_task_dic_rho_ye = [
            # {"v_n": "temp", "edges": 10.0 ** np.linspace(-2, 2, 300)},
            {"v_n": "rho",  "edges": 10.0 ** np.linspace(4.0, 13.0, 500) / rho_const},  # not in CGS :^
            {"v_n": "Ye",   "edges": np.linspace(0, 0.5, 300)}
        ]

        self.corr_task_dic_rho_theta = [
            {"v_n": "rho", "edges": 10.0 ** np.linspace(4.0, 13.0, 500) / rho_const},  # not in CGS :^
            {"v_n": "theta", "edges": np.linspace(0, 0.5*np.pi, 300)}
        ]

        self.corr_task_dic_rho_r = [
            {"v_n": "rho", "edges": 10.0 ** np.linspace(4.0, 13.0, 500) / rho_const},  # not in CGS :^
            {"v_n": "r", "edges": np.linspace(0, 100, 500)}
        ]

        self.corr_task_dic_rho_ang_mom = [
            {"v_n": "rho", "edges": 10.0 ** np.linspace(4.0, 13.0, 500) / rho_const},  # not in CGS :^
            {"v_n": "ang_mom", "points": 300, "scale": "log", "min":1e-9} # find min, max yourself
        ]

        self.corr_task_dic_rho_ang_mom_flux = [
            {"v_n": "rho", "edges": 10.0 ** np.linspace(4.0, 13.0, 500) / rho_const},  # not in CGS :^
            {"v_n": "ang_mom_flux", "points": 300, "scale": "log", "min":1e-12}
        ]

        self.corr_task_dic_rho_dens_unb_bern = [
            {"v_n": "rho", "edges": 10.0 ** np.linspace(4.0, 13.0, 500) / rho_const},  # not in CGS :^
            {"v_n": "dens_unb_bern", "edges": 10.0 ** np.linspace(-12., -6., 300)}
        ]

        self.corr_task_dic_rho_dens_unb_bern = [
            {"v_n": "rho", "edges": 10.0 ** np.linspace(4.0, 13.0, 500) / rho_const},  # not in CGS :^
            {"v_n": "dens_unb_bern", "edges": 10.0 ** np.linspace(-12., -6., 300)}
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

        self.corr_task_dic_r_phi = [
            {"v_n": "r", "edges": np.linspace(0, 50, 500)},
            {"v_n": "phi", "edges": np.linspace(-np.pi, np.pi, 500)},
        ]

        # -- 3D

        self.corr_task_dic_r_phi_ang_mom_flux = [
            {"v_n": "r", "edges": np.linspace(0, 100, 50)},
            {"v_n": "phi", "edges": np.linspace(-np.pi, np.pi, 300)},
            {"v_n": "ang_mom_flux", "points": 500, "scale": "log", "min": 1e-12}
        ]

    def get_total_mass(self, multiplier=2., save=False):
        mass = 0.
        for rl in range(self.nlevels):
            density = np.array(self.get_masked_data(rl, "density"))
            delta = self.get_prof_delta(rl)
            mass += float(multiplier * np.sum(density) * np.prod(delta))

        print("it:{} mass:{:3f}Msun".format(self.iteration, mass))

        if save:
            path = Paths.ppr_sims + self.sim + "/res_3d/"  + str(self.iteration) + '/'
            fname = "disk_mass.txt".format(self.iteration)

            if not os.path.exists(path):
                os.makedirs(path)

            np.savetxt(path + fname, np.array([mass]), fmt='%.5f')

        return mass

    def get_min_max(self, v_n):
        # self.check_v_n(v_n)
        min_, max_ = [], []
        for rl in range(self.nlevels):

            if v_n == 'inv_ang_mom_flux':
                v_n = 'ang_mom_flux'
                data = -1. * self.get_masked_data(rl, v_n)
            else:
                data = self.get_masked_data(rl, v_n)
            min_.append(data.min())
            max_.append(data.max())
        min_ = np.array(min_)
        max_ = np.array(max_)
        return min_.min(), max_.max()
            # print("rl:{} min:{} max:{}".format(rl, data.min(), data.max()))

    def get_edges(self, corr_task_dic):

        dic = dict(corr_task_dic)

        if "edges" in dic.keys():
            return dic["edges"]

        if "points" in dic.keys() and "scale" in dic.keys():
            min_, max_ = self.get_min_max(dic["v_n"])
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

    def get_correlation(self, corr_task_dic, multiplier=2., save=False):

        v_ns = []
        edges = []
        for setup_dictionary in corr_task_dic:
            v_ns.append(setup_dictionary["v_n"])
            edges.append(self.get_edges(setup_dictionary))
        edges = tuple(edges)

        correlation = np.zeros([len(edge) - 1 for edge in edges])
        for rl in range(self.nlevels):
            data = []
            weights = self.get_masked_data(rl, "density").flatten() * \
                      np.prod(self.get_prof_delta(rl)) * multiplier
            for i_vn, v_n in enumerate(v_ns):

                if v_n == 'inv_ang_mom_flux':
                    v_n = 'ang_mom_flux'
                    data.append(-1. * self.get_masked_data(rl, v_n).flatten())
                else:
                    data.append(self.get_masked_data(rl, v_n).flatten())


            data = tuple(data)
            tmp, _ = np.histogramdd(data, bins=edges, weights=weights)
            correlation += tmp

        if save:
            path = Paths.ppr_sims + self.sim + "/res_3d/" + str(self.iteration) + '/'
            fname = "corr_".format(self.iteration)
            for v_n in v_ns:
                fname += v_n
                if v_n != v_ns[-1]:
                    fname += '_'
            fname += '.h5'

            if not os.path.exists(path):
                os.makedirs(path)

            outfile = h5py.File(path + fname, "w")
            for v_n, edge in zip(v_ns, edges):
                outfile.create_dataset(v_n, data=edge)
            outfile.create_dataset("mass", data=correlation)
            outfile.close()

        return correlation




    def __delete__(self, instance):
        instance.dfile.close()
        instance.data_matrix = [[np.zeros(0, )
                                 for x in range(self.nlevels)]
                                 for y in range(len(self.list_all_v_ns))]
        instance.mask_matrix = [np.ones(0, dtype=bool) for x in range(self.nlevels)]

class INTERPOLATE_STORE(MAINMETHODS_STORE):

    def __init__(self, fname, sim, grid_object, symmetry=None):
        """
            fname - of the profile

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

        MAINMETHODS_STORE.__init__(self, fname, sim, symmetry)

        self.new_grid = grid_object

        self.list_int_grid_v_ns = grid_object.list_int_grid_v_ns
        self.list_int_v_ns = self.list_prof_v_ns + \
                             self.list_comp_v_ns + \
                             self.list_grid_v_ns

        self.int_data_matrix = [np.zeros(0,) for y in range(len(self.list_int_v_ns))]


    def check_int_v_n(self, v_n):
        if v_n not in self.list_int_v_ns:
            raise NameError("v_n: '{}' not in the v_n list \n{}"
                            .format(v_n, self.list_int_v_ns))

    def i_int_v_n(self, v_n):
        self.check_int_v_n(v_n)
        return int(self.list_int_v_ns.index(v_n))

    def do_append_grid_var(self, v_n):
        self.int_data_matrix[self.i_int_v_n(v_n)] = \
            self.new_grid.get_int_grid(v_n)

    def do_interpolate(self, v_n):

        tmp = []
        for rl in range(self.nlevels):
            data = self.get_comp_data(rl, v_n)
            tmp.append(data)

        xi = self.new_grid.get_xi()
        shape = self.new_grid.get_shape()

        print("\t\tInterpolating: {}".format(v_n))
        F = Interpolator(self.grid, tmp, interp=1)
        arr = F(xi).reshape(shape)

        self.int_data_matrix[self.i_int_v_n(v_n)] = arr

    def is_data_interpolated(self, v_n):

        if len(self.int_data_matrix[self.i_int_v_n(v_n)]) == 0:
            if v_n in self.list_int_grid_v_ns:
                self.do_append_grid_var(v_n)
            else:
                self.do_interpolate(v_n)




    def get_int(self, v_n):
        self.check_int_v_n(v_n)
        self.is_data_interpolated(v_n)
        return self.int_data_matrix[self.i_int_v_n(v_n)]

class INTMETHODS_STORE(INTERPOLATE_STORE):

    def __init__(self, fname, sim, grid_object, symmetry=None):

        INTERPOLATE_STORE.__init__(self, fname, sim, grid_object, symmetry)

    def save_new_grid(self):

        grid_type = self.new_grid.grid_info['type']

        path = Paths.ppr_sims + self.sim + "/res_3d/" + str(self.iteration) + '/'
        outfile = h5py.File(path + grid_type + '_grid.h5', "w")

        if not os.path.exists(path):
            os.makedirs(path)

        # print("Saving grid...")
        for v_n in self.list_int_grid_v_ns:
            outfile.create_dataset(v_n, data=self.new_grid.get_int_grid(v_n))
        outfile.close()

    def save_int_v_n(self, v_n, overwrite=False):

        path = Paths.ppr_sims + self.sim + "/res_3d/"
        if not os.path.isdir(path):
            os.mkdir(path)
        path = Paths.ppr_sims + self.sim + "/res_3d/" + str(self.iteration) + '/'
        if not os.path.isdir(path):
            os.mkdir(path)
        grid_type = self.new_grid.grid_type

        fname = path + grid_type + '_' + v_n + '.h5'

        if os.path.isfile(fname):
            if overwrite:
                print("File: {} already exists -- overwriting".format(fname))
                os.remove(fname)
                outfile = h5py.File(fname, "w")
                outfile.create_dataset(v_n, data=self.get_int(v_n))
                outfile.close()
            else:
                print("File: {} already exists -- skipping".format(fname))
        else:
            outfile = h5py.File(fname, "w")
            outfile.create_dataset(v_n, data=self.get_int(v_n))
            outfile.close()

    def save_vtk_file(self, v_n_s, overwrite=False, private_dir="vtk"):

        # This requires PyEVTK to be insalled. You can get it with:
        # $ hg clone https://bitbucket.org/pauloh/pyevtk PyEVTK
        from evtk.hl import gridToVTK

        if self.new_grid.grid_type != "cart":
            raise AttributeError("only 'cart' grid is supported")

        path = Paths.ppr_sims + self.sim + "/res_3d/" + str(self.iteration) + '/'
        if not os.path.isdir(path):
            os.mkdir(path)
        if private_dir != None and private_dir != '':
            path = path + private_dir + '/'
        if not os.path.isdir(path):
            os.mkdir(path)
        fname = "iter_" + str(self.iteration).zfill(10)
        fpath = path + fname

        if os.path.isfile(fpath) and not overwrite:
            print("Skipping it:{} ".format(self.iteration))
        else:

            xf = self.new_grid.get_int_grid("xf")
            yf = self.new_grid.get_int_grid("yf")
            zf = self.new_grid.get_int_grid("zf")

            celldata = {}
            for v_n in v_n_s:
                celldata[str(v_n)] = self.get_int(v_n)
            gridToVTK(fpath, xf, yf, zf, cellData=celldata)
'''

""" --- --- LOADING & PostPROCESSING RESILTS --- --- """

class LOAD_RES_CORR(LOAD_ITTIME):

    def __init__(self, sim):

        LOAD_ITTIME.__init__(self, sim)

        self.sim = sim

        self.list_iterations = get_list_iterationsfrom_res_3d(sim)
        # self.times = interpoate_time_form_it(self.list_iterations, Paths.gw170817+sim+'/')
        self.times = []
        for it in self.list_iterations:
            self.times.append(self.get_time_for_it(it, d1d2d3prof="prof"))
        self.times = np.array(self.times)

        self.list_corr_v_ns = ["temp", "Ye", "rho", "theta", "r", "phi",
                               "ang_mom", "ang_mom_flux", "dens_unb_bern",
                               "inv_ang_mom_flux", 'vr', 'velz', 'vely', 'velx'
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
        # self.check_v_n(v_n)
        fpath = Paths.ppr_sims + self.sim + "/res_3d/" + str(it) + "/corr_" + v_n + ".h5"
        if not os.path.isfile(fpath):
            raise IOError("Correlation file not found:\n{}".format(fpath))
        return fpath

    def load_corr_file_old(self, it, v_n):

        v_n = str(v_n)

        fpath = self.get_corr_fpath(it, v_n)

        dfile = h5py.File(fpath, "r")

        v_ns_in_data = []
        for v_n_ in dfile:
            v_ns_in_data.append(v_n_)

        if not "mass" in v_ns_in_data:
            raise NameError("mass is not found in file:{}".format(fpath))

        if len(v_ns_in_data) > 3:
            raise NameError("More than 3 datasets found in corr file: {}".format(fpath))

        v_ns_in_data.remove("mass")

        for v_n__ in v_ns_in_data:
            if not v_n__ in v_n:
                raise NameError("in_data_v_n: {} is not in corr name v_n: {}"
                                .format(v_n__, v_n))


        part1 = v_n.split(v_ns_in_data[0])
        part2 = v_n.split(v_ns_in_data[1])
        if v_ns_in_data[0] + '_' == part1[0]:
            v_n1 = v_ns_in_data[0]
            v_n2 = v_ns_in_data[1]
        elif '_' + v_ns_in_data[0] == part1[1]:
            v_n1 = v_ns_in_data[1]
            v_n2 = v_ns_in_data[0]
        elif v_ns_in_data[1] + '_' == part1[0]:
            v_n1 = v_ns_in_data[1]
            v_n2 = v_ns_in_data[0]
        elif '_' + v_ns_in_data[1] == part1[1]:
            v_n1 = v_ns_in_data[0]
            v_n2 = v_ns_in_data[1]
        else:
            print("v_n: {}".format(v_n))
            print("v_n_in_data: {}".format(v_ns_in_data))
            print("v_n.split({}): {}".format(v_ns_in_data[0], part1))
            print("v_n.split({}): {}".format(v_ns_in_data[1], part2))
            print("v_ns_in_data[0]: {}".format(v_ns_in_data[0]))
            print("v_ns_in_data[1]: {}".format(v_ns_in_data[1]))
            raise NameError("Get simpler for f*ck sake...")

        print("v_n1: {}".format(v_n1))
        print("v_n2: {}".format(v_n2))
        edge_x = np.array(dfile[v_n1])
        edge_y = np.array(dfile[v_n2])
        mass = np.array(dfile["mass"]).T

        arr_x = 0.5 * (edge_x[1:] + edge_x[:-1])
        arr_y = 0.5 * (edge_y[1:] + edge_y[:-1])

        result = combine(arr_x, arr_y, mass)

        self.corr_matrix[self.i_it(it)][self.i_v_n(v_n)] = result

    def load_corr_file(self, it, v_n_x, v_n_y):

        v_n_x = str(v_n_x)
        v_n_y = str(v_n_y)

        self.check_v_n(v_n_x)
        self.check_v_n(v_n_y)

        # check if the direct file exists or the inverse
        fpath_direct = Paths.ppr_sims + self.sim + "/res_3d/" + str(it) + "/corr_" + v_n_x + '_' + v_n_y + ".h5"
        fpath_inverse = Paths.ppr_sims + self.sim + "/res_3d/" + str(it) + "/corr_" + v_n_y + '_' + v_n_x + ".h5"
        if os.path.isfile(fpath_direct):
            fpath = fpath_direct
        elif os.path.isfile(fpath_inverse):
            fpath = fpath_inverse
        else:
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
        result = combine(arr_x, arr_y, mass)
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

        idx = find_nearest_index(self.times, t)
        return self.list_iterations[idx]

    def load_corr3d(self, it, v_n_x, v_n_y, v_n_z):

        v_n_x = str(v_n_x)
        v_n_y = str(v_n_y)
        v_n_z = str(v_n_z)

        self.check_v_n(v_n_x)
        self.check_v_n(v_n_y)
        self.check_v_n(v_n_z)

        fpath_direct = Paths.ppr_sims + self.sim + "/res_3d/" + str(it) + "/corr_" + v_n_x + '_' + v_n_y + '_' + v_n_z + ".h5"
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


class LOAD_INT_DATA(LOAD_ITTIME):

    def __init__(self, sim, grid_object):
        print("Warning. LOAD_INT_DATA is using only the '.grid_type' and '.list_int_grid_v_ns'\n"
              " It does not use the grid itself. Instead it loads the 'grid_type'_grid.h5 file")

        LOAD_ITTIME.__init__(self, sim)

        self.sim = sim

        self.list_iterations = list(get_list_iterationsfrom_res_3d(sim))
        self.times = []
        for it in self.list_iterations:
            self.times.append(self.get_time_for_it(it, d1d2d3prof="prof"))
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
            get_it_from_itdir(find_itdir_with_grid(sim, "{}_grid.h5"
                                                   .format(grid_object.grid_type)))

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

        path = Paths.ppr_sims + self.sim + "/res_3d/" + str(int(it)) + '/'
        fname = path + self.grid_type + '_grid.h5'
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

        path = Paths.ppr_sims + self.sim + "/res_3d/" + str(int(it)) + '/'
        fname = path + self.grid_type + '_' + v_n + ".h5"
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

        idx = find_nearest_index(self.times, t)
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


class COMPUTE_STORE_DESITYMODES(LOAD_INT_DATA):

    def __init__(self, sim):

        LOAD_INT_DATA.__init__(self, sim)

        self.gen_set = {
            'v_n': 'density',
            'v_n_r': 'r_cyl',
            'v_n_dr': 'dr_cyl',
            'v_n_phi': 'phi_cyl',
            'v_n_dphi': 'dphi_cyl',
            'v_n_dz': 'dz_cyl',
            'iterations': 'all',
            'do_norm':True,
            'm_to_norm': 0,
            'outfname': 'density_modes_int_lapse15.h5',
            'outdir': Paths.ppr_sims + sim + '/res_3d/',
            'lapse_mask': 0.15
        }

        self.list_modes = [0, 1, 2, 3, 4, 5, 6]
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

    def is_computed(self, it, mode, v_n):

        if len(self.data_dm_matrix[self.i_it(it)][self.i_mode(mode)][self.i_dm_v_n(v_n)]) == 0:
            self.compute_density_mode(it, mode)

    def get_density_mode(self, it, mode, v_n):
        self.check_it(it)
        self.check_mode(mode)
        self.check_dm_v_n(v_n)
        self.is_computed(it, mode, v_n)
        return self.data_dm_matrix[self.i_it(it)][self.i_mode(mode)][self.i_dm_v_n(v_n)]


class LOAD_DENSITY_MODES:

    def __init__(self, sim):

        self.sim = sim

        self.gen_set = {
            'maximum_modes': 50,
            'fname' :  Paths.ppr_sims + sim + '/res_3d/' + "density_modes.h5",
            'int_phi': 'int_phi',
            'int_phi_r': 'int_phi_r',
            'r_cyl': 'r_cyl',
            'times': 'times',
            'iterations':'iterations'
        }



        self.n_of_modes_max = 50
        self.list_data_v_ns = ["int_phi", "int_phi_r"]
        self.list_grid_v_ns = ["r_cyl", "times", "iterations"]

        self.data_dm_matrix = [[np.zeros(0,)
                              for k in range(len(self.list_data_v_ns))]
                              for z in range(self.n_of_modes_max)]

        self.grid_matrix = [np.zeros(0,)
                              for k in range(len(self.list_grid_v_ns))]

        self.list_modes = []

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

    def load_density_modes(self):

        dfile = h5py.File(self.gen_set['fname'], "r")

        for v_n in dfile:
            # extracting data for all modes
            if str(v_n).__contains__("m="):
                mode = int(v_n.split("m=")[-1])
                self.list_modes.append(mode)
                group = dfile[v_n]
                for v_n_ in group:
                    if str(v_n_) in self.list_data_v_ns:
                        self.data_dm_matrix[self.i_mode(mode)][self.i_v_n(v_n_)] = np.array(group[v_n_])
                    else:
                        raise NameError("{} group has a v_n: {} that is not in the data list:\n{}"
                                        .format(v_n, v_n_, self.list_data_v_ns))
                if len(self.list_modes) > self.n_of_modes_max - 1:
                    raise ValueError("too many modes (>{})".format(self.n_of_modes_max))

            # extracting grid data, for overall
            else:
                if v_n in self.list_grid_v_ns:
                    self.grid_matrix[self.i_v_n(v_n)] = np.array(dfile[v_n])
                else:
                    NameError("dfile v_n: {} not in list of grid v_ns\n{}"
                                    .format(v_n, self.list_grid_v_ns))

        print("  modes: {}".format(self.list_modes))

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

""" --- --- LOAD PROFILE.XY.XZ --- --- --- """

class LOAD_PROFILE_XYXZ(LOAD_ITTIME):

    def __init__(self, sim):

        LOAD_ITTIME.__init__(self, sim)

        self.nlevels = 7

        self.sim = sim

        self.list_iterations = get_list_iterationsfrom_res_3d(sim)
        # self.times = interpoate_time_form_it(self.list_iterations, Paths.gw170817+sim+'/')
        self.times = []
        for it in self.list_iterations:
            self.times.append(self.get_time_for_it(it, d1d2d3prof="prof"))
        self.times = np.array(self.times)

        self.list_v_ns = ["x", "y", "z", "rho", "w_lorentz", "vol", "press", "eps", "lapse", "velx", "vely", "velz",
                          "gxx", "gxy", "gxz", "gyy", "gyz", "gzz", "betax", "betay", "betaz", 'temp', 'Ye'] + \
                         ["density",  "enthalpy", "vphi", "vr", "dens_unb_geo", "dens_unb_bern", "dens_unb_garch",
                          "ang_mom", "ang_mom_flux", "theta", "r", "phi" ]
        self.list_planes = ["xy", "xz", "yz"]

        self.data_matrix = [[[[np.zeros(0,)
                             for v_n in range(len(self.list_v_ns))]
                             for p in range(len(self.list_planes))]
                             for x in range(self.nlevels)] # Here 2 * () as for correlation 2 v_ns are aneeded
                             for y in range(len(self.list_iterations))]

    def check_it(self, it):
        if not it in self.list_iterations:
            raise NameError("it:{} not in the list of iterations\n{}"
                            .format(it, self.list_iterations))

    def check_v_n(self, v_n):
        if not v_n in self.list_v_ns:
            raise NameError("v_n:{} not in list of corr_v_ns\n{}"
                            .format(v_n, self.list_v_ns))

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

    def i_v_n(self, v_n):
        self.check_v_n(v_n)
        return int(self.list_v_ns.index(v_n))

    def loaded_extract(self, it, plane):

        path = Paths.ppr_sims + self.sim + "/res_3d/" + str(it) + '/'
        fname = "profile" + '.' + plane + ".h5"
        fpath = path + fname

        if not os.path.isfile(fpath):
            raise IOError("file: {} not found".format(fpath))

        dfile = h5py.File(fpath, "r")
        for rl in np.arange(start=0, stop=self.nlevels, step=1):
            for v_n in self.list_v_ns:
                data = np.array(dfile["reflevel=%d" % rl][v_n], dtype=np.float32)
                self.data_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_v_n(v_n)] = data
        dfile.close()

    def is_data_loaded_extracted(self, it, rl, plane, v_n):
        data = self.data_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_v_n(v_n)]
        if len(data) == 0:
            self.loaded_extract(it, plane)

    def get_data(self, it, rl, plane, v_n):
        self.check_v_n(v_n)
        self.check_it(it)
        self.check_plane(plane)

        self.is_data_loaded_extracted(it, rl, plane, v_n)
        return self.data_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_v_n(v_n)]


""" --- --- PLOTTING RESILTS --- --- """


class PLOT_TASK:

    def __init__(self, o_data_class, plot_type):
        self.data = o_data_class
        self.plot_type = plot_type
        pass

    def plot_x_y_z2d_colormesh(self, ax, dic):

        if dic['name'] == "corr":

            table = self.data.get_res_corr(dic["it"], dic["v_n_x"], dic["v_n_y"])
            table = np.array(table)
            x_arr = table[0, 1:]  # * 6.176269145886162e+17
            y_arr = table[1:, 0]
            z_arr = table[1:, 1:]

            z_arr = z_arr / np.sum(z_arr)

            im = self.plot_colormesh(ax, dic, x_arr, y_arr, z_arr)

        elif dic['name'] == 'int':

            y_arr = self.data.get_grid_data(dic["it"], dic["v_n_x"]) # phi
            x_arr = self.data.get_grid_data(dic["it"], dic["v_n_y"]) # r
            z_arr = self.data.get_int_data(dic["it"], dic["v_n"])

            x_arr = np.array(x_arr[:, 0, 0])
            y_arr = np.array(y_arr[0, :, 0]) # phi
            z_arr = np.array(z_arr[:, :, 0]) # take a slice

            # print(x_arr.shape, y_arr.shape, z_arr.shape)
            # print(y_arr)

            im = self.plot_colormesh(ax, dic, y_arr, x_arr, z_arr) # phi, r, data

        else:
            raise NameError("plot type dic[name] is not recognised (given: {})".format(dic["name"]))

        return im

    def plot_colormesh(self, ax, dic, x_arr, y_arr, z_arr):


        # special treatment
        if dic["v_n_x"] == "theta": x_arr = 90 - (180 * x_arr / np.pi)
        if dic["v_n_y"] == "theta": y_arr = 90 - (180 * y_arr / np.pi)

        if dic["v_n_x"] == "phi": x_arr = 180 + (180 * x_arr / np.pi)
        if dic["v_n_y"] == "phi": y_arr = 180 + (180 * y_arr / np.pi)

        if dic["v_n_x"] == "rho": x_arr *= 6.176269145886162e+17
        if dic["v_n_y"] == "rho": y_arr *= 6.176269145886162e+17

        # z_arr = 2 * np.maximum(z_arr, 1e-12)  # WHAT'S THAT?

        # limits
        if self.plot_type == "cartesian":
            if dic["xmin"] != None and dic["xmax"] != None:
                ax.set_xlim(dic["xmin"], dic["xmax"])
            if dic["ymin"] != None and dic["ymax"] != None:
                ax.set_ylim(dic["ymin"], dic["ymax"])
            if dic["xscale"] == 'log':
                ax.set_xscale("log")
            if dic["yscale"] == 'log':
                ax.set_yscale("log")
            ax.set_ylabel(dic["v_n_y"].replace('_', '\_'))
        elif self.plot_type == "polar":
            if dic["phimin"] != None and dic["phimax"] != None:
                ax.set_philim(dic["phimin"], dic["phimax"])
            if dic["rmin"] != None and dic["rmax"] != None:
                ax.set_rlim(dic["rmin"], dic["rmax"])

        else:
            raise NameError("Unknown type of the plot: {}".format(self.plot_type))

        if dic["vmin"] == None: dic["vmin"] = z_arr.min()
        if dic["vmax"] == None: dic["vmax"] = z_arr.max()

        if dic["norm"] == "norm" or dic["norm"] == "line" or dic["norm"] == None:
            norm = Normalize(vmin=dic["vmin"], vmax=dic["vmax"])
        elif dic["norm"] == "log":
            norm = LogNorm(vmin=dic["vmin"], vmax=dic["vmax"])
        else:
            raise NameError("unrecognized norm: {} in task {}"
                            .format(dic["norm"], dic["v_n"]))

        ax.set_xlabel(dic["v_n_x"].replace('_', '\_'))



        im = ax.pcolormesh(x_arr, y_arr, z_arr, norm=norm, cmap=dic["cmap"])#, vmin=dic["vmin"], vmax=dic["vmax"])
        im.set_rasterized(True)

        return im

    def plot_countours(self, ax, dic):

        y_arr = self.data.get_grid_data(dic["it"], dic["v_n_x"])  # phi
        x_arr = self.data.get_grid_data(dic["it"], dic["v_n_y"])  # r
        z_arr = self.data.get_int_data(dic["it"], dic["v_n"])

        x_arr = np.array(x_arr[:, 0, 0])
        y_arr = np.array(y_arr[0, :, 0])  # phi
        z_arr = np.array(z_arr[:, :, 0])  # take a slice

        ax.contour(y_arr, x_arr, z_arr, dic["levels"], colors=dic["colors"])

        if dic["rmin"] != None and dic["rmax"] != None:
            ax.set_rlim(dic["rmin"], dic["rmax"])

    def plot_density_mode(self, ax, dic):

        dfile = dic["dfile"]
        it = dic["it"]
        p_int_phi = dic["plot_int_phi"]
        p_int_phi_r = dic["plot_int_phi_r"]
        mode = dic["mode"]
        r_max = dic["rmax"]

        # dfile = h5py.File(Paths.ppr_sims + sim + '/res_3d/' + load_file, "r")
        group = dfile["m=%d" % mode]
        times = np.array(dfile["times"])
        iterations = np.array(dfile["iterations"])
        r_cyl = np.array(dfile["r_cyl"])

        int_phi2d = np.array(group["int_phi"])
        int_phi_r1d = np.array(group["int_phi_r"])

        if not it in iterations:
            raise ValueError("it: {} not found in the iteration list fron density_modes.h5 file \n {}"
                             .format(it, iterations))

        print(int_phi2d.shape)

        int_phi_r1d_for_it = int_phi_r1d[iterations == it]
        int_phi2d_for_it = int_phi2d[int(np.where(iterations == it)[0]), :]

        if len(int_phi2d_for_it) != len(r_cyl):
            raise ValueError("Error len(int_phi2d_for_it) {} != {} len(r_cyl)".format(
                             len(int_phi2d_for_it), len(r_cyl)))

        if p_int_phi_r:
            # for one 'r'
            phi = np.zeros(r_cyl.shape)
            # print(np.angle(int_phi_r1d_for_it))
            phi.fill(float(np.angle(int_phi_r1d_for_it)))
            ax.plot(phi[r_cyl < r_max], r_cyl[r_cyl < r_max], '-', color='black')  # plot integrated

        if p_int_phi:
            # for every 'r'
            ax.plot(np.angle(int_phi2d_for_it)[r_cyl < r_max], r_cyl[r_cyl < r_max], '-.', color='black')


    def plot_summed_correlation_with_time(self):
        """
            this is temporatry function meade to show the summed histograms in ang_mom_flux_dens_unb_bern
            space

        :return:
        """
        times = []
        total_masses = []
        for it in self.data.list_iterations:

            table = self.data.get_res_corr(int(it), "ang_mom_flux", "dens_unb_bern")
            time_ = self.data.get_time(int(it))

            table = np.array(table)
            # table[0, 1:] = table[0, 1:] * 6.176269145886162e+17

            x_arr = table[0,1 :] #* 6.176269145886162e+17
            y_arr = table[1:, 0]
            z_arr = table[1:,1:]

            total_mass = np.sum(z_arr)



            times.append(time_)
            total_masses.append(total_mass)

        times, total_masses = x_y_z_sort(times, total_masses)

        inv_total_masses = []
        for it in self.data.list_iterations:

            table = self.data.get_res_corr(int(it), "inv_ang_mom_flux", "dens_unb_bern")
            # time_ = self.get_time(int(it))

            table = np.array(table)
            # table[0, 1:] = table[0, 1:] * 6.176269145886162e+17

            x_arr = table[0,1 :] #* 6.176269145886162e+17
            y_arr = table[1:, 0]
            z_arr = table[1:,1:]

            inv_total_mass = np.sum(z_arr)



            # times.append(time_)
            inv_total_masses.append(inv_total_mass)

        times, total_masses, inv_total_masses = x_y_z_sort(times, total_masses, inv_total_masses, 0)

        masses = []
        for it in self.data.list_iterations:
            mass = float(np.loadtxt(Paths.ppr_sims+sim+'/res_3d/'+str(int(it))+"/disk_mass.txt", unpack=True))
            masses.append(mass)


        masses = np.array(masses)
        times = np.array(times)
        inv_total_masses = np.array(inv_total_masses)
        total_masses = np.array(total_masses)


        plt.plot(times, masses/10., '-', color="black", label="Disk Mass * 0.1")
        plt.plot(times, total_masses, '.', color="black")
        plt.plot(times, inv_total_masses, '.', color="gray")
        plt.ylabel(r'$\sum(Jflux\_vs\_Dens\_unb\_bern)$', fontsize=12)
        plt.xlabel(r'time [s]', fontsize=12)
        plt.minorticks_on()
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title('Mass Flux', fontsize=20)
        plt.legend(loc='upper right', numpoints=1)
        plt.savefig(Paths.ppr_sims+sim+"/res_3d/"+"test_total_ang_mom_flux.png", bbox_inches='tight', dpi=128)
        plt.close()

    def plot_summed_correlation_density_modes_with_time(self):
        """
            this is temporatry function meade to show the summed histograms in ang_mom_flux_dens_unb_bern
            space

        :return:
        """
        times = []
        total_masses = []
        for it in self.data.list_iterations:

            table = self.data.get_res_corr(int(it), "ang_mom_flux", "dens_unb_bern")
            time_ = self.data.get_time(int(it))

            table = np.array(table)
            # table[0, 1:] = table[0, 1:] * 6.176269145886162e+17

            x_arr = table[0,1 :] #* 6.176269145886162e+17
            y_arr = table[1:, 0]
            z_arr = table[1:,1:]

            total_mass = np.sum(z_arr)



            times.append(time_)
            total_masses.append(total_mass)

        times, total_masses = x_y_z_sort(times, total_masses)

        inv_total_masses = []
        for it in self.data.list_iterations:

            table = self.data.get_res_corr(int(it), "inv_ang_mom_flux", "dens_unb_bern")
            # time_ = self.get_time(int(it))

            table = np.array(table)
            # table[0, 1:] = table[0, 1:] * 6.176269145886162e+17

            x_arr = table[0,1 :] #* 6.176269145886162e+17
            y_arr = table[1:, 0]
            z_arr = table[1:,1:]

            inv_total_mass = np.sum(z_arr)



            # times.append(time_)
            inv_total_masses.append(inv_total_mass)

        times, total_masses, inv_total_masses = x_y_z_sort(times, total_masses, inv_total_masses, 0)

        masses = []
        for it in self.data.list_iterations:
            mass = float(np.loadtxt(Paths.ppr_sims+sim+'/res_3d/'+str(int(it))+"/disk_mass.txt", unpack=True))
            masses.append(mass)


        masses = np.array(masses)
        times = np.array(times)
        inv_total_masses = np.array(inv_total_masses)
        total_masses = np.array(total_masses)







        plt.plot(times, masses/10., '-', color="black", label="Disk Mass * 0.1")
        plt.plot(times, total_masses, '.', color="black")
        plt.plot(times, inv_total_masses, '.', color="gray")
        plt.ylabel(r'$\sum(Jflux\_vs\_Dens\_unb\_bern)$', fontsize=12)
        plt.xlabel(r'time [s]', fontsize=12)
        plt.minorticks_on()
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title('Mass Flux', fontsize=20)
        plt.legend(loc='upper right', numpoints=1)
        plt.savefig(Paths.plots+"test_total_ang_mom_flux.png", bbox_inches='tight', dpi=128)
        plt.close()




# class PLOT_MANY_TASKS(PLOT_TASK):
#
#     def __init__(self, o_data_class, plot_type):
#
#         PLOT_TASK.__init__(self, o_data_class, plot_type)
#         it = 1818738
#
#         self.data = o_data_class
#
#         self.gen_set = {
#             "figdir": Paths.ppr_sims + self.data.sim + "/res_3d/{}/".format(it),
#             "figname": "inv_ang_mom_flux.png",
#             # "figsize": (13.5, 3.5), # <->, |
#             "figsize": (3.8, 3.5),  # <->, |
#             "type": "cartesian",
#             "subplots_adjust_h": 0.2,
#             "subplots_adjust_w": 0.3
#         }
#
#         self.set_plot_dics = []
#
#         corr_dic_temp_Ye = { # relies on the "get_res_corr(self, it, v_n): " method of data object
#             'name': 'corr', 'position': (1, 1), 'title': 'time [ms]', 'cbar': 'right .05 .0',
#             'it': 2237972, 'v_n_x': 'temp', 'v_n_y': 'Ye', 'v_n': 'mass',
#             'xmin': 2., 'xmax': 15., 'ymin': 0., 'ymax': 0.2, 'vmin': 1e-4, 'vmax': None,
#             'mask_below': None, 'mask_above': None, 'cmap': 'inferno_r', 'norm': 'log', 'todo': None
#         }
#         # self.set_plot_dics.append(corr_dic_temp_Ye)
#
#         corr_dic_rho_ang_mom = { # relies on the "get_res_corr(self, it, v_n): " method of data object
#             'name': 'corr', 'position': (1, 1), 'title': 'time [ms]', 'cbar':  'right .05 .0',
#             'it': 761856, 'v_n_x': 'rho', 'v_n_y': 'ang_mom', 'v_n': 'mass',
#             'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None, 'vmin': 1e-8, 'vmax': None,
#             'xscale':'log', 'yscale':'log',
#             'mask_below': None, 'mask_above': None, 'cmap': 'inferno_r', 'norm': 'log', 'todo': None
#         }
#         # self.set_plot_dics.append(corr_dic_rho_ang_mom)
#
#         corr_dic_rho_dens_unb_bern = { # relies on the "get_res_corr(self, it, v_n): " method of data object
#             'name': 'corr', 'position': (1, 1), 'title': 'time [ms]', 'cbar':  None, #  'right .05 .0',
#             'it': 1081344, 'v_n_x': 'rho', 'v_n_y': 'dens_unb_bern', 'v_n': 'mass',
#             'xmin': None, 'xmax': None, 'ymin': None, 'ymax': 1e-3, 'vmin': 1e-7, 'vmax': None,
#             'xscale':'log', 'yscale':'log',
#             'mask_below': None, 'mask_above': None, 'cmap': 'inferno_r', 'norm': 'log', 'todo': None
#         }
#         # self.set_plot_dics.append(corr_dic_rho_dens_unb_bern)
#
#         corr_dic_ang_mom_flux_theta = { # relies on the "get_res_corr(self, it, v_n): " method of data object
#             'name': 'corr', 'position': (1, 3), 'title': 'time [ms]', 'cbar':  None, #'right .05 .0',
#             'it': it, 'v_n_x': 'theta', 'v_n_y': 'ang_mom_flux', 'v_n': 'mass',
#             'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None, 'vmin': 1e-7, 'vmax': None,
#             'xscale':'line', 'yscale':'log',
#             'mask_below': None, 'mask_above': None, 'cmap': 'inferno_r', 'norm': 'log', 'todo': None
#         }
#         # self.set_plot_dics.append(corr_dic_ang_mom_flux_theta)
#
#         corr_dic_ang_mom_flux_dens_unb_bern = { # relies on the "get_res_corr(self, it, v_n): " method of data object
#             'name': 'corr', 'position': (1, 1), 'title': 'time [ms]', 'cbar': 'right .03 .0',
#             'it': it, 'v_n_x': 'dens_unb_bern', 'v_n_y': 'ang_mom_flux', 'v_n': 'mass',
#             'xmin': 1e-11, 'xmax': 1e-7, 'ymin': 1e-11, 'ymax': 1e-7, 'vmin': 1e-7, 'vmax': None,
#             'xscale':'log', 'yscale':'log',
#             'mask_below': None, 'mask_above': None, 'cmap': 'inferno_r', 'norm': 'log', 'todo': None
#         }
#         # self.set_plot_dics.append(corr_dic_ang_mom_flux_dens_unb_bern)
#
#         corr_dic_inv_ang_mom_flux_dens_unb_bern = { # relies on the "get_res_corr(self, it, v_n): " method of data object
#             'name': 'corr', 'position': (1, 1), 'title': 'time [ms]', 'cbar': 'right .03 .0',
#             'it': it, 'v_n_x': 'dens_unb_bern', 'v_n_y': 'inv_ang_mom_flux', 'v_n': 'mass',
#             'xmin': 1e-11, 'xmax': 1e-7, 'ymin': 1e-11, 'ymax': 1e-7, 'vmin': 1e-7, 'vmax': None,
#             'xscale':'log', 'yscale':'log',
#             'mask_below': None, 'mask_above': None, 'cmap': 'inferno_r', 'norm': 'log', 'todo': None
#         }
#         # self.set_plot_dics.append(corr_dic_inv_ang_mom_flux_dens_unb_bern)
#
#         corr_dic_rho_ang_mom_flux = { # relies on the "get_res_corr(self, it, v_n): " method of data object
#             'name': 'corr', 'position': (1, 2), 'title': 'time [ms]', 'cbar':  'right .05 .0',
#             'it': it, 'v_n_x': 'rho', 'v_n_y': 'ang_mom_flux', 'v_n': 'mass',
#             'xmin': None, 'xmax': None, 'ymin': None, 'ymax': 1e-3, 'vmin': 1e-7, 'vmax': None,
#             'xscale':'log', 'yscale':'log',
#             'mask_below': None, 'mask_above': None, 'cmap': 'inferno_r', 'norm': 'log', 'todo': None
#         }
#         # self.set_plot_dics.append(corr_dic_rho_ang_mom_flux)
#
#         corr_dic_rho_theta = { # relies on the "get_res_corr(self, it, v_n): " method of data object
#             'name': 'corr', 'position': (1, 2), 'title': 'time [ms]', 'cbar':None, #   'right .05 .0',
#             'it': it, 'v_n_x': 'theta', 'v_n_y': 'rho', 'v_n': 'mass',
#             'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None, 'vmin': 1e-7, 'vmax': None,
#             'xscale': 'line', 'yscale': 'log',
#             'mask_below': None, 'mask_above': None, 'cmap': 'inferno_r', 'norm': 'log', 'todo': None
#         }
#         # self.set_plot_dics.append(corr_dic_rho_theta)
#
#         corr_dic_rho_r = { # relies on the "get_res_corr(self, it, v_n): " method of data object
#             'name': 'corr', 'position': (1, 1), 'title': 'time [ms]', 'cbar': None,#  'right .05 .0',
#             'it': it, 'v_n_x': 'r', 'v_n_y': 'rho', 'v_n': 'mass',
#             'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None, 'vmin': 1e-7, 'vmax': None,
#             'xscale': 'line', 'yscale': 'log',
#             'mask_below': None, 'mask_above': None, 'cmap': 'inferno_r', 'norm': 'log', 'todo': None
#         }
#         # self.set_plot_dics.append(corr_dic_rho_r)
#
#         corr_dic_rho_Ye = { # relies on the "get_res_corr(self, it, v_n): " method of data object
#             'name': 'corr', 'position': (1, 2), 'title': 'time [ms]', 'cbar': 'right .05 .0',
#             'it': it, 'v_n_x': 'rho', 'v_n_y': 'Ye', 'v_n': 'mass',
#             'xmin': None, 'xmax': None, 'ymin': 0., 'ymax': 0.4, 'vmin': 1e-8, 'vmax': None,
#             'mask_below': None, 'mask_above': None, 'cmap': 'inferno_r', 'norm': 'log', 'todo': None
#         }
#         # self.set_plot_dics.append(corr_dic_rho_Ye)
#
#
#         int_ang_mom_flux_dic = {
#             'name': 'int', 'position': (1, 1), 'title': 'time [ms]', 'cbar': 'right .03 .0',
#             'it': it, 'v_n_x': 'phi_cyl', 'v_n_y': 'r_cyl', 'v_n': 'amg_mom_flux',
#             'rmin': 0, 'rmax': 50., 'ymin': None, 'ymax': None, 'vmin': -5e-6, 'vmax': 5e-6,
#             'xscale': None, 'yscale': None,
#             'mask_below': None, 'mask_above': None, 'cmap': 'RdBu_r', 'norm': 'log', 'todo': None
#         }
#
#
#     def set_ncols_nrows(self):
#
#         tmp_rows = []
#         tmp_cols = []
#
#         for dic in self.set_plot_dics:
#             tmp_cols.append(dic['position'][1])
#             tmp_rows.append(dic['position'][0])
#
#         max_row = max(tmp_rows)
#         max_col = max(tmp_cols)
#
#         for row in range(1, max_row):
#             if not row in tmp_rows:
#                 raise NameError("Please set vertical plot position in a subsequent order: 1,2,3... not 1,3...")
#
#         for col in range(1, max_col):
#             if not col in tmp_cols:
#                 raise NameError("Please set horizontal plot position in a subsequent order: 1,2,3... not 1,3...")
#
#         print("Set {} rows {} columns (total {}) of plots".format(max_row, max_col, len(self.set_plot_dics)))
#
#         return int(max_row), int(max_col)
#
#     def set_plot_dics_matrix(self):
#
#         plot_dic_matrix = [[0
#                              for x in range(self.n_rows)]
#                              for y in range(self.n_cols)]
#
#         # get a matrix of dictionaries describing plots (for ease of representation)
#         for dic in self.set_plot_dics:
#             col, row = int(dic['position'][1]-1), int(dic['position'][0]-1) # -1 as position starts with 1
#             # print(col, row)
#             for n_row in range(self.n_rows):
#                 for n_col in range(self.n_cols):
#                     if int(col) == int(n_col) and int(row) == int(n_row):
#                         plot_dic_matrix[n_col][n_row] = dic
#                         # print('adding {} {}'.format(col, row))
#
#             if isinstance(plot_dic_matrix[col][row], int):
#                 raise ValueError("Dictionary to found for n_row {} n_col {} in "
#                                  "creating matrix of dictionaries".format(col, row))
#
#         return plot_dic_matrix
#
#     def set_plot_matrix(self):
#
#         fig = plt.figure(figsize=self.gen_set['figsize'])  # (<->; v)
#
#
#         if self.gen_set['type'] == 'cartesian':
#             # initializing the matrix with dummy axis objects
#             sbplot_matrix = [[fig.add_subplot(self.n_rows, self.n_cols, 1)
#                                   for x in range(self.n_rows)]
#                                   for y in range(self.n_cols)]
#
#             i = 1
#             for n_row in range(self.n_rows):
#                 for n_col in range(self.n_cols):
#
#                     if n_col == 0 and n_row == 0:
#                         sbplot_matrix[n_col][n_row] = fig.add_subplot(self.n_rows, self.n_cols, i)
#                     elif n_col == 0 and n_row > 0:
#                         sbplot_matrix[n_col][n_row] = fig.add_subplot(self.n_rows, self.n_cols, i,
#                                                                       )#sharex=self.sbplot_matrix[n_col][0])
#                     elif n_col > 0 and n_row == 0:
#                         sbplot_matrix[n_col][n_row] = fig.add_subplot(self.n_rows, self.n_cols, i,
#                                                                       )#sharey=self.sbplot_matrix[0][n_row])
#                     else:
#                         sbplot_matrix[n_col][n_row] = fig.add_subplot(self.n_rows, self.n_cols, i,
#                                                                       #sharex=self.sbplot_matrix[n_col][0],
#                                                                       )#sharey=self.sbplot_matrix[0][n_row])
#
#                         # sbplot_matrix[n_col][n_row].axes.get_yaxis().set_visible(False)
#                     # sbplot_matrix[n_col][n_row] = fig.add_subplot(n_rows, n_cols, i)
#                     i += 1
#
#         elif self.gen_set['type'] == 'polar':
#             # initializing the matrix with dummy axis objects
#             sbplot_matrix = [[fig.add_subplot(self.n_rows, self.n_cols, 1, projection='polar')
#                                   for x in range(self.n_rows)]
#                                   for y in range(self.n_cols)]
#
#             i = 1
#             for n_row in range(self.n_rows):
#                 for n_col in range(self.n_cols):
#
#                     if n_col == 0 and n_row == 0:
#                         sbplot_matrix[n_col][n_row] = fig.add_subplot(self.n_rows, self.n_cols, i, projection='polar')
#                     elif n_col == 0 and n_row > 0:
#                         sbplot_matrix[n_col][n_row] = fig.add_subplot(self.n_rows, self.n_cols, i, projection='polar')
#                                                                       # sharex=self.sbplot_matrix[n_col][0])
#                     elif n_col > 0 and n_row == 0:
#                         sbplot_matrix[n_col][n_row] = fig.add_subplot(self.n_rows, self.n_cols, i, projection='polar')
#                                                                       # sharey=self.sbplot_matrix[0][n_row])
#                     else:
#                         sbplot_matrix[n_col][n_row] = fig.add_subplot(self.n_rows, self.n_cols, i, projection='polar')
#                                                                       # sharex=self.sbplot_matrix[n_col][0],
#                                                                       # sharey=self.sbplot_matrix[0][n_row])
#
#                         # sbplot_matrix[n_col][n_row].axes.get_yaxis().set_visible(False)
#                     # sbplot_matrix[n_col][n_row] = fig.add_subplot(n_rows, n_cols, i)
#                     i += 1
#         else:
#             raise NameError("type of the plot is not recognized. Use 'polar' or 'cartesian' ")
#
#         return fig, sbplot_matrix
#
#     def plot_one_task(self, ax, dic):
#
#         if dic["name"] == "corr" or dic["name"] == "int":
#             im = self.plot_x_y_z2d_colormesh(ax, dic)
#         elif dic["name"] == "count":
#             self.plot_countours(ax, dic)
#             im = 0
#         elif dic["name"] == 'densmode':
#             self.plot_density_mode(ax, dic)
#             im = 0
#         else:
#             raise NameError("name:{} is not recognized"
#                             .format(dic["name"]))
#
#         # self.time
#         return im
#
#     def set_plot_title(self, ax, plot_dic):
#         if plot_dic["title"] != '' and plot_dic["title"] != None:
#
#             if plot_dic["title"] == 'it':
#                 title = plot_dic["it"]
#             elif plot_dic["title"] == 'time [s]' or \
#                 plot_dic["title"] == 'time':
#                 title = "%.3f" % self.data.get_time(plot_dic["it"]) + " [s]"
#             elif plot_dic["title"] == 'time [ms]':
#                 title = "%.1f" % (self.data.get_time(plot_dic["it"]) * 1000) + " [ms]"
#             else:
#                 title = plot_dic["title"]
#             ax.title.set_text(r'{}'.format(title))
#
#     def plot_images(self):
#
#         # initializing the matrix of images for colorbars (if needed)
#         image_matrix = [[0
#                         for x in range(self.n_rows)]
#                         for y in range(self.n_cols)]
#
#         # for n_row in range(self.n_rows):
#         #     for n_col in range(self.n_cols):
#         #         print("Plotting n_row:{} n_col:{}".format(n_row, n_col))
#         #         ax = self.sbplot_matrix[n_col][n_row]
#         #         dic = self.plot_dic_matrix[n_col][n_row]
#         #         if isinstance(dic, int):
#         #             Printcolor.yellow("Dictionary for row:{} col:{} not set".format(n_row, n_col))
#         #             self.fig.delaxes(ax) # delets the axis for empty plot
#         #         else:
#         #             dic = dict(dic)
#         #             im = self.plot_one_task(ax, dic)
#         #             self.set_plot_title(ax, dic)
#         #             image_matrix[n_col][n_row] = im
#
#         for n_row in range(self.n_rows):
#             for n_col in range(self.n_cols):
#                 for dic in self.set_plot_dics:
#                     if n_col + 1 == int(dic['position'][1]) and n_row + 1 == int(dic['position'][0]):
#                         print("Plotting n_row:{} n_col:{}".format(n_row, n_col))
#                         ax = self.sbplot_matrix[n_col][n_row]
#                         # dic = self.plot_dic_matrix[n_col][n_row]
#                         if isinstance(dic, int):
#                             Printcolor.yellow("Dictionary for row:{} col:{} not set".format(n_row, n_col))
#                             self.fig.delaxes(ax)  # delets the axis for empty plot
#                         else:
#                             dic = dict(dic)
#                             im = self.plot_one_task(ax, dic)
#                             self.set_plot_title(ax, dic)
#                             if not isinstance(im, int):
#                                 image_matrix[n_col][n_row] = im
#
#
#         return image_matrix
#
#     def plot_one_cbar(self, im, dic, n_row, n_col):
#
#         if dic["cbar"] != None and dic["cbar"] != '':
#
#             location = dic["cbar"].split(' ')[0]
#             shift_h = float(dic["cbar"].split(' ')[1])
#             shift_w = float(dic["cbar"].split(' ')[2])
#             cbar_width = 0.02
#
#
#             if location == 'right':
#                 ax_to_use = self.sbplot_matrix[-1][n_row]
#                 pos1 = ax_to_use.get_position()
#                 pos2 = [pos1.x0 + pos1.width + shift_h,
#                         pos1.y0,
#                         cbar_width,
#                         pos1.height]
#             elif location == 'left':
#                 ax_to_use = self.sbplot_matrix[-1][n_row]
#                 pos1 = ax_to_use.get_position()
#                 pos2 = [pos1.x0 - pos1.width - shift_h,
#                         pos1.y0,
#                         cbar_width,
#                         pos1.height]
#             elif location == 'bottom':
#                 ax_to_use = self.sbplot_matrix[n_col][-1]
#                 pos1 = ax_to_use.get_position()
#                 pos2 = [pos1.x0,
#                         pos1.y0 - pos1.height + shift_w,
#                         cbar_width,
#                         pos1.height]
#             else:
#                 raise NameError("cbar location {} not recognized. Use 'right' or 'bottom' "
#                                 .format(location))
#
#             cax1 = self.fig.add_axes(pos2)
#             if location == 'right':
#                 cbar = plt.colorbar(im, cax=cax1, extend='both')#, format='%.1e')
#             elif location == 'left':
#                 cbar = plt.colorbar(im, cax=cax1, extend='both')#, format='%.1e')
#                 cax1.yaxis.set_ticks_position('left')
#                 cax1.yaxis.set_label_position('left')
#             else:
#                 raise NameError("cbar location {} not recognized. Use 'right' or 'bottom' "
#                                 .format(location))
#             cbar.ax.set_title(r"{}".format(str(dic["v_n"]).replace('_', '\_')))
#
#     def plot_colobars(self):
#
#         for n_row in range(self.n_rows):
#             for n_col in range(self.n_cols):
#                 for dic in self.set_plot_dics:
#                     if n_col + 1 == int(dic['position'][1]) and n_row + 1 == int(dic['position'][0]):
#                         print("Colobar for n_row:{} n_col:{}".format(n_row, n_col))
#                         # ax  = self.sbplot_matrix[n_col][n_row]
#                         # dic = self.plot_dic_matrix[n_col][n_row]
#                         im  = self.image_matrix[n_col][n_row]
#                         if isinstance(dic, int):
#                             Printcolor.yellow("Dictionary for row:{} col:{} not set".format(n_row, n_col))
#                         else:
#                             self.plot_one_cbar(im, dic, n_row, n_col)
#
#
#         # for n_row in range(self.n_rows):
#         #     for n_col in range(self.n_cols):
#         #         print("Colobar for n_row:{} n_col:{}".format(n_row, n_col))
#         #         # ax  = self.sbplot_matrix[n_col][n_row]
#         #         dic = self.plot_dic_matrix[n_col][n_row]
#         #         im  = self.image_matrix[n_col][n_row]
#         #         if isinstance(dic, int):
#         #             Printcolor.yellow("Dictionary for row:{} col:{} not set".format(n_row, n_col))
#         #         else:
#         #             self.plot_one_cbar(im, dic, n_row, n_col)
#
#     def save_plot(self):
#
#         plt.subplots_adjust(hspace=self.gen_set["subplots_adjust_h"])
#         plt.subplots_adjust(wspace=self.gen_set["subplots_adjust_w"])
#         # plt.tight_layout()
#         plt.savefig('{}{}'.format(self.gen_set["figdir"], self.gen_set["figname"]),
#                     bbox_inches='tight', dpi=128)
#         plt.close()
#
#     def main(self):
#
#         # initializing the n_cols, n_rows
#         self.n_rows, self.n_cols = self.set_ncols_nrows()
#         # initializing the matrix of dictionaries of the
#         self.plot_dic_matrix = self.set_plot_dics_matrix()
#         # initializing the axis matrix (for all subplots) and image matrix fo colorbars
#         self.fig, self.sbplot_matrix = self.set_plot_matrix()
#         # plotting
#         self.image_matrix = self.plot_images()
#         # adding colobars
#         self.plot_colobars()
#
#
#         # saving the result
#         self.save_plot()

""" --- --- LOADING & PLOTTING R-phi Phi-Z --- ---"""





""" --- --- TASK SPECIFIC FUNCTIONS --- --- --- """

def do_compute_save_vtk_for_it():

    sim = "DD2_M13641364_M0_LK_SR_R04"
    profs_loc = "/data/numrel/WhiskyTHC/Backup/2018/GW170817/{}/profiles/3d/".format(sim)
    it = 1111116

    o_grid = CARTESIAN_GRID()

    o_data = INTMETHODS_STORE(profs_loc+"{}.h5".format(it), sim, grid_object=o_grid)

    o_data.save_vtk_file(["rho", "lapse", "dens_unb_bern", "ang_mom_flux"], overwrite=True)



def do_histogram_processing_of_iterations():

    symmetry = "pi"
    sim =  "DD2_M13641364_M0_LK_LR_PI" #"SLy4_M13641364_M0_SR" # "DD2_M13641364_M0_SR"
    profs_loc = Paths.gw170817+sim+"/profiles/3d/"
    out_dir = Paths.ppr_sims + sim + "/res_3d/"

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)



    list_iterations = np.array(get_list_iterationsfrom_profiles_3d(sim, in_sim_dir="/profiles/3d/"))
    times = interpoate_time_form_it(list_iterations, Paths.gw170817 + sim + '/')

    # my dd2 pi: [times>0.039]
    # david's dd2 pi [times>0.029]

    iterations = list_iterations[times>0.039]
    # print(iterations)
    # exit(1)
    processed = []

    def do_for_iteration(it):

        print("| processing iteration: {} ({} out {})|".format(it, len(processed), len(iterations)))
        o_methods = MAINMETHODS_STORE(profs_loc + "{}.h5".format(it), sim, symmetry=symmetry)
        # o_methods.mask_setup["ang_mom_flux"] = [1e-12, 1.] # set a ADDITIONAL LIMIT
        #
        o_methods.get_total_mass_old(save=True)
        print("rho-ye",
              np.sum(o_methods.get_correlation_old(o_methods.corr_task_dic_rho_ye, save=True)))
        print("temp-ye",
              np.sum(o_methods.get_correlation_old(o_methods.corr_task_dic_temp_ye, save=True)))
        print("rho-r",
              np.sum(o_methods.get_correlation_old(o_methods.corr_task_dic_rho_r, save=True)))
        # print(np.sum(o_methods.get_correlation(o_methods.corr_task_dic_r_phi, save=True)))
        print("rho-theta",
              np.sum(o_methods.get_correlation_old(o_methods.corr_task_dic_rho_theta, save=True)))
        print("rho-ang_mom",
              np.sum(o_methods.get_correlation_old(o_methods.corr_task_dic_rho_ang_mom, save=True)))
        print("rho-ang_mom_flux",
              np.sum(o_methods.get_correlation_old(o_methods.corr_task_dic_rho_ang_mom_flux, save=True)))
        print("rho-dens_unb_bern",
              np.sum(o_methods.get_correlation_old(o_methods.corr_task_dic_rho_dens_unb_bern, save=True)))
        print("velz-dens_unb_bern",
              np.sum(o_methods.get_correlation_old(o_methods.corr_task_dic_velz_dens_unb_bern, save=True)))
        print("ang_mom_flux-theta",
              np.sum(o_methods.get_correlation_old(o_methods.corr_task_dic_ang_mom_flux_theta, save=True)))
        print("ang_mom_flux-dens_unb_bern",
              np.sum(o_methods.get_correlation_old(o_methods.corr_task_dic_ang_mom_flux_dens_unb_bern, save=True)))
        print("inv_ang_mom_flux-dens_unb_bern",
              np.sum(o_methods.get_correlation_old(o_methods.corr_task_dic_inv_ang_mom_flux_dens_unb_bern, save=True)))
        ### -- 3D
        # print(np.sum(o_methods.get_correlation(o_methods.corr_task_dic_r_phi_ang_mom_flux, save=True)))

        o_methods.__delete__(o_methods)
        processed.append(it)

    def do2_for_iteration(it):
        print("| processing iteration: {} ({} out {})|".format(it, len(processed), len(iterations)))
        o_methods = MAINMETHODS_STORE(profs_loc + "{}.h5".format(it), sim, symmetry=symmetry)
        o_methods.mask_setup["ang_mom_flux"] = [1e-12, 1.]  # set a ADDITIONAL LIMIT for POSITIVE Jf
        #
        o_methods.get_total_mass_old(save=False)

        print("r-phi",
              np.sum(o_methods.get_correlation_old(o_methods.corr_task_dic_r_phi, save=True)))

        o_methods.__delete__(o_methods)
        processed.append(it)

    for it in np.array(iterations, dtype=int):
        try:
            do_for_iteration(it)
        except:
            Printcolor.yellow("failed it:{}".format(it))


    for it in np.array(iterations, dtype=int):
        try:
            do2_for_iteration(it)
        except:
            Printcolor.yellow("failed it:{}".format(it))

    print("sim: {} histograms computed.\nDone.".format(sim))
    exit(1)

def old_do_produce_interpolated_data_for_iterations():
    """
    For all iterations (looks into the folders inside res_3d/ for list of them
    :return:
    """

    symmetry = "pi"
    sim = "DD2_M13641364_M0_LK_LR_PI"
    profs_loc = Paths.gw170817 + sim + "/profiles/3d/"

    list_iterations = get_list_iterationsfrom_profiles_3d(sim, in_sim_dir="/profiles/3d/")
    times = np.array(interpoate_time_form_it(list_iterations, path_to_sim=Paths.gw170817 + sim + '/'))
    if len(times) == 0:
        raise ValueError("Times not found.")

    print(list_iterations)
    print(times)
    # for David dd2 pi 0.029
    # for my dd2 pi 0.039
    iterations = np.array(list_iterations)[times>0.039]#[(times > 0.020) & (times < 0.020)]

    _, _, task_for_int = setup()
    processed = []

    # iterations = [630784]


    if click.confirm('Itertation: {} wish to continue?'.format(len(iterations)), default=True):

        for it in np.array(iterations, dtype=int):
            print("| interpolating iteration: {} ({} out {})|".format(it, len(processed), len(iterations)))

            int_ = INTMETHODS_STORE(profs_loc+"{}.h5".format(it), sim,
                                    CYLINDRICAL_GRID(task_for_int["grid"]),
                                    symmetry=symmetry)

            overwrite = False
            int_.save_int_v_n("lapse", overwrite=overwrite)
            int_.save_int_v_n("vr", overwrite=overwrite)
            int_.save_int_v_n("density", overwrite=overwrite)
            int_.save_int_v_n("ang_mom_flux", overwrite=overwrite)
            int_.save_int_v_n("dens_unb_bern", overwrite=overwrite)
            int_.save_int_v_n("rho", overwrite=overwrite)

            processed.append(it)
            if it == np.array(iterations, int).max():
                int_.save_new_grid()

    print("All done")
    exit(0)

def do_produce_interpolated_data_for_iterations():
    """
    For all iterations (looks into the folders inside res_3d/ for list of them
    :return:
    """

    symmetry = None #"pi"
    sim = "DD2_M13641364_M0_LK_SR_R04"
    v_ns = ["density"]
    o_grid = CYLINDRICAL_GRID()
    o_grid = SPHERICAL_GRID()
    # exit(1)
    o_int = INTMETHODS_STORE(sim, grid_object=o_grid, symmetry=symmetry)

    times = [0]#o_int.list_times
    iterations = [2213326]#o_int.list_iterations

    n = 1
    for t, it in zip(times, iterations):
        print("it:{:d} t:{:.1f}ms ({}/{})"
              .format(int(it), t*1e3, n, len(iterations)))
        for v_n in v_ns:
            # print("  interpolating {} onto {} grid"
            #       .format(v_n, o_grid.grid_type))
            o_int.save_int_v_n(it=it, v_n=v_n, overwrite=True)
        if n == 1:
            print("  saving {} grid "
                  .format(o_grid.grid_type))
            o_int.save_new_grid(it)
        n = n+1

    print("All done")
    # exit(0)

def do_plot_r_theta_summed_over_phi_interpolated_data():

    it = 2213326
    sim = "DD2_M13641364_M0_LK_SR_R04"
    v_n = "density"
    o_grid = SPHERICAL_GRID()
    o_data = LOAD_INT_DATA("DD2_M13641364_M0_LK_SR_R04", o_grid)
    o_data.flag_force_unique_grid = True

    r_req = 20 #[Msun]

    r = o_data.get_grid_data(it, "r_sph")
    # phi = o_data.get_grid_data(it, "phi_sph")[0, :, 0]
    # theta = o_data.get_grid_data(it, "theta_sph")[0, 0, :]
    # print('r', r)

    idx = find_nearest_index(r[:, 0, 0], r_req)
    # print('index of r_int', idx)

    # r = o_data.get_grid_data(it, "r_sph")[:, 0, 0]
    phi = o_data.get_grid_data(it, "phi_sph")
    theta = o_data.get_grid_data(it, "theta_sph")
    arr = o_data.get_int_data(it, v_n)

    print("phi [{}, {}] shape:{}".format(phi[0, :, 0].min(),
                                         phi[0, :, 0].max(),
                                         len(phi[0, :, 0])))

    print("r [{}, {}] shape:{}".format(r[:, 0, 0].min(),
                                         r[:, 0, 0].max(),
                                         len(r[:, 0, 0])))

    print("theta [{}, {}] shape:{}".format(theta[0, 0, :].min(),
                                         theta[0, 0, :].max(),
                                         len(theta[0, 0, :])))

    # assert len(phi[:, 0, 0]) == len(arr[0, 0, :])
    # assert len(r[0, :, 0]) == len(arr[0, :, 0])
    # assert len(theta[0, 0,: ]) == len(arr[0, 0, :])

    print(arr[idx, :, :].T)

    from matplotlib import colors
    fig = plt.figure()
    ax = fig.add_subplot(111)
    norm = colors.LogNorm(vmin=1e-10, vmax=1e-5)
    # norm = colors.Normalize(vmin=0.05, vmax=0.5)
    im = ax.pcolormesh(phi[idx, :, 0], theta[idx, 0, :], arr[idx, :, :].T, norm=norm, cmap="inferno_r")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xlabel("phi")
    ax.set_ylabel("theta")
    ax.minorticks_on()
    plt.title(r"{} map at r={}".format(v_n, r_req))
    plt.savefig('{}'.format(Paths.plots + "sph_int_test.png"), bbox_inches='tight', dpi=128)
    print("saved pi_test2.png")
    plt.close()

    # total mass
    dr = o_data.get_grid_data(it, "dr_sph")
    dtheta = o_data.get_grid_data(it, "dtheta_sph")
    dphi = o_data.get_grid_data(it, "dphi_sph")

    _m = np.sum(arr * (r ** 2) * dr, axis=0)
    __m = np.sum(_m * np.sin(phi[0, :, :]) * dphi[0, :, :], axis=0)
    ___m = np.sum(__m * dtheta[0,0,:])
    print(___m)


    # m = 2 * np.sum(np.sum(np.sum(arr * r ** 2 * dr, axis=0) * np.sin(phi[0, :, :]) * dphi[0, :, :], axis=1) * dtheta[0,0,:])
    # print(m)

    exit(1)

def do_compute_save_all_modes_for_all_it():


    print('-' * 20 + 'COMPUTING DENSITY MODES' + '-' * 20)

    sim = "DD2_M13641364_M0_LK_LR_PI"
    o_dm = COMPUTE_STORE_DESITYMODES(sim)
    # assume that the intepolated grid is the same for all the iterations
    o_dm.flag_force_unique_grid = True
    # o_dm.it_for_unique_grid = '' # 2254356
    # setting the mode list to be computed
    o_dm.list_modes = [0,1,2,3,4,5,6]
    o_dm.gen_set["do_norm"] = False # do not normalize

    r_cyl = o_dm.get_grid_data(o_dm.it_for_unique_grid, "r_cyl")
    r_cyl = np.array(r_cyl[:, 0, 0]).flatten() # 1d


    # print(o_dm.get_data(669810, "density")); exit(1)
    # print(o_dm.get_grid_data(669810, "dz_cyl").shape)
    # print(o_dm.get_density_mode(669810, 1, "int_phi")); exit(1)
    # print(r_cyl); exit(1)


    outpath = Paths.ppr_sims + sim + "/res_3d/"
    outfname = "density_modes.h5"
    dfile = h5py.File(outpath + outfname, "w")
    dfile.create_dataset("r_cyl", data=r_cyl)

    for mode in o_dm.list_modes:
        int_phi_all = []
        int_phi_r_all = []
        times = []
        iterations = []
        for it, time_ in zip(o_dm.list_iterations,
                             interpoate_time_form_it(o_dm.list_iterations, Paths.gw170817 + sim + '/')):
            try:
                print("\tComputing m:{:d} it:{:d}".format(mode, it))
                int_phi_all.append(o_dm.get_density_mode(it, mode, "int_phi"))
                int_phi_r_all.append(o_dm.get_density_mode(it, mode, "int_phi_r"))
                times.append(time_)
                iterations.append(it)
            except ValueError:
                Printcolor.yellow("Warning. it:{} failed with ValueError".format(it))
            except IOError:
                Printcolor.yellow("Warning. it:{} failed with IOError".format(it))

        # print(np.array(int_phi_all).shape, np.array(iterations).shape, np.array(r_cyl).shape)
        int_phi_reshaped = np.reshape(np.array(int_phi_all), (len(iterations), len(r_cyl)))
        group = dfile.create_group("m=%d" % mode)
        group["int_phi"] = int_phi_reshaped
        group["int_phi_r"] = np.array(int_phi_r_all).flatten()

    dfile.create_dataset("iterations", data=np.array(iterations, dtype=int).flatten())
    dfile.create_dataset("times", data=np.array(times, dtype=float).flatten())
    print("file: {} is created".format(outpath + outfname))
    print('-' * 25 + '------DONE-----' + '-' * 25)

    exit(0)

def plot_density_modes_old(load_file="density_modes.h5"):

    sim = "DD2_M13641364_M0_LK_HR_R04"
    m_list = [1,2]


    dfile = h5py.File(Paths.ppr_sims + sim + '/res_3d/' + load_file, "r")

    rows = 2
    cols = 1

    fig = plt.figure(figsize=(6.5, 1.5 * 3.6))  # figsize=(4.5, 2.5 * 3.6)  # (<->; v)
    ax_list = []
    for n in range(1, rows + 1):
        if n == 1:
            ax_list.append(fig.add_subplot(rows, cols, n))
        else:
            ax_list.append(fig.add_subplot(rows, cols, n, sharex=ax_list[n - 2]))  # sharex=axs[n - 2]))


    times = np.array(dfile["times"])
    iterations = np.array(dfile["iterations"])
    r_cyl = np.array(dfile["r_cyl"])

    for m in m_list:

        if m == 1:
            color = 'black'; ls = '-'; lw = 1.
        elif m == 2:
            color = 'gray';  ls = '-.'; lw = 1.
        elif m == 3:
            color = 'blue';  ls = '-.'; lw = 0.4
        elif m == 4:
            color = 'orange';  ls = '-.'; lw = 0.4
        elif m == 5:
            color = 'green';  ls = '-.'; lw = 0.4
        elif m == 6:
            color = 'pink';  ls = '-.'; lw = 0.4
        elif m == 7:
            color = 'purple';  ls = '-.'; lw = 0.4
        else:
            raise ValueError('m is not in color/ls list')

        group = dfile["m=%d" % m]
        int_phi2d = np.array(group["int_phi"])
        int_phi_r1d = np.array(group["int_phi_r"])  # | time   <-> r
        # times = np.array(group["times"])
        # r = np.array(group["r_cyl"])

        # phase plot
        ax_list[0].plot(times * 1e3, np.unwrap(np.angle(int_phi_r1d)), ls, lw=lw, color=color, label='m:{}'.format(m))

        ax_list[0].set_ylabel(r'$C_m/C_0$ Phase')
        # ax_list[0].annotate(r'$C_m = \int{\rho W \sqrt{-g}\cdot\exp(i m \phi) dz dr d\phi}$',
        #             xy=(-150 / 180 * np.pi, 50),  # theta, radius
        #             xytext=(0.65, 0.90),  # fraction, fraction
        #             xycoords='axes fraction',
        #             horizontalalignment='center',
        #             verticalalignment='center'
        #             )
        ax_list[0].legend()

        # magnitude plot
        ax_list[1].plot(times * 1e3, np.abs(int_phi_r1d), ls, lw=lw, color=color, label='m:{}'.format(m))

        ax_list[1].set_yscale('log')
        ax_list[1].set_ylabel(r'$C_m/C_0$ Magnitude')
        ax_list[1].legend()

    plt.savefig(Paths.ppr_sims + sim + '/res_3d/' + load_file.replace(".h5", ".png"), bbox_inches='tight', dpi=128)
    plt.close()
    exit(0)

def plot_density_modes(load_file="density_modes.h5"):

    sim = "DD2_M13641364_M0_LK_HR_R04"
    m_list = [1,2]


    dfile = h5py.File(Paths.ppr_sims + sim + '/res_3d/' + load_file, "r")

    rows = 2
    cols = 1

    fig = plt.figure(figsize=(6.5, 1.5 * 3.6))  # figsize=(4.5, 2.5 * 3.6)  # (<->; v)
    ax_list = []
    for n in range(1, rows + 1):
        if n == 1:
            ax_list.append(fig.add_subplot(rows, cols, n))
        else:
            ax_list.append(fig.add_subplot(rows, cols, n, sharex=ax_list[n - 2]))  # sharex=axs[n - 2]))


    times = np.array(dfile["times"])
    iterations = np.array(dfile["iterations"])
    r_cyl = np.array(dfile["r_cyl"])

    for m in m_list:

        if m == 1:
            color = 'black'; ls = '-'; lw = 1.
        elif m == 2:
            color = 'gray';  ls = '-.'; lw = 1.
        elif m == 3:
            color = 'blue';  ls = '-.'; lw = 0.4
        elif m == 4:
            color = 'orange';  ls = '-.'; lw = 0.4
        elif m == 5:
            color = 'green';  ls = '-.'; lw = 0.4
        elif m == 6:
            color = 'pink';  ls = '-.'; lw = 0.4
        elif m == 7:
            color = 'purple';  ls = '-.'; lw = 0.4
        else:
            raise ValueError('m is not in color/ls list')

        group = dfile["m=%d" % m]
        int_phi2d = np.array(group["int_phi"])
        int_phi_r1d = np.array(group["int_phi_r"])  # | time   <-> r
        # times = np.array(group["times"])
        # r = np.array(group["r_cyl"])

        # phase plot
        ax_list[0].plot(times * 1e3, np.unwrap(np.angle(int_phi_r1d)), ls, lw=lw, color=color, label='m:{}'.format(m))

        ax_list[0].set_ylabel(r'$C_m/C_0$ Phase')
        # ax_list[0].annotate(r'$C_m = \int{\rho W \sqrt{-g}\cdot\exp(i m \phi) dz dr d\phi}$',
        #             xy=(-150 / 180 * np.pi, 50),  # theta, radius
        #             xytext=(0.65, 0.90),  # fraction, fraction
        #             xycoords='axes fraction',
        #             horizontalalignment='center',
        #             verticalalignment='center'
        #             )
        ax_list[0].legend()

        # magnitude plot
        ax_list[1].plot(times * 1e3, np.abs(int_phi_r1d), ls, lw=lw, color=color, label='m:{}'.format(m))

        ax_list[1].set_yscale('log')
        ax_list[1].set_ylabel(r'$C_m/C_0$ Magnitude')
        ax_list[1].legend()

    plt.savefig(Paths.ppr_sims + sim + '/res_3d/' + load_file.replace(".h5", ".png"), bbox_inches='tight', dpi=128)
    plt.close()
    exit(0)

def plot_corr_ang_mom_flux_dens_unb_bern_movie():


    sim = "DD2_M13641364_M0_LK_HR_R04"
    movie_dir = Paths.ppr_sims + sim + "/res_3d/" + "corr_ang_mom_flux_dens_unb_bern_movie/"

    if not os.path.isdir(movie_dir):
        os.mkdir(movie_dir)

    o_data = LOAD_RES_CORR(sim)
    o_plot = PLOT_MANY_TASKS(o_data, "cartesian")
    o_plot.gen_set["figdir"] = movie_dir
    o_plot.gen_set["figsize"] = (7.2, 3.5)  # <->, |]


    for it, time_ in zip(o_data.list_iterations, o_data.times):

        o_plot.gen_set["figname"] = "{0:07d}.png".format(int(it))

        corr_dic_ang_mom_flux_dens_unb_bern = {  # relies on the "get_res_corr(self, it, v_n): " method of data object
            'name': 'corr', 'position': (1, 1), 'title': 'time [ms]', 'cbar': 'left .15 .0',
            'it': it, 'v_n_x': 'dens_unb_bern', 'v_n_y': 'ang_mom_flux', 'v_n': 'mass',
            'xmin': 1e-11, 'xmax': 1e-7, 'ymin': 1e-11, 'ymax': 1e-7, 'vmin': 1e-7, 'vmax': None,
            'xscale': 'log', 'yscale': 'log',
            'mask_below': None, 'mask_above': None, 'cmap': 'inferno_r', 'norm': 'log', 'todo': None
        }
        corr_dic_r_phi = {  # relies on the "get_res_corr(self, it, v_n): " method of data object
            'name': 'corr', 'position': (1, 2), 'title': 'time [ms]', 'cbar': 'right .03 .0',
            'it': it, 'v_n_x': 'phi', 'v_n_y': 'r', 'v_n': 'mass',
            'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None, 'vmin': 1e-6, 'vmax': None,
            'xscale': 'line', 'yscale': 'line',
            'mask_below': None, 'mask_above': None, 'cmap': 'inferno', 'norm': 'log', 'todo': None
        }

        o_plot.set_plot_dics = [corr_dic_ang_mom_flux_dens_unb_bern, corr_dic_r_phi]

        o_plot.main()

        # exit(1)

def plot_ang_mom_flux_dens_unb_bern_movie():

    sim = "DD2_M13641364_M0_LK_HR_R04"
    movie_dir = Paths.ppr_sims + sim + "/res_3d/" + "ang_mom_flux_dens_unb_bern_movie/"

    if not os.path.isdir(movie_dir):
        os.mkdir(movie_dir)

    o_data = LOAD_INT_DATA(sim)
    o_data.flag_force_unique_grid = True
    o_data.it_for_unique_grid = 1363968# 2254356
    o_plot = PLOT_MANY_TASKS(o_data, "polar")
    o_plot.gen_set["figdir"] = movie_dir
    o_plot.gen_set["type"] = "polar"
    o_plot.gen_set["figsize"] = (7.2, 3.5)  # <->, |]

    for it, time in zip(o_data.list_iterations, o_data.times):

        o_plot.gen_set["figname"] = "{0:07d}.png".format(int(it))

        int_ang_mom_flux_dic = {
            'name': 'int', 'position': (1, 1), 'title': 'time [ms]', 'cbar': 'left .15 .0',
            'it': it, 'v_n_x': 'phi_cyl', 'v_n_y': 'r_cyl', 'v_n': 'ang_mom_flux',
            'phimin': None, 'phimax': None, 'rmin': 0, 'rmax': 50, 'vmin': 1e-8, 'vmax': 1e-5,
            'xscale': None, 'yscale': None,
            'mask_below': None, 'mask_above': None, 'cmap': 'RdBu_r', 'norm': "log", 'todo': None
        }
        deisty_countour = {
            'name': 'count', 'position': (1, 1), 'title': None, 'cbar': None,
            'it': it, 'v_n_x': 'phi_cyl', 'v_n_y': 'r_cyl', 'v_n': 'density',
            'phimin': None, 'phimax': None, 'rmin': 0, 'rmax': 50, 'levels': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
            'xscale': None, 'yscale': None,
            'mask_below': None, 'mask_above': None, 'colors': 'gray', 'norm': "log", 'todo': None
        }
        densmode = {
            'name': 'densmode', 'position': (1, 1), 'title': None, 'cbar': None,
            'mode': 1, 'dfile': h5py.File(Paths.ppr_sims + sim + '/res_3d/' + "density_modes.h5", "r"),
            'it': it, 'rmax': 50, 'plot_int_phi': True, 'plot_int_phi_r': True,
        }


        int_deisty_dic = {
            'name': 'int', 'position': (1, 2), 'title': 'time [ms]', 'cbar': 'right .03 .0',
            'it': it, 'v_n_x': 'phi_cyl', 'v_n_y': 'r_cyl', 'v_n': 'density',
            'phimin': None, 'phimax': None, 'rmin': 0, 'rmax': 50, 'vmin': 1e-8, 'vmax': 1e-5,
            'xscale': None, 'yscale': None,
            'mask_below': None, 'mask_above': None, 'cmap': 'inferno', 'norm': "log", 'todo': None
        }
        rho_countour = {
            'name': 'count', 'position': (1, 2), 'title': None, 'cbar': None,
            'it': it, 'v_n_x': 'phi_cyl', 'v_n_y': 'r_cyl', 'v_n': 'rho',
            'phimin': None, 'phimax': None, 'rmin': 0, 'rmax': 50, 'levels': [Constants.ns_rho],
            'xscale': None, 'yscale': None,
            'mask_below': None, 'mask_above': None, 'colors': 'gray', 'norm': "log", 'todo': None
        }


        o_plot.set_plot_dics = [int_ang_mom_flux_dic, densmode, int_deisty_dic, rho_countour]

        o_plot.main()

        # exit(1)


def plot_max_of_corr_phi_r_density_mode_phase_for_all_it():

    sim = "DD2_M13641364_M0_LK_SR_R04"

    out_dir = Paths.ppr_sims + sim + "/res_3d/" + "spiral_arm_search/"

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    o_corr = LOAD_RES_CORR(sim)

    def get_all_for_r(r):

        all_phi_arr = []
        all_mass_for_r = []
        all_times = []

        for it, time_ in zip(o_corr.list_iterations, o_corr.times):
            try:
                table = o_corr.get_res_corr(it, "r", "phi")
                table = np.array(table)
                r_arr = table[0, 1:]  # * 6.176269145886162e+17
                phi_arr = table[1:, 0]
                mass_arr = table[1:, 1:]
                mass_arr = np.maximum(mass_arr, 1e-12)  # WHAT'S THAT?
                phi_arr = 180 + (180 * phi_arr / np.pi)

                # print(table); exit(1)

                # print("r:{} phi:{} mass:{}".format(r_arr.shape, phi_arr.shape, mass_arr.shape))

                r_idx = find_nearest_index(r_arr, r)

                # mass_for_r = mass_arr[r_idx, :]
                mass_for_r = mass_arr[:, r_idx] # correct

                all_phi_arr = phi_arr
                all_mass_for_r.append(mass_for_r)
                all_times.append(time_)
            except:
                Printcolor.yellow("Failed it:{} time:{}".format(it, time_))

            # plt.plot(phi_arr, mass_for_r, '.', color="black")
            # plt.ylabel(r'Mass [Jflux $>$ 1e-12]', fontsize=12)
            # plt.xlabel(r'$\phi$ [deg]', fontsize=12)
            # plt.minorticks_on()
            # plt.yscale("log")
            # plt.xticks(fontsize=12)
            # plt.yticks(fontsize=12)
            # plt.title('Spiral arm', fontsize=20)
            # plt.legend(loc='upper right', numpoints=1)
            # plt.savefig(Paths.ppr_sims+sim+"/res_3d/"+"spiral_arm.png", bbox_inches='tight', dpi=128)
            # plt.close()

        all_phi_arr = np.array(all_phi_arr)
        all_mass_for_r = np.array(all_mass_for_r)
        all_times = np.array(all_times)

        all_mass_for_r = np.reshape(all_mass_for_r, (len(all_times), len(all_phi_arr)))

        print("time:{} phi:{} mass:{}".format(all_times.shape, all_phi_arr.shape, all_mass_for_r.shape))

        return all_phi_arr, all_times, all_mass_for_r,

    def get_dens_mode(mode=1):

        dfile = h5py.File(Paths.ppr_sims + sim + '/res_3d/' + "density_modes_lap15.h5", "r")


        # dfile = h5py.File(Paths.ppr_sims + sim + '/res_3d/' + load_file, "r")
        group = dfile["m=%d" % mode]
        times = np.array(dfile["times"])
        # iterations = np.array(dfile["iterations"])
        # r_cyl = np.array(dfile["r_cyl"])

        # int_phi2d = np.array(group["int_phi"])
        int_phi_r1d = np.array(group["int_phi_r"])

        angles = np.array([np.angle(comp_mode, deg=True)+180 for comp_mode in int_phi_r1d])

        # angles = np.unwrap(np.angle(int_phi_r1d, deg=True))+180

        return angles, times



    fig = plt.figure(figsize=(3.2, 7.6))


    cmap = "inferno"
    vmin = 1e-7
    vmax = 1e-5


    m1_phi, m1_t = get_dens_mode()
    # ax.plot(m1_phi, m1_phi, '-', color='white')

    r = 12
    ax = fig.add_subplot(111)
    all_phi_arr, all_times, all_mass_for_r = get_all_for_r(r)
    norm = LogNorm(vmin=vmin, vmax=vmax)#all_mass_for_r.max())
    im = ax.pcolormesh(all_phi_arr, all_times, all_mass_for_r, norm=norm,
                       cmap=cmap)  # , vmin=dic["vmin"], vmax=dic["vmax"])
    im.set_rasterized(True)
    plt.minorticks_on()
    ax.set_xlabel(r'$\phi$ [deg]', fontsize=12)
    ax.set_ylabel(r'time [s]', fontsize=12)
    ax.set_title("R:{}".format(r))
    ax.plot(m1_phi, m1_t, '-', color='lightblue', lw=3.)

    # r = 14
    # ax = fig.add_subplot(142)
    # all_phi_arr, all_times, all_mass_for_r = get_all_for_r(r)
    # norm = LogNorm(vmin=vmin, vmax=vmax)#all_mass_for_r.max())
    # im = ax.pcolormesh(all_phi_arr, all_times, all_mass_for_r, norm=norm,
    #                    cmap=cmap)  # , vmin=dic["vmin"], vmax=dic["vmax"])
    # im.set_rasterized(True)
    # plt.minorticks_on()
    # ax.set_xlabel(r'$\phi$ [deg]', fontsize=12)
    # # ax.set_ylabel(r'time [s]', fontsize=12)
    # ax.set_title("R:{}".format(r))
    # ax.plot(m1_phi, m1_t, '-', color='blue')
    #
    # r = 16
    # ax = fig.add_subplot(143)
    # all_phi_arr, all_times, all_mass_for_r = get_all_for_r(r)
    # norm = LogNorm(vmin=vmin, vmax=vmax)#all_mass_for_r.max())
    # im = ax.pcolormesh(all_phi_arr, all_times, all_mass_for_r, norm=norm,
    #                    cmap=cmap)  # , vmin=dic["vmin"], vmax=dic["vmax"])
    # im.set_rasterized(True)
    # plt.minorticks_on()
    # ax.set_xlabel(r'$\phi$ [deg]', fontsize=12)
    # # ax.set_ylabel(r'time [s]', fontsize=12)
    # ax.set_title("R:{}".format(r))
    # ax.plot(m1_phi, m1_t, '-', color='blue')
    #
    # r = 18
    # ax = fig.add_subplot(144)
    # all_phi_arr, all_times, all_mass_for_r = get_all_for_r(r)
    # norm = LogNorm(vmin=vmin, vmax=vmax)#all_mass_for_r.max())
    # im = ax.pcolormesh(all_phi_arr, all_times, all_mass_for_r, norm=norm,
    #                    cmap=cmap)  # , vmin=dic["vmin"], vmax=dic["vmax"])
    # im.set_rasterized(True)
    # plt.minorticks_on()
    # ax.set_xlabel(r'$\phi$ [deg]', fontsize=12)
    # # ax.set_ylabel(r'time [s]', fontsize=12)
    # ax.set_title("R:{}".format(r))
    # ax.plot(m1_phi, m1_t, '-', color='blue')
    #
    plt.savefig(out_dir + "many_r.png", bbox_inches='tight', dpi=128)
    plt.close()

    exit(1)


def plot_phi0_slice_Jflux_with_time():
    sim = "DD2_M13641364_M0_SR"
    out_dir = Paths.ppr_sims + sim + "/res_3d/" + "spiral_arm_search/"

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    o_data = LOAD_INT_DATA(sim)
    o_data.flag_force_unique_grid = True


    jf_slice = []
    times = []
    r = []
    for it, time_ in zip(o_data.list_iterations, o_data.times):
        print("\tit:{} ({}/{})".format(it,len(times), len(o_data.list_iterations)))
        times.append(time_)
        jflux = o_data.get_int_data(it, "ang_mom_flux")
        r = o_data.get_grid_data(it, "r_cyl")[:,0,0] # 1d r
        jf_slice.append(jflux[:,0,0]) # slice for phiz=0 and z=0

    r = np.array(r)
    times = np.array(times)
    jf_slice = np.array(jf_slice)

    jf_slice = np.reshape(jf_slice, (len(times), len(r)))

    # print(jf_slice)
    # print('\n')
    # print(r)

    arr1 = np.ma.masked_array(jf_slice.T, jf_slice.T < 0)  # jf_slice.T[jf_slice.T>0]
    arr2 = -1 * np.ma.masked_array(jf_slice.T, jf_slice.T > 0)

    print(arr1)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    im1 = ax.pcolormesh(times * 1e3, r, arr1, norm=LogNorm(vmin=1e-8, vmax=1e-5), cmap='Reds')
    im2 = ax.pcolormesh(times * 1e3, r, arr2, norm=LogNorm(vmin=1e-8, vmax=1e-5), cmap='Blues')
    ax.set_xlabel("time [ms]", fontsize=12)
    ax.set_ylabel("radius [km]", fontsize=12)
    ax.minorticks_on()
    ax.set_ylim(0, 50)
    divider1 = make_axes_locatable(ax)
    cax = divider1.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im1, cax=cax)
    cbar.set_label(r"Jflux")
    cbar.ax.minorticks_off()
    # cbar = plt.colorbar(im, cax=cax1, extend='both')


    ax = fig.add_subplot(212)
    im1 = ax.pcolormesh(times * 1e3, r, arr1, norm=LogNorm(vmin=1e-12, vmax=1e-7), cmap='Reds')
    im2 = ax.pcolormesh(times * 1e3, r, arr2, norm=LogNorm(vmin=1e-12, vmax=1e-7), cmap='Blues')
    ax.set_xlabel("time [ms]", fontsize=12)
    ax.set_ylabel("radius [km]", fontsize=12)
    ax.minorticks_on()
    divider1 = make_axes_locatable(ax)
    cax = divider1.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im1, cax=cax)
    cbar.set_label(r"Jflux")
    cbar.ax.minorticks_off()
    ax.set_ylim(0, 500)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(out_dir + 'phi_slice_with_time.png', bbox_inches='tight', dpi=128)
    plt.close()
    exit(1)


""" --- --- TESTING/DEBUGGING --- --- """

def compute_density_modes_from_profiles():

    import numexpr as ne

    symmetry="pi"
    sim = "DD2_M13641364_M0_LK_LR_R04_PI"
    mmax = 8
    profs_loc = Paths.gw170817+sim+"/profiles/3d/"
    out_dir = Paths.ppr_sims + sim + "/res_3d/"
    outfname = Paths.ppr_sims+sim+'/res_3d/density_modes_lap15.h5'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)



    list_iterations = np.array(get_list_iterationsfrom_profiles_3d(sim, in_sim_dir="/profiles/3d/"))
    times_ = interpoate_time_form_it(list_iterations, Paths.gw170817 + sim + '/')

    iterations = list_iterations[times_>0.005]
    # print(iterations)
    # exit(1)


    times = []
    modes = [[] for m in range(mmax + 1)]
    xcs = []
    ycs = []

    for idx, it in enumerate(iterations):
        print("processing iteration: {}/{}".format(idx, len(iterations)))
        o_methods = MAINMETHODS_STORE(profs_loc+"{}.h5".format(it), sim, symmetry=symmetry)

        rl = o_methods.nlevels-1

        lapse =  o_methods.get_prof_arr(rl, "lapse")[:,:,0]
        rho = o_methods.get_prof_arr(rl, "rho")[:,:,0]
        vol = o_methods.get_prof_arr(rl, "vol")[:,:,0]
        w_lorentz = o_methods.get_prof_arr(rl, "w_lorentz")[:,:,0]


        delta = o_methods.get_prof_delta(rl)[:-1]
        # print(delta); exit(0)
        dxyz = np.prod(delta)
        x, y, z = o_methods.get_prof_x_y_z(rl)
        x=x[:,:,0]
        y=y[:,:,0]



        rho[lapse < 0.15] = 0


        # print(x); exit(1)

        # Exclude region outside refinement levels
        idx = np.isnan(rho)
        rho[idx] = 0.0
        vol[idx] = 0.0
        w_lorentz[idx] = 0.0

        # Compute center of mass
        modes[0].append(dxyz * ne.evaluate("sum(rho * w_lorentz * vol)"))
        Ix = dxyz * ne.evaluate("sum(rho * w_lorentz * vol * x)")
        Iy = dxyz * ne.evaluate("sum(rho * w_lorentz * vol * y)")
        xc = Ix / modes[0][-1]
        yc = Iy / modes[0][-1]
        phi = ne.evaluate("arctan2(y - yc, x - xc)")
        # phi = ne.evaluate("arctan2(y, x)")

        xcs.append(xc)
        ycs.append(yc)

        # Extract modes
        times.append(o_methods.time)
        for m in range(1, mmax + 1):
            modes[m].append(dxyz * ne.evaluate("sum(rho * w_lorentz * vol * exp(-1j * m * phi))"))

    dfile = h5py.File(outfname, "w")
    dfile.create_dataset("times", data=times)
    dfile.create_dataset("iterations", data=iterations)
    dfile.create_dataset("xc", data=xcs)
    dfile.create_dataset("yc", data=ycs)
    for m in range(mmax + 1):
        group = dfile.create_group("m=%d" % m)
        group["int_phi"] = np.zeros(0,)
        group["int_phi_r"] = np.array(modes[m]).flatten()
        # dfile.create_dataset(("m=%d" % m), data=modes[m])
    dfile.close()

    exit(1)


if __name__ == "__main__":

    ''' debug '''
    # do_produce_interpolated_data_for_iterations()
    do_plot_r_theta_summed_over_phi_interpolated_data()
    # profs = LOAD_PROFILE("SFHo_M14521283_M0_LK_HR")

    # print(profs.get_profile_dfile(704512))
    # print(profs.get_group(704512, 0))
    # print(profs.get_grid(704512))
    # print(profs.get_grid_data(704512, 0, "x"))
    print("delta",  profs.get_grid_data(704512, 0, "delta"))
    print("extent", profs.get_grid_data(704512, 0, "extent"))
    print("origin", profs.get_grid_data(704512, 0, "origin"))
    # print(profs.get_prof_arr(704512, 0, "rho"))
    exit(1)

    # data = COMPUTE_STORE("SFHo_M14521283_M0_LK_HR")
    # print(data.get_comp_data(704512, 0, "density"))
    # exit(1)

    # mask = MASK_STORE("SFHo_M14521283_M0_LK_HR")
    # print(mask.get_masked_data(704512, 0, "density"))
    # exit(1)

    # _, _, task_for_int = setup()
    # int_ = INTERPOLATE_STORE("SFHo_M14521283_M0_LK_HR", CYLINDRICAL_GRID(task_for_int["grid"]))
    # print(int_.get_int(704512, "density"))
    # exit(1)

    # for i in range(11, 21, 1):
    #     print("tar -xvf output-00{}.tar --directory /data1/numrel/WhiskyTHC/Backup/2018/GW170817/DD2_M13641364_M0_LK_LR_R04_PI/; ".format(i))


    # compute_density_modes_from_profiles()
    # do_compute_save_vtk_for_it()
    # plot_max_of_corr_phi_r_density_mode_phase_for_all_it()
    # plot_phi0_slice_Jflux_with_time()

    # corr = LOAD_RES_CORR("SFHo_M14521283_M0_LK_HR")
    # table = corr.get_res_corr(704512, "ang_mom_flux", "theta")
    # print(table[1:, 0])
    # print(table[0, 1:])
    # print(table[1:, 1:])
    # exit(1)
    # do_histogram_processing_of_iterations()
    # do_produce_interpolated_data_for_iterations()
    # do_compute_save_all_modes_for_all_it()
    # plot_corr_ang_mom_flux_dens_unb_bern_movie()
    # plot_ang_mom_flux_dens_unb_bern_movie()

    # times, files = get_profiles("DD2_M13641364_M0_LR_R04",
    #                             time_list=[], it_list=[], n_more=0, ftype='.h5', time_units='s', add_path='profiles/3d/')
    # exit(1)


    # --- DEBUGGING ---
    # o_prof = LOAD_PROFILE("/data1/numrel/WhiskyTHC/Backup/2018/GW170817/SLy4_M13641364_M0_SR/profiles/434176.h5")

    # o_data = COMPUTE_STORE("/data1/numrel/WhiskyTHC/Backup/2018/GW170817/SLy4_M13641364_M0_SR/profiles/434176.h5")
    # print(o_data.get_prof_arr(0, "rho").shape); exit(1) # (262, 262, 134)
    # print(o_data.get_comp_data(2, "dens_unb_bern"))

    # o_mask = MASK_STORE("/data1/numrel/WhiskyTHC/Backup/2018/GW170817/SLy4_M13641364_M0_SR/profiles/434176.h5")
    # print(o_mask.get_masked_data(2, "rho"))
    # plot_density_modes()

    ''' MAIN '''
    # sim = "DD2_M13641364_M0_SR"
    # it = 1111116

    # sim = "DD2_M13641364_M0_LK_SR_R04"; times: 0.084 0.073 0.062 0.051
    # sim = "DD2_M13641364_M0_LK_HR_R04"; times: 0.042 0.038 0.031
    # sim = "DD2_M13641364_M0_SR_R04";    times: 0.098, 0.094, 0.083


    #  ang_mom is in (-8.03540793958e-09->0.000251518721152)
    # DD2:  DD2_M13641364_M0_SR/profiles/3d/2025090.h5
    # SLy4: SLy4_M13641364_M0_SR/pr    print(o_methods.get_masked_data(it, rl=6, v_n="density"))
    # LS220:LS220_M13641364_M0_SR/profiles/3d/1081344.h5
    #
    it = 688128
    o_methods = MAINMETHODS_STORE("SFHo_M14521283_M0_LK_HR", symmetry=None)

    # print(o_methods.get_comp_data(it, rl=0, v_n="density"))
    # print(np.sum(o_methods.get_masked_data(it, rl=6, v_n="density")))
    # print(np.prod(o_methods.get_grid_data(it, rl=6, v_n="delta")))
    # print(np.prod(o_methods.get_comp_data(it, rl=6, v_n="delta")))
    # exit(1)
    # print(o_methods.get_total_mass(it, save=True, overwrite=True))
    # print(np.sum(o_methods.get_correlation(it, o_methods.corr_task_dic_rho_ye, save=True, overwrite=True)))
    # print(np.sum(o_methods.get_correlation(it, o_methods.corr_task_dic_temp_ye, save=True, overwrite=True)))
    # print(np.sum(o_methods.get_correlation(it, o_methods.corr_task_dic_rho_r, save=True, overwrite=True)))
    # print(np.sum(o_methods.get_correlation(it, o_methods.corr_task_dic_rho_theta, save=True, overwrite=True)))
    # print(np.sum(o_methods.get_correlation(it, o_methods.corr_task_dic_rho_ang_mom, save=True, overwrite=True)))
    # print(np.sum(o_methods.get_correlation(it, o_methods.corr_task_dic_rho_ang_mom_flux, save=True, overwrite=True)))
    # print(np.sum(o_methods.get_correlation(it, o_methods.corr_task_dic_rho_dens_unb_bern, save=True, overwrite=True)))
    # print(np.sum(o_methods.get_correlation(it, o_methods.corr_task_dic_ang_mom_flux_theta, save=True, overwrite=True)))
    # print(np.sum(o_methods.get_correlation(it, o_methods.corr_task_dic_ang_mom_flux_dens_unb_bern, save=True, overwrite=True)))
    # print(np.sum(o_methods.get_correlation(it, o_methods.corr_task_dic_inv_ang_mom_flux_dens_unb_bern, save=True, overwrite=True)))
    exit(1)
    ''' PLOTTING '''
    # o_plot = PLOT_MANY_TASKS(sim)
    # o_plot.main()
    # exit(1)
    ''' TESTING '''
    # test the pi symmetry runs plotting
    sim = "DD2_M13641364_M0_LK_LR_R04_PI"
    prof = "860160.h5"
    profs_loc = Paths.gw170817 + sim + "/profiles/3d/"
    _, _, task_for_int = setup()
    int_ = INTERPOLATE_STORE(profs_loc + prof,
                             sim,
                             CYLINDRICAL_GRID(task_for_int["grid"]),
                             symmetry='pi')
    x, y, z = int_.get_prof_x_y_z(4)
    data = int_.get_comp_data(4, "temp")

    from matplotlib import colors
    fig = plt.figure()
    ax = fig.add_subplot(111)
    norm = colors.LogNorm(vmin=data.min(), vmax=data.max())
    ax.pcolormesh(x[:,:,0], y[:,:,0], data[:,:,0], norm=norm, cmap="inferno_r")
    plt.title(r"copy $x>0$ and place to $x<0$ and invert $y$")
    plt.savefig('{}'.format(Paths.plots+"pi_test2.png"), bbox_inches='tight', dpi=128)
    print("saved pi_test2.png")
    plt.close()
    #
    #
    # int_.get_masked_data(0, 'x')
    # int_.get_int("x_cyl")
    # print(int_.get_int("rho"))

    # int_ = INTMETHODS_STORE("/data1/numrel/WhiskyTHC/Backup/2018/GW170817/"
    #                               "DD2_M13641364_M0_LK_SR_R04/profiles/3d/1818738.h5",
    #                          "DD2_M13641364_M0_LK_SR_R04",
    #                          CYLINDRICAL_GRID(task_for_int["grid"]))
    # int_.save_new_grid()
    # int_.save_int_v_n("density")
    # int_.save_int_v_n("ang_mom_flux")
    # int_.save_int_v_n("dens_unb_bern")

    # load interpolated data for plotting and processing
    # load_int = LOAD_INT_DATA("DD2_M13641364_M0_LK_SR_R04")
    # print(load_int.get_grid_data(1818738, "r_cyl"))

    # o_plot = PLOT_TASK(sim)
    # o_plot.plot_summed_correlation_with_time()

    # o_dm = LOAD_DENSITY_MODES(sim)
    # print(o_dm.get_data(1, 'int_phi_r'))
    # print(o_dm.get_data(1, 'int_phi'))
    # print(o_dm.get_grid('r_cyl'))


    ''''''





    # for rl in range(o_methods.nlevels):
    #     data = o_methods.get_masked_data(rl, "dens_unb_bern")
    #     print("rl:{} min:{} max:{}".format(rl, data.min(), data.max()))
    # exit(1)

    # o_res = LOAD_RES_CORR("DD2_M13641364_M0_SR")
    # print(np.sum(o_res.get_res_corr(2237972, "temp_Ye")[1:,1:]))# should yeid the disk mass


    # o_plot = PLOT_MANY_TASKS("SLy4_M13641364_M0_SR")
    # o_plot.main()

    # o_plot = PLOT_MANY()