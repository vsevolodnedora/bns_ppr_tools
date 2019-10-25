###################################################################################
#                                                                                 #
# This is a comprehensive set of analysis methods and tools                       #
# for the standart output of the Neutron Star Merger simulations                  #
# done with WhiskyTHC code.                                                       #
# For D1 analysis:                                                                #
#   :required: output of the Hydro Thorn of Cactus (for extraction spheres)       #
# For D2 analysis:                                                                #
#   :required: variable.xy.h5 files containing the xy ad xz slices of variables   #
# For D3 analysis:                                                                #
#   :required: iteration.h5 - profiles, containing 3D data for multiple variables #
#                             already put on one grid (for each refinment level)  #
# To use: run the analysis.py                                                     #
###################################################################################
from __future__ import division
from sys import path

from py.path import local

path.append('modules/')
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import warnings
import os.path
import click

warnings.filterwarnings("ignore",category=matplotlib.mplDeprecation)

from scidata.carpet.interp import Interpolator


from math import log10
import sys
from argparse import ArgumentParser

import h5py

import time



from preanalysis import SIM_STATUS, PRINT_SIM_STATUS, LOAD_ITTIME
from plotting_methods import PLOT_MANY_TASKS
from utils import *


""" ==============================================| SETTINGS |====================================================== """
__rootoutdir__ = "profiles/"
__profile__ = {"tasklist": ["all", "corr", "hist", "slice", "mass", "densmode", "vtk",
                            "plotall", "plotcorr", "plothist", "plotslice", "plotmass", "plotdensmode"]}
__d3slicesvns__ = ["x", "y", "z", "rho", "w_lorentz", "vol", "press", "eps", "lapse", "velx", "vely", "velz",
                    "gxx", "gxy", "gxz", "gyy", "gyz", "gzz", "betax", "betay", "betaz", 'temp', 'Ye'] + \
                    ["density",  "enthalpy", "vphi", "vr", "dens_unb_geo", "dens_unb_bern", "dens_unb_garch",
                    "ang_mom", "ang_mom_flux", "theta", "r", "phi" ]
__d3corrs__ = ["rho_r", "rho_Ye", "temp_Ye", "rho_theta", "velz_theta", "rho_ang_mom", "velz_Ye",
                    "rho_ang_mom_flux", "rho_dens_unb_bern", "ang_mom_flux_theta",
                    "ang_mom_flux_dens_unb_bern", "inv_ang_mom_flux_dens_unb_bern",
                    "velz_dens_unb_bern", "Ye_dens_unb_bern", "theta_dens_unb_bern"]
__d3histvns__      = ["r", "theta", "Ye", "temp", "velz", "rho", "dens_unb_bern"]
__d3slicesplanes__ = ["xy", "xz"]
__d3diskmass__ = "disk_mass.txt"
__d3densitymodesfame__ = "density_modes_lap15.h5"
# --- ploting ---
__d3sliceplotvns__ = ["Ye", "velx", "rho", "ang_mom_flux","ang_mom","dens_unb_garch",
                      "dens_unb_bern","dens_unb_geo","vr","vphi","enthalpy",
                        "density","temp","velz","vely","lapse","eps","press","vol","w_lorentz",
                      "Q_eff_nua", "Q_eff_nue", "Q_eff_nux"]
__d3sliceplotrls__ = [0, 1, 2, 3, 4, 5, 6]

""" ==========================================| GRID CLASSES |====================================================== """

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

    def __init__(self):

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

    def save_grid(self, sim, outdir="profiles/"):

        path = Paths.ppr_sims + sim + '/' + outdir
        outfile = h5py.File(path + str(self.grid_type) + '_grid.h5', "w")

        if not os.path.exists(path):
            os.makedirs(path)

        # print("Saving grid...")
        for v_n in self.list_int_grid_v_ns:
            outfile.create_dataset(v_n, data=self.get_int_grid(v_n))
        outfile.close()


class SPHERICAL_GRID:
    """
        Creates a stretched spherical grid and allows
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

        # print("\t GRID: [phi_sph:   ({},{})] {} pints".format(phi_sph[0], phi_sph[-1], len(phi_sph)))
        # print("\t GRID: [r_sph:     ({},{})] {} pints".format(r_sph[0], r_sph[-1], len(r_sph), len(r_sph)))
        # print("\t GRID: [theta_sph: ({},{})] {} pints".format(theta_sph[0], theta_sph[-1], len(theta_sph)))
        # print('   --- --- ---   ')
        # print("\t GRID: [x_sph_3d:  ({},{})] {} pints".format(self.x_sph_3d.min(), self.x_sph_3d.max(), len(self.x_sph_3d[:,0,0])))
        # print("\t GRID: [y_sph_3d:  ({},{})] {} pints".format(self.y_sph_3d.min(), self.y_sph_3d.max(), len(self.y_sph_3d[0,:,0])))
        # print("\t GRID: [z_sph_3d:  ({},{})] {} pints".format(self.z_sph_3d.min(), self.z_sph_3d.max(), len(self.z_sph_3d[0,0,:])))
        # print('   --- --- ---   ')
        # print("\t GRID: [x_sph_3d:  ({},{})] {} pints".format(self.x_sph_3d[0,0,0], self.x_sph_3d[0,-1,0], len(self.x_sph_3d[0,:,0])))
        # print("\t GRID: [y_sph_3d:  ({},{})] {} pints".format(self.y_sph_3d[0,0,0], self.y_sph_3d[-1,0,0], len(self.y_sph_3d[:,0,0])))
        # print("\t GRID: [z_sph_3d:  ({},{})] {} pints".format(self.z_sph_3d[0,0,0], self.z_sph_3d[0,0,-1], len(self.z_sph_3d[0,0,:])))

        print("\t GRID: [phi:r:theta] = [{}:{}:{}]".format(len(phi_sph), len(r_sph), len(theta_sph)))

        print('-' * 30 + '--------DONE-------' + '-' * 30)
        print('\n')

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

    def save_grid(self, sim, outdir="profiles/"):

        path = Paths.ppr_sims + sim + '/' + outdir
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
            "reflecting_xy": True,   # Apply reflection symmetry across the xy-plane
            "xmin": -150.0,          # Include region with x >= xmin
            "xmax": 150.0,           # Include region with x <= xmax
            "xix": 0.2,              # Stretch factor for the grid in the x-direction
            "nlinx": 120,            # Number of grid points in the linear portion of the x-grid
            "nlogx": 200,            # Number of grid points in the log portion of the x-grid
            "ymin": -150,            # Include region with y >= ymin
            "ymax": 150,             # Include region with y <= ymax
            "xiy": 0.2,              # Stretch factor for the grid in the y-direction
            "nliny": 120,            # Number of grid points in the linear portion of the y-grid
            "nlogy": 200,            # Number of grid points in the log portion of the y-grid
            "zmin": -100.0,           # Include region with z >= zmin
            "zmax": 100.0,            # Include region with z <= zmax
            "xiz": 0.2,              # Stretch factor for the grid in the z-direction
            "nlinz": 120,            # Number of grid points in the linear portion of the z-grid
            "nlogz": 200,            # Number of grid points in the log portion of the z-grid
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

    def save_grid(self, sim, outdir="profiles/"):

        path = Paths.ppr_sims + sim + "/" + outdir
        outfile = h5py.File(path + self.grid_type + '_grid.h5', "w")

        if not os.path.exists(path):
            os.makedirs(path)

        # print("Saving grid...")
        for v_n in self.list_int_grid_v_ns:
            outfile.create_dataset(v_n, data=self.get_int_grid(v_n))
        outfile.close()

""" =========================================| FORMULAS for METHODS |=============================================== """

class FORMULAS:

    def __init__(self):
        pass

    @staticmethod
    def r(x, y):
        return np.sqrt(x ** 2 + y ** 2)# + z ** 2)

    @staticmethod
    def density(rho, w_lorentz, vol):
        return rho * w_lorentz * vol

    @staticmethod
    def vup(velx, vely, velz):
        return [velx, vely, velz]

    @staticmethod
    def metric(gxx, gxy, gxz, gyy, gyz, gzz):
        return [[gxx, gxy, gxz], [gxy, gyy, gyz], [gxz, gyz, gzz]]

    @staticmethod
    def enthalpy(eps, press, rho):
        return 1 + eps + (press / rho)

    @staticmethod
    def shift(betax, betay, betaz):
        return [betax, betay, betaz]

    @staticmethod
    def shvel(shift, vlow):
        shvel = np.zeros(shift[0].shape)
        for i in range(len(shift)):
            shvel += shift[i] * vlow[i]
        return shvel

    @staticmethod
    def u_0(w_lorentz, shvel, lapse):
        return w_lorentz * (shvel - lapse)

    @staticmethod
    def vlow(metric, vup):
        vlow = [np.zeros_like(vv) for vv in [vup[0], vup[1], vup[2]]]
        for i in range(3):  # for x, y
            for j in range(3):
                vlow[i][:] += metric[i][j][:] * vup[j][:]  # v_i = g_ij * v^j (lowering index) for x y
        return vlow

    @staticmethod
    def vphi(x, y, vlow):
        return -y * vlow[0] + x * vlow[1]

    @staticmethod
    def vr(x, y, r, vup):
        # r = np.sqrt(x ** 2 + y ** 2)
        # print("x: {}".format(x.shape))
        # print("y: {}".format(y.shape))
        # print("r: {}".format(y.shape))
        # print("vup[0]: {}".format(vup[0].shape))

        return (x / r) * vup[0] + (y / r) * vup[1]

    @staticmethod
    def theta(r, z):
        # r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return np.arccos(z/r)

    @staticmethod
    def phi(x, y):
        return np.arctan2(y, x)

    @staticmethod
    def dens_unb_geo(u_0, rho, w_lorentz, vol):

        c_geo = -u_0 - 1.0
        mask_geo = (c_geo > 0).astype(int)  # 1 or 0
        rho_unbnd_geo = rho * mask_geo  # if 1 -> same, if 0 -> masked
        dens_unbnd_geo = rho_unbnd_geo * w_lorentz * vol

        return dens_unbnd_geo

    @staticmethod
    def dens_unb_bern(enthalpy, u_0, rho, w_lorentz, vol):

        c_ber = -enthalpy * u_0 - 1.0
        mask_ber = (c_ber > 0).astype(int)
        rho_unbnd_bernoulli = rho * mask_ber
        density_unbnd_bernoulli = rho_unbnd_bernoulli * w_lorentz * vol

        return density_unbnd_bernoulli

    @staticmethod
    def dens_unb_garch(enthalpy, u_0, lapse, press, rho, w_lorentz, vol):

        c_ber = -enthalpy * u_0 - 1.0
        c_gar = c_ber - (lapse / w_lorentz) * (press / rho)
        mask_gar = (c_gar > 0).astype(int)
        rho_unbnd_garching = rho * mask_gar
        density_unbnd_garching = rho_unbnd_garching * w_lorentz * vol

        return density_unbnd_garching

    @staticmethod
    def ang_mom(rho, eps, press, w_lorentz, vol, vphi):
        return (rho * (1 + eps) + press) * w_lorentz * w_lorentz * vol * vphi

    @staticmethod
    def ang_mom_flux(ang_mom, lapse, vr):
        return ang_mom * lapse * vr

    # data manipulation methods
    @staticmethod
    def get_slice(x3d, y3d, z3d, data3d, slice='xy'):

        if slice == 'yz':
            ix0 = np.argmin(np.abs(x3d[:, 0, 0]))
            if abs(x3d[ix0, 0, 0]) < 1e-15:
                res = data3d[ix0, :, :]
            else:
                if x3d[ix0, 0, 0] > 0:
                    ix0 -= 1
                res = 0.5 * (data3d[ix0, :, :] + data3d[ix0 + 1, :, :])
        elif slice == 'xz':
            iy0 = np.argmin(np.abs(y3d[0, :, 0]))
            if abs(y3d[0, iy0, 0]) < 1e-15:
                res = data3d[:, iy0, :]
            else:
                if y3d[0, iy0, 0] > 0:
                    iy0 -= 1
                res = 0.5 * (data3d[:, iy0, :] + data3d[:, iy0 + 1, :])
        elif slice == 'xy':
            iz0 = np.argmin(np.abs(z3d[0, 0, :]))
            if abs(z3d[0, 0, iz0]) < 1e-15:
                res = data3d[:, :, iz0]
            else:
                if z3d[0, 0, iz0] > 0 and iz0 > 0:
                    iz0 -= 1
                res = 0.5 * (data3d[:, :, iz0] + data3d[:, :, iz0 + 1])
        else:
            raise ValueError("slice:{} not recognized. Use 'xy', 'yz' or 'xz' to get a slice")
        return res


    # --------- OUTFLOWED -----

    @staticmethod
    def vinf(eninf):
        return np.sqrt(2 * eninf)

    @staticmethod
    def vinf_bern(eninf, enthalpy):
        return np.sqrt(2*(enthalpy*(eninf + 1) - 1))

    @staticmethod
    def vel(w_lorentz):
        return np.sqrt(1 - 1 / (w_lorentz**2))

    @staticmethod
    def get_tau(rho, vel, radius, lrho_b):

        rho_b = 10 ** lrho_b
        tau_0 = 0.5 * 2.71828182845904523536 * (radius / vel) * (0.004925794970773136) # in ms
        tau_b = tau_0 * ((rho/rho_b) ** (1.0 / 3.0))
        return tau_b # ms

""" ======================================| PROFILE PROCESSING MENTHODS |=========================================== """

class LOAD_PROFILE(LOAD_ITTIME):

    """
        Loads profile.h5 and extract grid object using scidata
    """

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

    # ---

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

    # ---

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
        self.hist_task_dic_theta = {"v_n": "theta", "edges": np.linspace(0., np.pi / 2., 500)}
        self.hist_task_dic_ye = {"v_n": "Ye",   "edges": np.linspace(0, 0.5, 500)}
        self.hist_task_dic_temp = {"v_n": "temp", "edges": 10.0 ** np.linspace(-2, 2, 300)}
        self.hist_task_dic_velz = {"v_n": "velz", "edges": np.linspace(-1,1, 500)}
        self.hist_task_dic_rho = {"v_n": "rho", "edges": 10.0 ** np.linspace(4.0, 13.0, 500) / rho_const}
        self.hist_task_dens_unb_bern = {"v_n": "dens_unb_bern", "edges": 10.0 ** np.linspace(-12., -6., 500)}

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
        # print(edge); exit(1)
        histogram = np.zeros(len(edge) - 1)
        _edge = []
        for rl in range(self.nlevels):
            weights = self.get_masked_data(it, rl, "density").flatten() * \
                      np.prod(self.get_grid_data(it, rl, "delta")) * multiplier
            data = self.get_masked_data(it, rl, v_n)
            tmp1, _ = np.histogram(data, bins=edge, weights=weights)
            histogram = histogram + tmp1
        # print(len(histogram), len(_edge), len(edge))
        # assert len(histogram) == len(edge)# 0.5 * (edge_x[1:] + edge_x[:-1])
        outarr = np.vstack((0.5*(edge[1:]+edge[:-1]), histogram)).T
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

    def delete_for_it(self, it, except_v_ns, rm_masks=True, rm_comp=True, rm_prof=True):
        self.check_it(it)
        # clean up mask array
        if rm_masks:
            for rl in range(self.nlevels):
                self.mask_matrix[self.i_it(it)][rl] = np.ones(0, dtype=bool)
        # clean up data
        if rm_masks:
            for v_n in self.list_all_v_ns:
                if v_n not in except_v_ns:
                    self.check_v_n(v_n)
                    for rl in range(self.nlevels):
                        self.data_matrix[self.i_it(it)][rl][self.i_v_n(v_n)] = np.zeros(0, )

        # clean up the initial data
        if rm_prof:
            self.dfile_matrix[self.i_it(it)] = 0
            self.grid_matrix[self.i_it(it)] = 0
            for v_n in self.list_grid_v_ns:
                if not v_n in except_v_ns:
                    for rl in range(self.nlevels):
                        self.grid_data_matrix[self.i_it(it)][rl][self.i_grid_v_n(v_n)] = np.zeros(0,)


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

    # ---

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

    def __init__(self, sim, grid_object, symmetry=None):

        INTERPOLATE_STORE.__init__(self, sim, grid_object, symmetry)

    def save_new_grid(self, it, outdir="profiles/"):
        self.check_it(it)

        grid_type = self.new_grid.grid_info['type']

        if not os.path.exists(Paths.ppr_sims + self.sim + '/' + outdir):
            os.makedirs(Paths.ppr_sims + self.sim + '/' + outdir)

        path = Paths.ppr_sims + self.sim + '/' + outdir + str(it) + '/'

        if os.path.isfile(path + grid_type + '_grid.h5'):
            os.remove(path + grid_type + '_grid.h5')

        outfile = h5py.File(path + grid_type + '_grid.h5', "w")

        if not os.path.exists(path):
            os.makedirs(path)

        # print("Saving grid...")
        for v_n in self.list_int_grid_v_ns:
            outfile.create_dataset(v_n, data=self.new_grid.get_int_grid(v_n))
        outfile.close()

    def save_int_v_n(self, it, v_n, outdir="profiles/", overwrite=False):

        self.check_it(it)

        path = Paths.ppr_sims + self.sim + '/' + outdir
        if not os.path.isdir(path):
            os.mkdir(path)

        path = Paths.ppr_sims + self.sim + '/' + outdir + str(it) + '/'
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

    def save_vtk_file(self, it, v_n_s, overwrite=False, outdir="profiles/", private_dir="vtk"):

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

        path = Paths.ppr_sims + self.sim + '/' + outdir + str(it) + '/'
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

""" ====================================| LOAD RESULTS OF PROFILE PROCESSING |====================================== """

class LOAD_RES_CORR(LOAD_ITTIME):

    def __init__(self, sim):

        LOAD_ITTIME.__init__(self, sim)

        self.set_rootdir = __rootoutdir__

        self.sim = sim

        self.list_iterations = Paths.get_list_iterations_from_res_3d(sim, self.set_rootdir)
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
        fpath = Paths.ppr_sims + self.sim + '/' + self.set_rootdir + str(it) + "/corr_" + v_n + ".h5"
        if not os.path.isfile(fpath):
            raise IOError("Correlation file not found:\n{}".format(fpath))
        return fpath

    def load_corr_file(self, it, v_n_x, v_n_y):

        v_n_x = str(v_n_x)
        v_n_y = str(v_n_y)

        self.check_v_n(v_n_x)
        self.check_v_n(v_n_y)

        # check if the direct file exists or the inverse
        fpath_direct = Paths.ppr_sims + self.sim + '/' + self.set_rootdir + str(it) \
                       + "/corr_" + v_n_x + '_' + v_n_y + ".h5"
        fpath_inverse = Paths.ppr_sims + self.sim + '/' + self.set_rootdir + str(it) \
                        + "/corr_" + v_n_y + '_' + v_n_x + ".h5"
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
        result = UTILS.combine(arr_x, arr_y, mass)
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

        idx = UTILS.find_nearest_index(self.times, t)
        return self.list_iterations[idx]

    def load_corr3d(self, it, v_n_x, v_n_y, v_n_z):

        v_n_x = str(v_n_x)
        v_n_y = str(v_n_y)
        v_n_z = str(v_n_z)

        self.check_v_n(v_n_x)
        self.check_v_n(v_n_y)
        self.check_v_n(v_n_z)

        fpath_direct = Paths.ppr_sims + self.sim + '/' + self.set_rootdir + str(it) + "/corr_" + v_n_x + '_' + v_n_y + '_' + v_n_z + ".h5"
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

        self.set_rootdir = __rootoutdir__

        self.list_iterations = list(Paths.get_list_iterations_from_res_3d(sim, self.set_rootdir))
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
            Paths.get_it_from_itdir(Paths.find_itdir_with_grid(sim, "{}_grid.h5"
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

        idx = UTILS.find_nearest_index(self.times, t)
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

    def __init__(self, sim, grid_object):

        LOAD_INT_DATA.__init__(self, sim, grid_object)

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
            'outdir': Paths.ppr_sims + sim + '/' + self.set_rootdir,
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

# old class (unused)
class LOAD_DENSITY_MODES:

    def __init__(self, sim):

        self.sim = sim

        self.set_rootdir = __rootoutdir__

        self.gen_set = {
            'maximum_modes': 50,
            'fname' :  Paths.ppr_sims + sim + '/' + self.set_rootdir + "density_modes.h5",
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

        if not os.path.isfile(self.gen_set['fname']):
            raise IOError("file {} npt found".format(self.gen_set['fname']))
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


class LOAD_PROFILE_XYXZ(LOAD_ITTIME):

    def __init__(self, sim):

        LOAD_ITTIME.__init__(self, sim)

        self.nlevels = 7

        self.sim = sim

        self.set_rootdir = __rootoutdir__

        self.list_iterations = Paths.get_list_iterations_from_res_3d(sim, self.set_rootdir)
        # isprof, itprof, tprof = self.get_ittime("profiles", "")
        # self.times = interpoate_time_form_it(self.list_iterations, Paths.gw170817+sim+'/')
        self.times = []
        for it in self.list_iterations:
            self.times.append(self.get_time_for_it(it, d1d2d3prof="prof")) # prof
        self.times = np.array(self.times)

        self.list_v_ns = ["x", "y", "z", "rho", "w_lorentz", "vol", "press", "eps", "lapse", "velx", "vely", "velz",
                          "gxx", "gxy", "gxz", "gyy", "gyz", "gzz", "betax", "betay", "betaz", 'temp', 'Ye'] + \
                         ["density",  "enthalpy", "vphi", "vr", "dens_unb_geo", "dens_unb_bern", "dens_unb_garch",
                          "ang_mom", "ang_mom_flux", "theta", "r", "phi"] + \
                         ["Q_eff_nua", "Q_eff_nue", "Q_eff_nux", "R_eff_nua", "R_eff_nue", "R_eff_nux",
                          "optd_0_nua", "optd_0_nue", "optd_0_nux", "optd_1_nua", "optd_1_nue", "optd_1_nux"]
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

        path = Paths.ppr_sims + self.sim + '/' + self.set_rootdir + str(it) + '/'
        fname = "profile" + '.' + plane + ".h5"
        fpath = path + fname

        if not os.path.isfile(fpath):
            raise IOError("file: {} not found".format(fpath))

        dfile = h5py.File(fpath, "r")
        for rl in np.arange(start=0, stop=self.nlevels, step=1):
            for v_n in self.list_v_ns:
                try:
                    data = np.array(dfile["reflevel=%d" % rl][v_n], dtype=np.float32)
                except KeyError:
                    # print("\tKeyerror: v_n:{} not in file:{}".format(v_n, fpath))
                    data = np.zeros(0,)
                self.data_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_v_n(v_n)] = data
        dfile.close()

    def is_data_loaded_extracted(self, it, rl, plane, v_n):
        data = self.data_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_v_n(v_n)]
        if len(data) == 0:
            self.loaded_extract(it, plane)

        data = self.data_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_v_n(v_n)]
        if len(data) == 0:
            raise NameError("failed tp extract data. it:{} rl:{} plane:{} v_n:{}"
                             .format(it, rl, plane, v_n))

    def get_data(self, it, rl, plane, v_n):
        self.check_v_n(v_n)
        self.check_it(it)
        self.check_plane(plane)

        self.is_data_loaded_extracted(it, rl, plane, v_n)
        return self.data_matrix[self.i_it(it)][rl][self.i_plane(plane)][self.i_v_n(v_n)]

""" ================================================================================================================ """

def select_number(list, ful_list, dtype=int):
    if not any(list):
        return np.array(ful_list, dtype=dtype)
    array = np.array(list, dtype=dtype)
    ref_array = np.array(ful_list, dtype=dtype)
    for element in array:
        if not element in ref_array:
            raise ValueError("number element: {} is not in the ref_array:{}"
                             .format(element, ref_array))
    return array

def select_string(list_str, ful_list, for_all="all"):
    if not any(list_str):
        return ful_list
    if len(list_str) == 1 and for_all in list_str:
        return ful_list
    for element in list_str:
        if not element in ful_list:
            raise ValueError("string element: {} is not in the ref_array:{}"
                             .format(element, ful_list))
    return list_str

""" ===========================================| FOR PRINTING |===================================================== """

def print_colored_string(parts, colors, comma=False):
    assert len(parts) ==len(colors)
    for color in colors:
        assert color in ["", "blue", "red", "yellow", "green"]

    for part, color in zip(parts, colors):
        if color == "":
            if isinstance(part, list):
                for _part in part: print(_part),
            else: print(part),
        elif color == "blue":
            if isinstance(part, list):
                for _part in part:
                    Printcolor.blue(_part, comma=True)
            else:
                Printcolor.blue(part, comma=True)
        elif color == "green":
            if isinstance(part, list):
                for _part in part:
                    Printcolor.green(_part, comma=True)
            else:
                Printcolor.green(part, comma=True)
        elif color == "red":
            if isinstance(part, list):
                for _part in part:
                    Printcolor.red(_part, comma=True)
            else:
                Printcolor.red(part, comma=True)
        elif color == "yellow":
            if isinstance(part, list):
                for _part in part:
                    Printcolor.yellow(_part, comma=True)
            else:
                Printcolor.yellow(part, comma=True)
        else:
            raise NameError("wrong color: {}".format(color))
    if comma:
        print(''),
    else:
        print('')

def get_xmin_xmax_ymin_ymax_zmin_zmax(rl):
    if rl == 6:
        xmin, xmax = -14, 14
        ymin, ymax = -14, 14
        zmin, zmax = 0, 14
    elif rl == 5:
        xmin, xmax = -28, 28
        ymin, ymax = -28, 28
        zmin, zmax = 0, 28
    elif rl == 4:
        xmin, xmax = -48, 48
        ymin, ymax = -48, +48
        zmin, zmax = 0, 48
    elif rl == 3:
        xmin, xmax = -88, 88
        ymin, ymax = -88, 88
        zmin, zmax = 0, 88
    elif rl == 2:
        xmin, xmax = -178, 178
        ymin, ymax = -178, +178
        zmin, zmax = 0, 178
    elif rl == 1:
        xmin, xmax = -354, 354
        ymin, ymax = -354, +354
        zmin, zmax = 0, 354
    elif rl == 0:
        xmin, xmax = -1044, 1044
        ymin, ymax = -1044, 1044
        zmin, zmax = 0, 1044
    else:
        # pass
        raise IOError("Set limits for rl:{}".format(rl))

    return xmin, xmax, ymin, ymax, zmin, zmax

""" ========================================| D3 COMPUTING MODULES |================================================ """

def d3_disk_mass_for_it(it, d3corrclass, outdir, rewrite=False):

    fpath = outdir + __d3diskmass__

    if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
        if os.path.isfile(fpath): os.remove(fpath)
        print_colored_string(["task:", "disk_mass", "it:", "{}".format(it), ":", "computing"],
                             ["blue", "green", "blue", "green", "", "green"])
        mass = d3corrclass.get_disk_mass(it, multiplier=2.)
        np.savetxt(fname=fpath, X=np.array([mass]))
    else:
        print_colored_string(["task:", "disk_mass", "it:", "{}".format(it), ":", "skipping"],
                             ["blue", "green", "blue", "green", "", "blue"])

def d3_hist_for_it(it, d3corrclass, outdir, rewrite=False):

    selected_vn1vn2s = select_string(glob_v_ns, __d3histvns__)

    for v_n in selected_vn1vn2s:
        # chose a dic
        if v_n == "r":       hist_dic = d3corrclass.hist_task_dic_r
        elif v_n == "theta": hist_dic = d3corrclass.hist_task_dic_theta
        elif v_n == "Ye":    hist_dic = d3corrclass.hist_task_dic_ye
        elif v_n == "velz":  hist_dic = d3corrclass.hist_task_dic_velz
        elif v_n == "temp":  hist_dic = d3corrclass.hist_task_dic_temp
        elif v_n == "rho":   hist_dic = d3corrclass.hist_task_dic_rho
        elif v_n == "dens_unb_bern": hist_dic = d3corrclass.hist_task_dens_unb_bern
        else:raise NameError("hist v_n:{} is not recognized".format(v_n))
        #
        fpath = outdir + "hist_{}.dat".format(v_n)
        #
        if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
            if os.path.isfile(fpath): os.remove(fpath)
            print_colored_string(["task:", "hist", "it:", "{}".format(it), "v_n:", v_n, ":", "computing"],
                                 ["blue", "green", "blue", "green", "blue", "green", "", "green"])
            edges_weights = d3corrclass.get_histogram(it, hist_dic, multiplier=2.)
            np.savetxt(fname=fpath, X=edges_weights, header="# {}   mass".format(v_n))
        else:
            print_colored_string(["task:", "hist", "it:", "{}".format(it), "v_n:", v_n, ":", "skipping"],
                                 ["blue", "green", "blue", "green", "blue", "green", "", "blue"])

def d3_corr_for_it(it, d3corrclass, outdir, rewrite=False):

    selected_vn1vn2s = select_string(glob_v_ns, __d3corrs__)

    for v_ns in selected_vn1vn2s:
        # chose a dictionary
        if v_ns == "rho_r":
            corr_task_dic = d3corrclass.corr_task_dic_rho_r
        elif v_ns == "rho_Ye":
            corr_task_dic = d3corrclass.corr_task_dic_rho_ye
        elif v_ns == "temp_Ye":
            corr_task_dic = d3corrclass.corr_task_dic_temp_ye
        elif v_ns == "velz_Ye":
            corr_task_dic = d3corrclass.corr_task_dic_velz_ye
        elif v_ns == "rho_theta":
            corr_task_dic = d3corrclass.corr_task_dic_rho_theta
        elif v_ns == "velz_theta":
            corr_task_dic = d3corrclass.corr_task_dic_velz_theta
        elif v_ns == "rho_ang_mom":
            corr_task_dic = d3corrclass.corr_task_dic_rho_ang_mom
        elif v_ns == "rho_ang_mom_flux":
            corr_task_dic = d3corrclass.corr_task_dic_rho_ang_mom_flux
        elif v_ns == "Ye_dens_unb_bern":
            corr_task_dic = d3corrclass.corr_task_dic_ye_dens_unb_bern
        elif v_ns == "rho_dens_unb_bern":
            corr_task_dic = d3corrclass.corr_task_dic_rho_dens_unb_bern
        elif v_ns == "velz_dens_unb_bern":
            corr_task_dic = d3corrclass.corr_task_dic_velz_dens_unb_bern
        elif v_ns == "theta_dens_unb_bern":
            corr_task_dic = d3corrclass.corr_task_dic_theta_dens_unb_bern
        elif v_ns == "ang_mom_flux_theta":
            corr_task_dic = d3corrclass.corr_task_dic_ang_mom_flux_theta
        elif v_ns == "ang_mom_flux_dens_unb_bern":
            corr_task_dic = d3corrclass.corr_task_dic_ang_mom_flux_dens_unb_bern
        elif v_ns == "inv_ang_mom_flux_dens_unb_bern":
            corr_task_dic = d3corrclass.corr_task_dic_inv_ang_mom_flux_dens_unb_bern
        else:
            raise NameError("unknown task for correlation computation: {}"
                            .format(v_ns))


        fpath = outdir + "corr_{}.h5".format(v_ns)

        try:
            if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                if os.path.isfile(fpath): os.remove(fpath)
                print_colored_string(["task:", "corr", "it:", "{}".format(it), "v_ns:", v_ns, ":", "computing"],
                                     ["blue", "green", "blue", "green", "blue", "green", "", "green"])
                edges, mass = d3corrclass.get_correlation(it, corr_task_dic, multiplier=2.)
                dfile = h5py.File(fpath, "w")
                dfile.create_dataset("mass", data=mass, dtype=np.float32)
                for dic, edge in zip(corr_task_dic, edges):
                    dfile.create_dataset("{}".format(dic["v_n"]), data=edge)
                dfile.close()
            else:
                print_colored_string(["task:", "corr", "it:", "{}".format(it), "v_ns:", v_ns, ":", "skipping"],
                                     ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
        except KeyboardInterrupt:
            exit(1)
        except:
            print_colored_string(["task:", "corr", "it:", "{}".format(it), "v_ns:", v_ns, ":", "failed"],
                                 ["blue", "green", "blue", "green", "blue", "green", "", "red"])

def d3_to_d2_slice_for_it(it, d3corrclass, outdir, rewrite=False):

    selected_planes = select_string(glob_v_ns, __d3slicesplanes__)

    for plane in selected_planes:
        fpath = outdir + "profile" + '.' + plane + ".h5"
        try:
            if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                if os.path.isfile(fpath): os.remove(fpath)
                print_colored_string(["task:", "prof slice", "it:", "{}".format(it), "plane:", plane, ":", "computing"],
                                     ["blue", "green", "blue", "green", "blue", "green", "", "green"])
                d3corrclass.make_save_prof_slice(it, plane, __d3slicesvns__, fpath)
            else:
                print_colored_string(["task:", "prof slice", "it:", "{}".format(it), "plane:", plane, ":", "skipping"],
                                     ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
        except:
            print_colored_string(["task:", "prof slice", "it:", "{}".format(it), "plane:", plane, ":", "failed"],
                                 ["blue", "green", "blue", "green", "blue", "green", "", "red"])

def d3_dens_modes(d3corrclass, outdir, rewrite=False):
    fpath = outdir + "density_modes_lap15.h5"
    rl = 3
    mmax = 8
    Printcolor.blue("\tNote: that for density mode computation, masks (lapse etc) are NOT used")
    try:
        if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
            if os.path.isfile(fpath): os.remove(fpath)
            print_colored_string(["task:", "dens modes", "rl:", str(rl), "mmax:", str(mmax), ":", "computing"],
                                 ["blue", "green", "blue", "green", "blue", "green", "", "green"])
            times, iterations, xcs, ycs, modes = d3corrclass.get_dens_modes_for_rl(rl=rl, mmax=mmax)
            dfile = h5py.File(fpath, "w")
            dfile.create_dataset("times", data=times)           # times that actually used
            dfile.create_dataset("iterations", data=iterations) # iterations for these times
            dfile.create_dataset("xc", data=xcs)                # x coordinate of the center of mass
            dfile.create_dataset("yc", data=ycs)                # y coordinate of the center of mass
            for m in range(mmax + 1):
                group = dfile.create_group("m=%d" % m)
                group["int_phi"] = np.zeros(0, ) # NOT USED (suppose to be data for every 'R' in disk and NS)
                group["int_phi_r"] = np.array(modes[m]).flatten() # integrated over 'R' data
            dfile.close()
        else:
            print_colored_string(["task:", "dens modes", "rl:", str(rl), "mmax:", str(mmax), ":", "skipping"],
                                 ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
    except:
        print_colored_string(["task:", "dens modes", "rl:", str(rl), "mmax:", str(mmax), ":", "failed"],
                             ["blue", "green", "blue", "green", "blue", "green", "", "red"])

def d3_int_data_to_vtk(d3intclass, outdir, rewrite=False):
    private_dir = "vtk/"

    selected_v_ns = select_string(glob_v_ns, __d3slicesvns__)

    try:
        from evtk.hl import gridToVTK
    except ImportError:
        raise ImportError("Error importing gridToVTK. Is evtk installed? \n"
                          "If not, do: hg clone https://bitbucket.org/pauloh/pyevtk PyEVTK ")

    for it in glob_its:
        # assert that path exists
        path = outdir + str(it) + '/'
        if not os.path.isdir(path):
            os.mkdir(path)
        if private_dir != None and private_dir != '':
            path = path + private_dir
        if not os.path.isdir(path):
            os.mkdir(path)
        fname = "iter_" + str(it).zfill(10)
        fpath = path + fname

        # preparing the data
        if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
            if os.path.isfile(fpath): os.remove(fpath)
            print_colored_string(
                ["task:", "vtk", "grid:", d3intclass.new_grid.grid_type, "it:", str(it), "v_ns:", selected_v_ns, ":", "computing"],
                ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "green"])
            #
            celldata = {}
            for v_n in selected_v_ns:
                #try:
                Printcolor.green("\tInterpolating. grid: {} it: {} v_n: {} ".format(d3intclass.new_grid.grid_type, it, v_n))
                celldata[str(v_n)] = d3intclass.get_int(it, v_n)
                # except:
                #     celldata[str(v_n)] = np.empty(0,)
                #     Printcolor.red("\tFailed to interpolate. grid: {} it: {}v_n: {} ".format(d3intclass.new_grid.type, it, v_n))

            xf = d3intclass.new_grid.get_int_grid("xf")
            yf = d3intclass.new_grid.get_int_grid("yf")
            zf = d3intclass.new_grid.get_int_grid("zf")
            Printcolor.green("\tProducing vtk. it: {} v_ns: {} ".format(it, selected_v_ns))
            gridToVTK(fpath, xf, yf, zf, cellData=celldata)
            Printcolor.blue("\tDone. File is saved: {}".format(fpath))
        else:
            print_colored_string(
                ["task:", "vtk", "grid:", d3intclass.new_grid.grid_type, "it:", str(it), "v_ns:", selected_v_ns, ":",
                 "skipping"],
                ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "blue"])
        #
        #     celldata = {}
        #     for v_n in selected_v_ns:
        #         try:
        #             print_colored_string(["task:", "int", "grid:", d3intclass.new_grid.type, "it:", str(it), "v_n:", v_n, ":", "interpolating"],
        #                                  ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "green"])
        #             celldata[str(v_n)] = d3intclass.get_int(it, v_n)
        #         except:
        #             print_colored_string(
        #                 ["task:", "int", "grid:", d3intclass.new_grid.type, "it:", str(it), "v_n:", v_n, ":",
        #                  "failed"],
        #                 ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])
        #     Printcolor.green("Data for v_ns:{} is interpolated and preapred".format(selected_v_ns))
        #     # producing the vtk file
        #     try:
        #         print_colored_string(
        #             ["task:", "vtk", "grid:", d3intclass.new_grid.type, "it:", str(it), "v_ns:", selected_v_ns, ":", "computing"],
        #             ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "green"])
        #         xf = d3intclass.new_grid.get_int_grid("xf")
        #         yf = d3intclass.new_grid.get_int_grid("yf")
        #         zf = d3intclass.new_grid.get_int_grid("zf")
        #
        #         gridToVTK(fpath, xf, yf, zf, cellData=celldata)
        #     except:
        #         print_colored_string(
        #             ["task:", "int", "grid:", d3intclass.new_grid.type, "it:", str(it), "v_ns:", selected_v_ns, ":",
        #              "failed"],
        #             ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])
        # else:
        #     print_colored_string(["task:", "prof slice", "it:", "{}".format(it), "plane:", plane, ":", "skipping"],
        #                          ["blue", "green", "blue", "green", "blue", "green", "", "blue"])

""" ==============================================| D3 PLOTS |======================================================="""

def plot_d3_prof_slices(d3class, figdir='slices/', rewritefigs=False):


    iterations = select_number(glob_its, d3class.list_iterations)
    v_ns = select_string(glob_v_ns, __d3sliceplotvns__, for_all="all")

    # tmerg = d1class.get_par("tmerger_gw")
    i = 1
    for it in iterations:
        for rl in __d3sliceplotrls__:
            for v_n in v_ns:
                #
                # print("v_n:{}".format(v_n))
                try:
                    data_arr = d3class.get_data(it, rl, "xz", v_n)
                    x_arr = d3class.get_data(it, rl, "xz", "x")
                    z_arr = d3class.get_data(it, rl, "xz", "z")
                    def_dic_xz = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                                  'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                                  'position': (1, 1),  # 'title': '[{:.1f} ms]'.format(time_),
                                  'cbar': {'location': 'right .04 .2', 'label': r'$\rho$ [geo]',  # 'fmt': '%.1e',
                                           'labelsize': 14,
                                           'fontsize': 14},
                                  'v_n_x': 'x', 'v_n_y': 'z', 'v_n': 'rho',
                                  'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None, 'vmin': 1e-10, 'vmax': 1e-4,
                                  'fill_vmin': False,  # fills the x < vmin with vmin
                                  'xscale': None, 'yscale': None,
                                  'mask': None, 'cmap': 'inferno_r', 'norm': "log",
                                  'fancyticks': True,
                                  'title': {"text": r'$t-t_{merg}:$' + r'${:.1f}$'.format(0), 'fontsize': 14},
                                  'sharex': True,  # removes angular citkscitks
                                  'fontsize': 14,
                                  'labelsize': 14
                                  }
                except KeyError:
                    print_colored_string(
                        ["task:", "plot prof slice", "it:", "{}".format(it), "rl:", "{:d}".format(rl), "v_ns:", v_n,
                         ":", "KeyError in getting xz {}".format(v_n)],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])
                except NameError:
                    print_colored_string(
                        ["task:", "plot prof slice", "it:", "{}".format(it), "rl:", "{:d}".format(rl), "v_ns:", v_n,
                         ":", "NameError in getting xz {}".format(v_n)],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])
                    continue
                try:
                    data_arr = d3class.get_data(it, rl, "xy", v_n)
                    x_arr = d3class.get_data(it, rl, "xy", "x")
                    y_arr = d3class.get_data(it, rl, "xy", "y")
                    def_dic_xy = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                                  'xarr': x_arr, "yarr": y_arr, "zarr": data_arr,
                                  'position': (2, 1),  # 'title': '[{:.1f} ms]'.format(time_),
                                  'cbar': {},
                                  'v_n_x': 'x', 'v_n_y': 'y', 'v_n': 'rho',
                                  'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None, 'vmin': 1e-10, 'vmax': 1e-4,
                                  'fill_vmin': False,  # fills the x < vmin with vmin
                                  'xscale': None, 'yscale': None,
                                  'mask': None, 'cmap': 'inferno_r', 'norm': "log",
                                  'fancyticks': True,
                                  'title': {},
                                  'sharex': False,  # removes angular citkscitks
                                  'fontsize': 14,
                                  'labelsize': 14
                                  }
                except KeyError:
                    print_colored_string(
                        ["task:", "plot prof slice", "it:", "{}".format(it), "rl:", "{:d}".format(rl), "v_ns:", v_n,
                         ":", "KeyError in getting xy {} ".format(v_n)],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])
                    continue
                except NameError:
                    print_colored_string(
                        ["task:", "plot prof slice", "it:", "{}".format(it), "rl:", "{:d}".format(rl), "v_ns:", v_n,
                         ":", "NameError in getting xy {} ".format(v_n)],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])
                    continue

                # "Q_eff_nua", "Q_eff_nue", "Q_eff_nux"
                if v_n in ["Q_eff_nua", "Q_eff_nue", "Q_eff_nux"]:
                    dens_arr = d3class.get_data(it, rl, "xz", "density")
                    data_arr = d3class.get_data(it, rl, "xz", v_n)
                    data_arr = data_arr / dens_arr
                    x_arr = d3class.get_data(it, rl, "xz", "x")
                    z_arr = d3class.get_data(it, rl, "xz", "z")
                    def_dic_xz['xarr'], def_dic_xz['yarr'], def_dic_xz['zarr'] = x_arr, z_arr, data_arr
                    #
                    dens_arr = d3class.get_data(it, rl, "xy", "density")
                    data_arr = d3class.get_data(it, rl, "xy", v_n)
                    data_arr = data_arr / dens_arr
                    x_arr = d3class.get_data(it, rl, "xy", "x")
                    y_arr = d3class.get_data(it, rl, "xy", "y")
                    def_dic_xy['xarr'], def_dic_xy['yarr'], def_dic_xy['zarr'] = x_arr, y_arr, data_arr


                if v_n == 'rho':
                    pass
                elif v_n == 'w_lorentz':
                    def_dic_xy['v_n'] = 'w_lorentz'
                    def_dic_xy['vmin'] = 1
                    def_dic_xy['vmax'] = 1.3
                    def_dic_xy['norm'] = None
                    def_dic_xz['v_n'] = 'w_lorentz'
                    def_dic_xz['vmin'] = 1
                    def_dic_xz['vmax'] = 1.3
                    def_dic_xz['norm'] = None
                elif v_n == 'vol':
                    def_dic_xy['v_n'] = 'vol'
                    def_dic_xy['vmin'] = 1
                    def_dic_xy['vmax'] = 10
                    # def_dic_xy['norm'] = None
                    def_dic_xz['v_n'] = 'vol'
                    def_dic_xz['vmin'] = 1
                    def_dic_xz['vmax'] = 10
                    # def_dic_xz['norm'] = None
                elif v_n == 'press':
                    def_dic_xy['v_n'] = 'press'
                    def_dic_xy['vmin'] = 1e-12
                    def_dic_xy['vmax'] = 1e-6

                    def_dic_xz['v_n'] = 'press'
                    def_dic_xz['vmin'] = 1e-12
                    def_dic_xz['vmax'] = 1e-6
                elif v_n == 'eps':
                    def_dic_xy['v_n'] = 'eps'
                    def_dic_xy['vmin'] = 5e-3
                    def_dic_xy['vmax'] = 5e-1
                    def_dic_xz['v_n'] = 'eps'
                    def_dic_xz['vmin'] = 5e-3
                    def_dic_xz['vmax'] = 5e-1
                elif v_n == 'lapse':
                    def_dic_xy['v_n'] = 'lapse'
                    def_dic_xy['vmin'] = 0.15
                    def_dic_xy['vmax'] = 1
                    def_dic_xy['norm'] = None
                    def_dic_xz['v_n'] = 'lapse'
                    def_dic_xz['vmin'] = 0.15
                    def_dic_xz['vmax'] = 1
                    def_dic_xz['norm'] = None
                elif v_n == 'velx':
                    def_dic_xy['v_n'] = 'velx'
                    def_dic_xy['vmin'] = 0.01
                    def_dic_xy['vmax'] = 1.
                    # def_dic_xy['norm'] = None
                    def_dic_xz['v_n'] = 'velx'
                    def_dic_xz['vmin'] = 0.01
                    def_dic_xz['vmax'] = 1.
                    # def_dic_xz['norm'] = None
                elif v_n == 'vely':
                    def_dic_xy['v_n'] = 'vely'
                    def_dic_xy['vmin'] = 0.01
                    def_dic_xy['vmax'] = 1.
                    # def_dic_xy['norm'] = None
                    def_dic_xz['v_n'] = 'vely'
                    def_dic_xz['vmin'] = 0.01
                    def_dic_xz['vmax'] = 1.
                    # def_dic_xz['norm'] = None
                elif v_n == 'velz':
                    def_dic_xy['v_n'] = 'velz'
                    def_dic_xy['vmin'] = 0.01
                    def_dic_xy['vmax'] = 1.
                    # def_dic_xy['norm'] = None
                    def_dic_xz['v_n'] = 'velz'
                    def_dic_xz['vmin'] = 0.01
                    def_dic_xz['vmax'] = 1.
                    # def_dic_xz['norm'] = None
                elif v_n == 'temp':
                    def_dic_xy['v_n'] = 'temp'
                    def_dic_xy['vmin'] =  1e-2
                    def_dic_xy['vmax'] = 1e2

                    def_dic_xz['v_n'] = 'temp'
                    def_dic_xz['vmin'] =  1e-2
                    def_dic_xz['vmax'] = 1e2
                elif v_n == 'Ye':
                    def_dic_xy['v_n'] = 'Ye'
                    def_dic_xy['vmin'] = 0.05
                    def_dic_xy['vmax'] = 0.5
                    def_dic_xy['norm'] = None
                    def_dic_xy['cmap'] = 'inferno'

                    def_dic_xz['v_n'] = 'Ye'
                    def_dic_xz['vmin'] = 0.05
                    def_dic_xz['vmax'] = 0.5
                    def_dic_xz['norm'] = None
                    def_dic_xz['cmap'] = 'inferno'
                elif v_n == 'density':
                    def_dic_xy['v_n'] = 'density'
                    def_dic_xy['vmin'] = 1e-9
                    def_dic_xy['vmax'] = 1e-5
                    # def_dic_xy['norm'] = None

                    def_dic_xz['v_n'] = 'density'
                    def_dic_xz['vmin'] = 1e-9
                    def_dic_xz['vmax'] = 1e-5
                    # def_dic_xz['norm'] = None
                elif v_n == 'enthalpy':
                    def_dic_xy['v_n'] = 'enthalpy'
                    def_dic_xy['vmin'] = 1.
                    def_dic_xy['vmax'] = 1.5
                    def_dic_xy['norm'] = None

                    def_dic_xz['v_n'] = 'enthalpy'
                    def_dic_xz['vmin'] = 1.
                    def_dic_xz['vmax'] = 1.5
                    def_dic_xz['norm'] = None
                elif v_n == 'vphi':
                    def_dic_xy['v_n'] = 'vphi'
                    def_dic_xy['vmin'] = 0.01
                    def_dic_xy['vmax'] = 10.
                    # def_dic_xy['norm'] = None
                    def_dic_xz['v_n'] = 'vphi'
                    def_dic_xz['vmin'] = 0.01
                    def_dic_xz['vmax'] = 10.
                    # def_dic_xz['norm'] = None
                elif v_n == 'vr':
                    def_dic_xy['v_n'] = 'vr'
                    def_dic_xy['vmin'] = 0.01
                    def_dic_xy['vmax'] = 0.5
                    # def_dic_xy['norm'] = None
                    def_dic_xz['v_n'] = 'vr'
                    def_dic_xz['vmin'] = 0.01
                    def_dic_xz['vmax'] = 0.5
                    # def_dic_xz['norm'] = None
                elif v_n == 'dens_unb_geo':
                    def_dic_xy['v_n'] = 'dens_unb_geo'
                    def_dic_xy['vmin'] = 1e-10
                    def_dic_xy['vmax'] = 1e-5
                    # def_dic_xy['norm'] = None
                    def_dic_xz['v_n'] = 'dens_unb_geo'
                    def_dic_xz['vmin'] = 1e-10
                    def_dic_xz['vmax'] = 1e-5
                    # def_dic_xz['norm'] = None
                elif v_n == 'dens_unb_bern':
                    def_dic_xy['v_n'] = 'dens_unb_bern'
                    def_dic_xy['vmin'] = 1e-10
                    def_dic_xy['vmax'] = 1e-5
                    # def_dic_xy['norm'] = None
                    def_dic_xz['v_n'] = 'dens_unb_bern'
                    def_dic_xz['vmin'] = 1e-10
                    def_dic_xz['vmax'] = 1e-5
                    # def_dic_xz['norm'] = None
                elif v_n == 'dens_unb_garch':
                    def_dic_xy['v_n'] = 'dens_unb_garch'
                    def_dic_xy['vmin'] = 1e-10
                    def_dic_xy['vmax'] = 1e-6
                    # def_dic_xy['norm'] = None
                    def_dic_xz['v_n'] = 'dens_unb_garch'
                    def_dic_xz['vmin'] = 1e-10
                    def_dic_xz['vmax'] = 1e-6
                    # def_dic_xz['norm'] = None
                elif v_n == 'ang_mom':
                    def_dic_xy['v_n'] = 'ang_mom'
                    def_dic_xy['vmin'] = 1e-8
                    def_dic_xy['vmax'] = 1e-3
                    # def_dic_xy['norm'] = None
                    def_dic_xz['v_n'] = 'ang_mom'
                    def_dic_xz['vmin'] = 1e-8
                    def_dic_xz['vmax'] = 1e-3
                    # def_dic_xz['norm'] = None
                elif v_n == 'ang_mom_flux':
                    def_dic_xy['v_n'] = 'ang_mom_flux'
                    def_dic_xy['vmin'] = 1e-9
                    def_dic_xy['vmax'] = 1e-5
                    # def_dic_xy['norm'] = None
                    def_dic_xz['v_n'] = 'ang_mom_flux'
                    def_dic_xz['vmin'] = 1e-9
                    def_dic_xz['vmax'] = 1e-5
                    # def_dic_xz['norm'] = None
                elif v_n == 'Q_eff_nua':
                    def_dic_xy['v_n'] = 'Q_eff_nua/D'
                    def_dic_xy['vmin'] = 1e-7
                    def_dic_xy['vmax'] = 1e-3
                    # def_dic_xy['norm'] = None

                    def_dic_xz['v_n'] = 'Q_eff_nua/D'
                    def_dic_xz['vmin'] = 1e-7
                    def_dic_xz['vmax'] = 1e-3
                    # def_dic_xz['norm'] = None
                elif v_n == 'Q_eff_nue':
                    def_dic_xy['v_n'] = 'Q_eff_nue/D'
                    def_dic_xy['vmin'] = 1e-7
                    def_dic_xy['vmax'] = 1e-3
                    # def_dic_xy['norm'] = None

                    def_dic_xz['v_n'] = 'Q_eff_nue/D'
                    def_dic_xz['vmin'] = 1e-7
                    def_dic_xz['vmax'] = 1e-3
                    # def_dic_xz['norm'] = None
                elif v_n == 'Q_eff_nux':
                    def_dic_xy['v_n'] = 'Q_eff_nux/D'
                    def_dic_xy['vmin'] = 1e-10
                    def_dic_xy['vmax'] = 1e-4
                    # def_dic_xy['norm'] = None

                    def_dic_xz['v_n'] = 'Q_eff_nux/D'
                    def_dic_xz['vmin'] = 1e-10
                    def_dic_xz['vmax'] = 1e-4
                    # def_dic_xz['norm'] = None
                    print("v_n: {} [{}->{}]".format(v_n, def_dic_xz['zarr'].min(), def_dic_xz['zarr'].max()))

                else:
                    raise NameError("v_n:{} not recogmized".format(v_n))

                def_dic_xy["xmin"], def_dic_xy["xmax"], def_dic_xy["ymin"], def_dic_xy["ymax"], _, _ \
                    = get_xmin_xmax_ymin_ymax_zmin_zmax(rl)
                def_dic_xz["xmin"], def_dic_xz["xmax"], _, _, def_dic_xz["ymin"], def_dic_xz["ymax"] \
                    = get_xmin_xmax_ymin_ymax_zmin_zmax(rl)

                """ --- --- --- """
                datafpath = Paths.ppr_sims + d3class.sim + '/' + __rootoutdir__

                figname = "{}_rl{}.png".format(v_n, rl)

                o_plot = PLOT_MANY_TASKS()
                o_plot.gen_set["figdir"] = datafpath
                o_plot.gen_set["type"] = "cartesian"
                o_plot.gen_set["figsize"] = (4.2, 8.0)  # <->, |] # to match hists with (8.5, 2.7)
                o_plot.gen_set["figname"] = figname
                o_plot.gen_set["sharex"] = False
                o_plot.gen_set["sharey"] = False
                o_plot.gen_set["subplots_adjust_h"] = -0.3
                o_plot.gen_set["subplots_adjust_w"] = 0.2
                o_plot.set_plot_dics = []

                # for it, t in zip(d3class.list_iterations, d3class.times):  # zip([346112],[0.020]):# #
                if not os.path.isdir(datafpath + str(it) + '/' + figdir):
                    os.mkdir(datafpath + str(it) + '/' + figdir)
                # tr = (t - tmerg) * 1e3  # ms
                if not os.path.isfile(datafpath + str(it) + '/' + "profile.xy.h5") \
                        or not os.path.isfile(datafpath + str(it) + '/' + "profile.xz.h5"):
                    Printcolor.yellow(
                        "Required data ia missing: {}".format(datafpath + str(it) + '/' + "profile.xy(or yz).h5"))
                    continue
                fpath = datafpath + str(it) + '/' + figdir + figname
                t = d3class.get_time_for_it(it, "prof")
                try:
                    if (os.path.isfile(fpath) and rewritefigs) or not os.path.isfile(fpath):
                        if os.path.isfile(fpath): os.remove(fpath)
                        print_colored_string(
                            ["task:", "plot prof slice", "it:", "{}".format(it), "rl:", "{:d}".format(rl), "v_ns:", v_n, ":", "plotting"],
                            ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "green"])
                        # ---------- PLOTTING -------------
                        if v_n in ["velx", "vely", "velz", "vphi", "vr", "ang_mom_flux"]:
                            print("\t\tUsing 2 colobars for v_n:{}".format(v_n))
                            # make separate plotting >0 and <0 with log scales
                            o_plot.gen_set["figdir"] = datafpath + str(it) + '/' + figdir

                            def_dic_xz['cmap'] = 'Reds'
                            def_dic_xz["mask"] = "negative"
                            def_dic_xz['cbar'] = {'location': 'right .04 0.00', 'label': v_n.replace('_', '\_') + r"$<0$",
                                                  'labelsize': 14,
                                                  'fontsize': 14}
                            def_dic_xz["it"] = int(it)
                            def_dic_xz["title"]["text"] = r'$t:{:.1f}$ [ms]'.format(float(t))

                            n_def_dic_xz = def_dic_xz.copy()  # copy.deepcopy(def_dic_xz)
                            def_dic_xz['data'] = d3class
                            o_plot.set_plot_dics.append(def_dic_xz)

                            n_def_dic_xz['data'] = d3class
                            n_def_dic_xz['cmap'] = 'Blues'
                            n_def_dic_xz["mask"] = "positive"
                            n_def_dic_xz['cbar'] = {}
                            n_def_dic_xz["it"] = int(it)
                            n_def_dic_xz["title"]["text"] = r'$t:{:.1f}$ [ms]'.format(float(t*1e3))

                            o_plot.set_plot_dics.append(n_def_dic_xz)

                            # --- ---
                            def_dic_xy["it"] = int(it)
                            def_dic_xy['cmap'] = 'Blues'
                            def_dic_xy['mask'] = "positive"
                            def_dic_xy['cbar'] = {'location': 'right .04 .0', 'label': v_n.replace('_', '\_') + r"$>0$",
                                                  # 'fmt': '%.1e',
                                                  'labelsize': 14,
                                                  'fontsize': 14}
                            # n_def_dic_xy = copy.deepcopy(def_dic_xy)
                            n_def_dic_xy = def_dic_xy.copy()
                            def_dic_xy['data'] = d3class
                            o_plot.set_plot_dics.append(def_dic_xy)

                            n_def_dic_xy['data'] = d3class
                            n_def_dic_xy['cbar'] = {}
                            n_def_dic_xy['cmap'] = 'Reds'
                            n_def_dic_xy['mask'] = "negative"
                            o_plot.set_plot_dics.append(n_def_dic_xy)

                            for dic in o_plot.set_plot_dics:
                                if not 'cbar' in dic.keys():
                                    raise IOError("dic:{} no cbar".format(dic))

                            # ---- ----
                            o_plot.main()
                            # del(o_plot.set_plot_dics)
                            o_plot.figure.clear()
                            n_def_dic_xy = {}
                            n_def_dic_xz = {}
                        else:
                            def_dic_xz['data'] = d3class
                            def_dic_xz['cbar']['label'] = v_n.replace('_', '\_')
                            def_dic_xz['cbar']['location'] = 'right .04 -.36'
                            def_dic_xz["it"] = int(it)
                            def_dic_xz["title"]["text"] = r'$t:{:.1f}$ [ms]'.format(float(t*1e3))
                            o_plot.gen_set["figdir"] = datafpath + str(it) + '/' + figdir
                            o_plot.set_plot_dics.append(def_dic_xz)

                            def_dic_xy['data'] = d3class
                            def_dic_xy["it"] = int(it)
                            # rho_dic_xy["title"]["text"] = r'$t-t_{merg}:$' + r'${:.2f}ms$'.format(float(tr))
                            # o_plot.gen_set["figname"] =   # 7 digit output
                            o_plot.set_plot_dics.append(def_dic_xy)

                            o_plot.main()
                            # del(o_plot.set_plot_dics)
                            o_plot.figure.clear()
                            def_dic_xy = {}
                            def_dic_xz = {}

                        # ------------------------
                    else:
                        print_colored_string(
                            ["task:", "plot prof slice", "it:", "{}".format(it), "rl:", "{:d}".format(rl), "v_ns:", v_n, ":", "skipping"],
                            ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "blue"])
                except KeyboardInterrupt:
                    exit(1)
                except IOError:
                    print_colored_string(
                        ["task:", "plot prof slice", "it:", "{}".format(it), "rl:", "{:d}".format(rl), "v_ns:", v_n,
                         ":", "IOError"],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])
                except ValueError:
                    print_colored_string(
                        ["task:", "plot prof slice", "it:", "{}".format(it), "rl:", "{:d}".format(rl), "v_ns:", v_n,
                         ":", "ValueError"],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])
                except:
                    print_colored_string(
                        ["task:", "plot prof slice", "it:", "{}".format(it),  "rl:", "{:d}".format(rl), "v_ns:", v_n, ":", "failed"],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])
                v_n = None
            rl = None
        it = None
        sys.stdout.flush()
        i = i + 1
    # exit(1)

def plot_d3_corr(d3histclass, rewrite=False):

    iterations = select_number(glob_its, d3histclass.list_iterations)
    v_ns = select_string(glob_v_ns, __d3corrs__, for_all="all")

    for it in iterations:
        for vn1vn2 in v_ns:

            default_dic = {  # relies on the "get_res_corr(self, it, v_n): " method of data object
                'task': 'corr2d', 'ptype': 'cartesian',
                'data': d3histclass,
                'position': (1, 1),
                'v_n_x': 'ang_mom_flux', 'v_n_y': 'dens_unb_bern', 'v_n': Labels.labels("mass"), 'normalize': True,
                'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None, 'vmin': 1e-7, 'vmax': 1e-3,
                'xscale': 'log', 'yscale': 'log',
                'mask_below': None, 'mask_above': None, 'cmap': 'inferno_r', 'norm': 'log', 'todo': None,
                'cbar': {'location': 'right .03 .0', 'label': r'mass',
                         'labelsize': 14,
                         'fontsize': 14},
                'title': {"text": r'$t-t_{merg}:$' + r'${:.1f}$'.format(0), 'fontsize': 14},
                'fontsize': 14,
                'labelsize': 14,
                'minorticks': True,
                'fancyticks': True,
                'sharey': False,
                'sharex': False,
            }

            if vn1vn2 == "rho_r":
                v_n_x = 'rho'
                v_n_y = 'r'
                # default_dic['v_n_x'] = 'rho'
                # default_dic['v_n_y'] = 'r'
                default_dic['xmin'] = 1e-9
                default_dic['xmax'] = 2e-5
                default_dic['ymin'] = 0
                default_dic['ymax'] = 250
                default_dic['yscale'] = None
            elif vn1vn2 == "rho_Ye":
                v_n_x = 'rho'
                v_n_y = 'Ye'
                # default_dic['v_n_x'] = 'rho'
                # default_dic['v_n_y'] = 'Ye'
                default_dic['xmin'] = 1e-9
                default_dic['xmax'] = 2e-5
                default_dic['ymin'] = 0.01
                default_dic['ymax'] = 0.5
                default_dic['yscale'] = None
            elif vn1vn2 == "temp_Ye":
                v_n_x = 'temp'
                v_n_y = 'Ye'
                # default_dic['v_n_x'] = 'temp'
                # default_dic['v_n_y'] = 'Ye'
                default_dic['xmin'] = 1e-2
                default_dic['xmax'] = 1e2
                default_dic['ymin'] = 0.01
                default_dic['ymax'] = 0.5
                default_dic['yscale'] = None
            elif vn1vn2 == "rho_theta":
                v_n_x = 'rho'
                v_n_y = 'theta'
                # default_dic['v_n_x'] = 'rho'
                # default_dic['v_n_y'] = 'theta'
                default_dic['xmin'] = 1e-9
                default_dic['xmax'] = 2e-5
                default_dic['ymin'] = 0
                default_dic['ymax'] = 1.7
                default_dic['yscale'] = None
            elif vn1vn2 == "velz_theta":
                v_n_x = 'velz'
                v_n_y = 'theta'
                # default_dic['v_n_x'] = 'velz'
                # default_dic['v_n_y'] = 'theta'
                default_dic['xmin'] = -.5
                default_dic['xmax'] = .5
                default_dic['ymin'] = 0
                default_dic['ymax'] = 90.
                default_dic['yscale'] = None
                default_dic['xscale'] = None
            elif vn1vn2 == "velz_Ye":
                v_n_x = 'velz'
                v_n_y = 'Ye'
                # default_dic['v_n_x'] = 'velz'
                # default_dic['v_n_y'] = 'Ye'
                default_dic['xmin'] = -.5
                default_dic['xmax'] = .5
                default_dic['ymin'] = 0.01
                default_dic['ymax'] = 0.5
                default_dic['yscale'] = None
                default_dic['xscale'] = None
            elif vn1vn2 == "rho_ang_mom":
                v_n_x = 'rho'
                v_n_y = 'ang_mom'
                # default_dic['v_n_x'] = 'rho'
                # default_dic['v_n_y'] = 'ang_mom'
                default_dic['xmin'] = 1e-9
                default_dic['xmax'] = 2e-5
                default_dic['ymin'] = 1e-9
                default_dic['ymax'] = 1e-3
            elif vn1vn2 == "theta_dens_unb_bern":
                v_n_x = 'theta'
                v_n_y = 'dens_unb_bern'
                # default_dic['v_n_x'] = 'theta'
                default_dic['xmin'] = 0.
                default_dic['xmax'] = 90.
                default_dic['xscale'] = None
                # default_dic['v_n_y'] = 'dens_unb_bern'
                default_dic['ymin'] = 1e-9
                default_dic['ymax'] = 2e-6
            elif vn1vn2 == "velz_dens_unb_bern":
                v_n_x = 'velz'
                v_n_y = 'dens_unb_bern'
                # default_dic['v_n_x'] = 'velz'
                default_dic['xmin'] = -.5
                default_dic['xmax'] = .5
                default_dic['xscale'] = None
                # default_dic['v_n_y'] = 'dens_unb_bern'
                default_dic['ymin'] = 1e-9
                default_dic['ymax'] = 2e-6
            elif vn1vn2 == "rho_ang_mom_flux":
                v_n_x = 'rho'
                v_n_y = 'ang_mom_flux'
                # default_dic['v_n_x'] = 'rho'
                # default_dic['v_n_y'] = 'ang_mom_flux'
                default_dic['xmin'] = 1e-9
                default_dic['xmax'] = 2e-5
                default_dic['ymin'] = 1e-9
                default_dic['ymax'] = 8e-5
            elif vn1vn2 == "rho_dens_unb_bern":
                v_n_x = 'rho'
                v_n_y = 'dens_unb_bern'
                # default_dic['v_n_x'] = 'rho'
                # default_dic['v_n_y'] = 'dens_unb_bern'
                default_dic['xmin'] = 1e-9
                default_dic['xmax'] = 2e-5
                default_dic['ymin'] = 1e-9
                default_dic['ymax'] = 2e-6
            elif vn1vn2 == "Ye_dens_unb_bern":
                v_n_x = 'Ye'
                v_n_y = 'dens_unb_bern'
                # default_dic['v_n_x'] = 'Ye'
                default_dic['xmin'] = 0.01
                default_dic['xmax'] = 0.5
                default_dic['xscale'] = None
                # default_dic['v_n_y'] = 'dens_unb_bern'
                default_dic['ymin'] = 1e-9
                default_dic['ymax'] = 2e-6
                default_dic['yscale'] = "log"
            elif vn1vn2 == "ang_mom_flux_theta":
                v_n_x = 'ang_mom_flux'
                v_n_y = 'theta'
                # default_dic['v_n_x'] = 'ang_mom_flux'
                # default_dic['v_n_y'] = 'theta'
                default_dic['xmin'] = 1e-9
                default_dic['xmax'] = 8e-5
                default_dic['ymin'] = 0
                default_dic['ymax'] = 1.7
                default_dic['yscale'] = None
            elif vn1vn2 == "ang_mom_flux_dens_unb_bern":
                v_n_x = 'ang_mom_flux'
                v_n_y = 'dens_unb_bern'
                default_dic['xmin'] = 1e-11
                default_dic['xmax'] = 1e-7
                default_dic['ymin'] = 1e-11
                default_dic['ymax'] = 1e-7
            elif vn1vn2 == "inv_ang_mom_flux_dens_unb_bern":
                v_n_x = 'inv_ang_mom_flux'
                v_n_y = 'dens_unb_bern'
                default_dic['xmin'] = 1e-11
                default_dic['xmax'] = 1e-7
                default_dic['ymin'] = 1e-11
                default_dic['ymax'] = 1e-7
                # default_dic['v_n_x'] = 'inv_ang_mom_flux'
            else:
                raise NameError("vn1vn2:{} is not recognized"
                                .format(vn1vn2))
            outfpath = resdir + __rootoutdir__ + str(it) + "/corr_plots/"
            if not os.path.isdir(outfpath):
                os.mkdir(outfpath)
            fpath = outfpath + "{}.png".format(vn1vn2)
            try:
                if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                    if os.path.isfile(fpath): os.remove(fpath)
                    print_colored_string(["task:", "plot corr", "it:", "{}".format(it), "v_ns:", vn1vn2, ":", "computing"],
                                         ["blue", "green", "blue", "green", "blue", "green", "", "green"])

                    table = d3histclass.get_res_corr(it, v_n_x, v_n_y)
                    default_dic["data"] = table
                    default_dic["v_n_x"] = v_n_x
                    default_dic["v_n_y"] = v_n_y
                    default_dic["xlabel"] = Labels.labels(v_n_x)
                    default_dic["ylabel"] = Labels.labels(v_n_y)


                    o_plot = PLOT_MANY_TASKS()
                    o_plot.gen_set["figdir"] = outfpath
                    o_plot.gen_set["type"] = "cartesian"
                    o_plot.gen_set["figsize"] = (4.2, 3.8)  # <->, |] # to match hists with (8.5, 2.7)
                    o_plot.gen_set["figname"] = "{}.png".format(vn1vn2)
                    o_plot.gen_set["sharex"] = False
                    o_plot.gen_set["sharey"] = False
                    o_plot.gen_set["subplots_adjust_h"] = 0.0
                    o_plot.gen_set["subplots_adjust_w"] = 0.2
                    o_plot.set_plot_dics = []

                    #-------------------------------
                    # tr = (t - tmerg) * 1e3  # ms
                    t = d3histclass.get_time_for_it(it, d1d2d3prof="prof")
                    default_dic["it"] = it
                    default_dic["title"]["text"] = r'$t:{:.1f}$ [ms]'.format(float(t*1e3))
                    o_plot.set_plot_dics.append(default_dic)

                    o_plot.main()
                    o_plot.set_plot_dics = []
                    o_plot.figure.clear()
                    #-------------------------------
                else:
                    print_colored_string(["task:", "plot corr", "it:", "{}".format(it), "v_ns:", vn1vn2, ":", "skipping"],
                                         ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
            except IOError:
                print_colored_string(["task:", "plot corr", "it:", "{}".format(it), "v_ns:", vn1vn2, ":", "missing file"],
                                     ["blue", "green", "blue", "green", "blue", "green", "", "red"])
            except KeyboardInterrupt:
                exit(1)
            except:
                print_colored_string(["task:", "plot corr", "it:", "{}".format(it), "v_ns:", vn1vn2, ":", "failed"],
                                     ["blue", "green", "blue", "green", "blue", "green", "", "red"])
            default_dic = {}

def plot_d3_hist(d3histclass, rewrite=False):

    iterations = select_number(glob_its, d3histclass.list_iterations)
    v_ns = select_string(glob_v_ns, __d3histvns__, for_all="all")

    for it in iterations:
        for v_n in v_ns:

            fpath = resdir + __rootoutdir__ + str(it) + "/" + "hist_{}.dat".format(v_n)
            data = np.loadtxt(fpath, unpack=False)
            # print(data)
            default_dic = {
                'task': 'hist1d', 'ptype': 'cartesian',
                'position': (1, 1),
                'data': data, 'normalize': False,
                'v_n_x': 'var', 'v_n_y': 'mass',
                'color': "black", 'ls': '-', 'lw': 0.8, 'ds': 'steps', 'alpha':1.0,
                'ymin': 1e-4, 'ymax': 1e-1,
                'xlabel': None,  'ylabel': "mass",
                'label': None, 'yscale': 'log',
                'fancyticks': True, 'minorticks': True,
                'fontsize': 14,
                'labelsize': 14,
                'legend': {}#'loc': 'best', 'ncol': 2, 'fontsize': 18
            }

            if v_n == "r":
                default_dic['v_n_x'] = 'r'
                default_dic['xlabel'] = 'cylindrical radius'
                default_dic['xmin'] = 10.
                default_dic['xmax'] = 50.
            elif v_n == "theta":
                default_dic['v_n_x'] = 'theta'
                default_dic['xlabel'] = 'angle from binary plane'
                default_dic['xmin'] = 0
                default_dic['xmax'] = 90.
            elif v_n == "Ye":
                default_dic['v_n_x'] = 'Ye'
                default_dic['xlabel'] = 'Ye'
                default_dic['xmin'] = 0.
                default_dic['xmax'] = 0.5
            elif v_n == "temp":
                default_dic['v_n_x'] = "temp"
                default_dic["xlabel"] = "temp"
                default_dic['xmin'] = 1e-2
                default_dic['xmax'] = 1e2
                default_dic['xscale'] = "log"
            elif v_n == "velz":
                default_dic['v_n_x'] = "velz"
                default_dic["xlabel"] = "velz"
                default_dic['xmin'] = -0.7
                default_dic['xmax'] = 0.7
            elif v_n == "rho":
                default_dic['v_n_x'] = "rho"
                default_dic["xlabel"] = "rho"
                default_dic['xmin'] = 1e-10
                default_dic['xmax'] = 1e-6
                default_dic['xscale'] = "log"
            elif v_n == "dens_unb_bern":
                default_dic['v_n_x'] = "temp"
                default_dic["xlabel"] = "temp"
                default_dic['xmin'] = 1e-10
                default_dic['xmax'] = 1e-6
                default_dic['xscale'] = "log"
            else:
                raise NameError("hist v_n:{} is not recognized".format(v_n))

            outfpath = resdir + __rootoutdir__ + str(it) + "/hist_plots/"
            if not os.path.isdir(outfpath):
                os.mkdir(outfpath)

            o_plot = PLOT_MANY_TASKS()
            o_plot.gen_set["figdir"] = outfpath
            o_plot.gen_set["type"] = "cartesian"
            o_plot.gen_set["figsize"] = (4.2, 3.8)  # <->, |] # to match hists with (8.5, 2.7)
            o_plot.gen_set["figname"] = "{}.png".format(v_n)
            o_plot.gen_set["sharex"] = False
            o_plot.gen_set["sharey"] = False
            o_plot.gen_set["subplots_adjust_h"] = 0.0
            o_plot.gen_set["subplots_adjust_w"] = 0.2
            o_plot.set_plot_dics = []

            fpath = outfpath + "{}.png".format(v_n)

            try:
                if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                    if os.path.isfile(fpath): os.remove(fpath)
                    print_colored_string(["task:", "plot hist", "it:", "{}".format(it), "v_ns:", v_n, ":", "computing"],
                                         ["blue", "green", "blue", "green", "blue", "green", "", "green"])
                    #-------------------------------
                    default_dic["it"] = it
                    o_plot.set_plot_dics.append(default_dic)

                    o_plot.main()
                    o_plot.set_plot_dics = []
                    o_plot.figure.clear()
                    #-------------------------------
                else:
                    print_colored_string(["task:", "plot hist", "it:", "{}".format(it), "v_ns:", v_n, ":", "skipping"],
                                         ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
            except IOError:
                print_colored_string(["task:", "plot hist", "it:", "{}".format(it), "v_ns:", v_n, ":", "missing file"],
                                     ["blue", "green", "blue", "green", "blue", "green", "", "red"])
            except KeyboardInterrupt:
                exit(1)
            except:
                print_colored_string(["task:", "plot hist", "it:", "{}".format(it), "v_ns:", v_n, ":", "failed"],
                                     ["blue", "green", "blue", "green", "blue", "green", "", "red"])

def plot_density_modes(dmclass, rewrite=False):
    plotfname = __d3densitymodesfame__.replace(".h5", ".png")
    path = Paths.ppr_sims + dmclass.sim + '/' + __rootoutdir__
    # fpath = path + fname
    dmclass.gen_set['fname'] = path + __d3densitymodesfame__ #"density_modes_lap15.h5"

    fpath = path + plotfname

    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = path
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = plotfname
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = False
    o_plot.set_plot_dics = []

    mags = dmclass.get_data(1, "int_phi_r")
    times = dmclass.get_grid("times")
    densmode_m1 = {
        'task': 'line',  'ptype': 'cartesian',
        'xarr':times*1e3, 'yarr':mags,
        'position': (1, 1),
        'v_n_x': 'times', 'v_n_y': 'int_phi_r abs',
        'mode': 1, 'norm_to_m': 0,
        'ls': '-', 'color': 'black', 'lw': 1., 'ds': 'default', 'alpha': 1.,
        'label': r'$m=1$', 'ylabel': r'$C_m/C_0$ Magnitude', 'xlabel': r'time [ms]',
        'xmin': None, 'xmax': None, 'ymin': 1e-4, 'ymax': 1e0,
        'xscale': None, 'yscale': 'log', 'legend': {},
        'fancyticks': True, 'minorticks': True,
        'fontsize': 14,
        'labelsize': 14,
    }

    mags = dmclass.get_data(2, "int_phi_r")
    times = dmclass.get_grid("times")
    densmode_m2 = {
        'task': 'line',  'ptype': 'cartesian',
        'xarr':times*1e3, 'yarr':mags,
        'position': (1, 1),
        'v_n_x': 'times', 'v_n_y': 'int_phi_r abs',
        'mode': 2, 'norm_to_m': 0,
        'ls': ':', 'color': 'black', 'lw': 0.8, 'ds': 'default', 'alpha': 1.,
        'label': r'$m=2$', 'ylabel': r'$C_m/C_0$ Magnitude', 'xlabel': r'time [ms]',
        'xmin': None, 'xmax': None, 'ymin': 1e-4, 'ymax': 1e0,
        'xscale': None, 'yscale': 'log',
        'fancyticks': True, 'minorticks': True,
        'legend': {'loc': 'best', 'ncol': 1, 'fontsize': 14},
        'fontsize': 14,
        'labelsize': 14,
    }
    # o_plot.set_plot_dics.append(densmode_m0)

    try:
        if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
            if os.path.isfile(fpath): os.remove(fpath)
            print_colored_string(["task:", "plot dens modes", "fname:", plotfname, "mmodes:", "[1,2]", ":", "computing"],
                                 ["blue", "green", "blue", "green", "blue", "green", "", "green"])
            o_plot.set_plot_dics.append(densmode_m1)
            o_plot.set_plot_dics.append(densmode_m2)

            o_plot.main()
        else:
            print_colored_string(["task:", "plot dens modes", "fname:", plotfname, "mmodes:", "[1,2]", ":", "skipping"],
                                 ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
    except IOError:
        print_colored_string(["task:", "plot dens modes", "fname:", plotfname, "mmodes:", "[1,2]", ":", "missing file"],
                             ["blue", "green", "blue", "green", "blue", "green", "", "red"])
    except KeyboardInterrupt:
        exit(1)
    except:
        print_colored_string(["task:", "plot dens modes", "fname:", plotfname, "mmodes:", "[1,2]", ":", "failed"],
                             ["blue", "green", "blue", "green", "blue", "green", "", "red"])

""" ==============================================| D3 OTHER |======================================================="""

""" ===============================================| D3 ALL |======================================================= """

def d3_main_computational_loop():

    outdir = glob_outdir_sim
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    outdir += __rootoutdir__
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # methods that required inteprolation [No masks used!]
    if "vtk" in glob_tasklist:
        o_grid = CARTESIAN_GRID()
        o_d3int = INTMETHODS_STORE(glob_sim, o_grid, glob_symmetry)

        d3_int_data_to_vtk(o_d3int, outdir=outdir, rewrite=glob_overwrite)

        for it in glob_its:
            sys.stdout.flush()
            o_d3int.save_vtk_file(it, glob_v_ns, glob_overwrite, outdir="profiles/", private_dir="vtk/")
            sys.stdout.flush()

    # methods that do not require interplation [Use masks for reflevels and lapse]
    d3corr_class = MAINMETHODS_STORE(glob_sim)
    d3corr_class.mask_setup = {'rm_rl': True, # REMOVE previouse ref. level from the next
                                'rho': [6.e4 / 6.176e+17, 1.e13 / 6.176e+17],  # REMOVE atmo and NS
                                'lapse': [0.15, 1.]}  # remove apparent horizon

    # tasks for each iteration
    for it in glob_its:
        sys.stdout.flush()
        _outdir = outdir + str(it) + '/'
        if not os.path.isdir(_outdir):
            os.mkdir(_outdir)
        for task in glob_tasklist:
            # if task in ["all", "plotall", "densmode"]:   pass
            if task == "corr":  d3_corr_for_it(it, d3corr_class, outdir=_outdir, rewrite=glob_overwrite)
            if task == "hist":  d3_hist_for_it(it, d3corr_class, outdir=_outdir, rewrite=glob_overwrite)
            if task == "mass":  d3_disk_mass_for_it(it, d3corr_class, outdir=_outdir, rewrite=glob_overwrite)
            if task == "slice": d3_to_d2_slice_for_it(it, d3corr_class, outdir=_outdir, rewrite=glob_overwrite)
            # else:
            #     raise NameError("d3 method is not recognized: {}".format(task))
            sys.stdout.flush()
        d3corr_class.delete_for_it(it=it, except_v_ns=[], rm_masks=True, rm_comp=True, rm_prof=False)
        print("\n")

    # methods that require all iterations loaded
    if "densmode" in glob_tasklist:
        d3_dens_modes(d3corr_class, outdir=outdir, rewrite=glob_overwrite)

    # plotting tasks
    d3_slices = LOAD_PROFILE_XYXZ(glob_sim)
    d3_corr = LOAD_RES_CORR(glob_sim)
    dm_class = LOAD_DENSITY_MODES(glob_sim)

    for task in glob_tasklist:
        if task.__contains__("plot"):
            # if task in ["all", "plotall", "densmode"]:  pass
            if task == "plotcorr":        plot_d3_corr(d3_corr, rewrite=glob_overwrite)
            if task == "plotslice":       plot_d3_prof_slices(d3_slices, rewritefigs=glob_overwrite)
            if task == "plothist":        plot_d3_hist(d3_corr, rewrite=glob_overwrite)
            if task == "plotdensmode":    plot_density_modes(dm_class, rewrite=glob_overwrite)
            sys.stdout.flush()
            # else:
            #     raise NameError("glob_task for plotting is not recognized: {}"
            #                     .format(task))



if __name__ == '__main__':
    #
    parser = ArgumentParser(description="postprocessing pipeline")
    parser.add_argument("-s", dest="sim", required=True, help="task to perform")
    parser.add_argument("-t", dest="tasklist", required=False, nargs='+', default=[], help="tasks to perform")
    parser.add_argument("--v_n", dest="v_ns", required=False, nargs='+', default=[], help="variable (or group) name")
    parser.add_argument("--rl", dest="reflevels", required=False, nargs='+', default=[], help="reflevels")
    parser.add_argument("--it", dest="iterations", required=False, nargs='+', default=[], help="iterations")
    parser.add_argument('--time', dest="times", required=False, nargs='+', default=[], help='Timesteps')
    #
    parser.add_argument("-o", dest="outdir", required=False, default=Paths.ppr_sims, help="path for output dir")
    parser.add_argument("-i", dest="simdir", required=False, default=Paths.gw170817, help="path to simulation dir")
    parser.add_argument("--overwrite", dest="overwrite", required=False, default="no", help="overwrite if exists")
    #
    parser.add_argument("--sym", dest="symmetry", required=False, default=None, help="symmetry (like 'pi')")
    # Info/checks
    args = parser.parse_args()
    glob_tasklist = args.tasklist
    glob_sim = args.sim
    glob_simdir = args.simdir
    glob_outdir = args.outdir
    glob_v_ns = args.v_ns
    glob_rls = args.reflevels
    glob_its = args.iterations
    glob_times=args.times
    glob_symmetry = args.symmetry
    glob_overwrite = args.overwrite
    simdir = Paths.gw170817 + glob_sim + '/'
    resdir = Paths.ppr_sims + glob_sim + '/'

    # check given data
    if glob_symmetry != None:
        if not click.confirm("Selected symmetry: {} Is it correct?".format(glob_symmetry),
                             default=True, show_default=True):
            exit(1)

    # check if the simulations dir exists
    if not os.path.isdir(glob_simdir + glob_sim):
        raise NameError("simulation dir: {} does not exist in rootpath: {} "
                        .format(glob_sim, glob_simdir))
    # check if tasks are set properly
    if len(glob_tasklist) == 0:
        raise NameError("tasklist is empty. Set what tasks to perform with '-t' option")
    elif len(glob_tasklist) == 1 and "all" in glob_tasklist:
        glob_tasklist = __profile__["tasklist"]
        glob_tasklist.remove("vtk")
        Printcolor.print_colored_string(["Set", "All", "tasks"],
                                        ["blue", "green", "blue"])
    else:
        for task in glob_tasklist:
            if not task in __profile__["tasklist"]:
                raise NameError("task: {} is not among available ones: {}"
                                .format(task, __profile__["tasklist"]))
    # check if there any profiles to use
    ittime = LOAD_ITTIME(glob_sim)
    isprof, itprof, tprof = ittime.get_ittime("profiles", d1d2d3prof="prof")
    #
    if len(itprof) == 0:
        Printcolor.red("No profiles found. Please, extract profiles for {} "
                         "and save them in /sim_dir/profiles/3d/".format(glob_sim))
        exit(0)
    else:
        Printcolor.print_colored_string(["Available", "{}".format(len(itprof)), "profiles to postprocess"],
                                        ["blue", "green", "blue"])
        for it, t in zip(itprof, tprof):
            Printcolor.print_colored_string(["\tit:", "{:d}".format(it), "time:", "{:.1f}".format(t*1e3), "[ms]"],
                                            ["blue", "green", "blue", "green", "blue"])
    # check which iterations/timesteps to use
    if len(glob_its) > 0 and len(glob_times) > 0:
        raise ValueError("Please, set either iterations (--it) or times (--time) "
                         "but NOT both")
    elif len(glob_its) == 0 and len(glob_times) == 0:
        raise ValueError("Please, set either iterations (--it) or times (--time)")
    elif (len(glob_times) == 1 and "all" in glob_times) or (len(glob_its) == 1 and "all" in glob_its):
        Printcolor.print_colored_string(["Selected All", "{}".format(len(itprof)), "iterations to postprocess"],
                                        ["blue", "green", "blue"])
        glob_its = itprof
        glob_times = tprof
    elif len(glob_its) > 0 and len(glob_times) == 0:
        glob_its = np.array(glob_its, dtype=int)
        _glob_its = []
        _glob_times = []
        for it in glob_its:
            if int(it) in itprof:
                _glob_its = np.append(_glob_its, it)
                _glob_times = np.append(_glob_times, ittime.get_time_for_it(it, "prof"))
            else:
                raise ValueError("For given iteraton:{} profile is not found (in ittime.h5)"
                                 .format(it))
        glob_its = _glob_its
        glob_times = _glob_times
        assert len(glob_its) > 0
    elif len(glob_its) == 0 and len(glob_times) > 0:
        glob_times = np.array(glob_times, dtype=float) / 1e3 # back to [s]
        _glob_its = []
        _glob_times = []
        for t in glob_times:
            idx = UTILS.find_nearest_index(tprof, t)
            _t =  tprof[idx]
            _it = ittime.get_it_for_time(_t, "d1")
            _glob_its = np.append(_glob_its, _it)
            _glob_times = np.append(_glob_times, _t)
        glob_its = np.unique(_glob_its)
        glob_times = np.unique(_glob_times)
        assert len(glob_its) > 0
        assert len(glob_times) == len(glob_its)
    else:
        raise IOError("Input iterations (--it) or times (--time) are not recognized")

    glob_its = np.array(glob_its, dtype=int)
    Printcolor.print_colored_string(["Set iterations", "{}".format(glob_its)],["blue","green"])
    Printcolor.print_colored_string(["Set timesteps ", "{}".format(glob_times*1e3, fmt=".1f")], ["blue", "green"])

    if glob_overwrite == "no":
        glob_overwrite = False
    elif glob_overwrite == "yes":
        glob_overwrite = True
    else:
        raise NameError("for '--overwrite' option use 'yes' or 'no'. Given: {}"
                        .format(glob_overwrite))
    glob_outdir_sim = Paths.ppr_sims + glob_sim + '/'

    # set globals
    Paths.gw170817 = glob_simdir
    Paths.ppr_sims = glob_outdir

    # tasks

    d3_main_computational_loop()

    Printcolor.blue("Done.")
