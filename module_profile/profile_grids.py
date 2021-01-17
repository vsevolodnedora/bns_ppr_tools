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


class POLAR_GRID:
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

        self.grid_info = {'type': 'pol', 'n_r': 150, 'n_phi': 150}

        self.grid_type = self.grid_info['type']

        # self.carpet_grid = carpet_grid

        self.list_int_grid_v_ns = ["x_pol", "y_pol",
                                  "r_pol", "phi_pol",
                                  "dr_pol", "dphi_pol"]

        print('-' * 25 + 'INITIALIZING POLAR GRID' + '-' * 25)

        phi_pol, r_pol, self.dphi_pol_2d, self.dr_pol_2d = self.get_phi_r_grid()

        self.r_pol_2d, self.phi_pol_2d = np.meshgrid(r_pol, phi_pol, indexing='ij')
        self.x_pol_2d = self.r_pol_2d * np.cos(self.phi_pol_2d)
        self.y_pol_2d = self.r_pol_2d * np.sin(self.phi_pol_2d)

        print("\t GRID: [phi:r] = [{}:{}]".format(len(phi_pol), len(r_pol)))

        print("\t GRID: [x_pol_2d:  ({},{})] {} pints".format(self.x_pol_2d.min(), self.x_pol_2d.max(), len(self.x_pol_2d[:,0])))
        print("\t GRID: [y_pol_2d:  ({},{})] {} pints".format(self.y_pol_2d.min(), self.y_pol_2d.max(), len(self.y_pol_2d[0,:])))

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

    def get_phi_r_grid(self):

        # extracting grid info
        n_r = self.grid_info["n_r"]
        n_phi = self.grid_info["n_phi"]

        # constracting the grid
        r_cyl_f = self.make_stretched_grid(0., 15., 512., n_r, n_phi)
        phi_cyl_f = np.linspace(0, 2 * np.pi, n_phi)

        # edges -> bins (cells)
        r_cyl = 0.5 * (r_cyl_f[1:] + r_cyl_f[:-1])
        phi_cyl = 0.5 * (phi_cyl_f[1:] + phi_cyl_f[:-1])

        # 1D grind -> 3D grid (to mimic the r, z, phi structure)
        dr_cyl = np.diff(r_cyl_f)[:, np.newaxis]
        dphi_cyl = np.diff(phi_cyl_f)[np.newaxis, :]

        return phi_cyl, r_cyl, dphi_cyl, dr_cyl

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
        return np.column_stack([self.x_pol_2d.flatten(),
                                self.y_pol_2d.flatten()])

    def get_shape(self):
        return self.x_pol_2d.shape

    def get_int_grid(self, v_n):

        if v_n == "x_pol":
            return self.x_pol_2d
        elif v_n == "y_pol":
            return self.y_pol_2d
        elif v_n == "r_pol":
            return self.r_pol_2d
        elif v_n == "phi_pol":
            return self.phi_pol_2d
        elif v_n == "dr_pol":
            return self.dr_pol_2d
        elif v_n == "dphi_pol":
            return self.dphi_pol_2d
        else:
            raise NameError("v_n: {} not recogized in grid. Available:{}"
                            .format(v_n, self.list_int_grid_v_ns))

    def save_grid(self, sim, outdir):

        # path = Paths.ppr_sims + sim + '/' + outdir
        outfile = h5py.File(outdir + str(self.grid_type) + '_grid.h5', "w")

        # if not os.path.exists(outdir):
        #     os.makedirs(outdir)

        # print("Saving grid...")
        for v_n in self.list_int_grid_v_ns:
            outfile.create_dataset(v_n, data=self.get_int_grid(v_n))
        outfile.close()


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

    def __init__(self, grid_info = None):
        if grid_info == None:
            self.grid_info = {'type': 'cyl', 'n_r': 150, 'n_phi': 150, 'n_z': 100}
        else:
            self.grid_info = grid_info

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

    def save_grid(self, sim, outdir):

        # path = Paths.ppr_sims + sim + '/' + outdir
        outfile = h5py.File(outdir + str(self.grid_type) + '_grid.h5', "w")

        # if not os.path.exists(path):
        #     os.makedirs(path)

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

    def save_grid(self, sim, outdir):

        # path = Paths.ppr_sims + sim + '/' + outdir
        outfile = h5py.File(outdir + str(self.grid_type) + "_grid.h5", "w")

        # if not os.path.exists(path):
        #     os.makedirs(path)

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

        # path = Paths.ppr_sims + sim + "/" + outdir
        outfile = h5py.File(outdir + self.grid_type + '_grid.h5', "w")

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # print("Saving grid...")
        for v_n in self.list_int_grid_v_ns:
            outfile.create_dataset(v_n, data=self.get_int_grid(v_n))
        outfile.close()
