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


class LOAD_NU_PROFILE:

    def __init__(self, flist, itlist, timesteplist, symmetry=None):

        # LOAD_ITTIME.__init__(self, sim, pprdir=pprdir)

        assert len(flist) == len(itlist)
        assert len(timesteplist) == len(flist)

        self.nuprof_name = "nu" # -> 12345nu.h5

        self.symmetry = symmetry

        self.list_files = flist

        # self.profpath = profdir#Paths.gw170817 + sim + '/' + "profiles/3d/"

        # _, itnuprofs, timenuprofs = self.get_ittime("nuprofiles", "nuprof")
        # if not len(itnuprofs) == 0:
        #     is3ddata, it3d, t3d = self.get_ittime("overall", d1d2d3prof="d3")
        #     if is3ddata:
        #         raise IOError("ittime.h5 says there are NO nuprofiles, while there IS 3D data for times:\n{}"
        #                       "\n Extract nuprofiles before proceeding"
        #                       .format(t3d))
        #     else:
        #         raise IOError("ittime.h5 says there are no profiles, and no 3D data found.")

        self.list_iterations = list(itlist)
        self.list_times = timesteplist

        self.list_nuprof_v_ns = ['abs_energy', 'abs_nua', 'abs_nue', 'abs_number', 'eave_nua', 'eave_nue',
                                 'eave_nux', 'E_nua', 'E_nue', 'E_nux', 'flux_fac', 'ndens_nua', 'ndens_nue',
                                 'ndens_nux','N_nua', 'N_nue', 'N_nux']

        self.list_nugrid_v_ns = ["x", "y", "z", "r", "theta", "phi"]

        self.nudfile_matrix = [0 for it in range(len(self.list_iterations))]

        # self.nugrid_matrix = [0 for it in range(len(self.list_iterations))]

        self.nuprof_arr_matrix = [[np.zeros(0, )
                                   for v_n in range(len(self.list_nuprof_v_ns))]
                                   for it in range(len(self.list_iterations))]

        self.nuprof_grid_params_matrix = [[-1.
                                   for v_n in range(len(self.list_nugrid_v_ns))]
                                   for it in range(len(self.list_iterations))]

    def check_nuprof_v_n(self, v_n):
        if not v_n in self.list_nuprof_v_ns:
            raise NameError("v_n:{} not in list of nuprofile v_ns:{}"
                            .format(v_n, self.list_nuprof_v_ns))

    def check_it(self, it):
        if not int(it) in self.list_iterations:
            raise NameError("it:{} not in list of iterations:{}"
                            .format(it, self.list_iterations))

    def i_nu_it(self, it):
        return int(self.list_iterations.index(it))

    # --- ---

    def load_nudfile(self, it):
        idx = self.list_iterations.index(it)
        fname = self.list_files[idx]
        if not os.path.isfile(fname):
            raise IOError("Expected file:{} NOT found".format(fname))
        dfile = h5py.File(fname, "r")
        self.nudfile_matrix[self.i_nu_it(it)] = dfile

    def is_nudfile_loaded(self, it):
        if isinstance(self.nudfile_matrix[self.i_nu_it(it)], int):
            self.load_nudfile(it)

    def get_nuprofile_dfile(self, it):
        self.check_it(it)
        self.is_nudfile_loaded(it)
        return self.nudfile_matrix[self.i_nu_it(it)]

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

    def i_nu_v_n(self, v_n):
        return int(self.list_nuprof_v_ns.index(v_n))

    def extract_arr_from_nuprof(self, it, v_n):
        nudfile = self.get_nuprofile_dfile(it)
        arr = np.array(nudfile[v_n])
        self.nuprof_arr_matrix[self.i_nu_it(it)][self.i_nu_v_n(v_n)] = arr

    def is_nuprofarr_extracted(self, it, v_n):
        if len(self.nuprof_arr_matrix[self.i_nu_it(it)][self.i_nu_v_n(v_n)]) == 0:
            self.extract_arr_from_nuprof(it, v_n)

    def get_nuprof_arr(self, it, v_n):

        self.check_nuprof_v_n(v_n)

        self.is_nuprofarr_extracted(it, v_n)

        return self.nuprof_arr_matrix[self.i_nu_it(it)][self.i_nu_v_n(v_n)]

    # grid

    def get_nrad(self, it):
        nudfile = self.get_nuprofile_dfile(it)
        return int(nudfile.attrs["nrad"])

    def get_nphi(self, it):
        nudfile = self.get_nuprofile_dfile(it)
        return int(nudfile.attrs["nphi"])

    def get_ntheta(self, it):
        nudfile = self.get_nuprofile_dfile(it)
        return int(nudfile.attrs["ntheta"])

    def get_sph_grid(self, it, nextra=0):

        rad, phi, theta = np.mgrid[
                          0:self.get_nrad(it) + nextra, \
                          0:self.get_nphi(it) + nextra, \
                          0:self.get_ntheta(it) + nextra].astype(np.float32)

        return rad, phi, theta

    def get_x_y_z_grid(self, it, plane=None, dual=False, rmax=50):

        if dual:
            nextra = 1
            shift  = -0.5
        else:
            nextra = 0
            shift  = 0.0

        nrad, nphi, ntheta = self.get_nrad(it), self.get_nphi(it), self.get_ntheta(it)


        if plane is None:
            rad, phi, theta = np.mgrid[0:nrad+nextra, 0:nphi+nextra,\
                    0:ntheta+nextra].astype(np.float32)
            rad = (rad + shift) * rmax/(nrad - 1)
            phi = (phi + shift) * (2*np.pi)/(nphi - 1)
            theta = (theta + shift) * np.pi/(ntheta - 1)
            x = rad * np.cos(phi) * np.sin(theta)
            y = rad * np.sin(phi) * np.sin(theta)
            z = rad * np.cos(theta)
            return x, y, z
        if plane == "xy":
            rad, phi = np.mgrid[0:nrad+nextra,\
                    0:nphi+nextra].astype(np.float32)
            rad = (rad + shift) * rmax/(nrad - 1)
            phi = (phi + shift) * (2*np.pi)/(nphi - 1)
            x = rad * np.cos(phi)
            y = rad * np.sin(phi)
            return x, y
        if plane == "xz" or plane == "yz":
            rad, theta = np.mgrid[0:nrad+nextra,\
                    0:2*ntheta+nextra-1].astype(np.float32)
            rad = (rad  + shift) * rmax/(nrad - 1)
            theta = (theta + shift) * np.pi/(ntheta - 1)
            x = rad * np.sin(theta)
            z = rad * np.cos(theta)
            return x, z
        raise Exception("This is a bug in the code")

    # def check_nugrid_v_n(self, v_n):
    #     if not v_n in self.list_nugrid_v_ns:
    #         raise NameError("v_n:{} is not in the list of nugrid v_ns:{}"
    #                         .format(v_n, self.list_nugrid_v_ns))
    #
    # def is_grid_params_extracted(self, it, v_n):
    #     pass
    #
    # def get_sph_grid_params(self, it, v_n):
    #     self.check_nugrid_v_n(v_n)
    #     self.check_it(it)
    #     self.is_grid_params_extracted(it, v_n)


class MODIFY_NU_DATA(LOAD_NU_PROFILE):

    def __init__(self, flist, itlist, timesteplist, symmetry=None):
        LOAD_NU_PROFILE.__init__(self, flist=flist, itlist=itlist, timesteplist=timesteplist, symmetry=symmetry)

    def get_nuprof_arr_sph(self, it, v_n):

        nrad = self.get_nrad(it)
        nphi = self.get_nphi(it)
        ntheta = self.get_ntheta(it)

        arr = self.get_nuprof_arr(it, v_n)
        reshaped_arr = arr.reshape((nrad, nphi, ntheta))

        return reshaped_arr

    def get_nuprof_arr_slice(self, it, plane, v_n):

        if not plane in ["xy", "xz", "yz"]:
            raise NameError("plane:{} is not recognized"
                            .format(plane))

        nrad = self.get_nrad(it)
        nphi = self.get_nphi(it)
        ntheta = self.get_ntheta(it)

        fnew = self.get_nuprof_arr_sph(it, v_n)

        if plane == "xy":
            out = np.empty((nrad, nphi), dtype=fnew.dtype)
            out[:] = np.NAN
            if 0 != ntheta % 2:
                out[:,:] = fnew[:,:,ntheta/2]
            else:
                itheta = int(ntheta/2)
                out[:,:] = 0.5*(fnew[:,:,itheta-1] + fnew[:,:,itheta])
        elif plane == "xz":
            out = np.empty((nrad, 2*ntheta-1), dtype=fnew.dtype)
            out[:] = np.NAN
            out[:,:ntheta] = fnew[:,0,:]
            iphi = int(nphi/2)
            out[:,ntheta:] = 0.5*(fnew[:,iphi-1,-2::-1] + fnew[:,iphi,-2::-1])
        elif plane == "yz":
            out = np.empty((nrad, 2*ntheta-1), dtype=fnew.dtype)
            out[:] = np.NAN
            iphi1 = int(nphi/4)
            iphi2 = int(3*iphi1)
            out[:,:ntheta] = 0.5*(fnew[:,iphi1+1,:] + fnew[:,iphi1,:])
            out[:,ntheta:] = 0.5*(fnew[:,iphi2+1,-2::-1] + fnew[:,iphi2,-2::-1])
        else: raise Exception("This is a bug in the code. Deal with it.")
        return np.ma.masked_invalid(out)
