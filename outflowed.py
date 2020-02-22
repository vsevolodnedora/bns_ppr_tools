from __future__ import division
import sys
from sys import path
path.append('modules/')
import units as ut # for tmerg
from math import pi, log10, sqrt
import os.path
import copy
import h5py
import click
from argparse import ArgumentParser
from scipy.interpolate import RegularGridInterpolator

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

import multiprocessing as mp
from functools import partial

# from _curses import raw
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib import ticker
# import matplotlib.pyplot as plt
# from matplotlib import rc
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

# import statsmodels.formula.api as smf

# import scipy.optimize as opt
# import matplotlib as mpl
# import pandas as pd
# import numpy as np
# import itertools-

# import cPickle
# import time

# import csv

# import os
# import re


# from multiprocessing import Pool



# from scidata.utils import locate
# import scidata.carpet.hdf5 as h5
# import scidata.xgraph as xg
#
# from matplotlib.ticker import AutoMinorLocator, FixedLocator, NullFormatter, \
#     MultipleLocator
# from matplotlib.colors import LogNorm, Normalize
# from matplotlib.colors import Normalize, LogNorm



from utils import *

from preanalysis import LOAD_ITTIME

from plotting_methods import PLOT_MANY_TASKS

""" ==============================================| SETTINGS |======================================================="""

__outflowed__ = {
    "name": "outflowed",
    "tasklist": ["reshape", "all", "hist", "timecorr", "corr", "totflux",
                 "massave", "ejtau", "yeilds", "mknprof", "summary"],
    "detectors":[0,1]
}

""" =======================================| OUTFLOW.ASC -> OUTFLOW.h5 |============================================ """

# David Radice's spherical Grid class for 'outflowed data'
class SphericalSurface:
    def __init__(self, ntheta, nphi, radius=1.0):
        self.ntheta = ntheta
        self.nphi   = nphi
        self.radius = radius
        self.dtheta = pi/self.ntheta
        self.dphi   = 2*pi/self.nphi
    def mesh(self):
        theta = (np.arange(self.ntheta) + 0.5)*self.dtheta
        phi   = np.linspace(0, 2*pi, self.nphi+1)
        return np.meshgrid(theta, phi, indexing="ij")
    def area(self):
        theta, phi = self.mesh()
        dA = self.radius**2 * np.sin(theta) * self.dtheta * self.dphi
        dA[:,-1] = 0.0  # avoid double counting data at phi = 2 pi
        return dA
    # This mesh is used for the visualization
    def mesh_vis(self):
        dtheta = 180./(self.ntheta)
        dphi   = 360./(self.nphi + 1)
        theta  = (np.arange(self.ntheta) + 0.5)*dtheta - 90.0
        phi    = (np.arange(self.nphi + 1) + 0.5)*dphi
        return np.meshgrid(theta, phi, indexing='ij')
    def reshape(self, vector):
        return vector.reshape((self.ntheta, self.nphi + 1))
    def size(self):
        return (self.nphi + 1)*self.ntheta

# DAVID RADICE A tabulated nuclear equation of state
class EOSTable(object):
    def __init__(self):
        """
        Create an empty table
        """
        self.log_rho = None
        self.log_temp = None
        self.ye = None
        self.table = {}
        self.interp = {}

    def read_table(self, fname):
        """
        Initialize the EOS object from a table file

        * fname : must be the filename of a table in EOS_Thermal format
        """
        assert os.path.isfile(fname)

        dfile = h5py.File(fname, "r")
        for k in dfile.keys():
            self.table[k] = np.array(dfile[k])
        del dfile

        self.log_rho = np.log10(self.table["density"])
        self.log_temp = np.log10(self.table["temperature"])
        self.ye = self.table["ye"]

    def get_names(self):
        return self.table.keys()

    def evaluate(self, prop, rho, temp, ye):
        """
        * prop  : name of the thermodynamical quantity to compute
        * rho   : rest mass density (in Msun^-2)
        * temp  : temperature (in MeV)
        * ye    : electron fraction
        """
        assert self.table.has_key(prop)
        assert self.log_rho is not None
        assert self.log_temp is not None
        assert self.ye is not None

        assert rho.shape == temp.shape
        assert temp.shape == ye.shape

        log_rho = np.log10(ut.conv_dens(ut.cactus, ut.cgs, rho))
        log_temp = np.log10(temp)
        xi = np.array((ye.flatten(), log_temp.flatten(),
            log_rho.flatten())).transpose()

        if not self.interp.has_key(prop):
            self.interp[prop] = RegularGridInterpolator(
                (self.ye, self.log_temp, self.log_rho), self.table[prop],
                method="linear", bounds_error=False, fill_value=None)

        return self.interp[prop](xi).reshape(rho.shape)


class LOAD_OUTFLOW_SURFACE(LOAD_ITTIME):

    def __init__(self, sim):

        LOAD_ITTIME.__init__(self, sim)

        _, itd1, td1 = self.get_ittime("overall", d1d2d3prof="outflow")

        self.list_iterations = list(itd1)
        self.list_times = list(td1)

        self.list_detectors = [0, 1]

        self.list_eos_v_ns = ['eps', 'press']  # 'press', 'entropy'

        self.list_v_ns = ['it', 'time',
                          "fluxdens", "w_lorentz", "eninf", "surface_element",
                          "alp", "rho", "vel[0]", "vel[1]", "vel[2]", "Y_e",
                          "entropy", "temperature"] + self.list_eos_v_ns

        # dummy object to allocate space
        self.matrix_grid_objects = [0 for i in range(len(self.list_detectors))]

        self.list_outputs = self.get_list_outputs()

        self.matrix_raw_data = [[[np.zeros(0,)
                                  for v in range(len(self.list_v_ns))]
                                 for o in range(len(self.list_outputs))]
                                for d in range(len(self.list_detectors))]


        self.v_n_to_file_dic = {
            'it'                : 0,
            'time'              : 1,
            'fluxdens'          : 5,
            'w_lorentz'         : 6,
            'eninf'             : 7,
            'surface_element'   : 8,
            'alp'               : 9,
            'rho'               : 10,
            'vel[0]'            : 11,
            'vel[1]'            : 12,
            'vel[2]'            : 13,
            'Y_e'               : 14,
            'entropy'           : 15,
            'temperature'       : 16
        }

        self.clean = True

    def check_det(self, det):
        if not det in self.list_detectors:
            raise NameError("detector: {} is not in the list: {}"
                            .format(det,self.list_detectors))

    def i_det(self, det):
        return int(self.list_detectors.index(det))

    def _grid_object(self, det=0):

        fname = "outflow_surface_det_%d_fluxdens.asc" % det
        fpath = Paths.gw170817 + self.sim + "/" + self.get_list_outputs()[0] + "/data/" + fname
        assert os.path.isfile(fpath)
        dfile = open(fpath, "r")
        dfile.readline() # move down one line
        match = re.match('# detector no.=(\d+) ntheta=(\d+) nphi=(\d+)$', dfile.readline())
        assert int(det) == int(match.group(1))
        ntheta = int(match.group(2))
        nphi =int(match.group(3))
        dfile.readline()
        dfile.readline()
        line = dfile.readline().split()
        radius = round(sqrt(float(line[2])**2 + float(line[3])**2 + float(line[4])**2))
        if not self.clean:
            print("\t\tradius = {}".format(radius))
            print("\t\tntheta = {}".format(ntheta))
            print("\t\tnphi   = {}".format(nphi))
        del dfile
        return SphericalSurface(ntheta, nphi, radius)

    def is_grid_object_loaded(self, det):
        if isinstance(self.matrix_grid_objects[self.i_det(det)], int):
            grid = self._grid_object(det)
            self.matrix_grid_objects[self.i_det(det)]=grid

    def get_grid_object(self, det=0):
        self.check_det(det)
        self.is_grid_object_loaded(det)
        return self.matrix_grid_objects[self.i_det(det)]

    def check_v_n(self, v_n):
        if not v_n in self.list_v_ns:
            raise NameError("v_n: {} is not in the list: {} "
                            .format(v_n, self.list_v_ns))

    def i_v_n(self, v_n):
        return int(self.list_v_ns.index(v_n))

    def check_output(self, output):
        if not output in self.list_outputs:
            raise NameError("output: {} is not on the list: {}"
                            .format(output, self.list_outputs))

    def i_output(self, output):
        return int(self.list_outputs.index(output))

    def load_raw_data(self, det, output):

        fname = "outflow_surface_det_%d_fluxdens.asc" % det
        fpath = Paths.gw170817 + self.sim + "/" + output + "/data/" + fname
        assert os.path.isfile(fpath)
        if not self.clean: print("\t\tReading %s..." % (output)),
        sys.stdout.flush()
        fdata = np.loadtxt(fpath, usecols=self.v_n_to_file_dic.values(), unpack=True) # dtype=np.float64
        for i_v_n, v_n in enumerate(self.v_n_to_file_dic.keys()):
            data = np.array(fdata[i_v_n])
            self.matrix_raw_data[self.i_det(det)][self.i_output(output)][self.i_v_n(v_n)] = data
        if not self.clean: print("done!")
        sys.stdout.flush()

    def is_raw_data_loaded(self, det, output, v_n):
        data = self.matrix_raw_data[self.i_det(det)][self.i_output(output)][self.i_v_n(v_n)]
        if len(data) == 0:
            self.load_raw_data(det, output)

    def get_raw_data(self, det, output, v_n):
        self.check_det(det)
        self.check_output(output)
        self.check_v_n(v_n)
        self.is_raw_data_loaded(det, output, v_n)
        data = self.matrix_raw_data[self.i_det(det)][self.i_output(output)][self.i_v_n(v_n)]
        return data


class EXTRACT_OUTFLOW_SURFACE(LOAD_OUTFLOW_SURFACE):

    def __init__(self, sim):

        LOAD_OUTFLOW_SURFACE.__init__(self, sim)

        self.eos_fname = Paths.get_eos_fname_from_curr_dir(self.sim)

        self.o_eos = EOSTable()
        self.is_eos_table_red = False

        self.list_eos_v_ns = ['eps', 'press'] # 'press', 'entropy'

        self.v_n_to_eos_dic = {
            'eps': "internalEnergy",
            'press': "pressure",
            'entropy': "entropy"
        }

        self.matrix_reshaped_data = [[[np.empty(0,)
                                     for v in range(len(self.list_v_ns))]
                                     for i in range(len(self.list_iterations))]
                                     for d in range(len(self.list_detectors))]

    def check_it(self, it):
        if not it in self.list_iterations:
            raise NameError("it:{} not in the list of iterations: {}"
                            .format(it, self.list_iterations))

    def i_it(self, it):
        return int(self.list_iterations.index(it))

    def extract_data(self, det, it, v_n):
        if not self.clean: print("\tExtracting: det:{} it:{} v_n:{}...".format(det, it, v_n)),
        output = self.get_output_for_it(it, d1d2d3="outflow")
        raw_iterations = np.array(self.get_raw_data(det, output, "it"), dtype=np.int)
        raw_data = self.get_raw_data(det, output, v_n)
        o_grid = self.get_grid_object(det)
        tmp = raw_data[np.array(raw_iterations, dtype=int) == int(it)][:o_grid.size()]
        # if v_n == "fluxdens":
        #     print("raw_data: {}".format(np.sum(tmp))),
        assert len(tmp) > 0
        reshaped_data = o_grid.reshape(tmp)
        # if not self.clean: print("\tdone")
        # if v_n == "fluxdens":
        #     print("reshaped_data: {}".format(np.sum(reshaped_data))),
        if not self.clean: print(' ')
        return reshaped_data

    def extract_eos_data(self, det, it, v_n):

        if not self.is_eos_table_red:
            self.o_eos.read_table(self.eos_fname)
            self.is_eos_table_red = True

        arr_rho = self.get_reshaped_data(det, it, "rho")
        arr_ye = self.get_reshaped_data(det, it, "Y_e")
        arr_temp = self.get_reshaped_data(det, it, "temperature")

        if not self.clean: print("\tExtracting EOS: det:{} it:{} v_n:{}...".format(det, it, v_n))

        data_arr = self.o_eos.evaluate(self.v_n_to_eos_dic[v_n], arr_rho, arr_temp, arr_ye)

        if v_n == 'eps':
            data_arr = ut.conv_spec_energy(ut.cgs, ut.cactus, data_arr)
        elif v_n == 'press':
            data_arr = ut.conv_press(ut.cgs, ut.cactus, data_arr)
        elif v_n == 'entropy':
            data_arr = data_arr
        else:
            raise NameError("EOS quantity: {}".format(v_n))

        return data_arr

    def is_data_extracted(self, det, it, v_n):
        data = self.matrix_reshaped_data[self.i_det(det)][self.i_it(it)][self.i_v_n(v_n)]
        if len(data) == 0:
            if v_n in self.list_eos_v_ns:
                data = self.extract_eos_data(det, it, v_n)
            else:
                data = self.extract_data(det, it, v_n)
            self.matrix_reshaped_data[self.i_det(det)][self.i_it(it)][self.i_v_n(v_n)] = data
        if len(data) == 0:
            raise ValueError("data extraction and reshaping failed")

    def get_reshaped_data(self, det, it, v_n):
        self.check_it(it)
        self.check_v_n(v_n)
        self.check_det(det)
        self.is_data_extracted(det, it, v_n)
        return self.matrix_reshaped_data[self.i_det(det)][self.i_it(it)][self.i_v_n(v_n)]


class COMPUTE_OUTFLOW_SURFACE(EXTRACT_OUTFLOW_SURFACE):

    def __init__(self, sim):

        EXTRACT_OUTFLOW_SURFACE.__init__(self, sim)

    def get_total_flux(self, det, it1=None, it2=None):

        if it1 == None:
            it1 = self.list_iterations[0]
        if it2 == None:
            it2 = self.list_iterations[-1]
        assert it1 < it2
        iterations = np.array(self.list_iterations, dtype=int)
        times = np.array(self.list_times, dtype=float)
        iterations = iterations[(iterations>=it1)&(iterations<=it2)]
        times = times[(iterations>=it1)&(iterations<=it2)] * 1e3 / 0.004925794970773136

        o_grid = self.get_grid_object(det)
        dA = o_grid.area()

        total_flux = np.zeros((o_grid.ntheta, o_grid.nphi + 1))
        # int_flux = 0.0
        told = times[0]

        for t, it in zip(times, iterations):
            dt = (t - told)
            told = t
            print("dt:{}".format(dt))
            data = self.get_reshaped_data(det, it, "fluxdens")
            mask = np.isnan(data)
            data[mask] = 0.0
            # flux = np.sum(data * dA)
            # int_flux += flux*dt
            total_flux[:] += data * dt
        print("Total ejecta mass = {} Msun".format(np.sum(total_flux*dA)))


        return(total_flux)

    def save_outflow(self, det, rewrite=True):

        fname = "outflow_surface_det_{:d}_fluxdens.h5".format(det)
        fpath = Paths.ppr_sims+self.sim+'/'+fname
        # print(self.list_v_ns)
        if os.path.isfile(fpath) and rewrite:
            os.remove(fpath)
        # exit(1)

        # self.load_all_data_in_parallel(det, 12)

        dfile = h5py.File(fpath, "w")

        dfile.create_dataset("iterations", data=np.array(self.list_iterations, dtype=int))
        dfile.create_dataset("times", data=np.array(np.array(self.list_times) * 1e3 / 0.004925794970773136, dtype=np.float32))

        o_grid = self.get_grid_object(det)
        dfile.attrs.create("ntheta", o_grid.ntheta)
        dfile.attrs.create("nphi", o_grid.nphi)
        dfile.attrs.create("radius", o_grid.radius)
        dfile.attrs.create("dphi", 2*pi/o_grid.nphi)
        dfile.attrs.create("dtheta", pi/o_grid.ntheta)

        dfile.create_dataset("area", data=o_grid.area(), dtype=np.float32)
        theta, phi = o_grid.mesh()
        dfile.create_dataset("phi", data=phi, dtype=np.float32)
        dfile.create_dataset("theta", data=theta, dtype=np.float32)

        v_ns = copy.deepcopy(self.list_v_ns)
        v_ns.remove("it")
        v_ns.remove("time")

        if not self.clean:
            print("\tGrid data saved")

        for v_n in v_ns:
            arr = []
            print("\t--- {} ---".format(v_n))
            for it in self.list_iterations:
                data = self.get_reshaped_data(det, it, v_n)
                arr.append(data)
                # print(np.sum(data))
            arr = np.reshape(arr, (len(self.list_iterations), o_grid.ntheta, o_grid.nphi + 1))
            dfile.create_dataset(v_n, data=arr, dtype=np.float32)
        dfile.close()
        print("File saved {}".format(fpath))
        print("done.")

    # def get_raw_data_array(self, det, v_n, it1=None, it2=None):
    #
    #     _, itd1, td1 = self.get_ittime("overall", d1d2d3prof="d1")
    #
    #     if it1 == None:
    #         it1 = itd1[0]
    #     if it2 == None:
    #         it2 = itd1[-1]
    #
    #     outputs = self.get_outputs_between_it1_it2(it1, it2, d1d2d3="d1")
    #     print(outputs)
    #     arr = []
    #     for output in outputs:
    #         data = self.get_raw_data(det, output, v_n)
    #         arr = np.append(arr, data)
    #     return arr
    #
    # def get_data_arr(self, det, v_n, it):
    #     raw_arr = self.get_raw_data_array(det, v_n, it, it)
    #     o_grid = self.get_grid_object(det)
    #
    #
    #
    # def get_total_flux(self, det):
    #     grid = self.get_grid_object(det)
    #     total_flux = np.zeros((grid.ntheta, grid.nphi + 1))
    #     int_flux = 0.0

# with parallelalisation
class LOAD_RESHAPE_SAVE_PARALLEL(LOAD_ITTIME):

    def __init__(self, sim, det, n_proc, eosfname):

        LOAD_ITTIME.__init__(self, sim)

        self.det = det
        self.sim = sim
        n_procs = n_proc
        #
        self.v_ns = ['it', 'time', "fluxdens", "w_lorentz", "eninf", "surface_element",
                "alp", "rho", "vel[0]", "vel[1]", "vel[2]", "Y_e", "entropy", "temperature"]
        self.eos_v_ns = ['eps', 'press']
        #
        self.eos_fpath = eosfname#Paths.get_eos_fname_from_curr_dir(self.sim)
        #
        self.outdirtmp = Paths.ppr_sims+sim+'/tmp/'
        if not os.path.isdir(self.outdirtmp):
            os.mkdir(self.outdirtmp)
        # selecting maximum time
        _, d1it, ditimes = self.get_ittime("overall", "outflow")
        if glob_usemaxtime and (~np.isnan(glob_maxtime) or ~np.isnan(self.maxtime)):
            # use maxtime, just chose which
            if np.isnan(glob_maxtime) and not np.isnan(self.maxtime):
                maxtime = self.maxtime
            elif not np.isnan(glob_maxtime) and not np.isnan(self.maxtime):
                maxtime = glob_maxtime
                Printcolor.yellow("\tOverwriting ittime maxtime:{:.1f}ms with {:.1f}ms"
                                  .format(self.maxtime*1.e3, glob_maxtime*1.e3))
            elif np.isnan(glob_maxtime) and np.isnan(self.maxtime):
                maxtime = d1it.max()
            else:
                maxtime = glob_maxtime
            maxit = self.get_it_for_time(maxtime, "d1")
            Printcolor.print_colored_string(["Max. it set:", "{}".format(maxit), "out of", "{}".format(d1it[-1])],
                                            ["yellow", "green", "yellow", "green"])
        else:
            maxit = -1
        #
        fname = "outflow_surface_det_%d_fluxdens.asc" % det
        if not os.path.isdir(Paths.gw170817 + sim + "/"):
            raise IOError("directory does not exist: {}".format(Paths.gw170817 + sim + "/"))
        self.flist = glob(Paths.gw170817 + sim + "/" + "output-????" + "/data/" + fname)
        if len(self.flist) == 0:
            raise IOError("No files found. Searching for: {} in {}".format(
                          fname, Paths.gw170817 + sim + "/" + "output-????" + "/data/"))
        assert len(self.flist) > 0
        #
        self.grid = self.get_grid()
        #
        print("Pool procs = %d" % n_procs)
        pool = mp.Pool(processes=int(n_procs))
        task = partial(serial_load_reshape_save, grid_object=self.grid, outdir=self.outdirtmp, maxit=maxit)
        result_list = pool.map(task, self.flist)
        #
        tmp_flist = [Paths.ppr_sims + sim + '/tmp/' + outfile.split('/')[-3] + ".h5" for outfile in self.flist]
        tmp_flist = sorted(tmp_flist)
        assert len(tmp_flist) == len(self.flist)
        # load reshaped data
        iterations, times, data_matrix = self.load_tmp_files(tmp_flist)
        # concatenate data into [ntimes, ntheta, nphi] arrays
        self.iterations = np.sort(iterations)
        self.times = np.sort(times)
        concatenated_data = {}
        for v_n in self.v_ns:
            concatenated_data[v_n] = np.stack(([data_matrix[it][v_n] for it in sorted(data_matrix.keys())]))
        # compute EOS quantities
        concatenated_data = self.add_eos_quantities(concatenated_data)
        # save data
        outfname = Paths.ppr_sims + sim + '/' + fname.replace(".asc", ".h5")
        self.save_result(outfname, concatenated_data)
        # removing outflow-xxxx.h5 fiels and /tmp/
        print("...removing temporary files...")
        if os.path.isdir(self.outdirtmp):
            for fname in tmp_flist:
                if os.path.isfile(fname):
                    os.remove(fname)
            os.rmdir(self.outdirtmp)
        print("Done. {} is saved".format(outfname))

    def get_grid(self):

        dfile = open(self.flist[0], "r")
        dfile.readline()  # move down one line
        match = re.match('# detector no.=(\d+) ntheta=(\d+) nphi=(\d+)$', dfile.readline())
        assert int(self.det) == int(match.group(1))
        ntheta = int(match.group(2))
        nphi = int(match.group(3))
        dfile.readline()
        dfile.readline()
        line = dfile.readline().split()
        radius = round(sqrt(float(line[2]) ** 2 + float(line[3]) ** 2 + float(line[4]) ** 2))
        # if not self.clean:
        print("\t\tradius = {}".format(radius))
        print("\t\tntheta = {}".format(ntheta))
        print("\t\tnphi   = {}".format(nphi))
        del dfile

        grid = SphericalSurface(ntheta, nphi, radius)
        return grid

    def load_tmp_files(self, tmp_flist):

        iterations = []
        times = []
        data_matrix = {}
        dum_i = 1
        for ifile, fpath in enumerate(tmp_flist):
            assert os.path.isfile(fpath)
            dfile = h5py.File(fpath, "r")
            for v_n in dfile:
                match = re.match('iteration=(\d+)$', v_n)
                it = int(match.group(1))
                if not it in iterations:
                    i_data_matrix = {}
                    for var_name in self.v_ns:
                        data = np.array(dfile[v_n][var_name])
                        i_data_matrix[var_name] = data
                    data_matrix[it] = i_data_matrix
                    times.append(float(i_data_matrix["time"][0, 0]))
                    iterations.append(int(match.group(1)))
                    print("\tit:{} output:({}/{})".format(it, dum_i, len(tmp_flist)))
                else:
                    pass
            dum_i+=1
            dfile.close()
        return iterations, times, data_matrix

    def add_eos_quantities(self, concatenated_data):

        o_eos = EOSTable()
        o_eos.read_table(self.eos_fpath)
        v_n_to_eos_dic = {
            'eps': "internalEnergy",
            'press': "pressure",
            'entropy': "entropy"
        }
        for v_n in self.eos_v_ns:
            print("Evaluating eos: {}".format(v_n))
            data_arr = o_eos.evaluate(v_n_to_eos_dic[v_n], concatenated_data["rho"],
                                      concatenated_data["temperature"],
                                      concatenated_data["Y_e"])
            if v_n == 'eps':
                data_arr = ut.conv_spec_energy(ut.cgs, ut.cactus, data_arr)
            elif v_n == 'press':
                data_arr = ut.conv_press(ut.cgs, ut.cactus, data_arr)
            elif v_n == 'entropy':
                data_arr = data_arr
            else:
                raise NameError("EOS quantity: {}".format(v_n))

            concatenated_data[v_n] = data_arr
        return concatenated_data

    def save_result(self, outfpath, concatenated_data):
        if os.path.isfile(outfpath):
            os.remove(outfpath)

        outfile = h5py.File(outfpath, "w")

        outfile.create_dataset("iterations", data=np.array(self.iterations, dtype=int))
        outfile.create_dataset("times", data=self.times, dtype=np.float32)

        outfile.attrs.create("ntheta", self.grid.ntheta)
        outfile.attrs.create("nphi", self.grid.nphi)
        outfile.attrs.create("radius", self.grid.radius)
        outfile.attrs.create("dphi", 2 * np.pi / self.grid.nphi)
        outfile.attrs.create("dtheta", np.pi / self.grid.ntheta)

        outfile.create_dataset("area", data=self.grid.area(), dtype=np.float32)
        theta, phi = self.grid.mesh()
        outfile.create_dataset("phi", data=phi, dtype=np.float32)
        outfile.create_dataset("theta", data=theta, dtype=np.float32)

        self.v_ns.remove("it")
        self.v_ns.remove("time")

        for v_n in self.v_ns + self.eos_v_ns:
            outfile.create_dataset(v_n, data=concatenated_data[v_n], dtype=np.float32)
        outfile.close()

""" ==========================================| ANALYZE OUTFLOW.h5 |================================================ """

class HISTOGRAM_EDGES:

    @staticmethod
    def get_edge(v_n):
        if v_n == "Y_e": return np.linspace(0.035, 0.55, 50)
        elif v_n == "theta": return np.linspace(0.031, 3.111, 50) # return np.linspace(0.0, np.pi, 50)
        elif v_n == "phi": return np.linspace(0.06, 6.29, 93)
        elif v_n == "vel_inf" or v_n == "vel_inf_bern": return np.linspace(0., 1., 50)
        elif v_n == "entropy": return np.linspace(0, 200, 100)
        elif v_n == "temperature": #return 10.0 ** np.linspace(-2, 2, 100)
            return np.linspace(0, 5, 100)
        else:
            raise NameError("no hist edges found for v_n:{}".format(v_n))


class LOAD_OUTFLOW_SURFACE_H5(LOAD_ITTIME):

    def __init__(self, sim):
        LOAD_ITTIME.__init__(self, sim)

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

    def update_v_n(self, new_v_n=None):
        if new_v_n != None:
            if not new_v_n in self.list_v_ns:
                self.list_v_ns.append(v_n)

                self.matrix_data = [[np.empty(0, )
                                     for v in range(len(self.list_v_ns) + len(self.list_grid_v_ns))]
                                    for d in range(len(self.list_detectors))]


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
        fpath = Paths.ppr_sims + self.sim + '/' + "outflow_surface_det_{:d}_fluxdens.h5".format(det)
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

    def __init__(self, sim):

        LOAD_OUTFLOW_SURFACE_H5.__init__(self, sim)

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

    def __init__(self, sim, add_mask=None):

        COMPUTE_OUTFLOW_SURFACE_H5.__init__(self, sim)

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
            # 1 - data below tmax and 0 - above
            base_mask_name = str(str(mask).split("_tmax")[0])
            base_mask = self.get_mask(det, base_mask_name)
            #
            tmax = float(str(mask).split("_tmax")[-1])
            tmax = tmax / Constants.time_constant # Msun
            # tmax loaded is postmerger tmax. Thus it need to be added to merger time
            fpath = Paths.ppr_sims+self.sim+"/waveforms/tmerger.dat"
            try:
                tmerg = float(np.loadtxt(fpath, unpack=True)) # Msun
                Printcolor.yellow("\tWarning! using defauled M_Inf=2.748, R_GW=400.0 for retardet time")
                ret_time = PHYSICS.get_retarded_time(tmerg, M_Inf=2.748, R_GW=400.0)
                tmerg = ret_time
                # tmerg = ut.conv_time(ut.cactus, ut.cgs, ret_time)
                # tmerg = tmerg / (Constants.time_constant *1e-3)
            except IOError:
                raise IOError("For the {} mask, the tmerger.dat is needed at {}"
                              .format(mask, fpath))
            except:
                raise ValueError("failed to extract tmerg for outflow tmax mask analysis")

            t = self.get_full_arr(det, "times") # Msun
            # tmax = tmax + tmerg       # Now tmax is absolute time (from the begniing ofthe simulation
            print("t[-1]:{} tmax:{} tmerg:{} -> {}".format(t[-1]*Constants.time_constant*1e-3,
                                            tmax*Constants.time_constant*1e-3,
                                            tmerg*Constants.time_constant*1e-3,
                                            (tmax+tmerg)*Constants.time_constant*1e-3))
            tmax = tmax + tmerg
            if tmax > t[-1]:
                raise ValueError("tmax:{} for the mask is > t[-1]:{}".format(tmax*Constants.time_constant*1e-3,
                                                                             t[-1]*Constants.time_constant*1e-3))
            if tmax < t[0]:
                raise ValueError("tmax:{} for the mask is < t[0]:{}".format(tmax * Constants.time_constant * 1e-3,
                                                                             t[0] * Constants.time_constant * 1e-3))
            fluxdens = self.get_full_arr(det, "fluxdens")
            i_mask = t < t[UTILS.find_nearest_index(t, tmax)]
            newmask = np.zeros(fluxdens.shape)
            for i in range(len(newmask[:, 0, 0])):
                newmask[i, :, :].fill(i_mask[i])

            # print(base_mask.shape,newmask.shape)

            return base_mask & newmask.astype(bool)
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

    def __init__(self, sim, add_mask=None):
        ADD_MASK.__init__(self, sim, add_mask)

        self.list_hist_v_ns = ["Y_e", "theta", "phi", "vel_inf", "entropy", "temperature"]

        self.list_corr_v_ns = ["Y_e theta", "vel_inf theta"]

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

        self.set_skynet_densmap_fpath = Paths.skynet + "densmap.h5"
        self.set_skyent_grid_fpath = Paths.skynet + "grid.h5"

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
            raise NameError("ejecta v_n: {} is not in the list of ejecta v_ns {}"
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
            raise NameError("no method found for computing ejecta arr for det:{} mask:{} v_n:{}"
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
            raise ValueError("Failed to compute ejecta array for "
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

    def __init__(self, sim, add_mask=None):
        EJECTA.__init__(self, sim, add_mask)


        self._list_tab_nuc_v_ns = ["Y_final", "A", "Z"]
        self._list_sol_nuc_v_ns = ["Ysun", "Asun"]

        self.list_nucleo_v_ns = ["sim final", "solar final", "yields", "Ye", "mass"] \
                                + self._list_tab_nuc_v_ns + self._list_sol_nuc_v_ns

        self.matrix_ejecta_nucleo = [[[np.zeros(0,)
                                     for i in range(len(self.list_nucleo_v_ns))]
                                     for j in range(len(self.list_masks))]
                                     for k in range(len(self.list_detectors))]


        self.set_table_solar_r_fpath = Paths.skynet + "solar_r.dat"
        self.set_tabulated_nuc_fpath = Paths.skynet + "tabulated_nucsyn.h5"

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
    def __init__(self, sim, add_mask=None):
        EJECTA_NUCLEO.__init__(self, sim, add_mask)

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

    def __init__(self, sim, add_mask=None):
        EJECTA_NORMED_NUCLEO.__init__(self, sim, add_mask)

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
        if ye_ave > 0.6: raise ValueError("Ye_ave > 0.6 "
                                          "det:{} mask:{} v_n:{}"
                                          .format(det, mask, v_n))
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
            raise NameError("ejecta par v_n: {} (det:{}, mask:{}) does not have a"
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
            raise ValueError("failed to compute ejecta par v_n:{} det:{} mask:{}"
                             .format(v_n, det, mask))

    def get_ejecta_par(self, det, mask, v_n):
        self.check_mask(mask)
        self.check_ej_par_v_n(v_n)
        self.check_det(det)
        self.is_ej_par_computed(det, mask, v_n)
        data = self.matrix_ejecta_pars[self.i_det(det)][self.i_mask(mask)][self.i_ej_par(v_n)]
        return data

""" ====================================| OUTFLOW.ASC -> OUTFLOW.h5 |=============================================== """
# for parallelasiation code
def serial_load_reshape_save(outflow_ascii_file, outdir, grid_object, maxit=-1):
    v_n_to_file_dic = {
        'it': 0,
        'time': 1,
        'fluxdens': 5,
        'w_lorentz': 6,
        'eninf': 7,
        'surface_element': 8,
        'alp': 9,
        'rho': 10,
        'vel[0]': 11,
        'vel[1]': 12,
        'vel[2]': 13,
        'Y_e': 14,
        'entropy': 15,
        'temperature': 16
    }
    data_matrix = {}
    # load ascii
    fdata = np.loadtxt(outflow_ascii_file, usecols=v_n_to_file_dic.values(), unpack=True)
    for i_v_n, v_n in enumerate(v_n_to_file_dic.keys()):
        data = np.array(fdata[i_v_n])
        data_matrix[v_n] = np.array(data)
    iterations = np.sort(np.unique(data_matrix["it"]))
    if maxit > -1.:
        iterations = iterations[iterations <= maxit]
    reshaped_data_matrix = [{} for i in range(len(iterations))]
    # extract the data and reshape to [ntheta, nphi] grid for every iteration
    for i_it, it in enumerate(iterations):
        for i_v_n, v_n in enumerate(v_n_to_file_dic.keys()):
            raw_data = np.array(data_matrix[v_n])
            raw_iterations = np.array(data_matrix["it"], dtype=int)
            tmp = raw_data[np.array(raw_iterations, dtype=int) == int(it)][:grid_object.size()]
            assert len(tmp) > 0
            reshaped_data = grid_object.reshape(tmp)
            reshaped_data_matrix[i_it][v_n] = reshaped_data
    # saving data
    fname = outflow_ascii_file.split('/')[-3]  # output-xxxx
    if os.path.isfile(outdir + fname + ".h5"):
        os.remove(outdir + fname + ".h5")
    dfile = h5py.File(outdir + fname + ".h5", "w")
    for i_it, it in enumerate(iterations):
        gname = "iteration=%d" % it
        dfile.create_group(gname)
        for i_v_n, v_n in enumerate(v_n_to_file_dic.keys()):
            data = reshaped_data_matrix[i_it][v_n]
            dfile[gname].create_dataset(v_n, data=data, dtype=np.float32)
    dfile.close()
    print("Done: {}".format(fname))

""" ============================================| METHODS |=========================================================="""

def outflowed_historgrams(o_outflow, detectors, masks, v_ns, rewrite=False):

    # exit(1)

    # creating histograms
    for det in detectors:
        for mask in masks:
            outdir = Paths.ppr_sims+o_outflow.sim+'/' + "outflow_{}/".format(det) + mask + '/'
            for v_n in v_ns:
                fpath = outdir + "/hist_{}.dat".format(v_n)
                try:
                    if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                        if os.path.isfile(fpath): os.remove(fpath)
                        Printcolor.print_colored_string(
                            ["task:", "d1hist", "det:", "{}".format(det), "mask:", mask, "v_n:", v_n, ":", "computing"],
                            ["blue",   "green", "blue", "green",          "blue", "green","blue","green","", "green"])
                        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                        hist = o_outflow.get_ejecta_arr(det, mask, "hist {}".format(v_n))
                        np.savetxt(outdir + "/hist_{}.dat".format(v_n), X=hist)

                        o_plot = PLOT_MANY_TASKS()
                        o_plot.gen_set["figdir"] = outdir
                        o_plot.gen_set["type"] = "cartesian"
                        o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
                        o_plot.gen_set["figname"] = "hist_{}.png".format(v_n)
                        o_plot.gen_set["sharex"] = False
                        o_plot.gen_set["sharey"] = False
                        o_plot.gen_set["dpi"] = 128
                        o_plot.gen_set["subplots_adjust_h"] = 0.3
                        o_plot.gen_set["subplots_adjust_w"] = 0.0
                        o_plot.set_plot_dics = []

                        plot_dic = {
                            'task': 'hist1d', 'ptype': 'cartesian',
                            'position': (1, 1),
                            'data': hist, 'normalize': True,
                            'v_n_x': v_n, 'v_n_y': None,
                            'color': "black", 'ls': ':', 'lw': 0.8, 'ds': 'steps', 'alpha': 1.0,
                            'xmin':None, 'xamx':None, 'ymin': 1e-4, 'ymax': 1e0,
                            'xlabel': Labels.labels(v_n), 'ylabel': Labels.labels("mass"),
                            'label': None, 'yscale': 'log',
                            'fancyticks': True, 'minorticks': True,
                            'fontsize': 14,
                            'labelsize': 14,
                            'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
                        }
                        # if v_n == "tempterature":


                        plot_dic = Limits.in_dic(plot_dic)

                        o_plot.set_plot_dics.append(plot_dic)
                        o_plot.main()
                        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                    else:
                        Printcolor.print_colored_string(
                            ["task:", "d1hist", "det:", "{}".format(det), "mask:", mask, "v_n:", v_n, ":", "skipping"],
                            ["blue",   "green", "blue", "green",          "blue", "green","blue","green","", "blue"])
                except KeyboardInterrupt:
                    Printcolor.red("Forced termination... done")
                    exit(1)
                except ValueError:
                    Printcolor.print_colored_string(
                        ["task:", "d1hist", "det:", "{}".format(det), "mask:", mask, "v_n:", v_n, ":", "ValueError"],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])
                except:
                    Printcolor.print_colored_string(
                        ["task:", "d1hist", "det:", "{}".format(det), "mask:", mask, "v_n:", v_n, ":", "failed"],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])

def outflowed_correlations(o_outflow, detectors, masks, v_ns, rewrite=False):

    assert len(v_ns) % 2 == 0

    for det in detectors:
        for mask in masks:
            outdir = Paths.ppr_sims+o_outflow.sim+'/' + "outflow_{}/".format(det) + mask + '/'
            for v_n1, v_n2 in zip(v_ns[::2],v_ns[1::2]):
                fpath = outdir + "corr_{}_{}.h5".format(v_n1, v_n2)
                try:
                    if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                        if os.path.isfile(fpath): os.remove(fpath)
                        Printcolor.print_colored_string(
                            ["task:", "d1corr", "det:", "{}".format(det), "mask:", mask, "v_n:", "{}_{}".format(v_n1, v_n2), ":", "computing"],
                            ["blue",   "green", "blue", "green",          "blue", "green","blue","green","", "green"])
                        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                        corr = o_outflow.get_ejecta_arr(det, mask, "corr2d {} {}".format(v_n1, v_n2))
                        y_arr = corr[1:, 0]
                        x_arr = corr[0, 1:]
                        z_arr = corr[1:, 1:]
                        dfile = h5py.File(fpath, "w")
                        dfile.create_dataset(v_n1, data=x_arr)
                        dfile.create_dataset(v_n2, data=y_arr)
                        dfile.create_dataset("mass", data=z_arr)
                        dfile.close()
                        # print(x_arr)
                        # exit(1)

                        # np.savetxt(outdir + "/hist_{}.dat".format(v_n), X=hist)
                        #
                        o_plot = PLOT_MANY_TASKS()
                        o_plot.gen_set["figdir"] = outdir
                        o_plot.gen_set["type"] = "cartesian"
                        o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
                        o_plot.gen_set["figname"] = "corr_{}_{}.png".format(v_n1, v_n2)
                        o_plot.gen_set["sharex"] = False
                        o_plot.gen_set["sharey"] = False
                        o_plot.gen_set["dpi"] = 128
                        o_plot.gen_set["subplots_adjust_h"] = 0.3
                        o_plot.gen_set["subplots_adjust_w"] = 0.0
                        o_plot.set_plot_dics = []
                        #
                        corr_dic2 = {  # relies on the "get_res_corr(self, it, v_n): " method of data object
                            'task': 'corr2d', 'dtype': 'corr', 'ptype': 'cartesian',
                            'data': corr,
                            'position': (1, 1),
                            'v_n_x': v_n1, 'v_n_y': v_n2, 'v_n': 'mass', 'normalize':True,
                            'cbar': {
                                'location': 'right .03 .0', 'label': Labels.labels("mass"),#  'fmt': '%.1f',
                                'labelsize': 14, 'fontsize': 14},
                            'cmap': 'inferno_r', 'set_under': 'white', 'set_over': 'black',
                            'xlabel': Labels.labels(v_n1), 'ylabel': Labels.labels(v_n2),
                            'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None, 'vmin': 1e-4, 'vmax': 1e-1,
                            'xscale': "linear", 'yscale': "linear", 'norm': 'log',
                            'mask_below': None, 'mask_above': None,
                            'title': {},#{"text": o_corr_data.sim.replace('_', '\_'), 'fontsize': 14},
                            'fancyticks': True,
                            'minorticks': True,
                            'sharex': False,  # removes angular citkscitks
                            'sharey': False,
                            'fontsize': 14,
                            'labelsize': 14
                        }
                        corr_dic2 = Limits.in_dic(corr_dic2)

                        corr_dic2["axhline"] = {"y":60, "linestyle":"-", "linewidth":0.5,"color":"black"}
                        corr_dic2["axvline"] = {"x":0.4, "linestyle":"-", "linewidth":0.5, "color":"black"}

                        o_plot.set_plot_dics.append(corr_dic2)
                        #


                        # if v_n1 in ["Y_e", "ye", "Ye"]:
                        #     corr_dic2["xmin"] = 0.
                        #     corr_dic2["xmax"] = 0.5
                        # if v_n1 in ["vel_inf", "vinf", "velinf"]:
                        #     corr_dic2["xmin"] = 0.
                        #     corr_dic2["xmax"] = 1.
                        # if v_n1 in ["vel_inf", "vinf", "velinf"]:
                        #     corr_dic2["xmin"] = 0.
                        #     corr_dic2["xmax"] = 1.
                        # if v_n2 in ["Y_e", "ye", "Ye"]:
                        #     corr_dic2["ymin"] = 0.
                        #     corr_dic2["ymax"] = 0.5
                        # if v_n2 in ["vel_inf", "vinf", "velinf"]:
                        #     corr_dic2["ymin"] = 0.
                        #     corr_dic2["ymax"] = 1.
                        # if v_n2 in ["vel_inf", "vinf", "velinf"]:
                        #     corr_dic2["ymin"] = 0.
                        #     corr_dic2["ymax"] = 1.
                        # if v_n1 in ["theta"]:
                        #     corr_dic2["xmin"] = 0.
                        #     corr_dic2["xmax"] = 90.
                        # if v_n1 in ["phi"]:
                        #     corr_dic2["xmin"] = 0.
                        #     corr_dic2["xmax"] = 360
                        # if v_n2 in ["theta"]:
                        #     corr_dic2["ymin"] = 0.
                        #     corr_dic2["ymax"] = 90.
                        # if v_n2 in ["phi"]:
                        #     corr_dic2["ymin"] = 0.
                        #     corr_dic2["ymax"] = 360
                        #
                        # o_plot.set_plot_dics.append(plot_dic)
                        o_plot.main()
                        # del v_ns[0:2]
                        # del v_ns[0]
                        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                    else:
                        Printcolor.print_colored_string(
                            ["task:", "d1corr", "det:", "{}".format(det), "mask:", mask, "v_n:", "{}_{}".format(v_n1, v_n2), ":", "skipping"],
                            ["blue",   "green", "blue", "green",          "blue", "green","blue","green","", "blue"])
                except KeyboardInterrupt:
                    Printcolor.red("Forced termination... done")
                    exit(1)
                except ValueError:
                    Printcolor.print_colored_string(
                        ["task:", "d1corr", "det:", "{}".format(det), "mask:", mask, "v_n:", "{}_{}".format(v_n1, v_n2),
                         ":", "ValueError"],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])
                except:
                    Printcolor.print_colored_string(
                        ["task:", "d1corr", "det:", "{}".format(det), "mask:", mask, "v_n:", "{}_{}".format(v_n1, v_n2), ":", "failed"],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])

def outflowed_timecorr(o_outflow, detectors, masks, v_ns, rewrite=False):

    # assert len(v_ns) % 2 == 0

    for det in detectors:
        for mask in masks:
            outdir = Paths.ppr_sims+o_outflow.sim+'/' + "outflow_{}/".format(det) + mask + '/'
            for v_n in v_ns:
                fpath = outdir + "timecorr_{}.h5".format(v_n)
                try:
                    if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                        if os.path.isfile(fpath): os.remove(fpath)
                        Printcolor.print_colored_string(
                            ["task:", "timecorr", "det:", "{}".format(det), "mask:", mask, "v_n:", "{}".format(v_n), ":", "computing"],
                            ["blue",   "green", "blue", "green",          "blue", "green","blue","green","", "green"])
                        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                        table = o_outflow.get_ejecta_arr(det, mask, "timecorr {}".format(v_n))
                        table[0, 1:] *= Constants.time_constant
                        timearr = table[0, 1:]
                        yarr = table[1:, 0]
                        zarr =  table[1:, 1:]

                        # print (timearr)

                        # table[0, 1:] *= Constants.time_constant

                        # corr = o_outflow.get_ejecta_arr(det, mask, "timehist {} {}".format(v_n1, v_n2))

                        # y_arr = corr[1:, 0]
                        # x_arr = corr[0, 1:]
                        # z_arr = corr[1:, 1:]
                        dfile = h5py.File(fpath, "w")
                        dfile.create_dataset("time", data=timearr)
                        dfile.create_dataset(v_n, data=yarr)
                        dfile.create_dataset("mass", data=zarr)
                        dfile.close()
                        # print(x_arr)
                        # exit(1)

                        # np.savetxt(outdir + "/hist_{}.dat".format(v_n), X=hist)
                        #
                        o_plot = PLOT_MANY_TASKS()
                        o_plot.gen_set["figdir"] = outdir
                        o_plot.gen_set["type"] = "cartesian"
                        o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
                        o_plot.gen_set["figname"] = "timecorr_{}.png".format(v_n)
                        o_plot.gen_set["sharex"] = False
                        o_plot.gen_set["sharey"] = False
                        o_plot.gen_set["dpi"] = 128
                        o_plot.gen_set["subplots_adjust_h"] = 0.3
                        o_plot.gen_set["subplots_adjust_w"] = 0.0
                        o_plot.set_plot_dics = []
                        #
                        corr_dic2 = {  # relies on the "get_res_corr(self, it, v_n): " method of data object
                            'task': 'corr2d', 'dtype': 'corr', 'ptype': 'cartesian',
                            'data': table,
                            'position': (1, 1),
                            'v_n_x': "time", 'v_n_y': v_n, 'v_n': 'mass', 'normalize':True,
                            'cbar': {
                                'location': 'right .03 .0', 'label': Labels.labels("mass"),#  'fmt': '%.1f',
                                'labelsize': 14, 'fontsize': 14},
                            'cmap': 'inferno',
                            'xlabel': Labels.labels("time"), 'ylabel': Labels.labels(v_n),
                            'xmin': timearr[0], 'xmax': timearr[-1], 'ymin': None, 'ymax': None, 'vmin': 1e-4, 'vmax': 1e-1,
                            'xscale': "linear", 'yscale': "linear", 'norm': 'log',
                            'mask_below': None, 'mask_above': None,
                            'title': {},#{"text": o_corr_data.sim.replace('_', '\_'), 'fontsize': 14},
                            'fancyticks': True,
                            'minorticks': True,
                            'sharex': False,  # removes angular citkscitks
                            'sharey': False,
                            'fontsize': 14,
                            'labelsize': 14
                        }


                        corr_dic2 = Limits.in_dic(corr_dic2)
                        o_plot.set_plot_dics.append(corr_dic2)
                        #


                        # if v_n1 in ["Y_e", "ye", "Ye"]:
                        #     corr_dic2["xmin"] = 0.
                        #     corr_dic2["xmax"] = 0.5
                        # if v_n1 in ["vel_inf", "vinf", "velinf"]:
                        #     corr_dic2["xmin"] = 0.
                        #     corr_dic2["xmax"] = 1.
                        # if v_n1 in ["vel_inf", "vinf", "velinf"]:
                        #     corr_dic2["xmin"] = 0.
                        #     corr_dic2["xmax"] = 1.
                        # if v_n2 in ["Y_e", "ye", "Ye"]:
                        #     corr_dic2["ymin"] = 0.
                        #     corr_dic2["ymax"] = 0.5
                        # if v_n2 in ["vel_inf", "vinf", "velinf"]:
                        #     corr_dic2["ymin"] = 0.
                        #     corr_dic2["ymax"] = 1.
                        # if v_n2 in ["vel_inf", "vinf", "velinf"]:
                        #     corr_dic2["ymin"] = 0.
                        #     corr_dic2["ymax"] = 1.
                        # if v_n1 in ["theta"]:
                        #     corr_dic2["xmin"] = 0.
                        #     corr_dic2["xmax"] = 90.
                        # if v_n1 in ["phi"]:
                        #     corr_dic2["xmin"] = 0.
                        #     corr_dic2["xmax"] = 360
                        # if v_n2 in ["theta"]:
                        #     corr_dic2["ymin"] = 0.
                        #     corr_dic2["ymax"] = 90.
                        # if v_n2 in ["phi"]:
                        #     corr_dic2["ymin"] = 0.
                        #     corr_dic2["ymax"] = 360
                        #
                        # o_plot.set_plot_dics.append(plot_dic)
                        o_plot.main()
                        # del v_ns[0:2]
                        # del v_ns[0]
                        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                    else:
                        Printcolor.print_colored_string(
                            ["task:", "timecorr", "det:", "{}".format(det), "mask:", mask, "v_n:", "{}".format(v_n), ":", "skipping"],
                            ["blue",   "green", "blue", "green",          "blue", "green","blue","green","", "blue"])
                except KeyboardInterrupt:
                    Printcolor.red("Forced termination... done")
                    exit(1)
                except ValueError:
                    Printcolor.print_colored_string(
                        ["task:", "timecorr", "det:", "{}".format(det), "mask:", mask, "v_n:", "{}".format(v_n), ":",
                         "ValueError"],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])
                except:
                    Printcolor.print_colored_string(
                        ["task:", "timecorr", "det:", "{}".format(det), "mask:", mask, "v_n:", "{}".format(v_n), ":", "failed"],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])

def outflowed_totmass(o_outflow, detectors, masks, rewrite=False):

    for det in detectors:
        for mask in masks:
            outdir = Paths.ppr_sims+o_outflow.sim+'/' + "outflow_{}/".format(det) + mask + '/'
            fpath = outdir + "total_flux.dat"
            try:
                if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                    if os.path.isfile(fpath): os.remove(fpath)
                    Printcolor.print_colored_string(
                        ["task:", "mass flux", "det:", "{}".format(det), "mask:", mask, ":", "computing"],
                        ["blue", "green", "blue", "green", "blue", "green",  "", "green"])
                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                    data = o_outflow.get_ejecta_arr(det, mask, "tot_flux")
                    np.savetxt(fpath, X=data, header=" 1:time 2:flux 3:mass")

                    o_plot = PLOT_MANY_TASKS()
                    o_plot.gen_set["figdir"] = outdir
                    o_plot.gen_set["type"] = "cartesian"
                    o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
                    o_plot.gen_set["figname"] = "total_flux.png"
                    o_plot.gen_set["sharex"] = False
                    o_plot.gen_set["sharey"] = False
                    o_plot.gen_set["dpi"] = 128
                    o_plot.gen_set["subplots_adjust_h"] = 0.3
                    o_plot.gen_set["subplots_adjust_w"] = 0.0
                    o_plot.set_plot_dics = []

                    plot_dic = {
                        'task': 'line', 'ptype': 'cartesian',
                        'position': (1, 1),
                        'xarr': data[:,0]*1e3, 'yarr':data[:,2]*1e2,
                        'v_n_x': "time", 'v_n_y': "mass",
                        'color': "black", 'ls': '-', 'lw': 0.8, 'ds': 'default', 'alpha': 1.0,
                        'ymin': 0, 'ymax': 3.0, 'xmin': np.array(data[:,0]*1e3).min(), 'xmax': np.array(data[:,0]*1e3).max(),
                        'xlabel': Labels.labels("time"), 'ylabel': Labels.labels("ejmass"),
                        'label': None, 'yscale': 'linear',
                        'fancyticks': True, 'minorticks': True,
                        'fontsize': 14,
                        'labelsize': 14,
                        'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
                    }

                    o_plot.set_plot_dics.append(plot_dic)
                    o_plot.main()

                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                else:
                    Printcolor.print_colored_string(
                        ["task:", "mass flux", "det:", "{}".format(det), "mask:", mask, ":", "skipping"],
                        ["blue", "green", "blue", "green", "blue", "green",  "", "blue"])
            except KeyboardInterrupt:
                Printcolor.red("Forced termination... done")
                exit(1)
            except ValueError:
                Printcolor.print_colored_string(
                    ["task:", "mass flux", "det:", "{}".format(det), "mask:", mask, ":", "ValueError"],
                    ["blue", "green", "blue", "green", "blue", "green", "", "red"])
            except:
                Printcolor.print_colored_string(
                    ["task:", "mass flux", "det:", "{}".format(det), "mask:", mask, ":", "failed"],
                    ["blue", "green", "blue", "green", "blue", "green", "", "red"])

def outflowed_massaverages(o_outflow, detectors, masks, rewrite=False):

    for det in detectors:
        for mask in masks:
            outdir = Paths.ppr_sims+o_outflow.sim+'/' + "outflow_{}/".format(det) + mask + '/'
            fpath = outdir + "mass_averages.h5"
            try:
                if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                    if os.path.isfile(fpath): os.remove(fpath)
                    Printcolor.print_colored_string(
                        ["task:", "mass averages", "det:", "{}".format(det), "mask:", mask, ":", "computing"],
                        ["blue", "green", "blue", "green", "blue", "green",  "", "green"])
                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

                    v_ns = ["fluxdens", "w_lorentz", "eninf", "surface_element", "rho", "Y_e", "entropy", "temperature"]
                    dfile = h5py.File(fpath, "w")
                    for v_n in v_ns:
                        arr = o_outflow.get_ejecta_arr(0, "geo", "mass_ave Y_e")
                        # print(arr.shape)
                        dfile.create_dataset(v_n, data=arr)
                    # print("end")
                    dfile.create_dataset("theta", data=o_outflow.get_full_arr(0, "theta"))
                    dfile.create_dataset("phi", data=o_outflow.get_full_arr(0, "phi"))
                    dfile.close()

                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                else:
                    Printcolor.print_colored_string(
                        ["task:", "mass averages", "det:", "{}".format(det), "mask:", mask, ":", "skipping"],
                        ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
            except KeyboardInterrupt:
                Printcolor.red("Forced termination... done")
                exit(1)
            except ValueError:
                Printcolor.print_colored_string(
                    ["task:", "mass averages", "det:", "{}".format(det), "mask:", mask, ":", "ValueError"],
                    ["blue", "green", "blue", "green", "blue", "green", "", "red"])
            except:
                Printcolor.print_colored_string(
                    ["task:", "mass averages", "det:", "{}".format(det), "mask:", mask, ":", "failed"],
                    ["blue", "green", "blue", "green", "blue", "green", "", "red"])

def outflowed_ejectatau(o_outflow, detectors, masks, rewrite=False):

    for det in detectors:
        for mask in masks:
            outdir = Paths.ppr_sims+o_outflow.sim+'/' + "outflow_{}/".format(det) + mask + '/'
            fpath = outdir + "ejecta.h5"
            try:
                if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                    if os.path.isfile(fpath): os.remove(fpath)
                    Printcolor.print_colored_string(
                        ["task:", "ejecta tau", "det:", "{}".format(det), "mask:", mask, ":", "computing"],
                        ["blue", "green", "blue", "green", "blue", "green",  "", "green"])
                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

                    arr = o_outflow.get_ejecta_arr(det, mask, "corr3d Y_e entropy tau")
                    ye, entropy, tau = arr[1:, 0, 0], arr[0, 1:, 0], arr[0, 0, 1:]
                    mass = arr[1:,1:,1:]

                    assert ye.min() > 0. and ye.max() < 0.51
                    assert entropy.min() > 0. and entropy.max() < 201.

                    dfile = h5py.File(fpath, "w")
                    dfile.create_dataset("Y_e", data=ye)
                    dfile.create_dataset("entropy", data=entropy)
                    dfile.create_dataset("tau", data=tau)
                    dfile.create_dataset("mass", data=mass)
                    dfile.close()

                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                else:
                    Printcolor.print_colored_string(
                        ["task:", "ejecta tau", "det:", "{}".format(det), "mask:", mask, ":", "skipping"],
                        ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
            except KeyboardInterrupt:
                Printcolor.red("Forced termination... done")
                exit(1)
            except ValueError:
                Printcolor.print_colored_string(
                    ["task:", "ejecta tau", "det:", "{}".format(det), "mask:", mask, ":", "ValueError"],
                    ["blue", "green", "blue", "green", "blue", "green", "", "red"])
            except:
                Printcolor.print_colored_string(
                    ["task:", "ejecta tau", "det:", "{}".format(det), "mask:", mask, ":", "failed"],
                    ["blue", "green", "blue", "green", "blue", "green", "", "red"])

def outflowed_yields(o_outflow, detectors, masks, rewrite=False):

    for det in detectors:
        for mask in masks:
            outdir = Paths.ppr_sims+o_outflow.sim+'/' + "outflow_{}/".format(det) + mask + '/'
            fpath = outdir + "yields.h5"
            try:
                if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                    if os.path.isfile(fpath): os.remove(fpath)
                    Printcolor.print_colored_string(
                        ["task:", "ejecta nucleo", "det:", "{}".format(det), "mask:", mask, ":", "computing"],
                        ["blue", "green", "blue", "green", "blue", "green",  "", "green"])
                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

                    yields = o_outflow.get_nucleo_arr(det, mask, "yields")
                    a = o_outflow.get_nucleo_arr(det, mask, "A")
                    z = o_outflow.get_nucleo_arr(det, mask, "Z")
                    dfile = h5py.File(fpath, "w")
                    dfile.create_dataset("Y_final", data=yields)
                    dfile.create_dataset("A", data=a)
                    dfile.create_dataset("Z", data=z)
                    dfile.close()

                    o_plot = PLOT_MANY_TASKS()
                    o_plot.gen_set["figdir"] = outdir
                    o_plot.gen_set["type"] = "cartesian"
                    o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
                    o_plot.gen_set["figname"] = "yields.png"
                    o_plot.gen_set["sharex"] = False
                    o_plot.gen_set["sharey"] = False
                    o_plot.gen_set["dpi"] = 128
                    o_plot.gen_set["subplots_adjust_h"] = 0.3
                    o_plot.gen_set["subplots_adjust_w"] = 0.0
                    o_plot.set_plot_dics = []

                    sim_nuc = o_outflow.get_normed_sim_abund(0, "geo", "Asol=195")
                    sol_nuc = o_outflow.get_nored_sol_abund("sum")
                    sim_nucleo = {
                        'task': 'line', 'ptype': 'cartesian',
                        'position': (1, 1),
                        'xarr': sim_nuc[:,0], 'yarr': sim_nuc[:,1],
                        'v_n_x': 'A', 'v_n_y': 'abundances',
                        'color': 'black', 'ls': '-', 'lw': 0.8, 'ds': 'steps', 'alpha': 1.0,
                        'ymin': 1e-5, 'ymax': 2e-1, 'xmin': 50, 'xmax': 210,
                        'xlabel': Labels.labels("A"), 'ylabel': Labels.labels("Y_final"),
                        'label': None, 'yscale': 'log',
                        'fancyticks': True, 'minorticks': True,
                        'fontsize': 18,
                        'labelsize': 14,
                    }
                    o_plot.set_plot_dics.append(sim_nucleo)

                    sol_yeilds = {
                        'task': 'line', 'ptype': 'cartesian',
                        'position': (1, 1),
                        'xarr': sol_nuc[:,0], 'yarr': sol_nuc[:,1],
                        'v_n_x': 'Asun', 'v_n_y': 'Ysun',
                        'color': 'gray', 'marker': 'o', 'ms': 4, 'alpha': 0.4,
                        'ymin': 1e-5, 'ymax': 2e-1, 'xmin': 50, 'xmax': 210,
                        'xlabel': Labels.labels("A"), 'ylabel': Labels.labels("Y_final"),
                        'label': 'solar', 'yscale': 'log',
                        'fancyticks': True, 'minorticks': True,
                        'fontsize': 14,
                        'labelsize': 14,
                    }
                    o_plot.set_plot_dics.append(sol_yeilds)
                    o_plot.main()


                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                else:
                    Printcolor.print_colored_string(
                        ["task:", "ejecta nucleo", "det:", "{}".format(det), "mask:", mask, ":", "skipping"],
                        ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
            except KeyboardInterrupt:
                Printcolor.red("Forced termination... done")
                exit(1)
            except ValueError:
                Printcolor.print_colored_string(
                    ["task:", "ejecta nucleo", "det:", "{}".format(det), "mask:", mask, ":", "failed"],
                    ["blue", "green", "blue", "green", "blue", "green", "", "red"])
            except:
                Printcolor.print_colored_string(
                    ["task:", "ejecta nucleo", "det:", "{}".format(det), "mask:", mask, ":", "failed"],
                    ["blue", "green", "blue", "green", "blue", "green", "", "red"])

def outflowed_mkn_profile(o_outflow, detectors, masks, rewrite=False):

    for det in detectors:
        for mask in masks:
            outdir = Paths.ppr_sims+o_outflow.sim+'/' + "outflow_{}/".format(det) + mask + '/'
            fpath = outdir + "ejecta_profile.dat"
            try:
                if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                    if os.path.isfile(fpath): os.remove(fpath)
                    Printcolor.print_colored_string(
                        ["task:", "ejecta nucleo", "det:", "{}".format(det), "mask:", mask, ":", "computing"],
                        ["blue", "green", "blue", "green", "blue", "green",  "", "green"])
                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

                    corr_ye_theta = o_outflow.get_ejecta_arr(det, mask, "corr2d Y_e theta")
                    corr_vel_inf_theta = o_outflow.get_ejecta_arr(det, mask, "corr2d vel_inf theta")

                    # print(corr_vel_inf_theta[0, 1:]) # velocity
                    # print(corr_vel_inf_theta[1:, 0])  # theta
                    assert corr_vel_inf_theta[0, 1:].min() > 0. and corr_vel_inf_theta[0, 1:].max() < 1.
                    assert corr_vel_inf_theta[1:, 0].min() > 0. and corr_vel_inf_theta[1:, 0].max() < 3.14
                    assert corr_ye_theta[0, 1:].min() > 0.035 and corr_ye_theta[0, 1:].max() < 0.55

                    vel_v = np.array(corr_vel_inf_theta[0, 1:])
                    thf = np.array(corr_vel_inf_theta[1:, 0])
                    vel_M = np.array(corr_vel_inf_theta[1:, 1:]).T

                    ye_ye = np.array(corr_ye_theta[0,1:])
                    ye_M = np.array(corr_ye_theta[1:,1:]).T

                    M_of_th = np.sum(vel_M, axis=0)  # Sum of all Mass for all velocities
                    vel_ave = np.sum(vel_v[:, np.newaxis] * vel_M, axis=0) / M_of_th  # average velocity per unit mass ?
                    ye_ave = np.sum(ye_ye[:, np.newaxis] * ye_M, axis=0) / M_of_th  # average Ye per unit mass ?

                    out = np.stack((thf, M_of_th, vel_ave, ye_ave), axis=1)

                    np.savetxt(fpath, out, '%.6f', '  ', '\n', '1:theta 2:M 3:vel 4:ye  ')

                    # -----------------------------------------------------------------

                    o_plot = PLOT_MANY_TASKS()
                    o_plot.gen_set["figdir"] = outdir
                    o_plot.gen_set["type"] = "cartesian"
                    o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
                    o_plot.gen_set["figname"] = "ejecta_profile.png"
                    o_plot.gen_set["sharex"] = False
                    o_plot.gen_set["sharey"] = False
                    o_plot.gen_set["dpi"] = 128
                    o_plot.gen_set["subplots_adjust_h"] = 0.3
                    o_plot.gen_set["subplots_adjust_w"] = 0.0
                    o_plot.set_plot_dics = []


                    mass_dic = {
                        'task': 'line', 'ptype': 'cartesian',
                        'position': (1, 1),
                        'xarr': M_of_th*1e3, 'yarr': 90. - (thf / np.pi * 180.),
                        'v_n_x': 'A', 'v_n_y': 'abundances',
                        'color': 'black', 'ls': '-', 'lw': 0.8, 'ds': 'steps', 'alpha': 1.0,
                        'ymin': 0, 'ymax': 90, 'xmin': 0, 'xmax': 1.,
                        'xlabel': None, 'ylabel': Labels.labels("theta"),
                        'label': Labels.labels("ejmass3"), 'yscale': 'linear',
                        'fancyticks': True, 'minorticks': True,
                        'fontsize': 14,
                        'labelsize': 14,
                    }
                    o_plot.set_plot_dics.append(mass_dic)

                    vel_dic = {
                        'task': 'line', 'ptype': 'cartesian',
                        'position': (1, 1),
                        'xarr': vel_ave, 'yarr': 90. - (thf / np.pi * 180.),
                        'v_n_x': 'A', 'v_n_y': 'abundances',
                        'color': 'red', 'ls': '-', 'lw': 0.8, 'ds': 'steps', 'alpha': 1.0,
                        'ymin': 0, 'ymax': 90, 'xmin': 0, 'xmax': 1.,
                        'xlabel': None, 'ylabel': Labels.labels("theta"),
                        'label': Labels.labels("vel_inf"), 'yscale': 'linear',
                        'fancyticks': True, 'minorticks': True,
                        'fontsize': 14,
                        'labelsize': 14,
                    }
                    o_plot.set_plot_dics.append(vel_dic)

                    ye_dic = {
                        'task': 'line', 'ptype': 'cartesian',
                        'position': (1, 1),
                        'xarr': ye_ave, 'yarr': 90. - (thf / np.pi * 180.),
                        'v_n_x': 'A', 'v_n_y': 'abundances',
                        'color': 'blue', 'ls': '-', 'lw': 0.8, 'ds': 'steps', 'alpha': 1.0,
                        'ymin': 0, 'ymax': 90, 'xmin': 0, 'xmax': 1.,
                        'xlabel': None, 'ylabel': Labels.labels("theta"),
                        'label': Labels.labels("Y_e"), 'yscale': 'linear',
                        'fancyticks': True, 'minorticks': True,
                        'fontsize': 14,
                        'labelsize': 14,
                        'legend':{'loc': 'best', 'ncol':1, 'fontsize':14}
                    }
                    o_plot.set_plot_dics.append(ye_dic)
                    o_plot.main()

                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                else:
                    Printcolor.print_colored_string(
                        ["task:", "ejecta nucleo", "det:", "{}".format(det), "mask:", mask, ":", "skipping"],
                        ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
            except KeyboardInterrupt:
                Printcolor.red("Forced termination... done")
                exit(1)
            except ValueError:
                Printcolor.print_colored_string(
                    ["task:", "ejecta nucleo", "det:", "{}".format(det), "mask:", mask, ":", "failed"],
                    ["blue", "green", "blue", "green", "blue", "green", "", "red"])
            except:
                Printcolor.print_colored_string(
                    ["task:", "ejecta nucleo", "det:", "{}".format(det), "mask:", mask, ":", "failed"],
                    ["blue", "green", "blue", "green", "blue", "green", "", "red"])

def outflowed_summary(o_outflow, detectors, masks, rewrite=False):
    #
    for det in detectors:
        for mask in masks:
            outdir = Paths.ppr_sims + o_outflow.sim + '/' + "outflow_{}/".format(det) + mask + '/'
            #
            outfpath = outdir + "summary.txt"
            try:
                if (os.path.isfile(outfpath) and rewrite) or not os.path.isfile(outfpath):
                    if os.path.isfile(outfpath): os.remove(outfpath)
                    Printcolor.print_colored_string(
                        ["task:", "summary", "det:", "{}".format(det), "mask:", mask, ":", "computing"],
                        ["blue", "green", "blue", "green", "blue", "green", "", "green"])
                    # total flux
                    fpath = outdir + "total_flux.dat"
                    if os.path.isfile(fpath):
                        data = np.array(np.loadtxt(fpath))
                        mass_arr = data[:, 2]
                        time_arr = data[:, 0]
                        total_ej_mass = float(mass_arr[-1])
                        time_end = float(time_arr[-1])# * Constants.time_constant * 1e-3 # s
                    else:
                        total_ej_mass = np.nan
                        time_end = np.nan
                        Printcolor.red("Missing: {}".format(fpath))
                    # Y_e ave
                    v_n = "Y_e"
                    fpath = outdir + "hist_{}.dat".format(v_n)
                    if os.path.isfile(fpath):
                        hist = np.array(np.loadtxt(fpath))
                        ye_ave = o_outflow.compute_ave_ye(np.sum(hist[:,1]), hist)
                    else:
                        ye_ave = np.nan
                        Printcolor.red("Missing: {}".format(fpath))
                    # s_ave
                    v_n = "entropy"
                    fpath = outdir + "hist_{}.dat".format(v_n)
                    if os.path.isfile(fpath):
                        hist = np.array(np.loadtxt(fpath))
                        s_ave = o_outflow.compute_ave_s(np.sum(hist[:, 1]), hist)
                    else:
                        s_ave = np.nan
                        Printcolor.red("Missing: {}".format(fpath))
                    # vel inf
                    v_n = "vel_inf"
                    fpath = outdir + "hist_{}.dat".format(v_n)
                    if os.path.isfile(fpath):
                        hist = np.array(np.loadtxt(fpath))
                        vel_inf_ave = o_outflow.compute_ave_vel_inf(np.sum(hist[:, 1]), hist)
                    else:
                        vel_inf_ave = np.nan
                        Printcolor.red("Missing: {}".format(fpath))
                    # E kin
                    v_n = "vel_inf"
                    fpath = outdir + "hist_{}.dat".format(v_n)
                    if os.path.isfile(fpath):
                        hist = np.array(np.loadtxt(fpath))
                        e_kin_ave = o_outflow.compute_ave_ekin(np.sum(hist[:, 1]), hist)
                    else:
                        e_kin_ave = np.nan
                        Printcolor.red("Missing: {}".format(fpath))
                    # theta
                    v_n = "theta"
                    fpath = outdir + "hist_{}.dat".format(v_n)
                    if os.path.isfile(fpath):
                        hist = np.array(np.loadtxt(fpath))
                        theta_rms = o_outflow.compute_ave_theta_rms(hist)
                    else:
                        theta_rms = np.nan
                        Printcolor.red("Missing: {}".format(fpath))
                    # writing the result
                    with open(outfpath, 'w') as f1:
                        f1.write("# ejecta properties for det:{} mask:{} \n".format(det, mask))
                        f1.write("m_ej      {:.5f} [M_sun]  total ejected mass \n".format(total_ej_mass))
                        f1.write("<Y_e>     {:.3f}            mass-averaged electron fraction \n".format(ye_ave))
                        f1.write("<s>       {:.3f} [k_b]      mass-averaged entropy \n".format(s_ave))
                        f1.write("<v_inf>   {:.3f} [c]        mass-averaged terminal velocity \n".format(vel_inf_ave))
                        f1.write("<E_kin>   {:.3f} [c^2]      mass-averaged terminal kinetical energy \n".format(e_kin_ave))
                        f1.write("theta_rms {:.2f} [degrees]  root mean squared angle of the ejecta (2 planes) \n".format(2. * theta_rms))
                        f1.write("time_end  {:.3f} [s]        end data time \n".format(time_end))
                else:
                    Printcolor.print_colored_string(
                        ["task:", "summary", "det:", "{}".format(det), "mask:", mask, ":", "skipping"],
                        ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
            except KeyboardInterrupt:
                Printcolor.red("Forced termination... done")
                exit(1)
            except ValueError:
                Printcolor.print_colored_string(
                    ["task:", "summary: total flux", "det:", "{}".format(det), "mask:", mask, ":", "ValueError"],
                    ["blue", "green", "blue", "green", "blue", "green", "", "red"])
            except:
                Printcolor.print_colored_string(
                    ["task:", "summary: total flux", "det:", "{}".format(det), "mask:", mask, ":", "failed"],
                    ["blue", "green", "blue", "green", "blue", "green", "", "red"])
            #

if __name__ == '__main__':
    #
    parser = ArgumentParser(description="postprocessing pipeline")
    parser.add_argument("-s", dest="sim", required=True, help="name of the simulation dir")
    parser.add_argument("-t", dest="tasklist", nargs='+', required=False, default=[], help="list of tasks to to")
    parser.add_argument("-d", dest="detectors", nargs='+', required=False, default=[], help="detectors to use (0, 1...)")
    parser.add_argument("-m", dest="masks", nargs='+', required=False, default=[], help="mask names")
    parser.add_argument("-p", dest="num_proc", required=False, default=0, help="number of processes in parallel")
    #
    parser.add_argument("--v_n", dest="v_ns", nargs='+', required=False, default=[], help="variable names to compute")
    #
    parser.add_argument("--usemaxtime", dest="usemaxtime", required=False, default="no",
                        help=" auto/no to use ittime.h5 set value. Or set a float [ms] to overwrite ")
    # parser.add_argument("--maxtime", dest="maxtime", required=False, default=-1., help="Time limiter for 'reshape' task only")
    parser.add_argument("-o", dest="outdir", required=False, default=Paths.ppr_sims, help="path for output dir")
    parser.add_argument("-i", dest="simdir", required=False, default=Paths.gw170817, help="path to simulation dir")
    parser.add_argument("--overwrite", dest="overwrite", required=False, default="no", help="overwrite if exists")
    #
    parser.add_argument("--eos", dest="eosfpath", required=False, default=None, help="Hydro EOS to use")
    #
    # examples
    # python outflowed.py -s SLy4_M13641364_M0_SR -t d1hist -v Y_e vel_inf theta phi entropy -d 0 -m geo --overwrite yes
    # python outflowed.py -s SLy4_M13641364_M0_SR -t d1corr -v Y_e theta vel_inf theta -d 0 -m geo --overwrite yes
    #
    args = parser.parse_args()
    glob_sim = args.sim
    glob_eos = args.eosfpath
    glob_simdir = args.simdir
    glob_outdir = args.outdir
    glob_tasklist = args.tasklist
    glob_overwrite = args.overwrite
    glob_detectors = np.array(args.detectors, dtype=int)
    glob_v_ns = args.v_ns
    glob_masks = args.masks
    glob_nproc = int(args.num_proc)
    glob_usemaxtime = args.usemaxtime
    glob_maxtime = np.nan
    # check given data
    if not glob_eos == None:
        if not os.path.isfile(glob_eos):
            raise NameError("given eos file paths does not exist: {}".format(glob_eos))
    #
    if not os.path.isdir(glob_simdir + glob_sim):
        raise NameError("simulation dir: {} does not exist in rootpath: {} "
                        .format(glob_sim, glob_simdir))
    if len(glob_tasklist) == 0:
        raise NameError("tasklist is empty. Set what tasks to perform with '-t' option")
    else:
        for task in glob_tasklist:
            if task not in __outflowed__["tasklist"]:
                raise NameError("task: {} is not among available ones: {}"
                                .format(task, __outflowed__["tasklist"]))
    if glob_overwrite == "no":
        glob_overwrite = False
    elif glob_overwrite == "yes":
        glob_overwrite = True
    else:
        raise NameError("for '--overwrite' option use 'yes' or 'no'. Given: {}"
                        .format(glob_overwrite))
    glob_outdir_sim = Paths.ppr_sims + glob_sim
    if not os.path.isdir(glob_outdir_sim):
        os.mkdir(glob_outdir_sim)
    if len(glob_detectors) == 0:
        raise NameError("No detectors selected. Set '-d' option to 0, 1, etc")
    # checking if to use maxtime
    if glob_usemaxtime == "no":
        glob_usemaxtime = False
        glob_maxtime = np.nan
    elif glob_usemaxtime == "auto":
        glob_usemaxtime = True
        glob_maxtime = np.nan
    elif re.match(r'^-?\d+(?:\.\d+)?$', glob_usemaxtime):
        glob_maxtime = float(glob_usemaxtime) / 1.e3 # [s]
        glob_usemaxtime = True
    else: raise NameError("for '--usemaxtime' option use 'yes' or 'no' or float. Given: {}"
                          .format(glob_usemaxtime))

    # set globals
    Paths.gw170817 = glob_simdir
    Paths.ppr_sims = glob_outdir

    # do tasks

    if "reshape" in glob_tasklist and glob_nproc == 0:
        assert len(glob_detectors) > 0
        o_os = COMPUTE_OUTFLOW_SURFACE(glob_sim)
        # check if EOS file is correclty set
        if glob_eos != None:
            pass
        else:
            glob_eos = Paths.get_eos_fname_from_curr_dir(glob_sim)
        assert os.path.isfile(glob_eos)
        #
        o_os.eos_fname = glob_eos
        #
        if os.path.isfile(glob_eos) and glob_eos.__contains__(str(glob_sim.split('_')[0])): # is sim EOS is in eosfname
            Printcolor.green("\tSetting EOS file as: {}".format(glob_eos))
            print("Initializing serial reshape...")
        else:# click.confirm('Is the EOS fname correct? {}'.format(glob_eos), default=True):
            Printcolor.yellow("\tSetting EOS file as: {}".format(glob_eos))
            print("Initializing serial reshape...")
        #
        assert os.path.isfile(glob_eos)
        for det in glob_detectors:
            Printcolor.print_colored_string(
                ["Task:", "reshape", "detector:", "{}".format(det), "Executing..."],
                ["blue", "green", "blue", "green", "blue"])
            if not glob_eos == None: o_os.eos_fname = glob_eos
            o_os.save_outflow(det, glob_overwrite)
            Printcolor.print_colored_string(
                ["Task:", "reshape", "detector:", "{}".format(det), "DONE..."],
                ["blue", "green", "blue", "green", "green"])
        exit(0)
    elif "reshape" in glob_tasklist and glob_nproc > 0:
        assert len(glob_detectors) > 0
        # check if EOS file is correclty set
        if not glob_eos == None: pass
        else: glob_eos = Paths.get_eos_fname_from_curr_dir(glob_sim)
        if os.path.isfile(glob_eos) and glob_eos.__contains__(glob_sim.split('_')[0]): # is sim EOS is in eosfname
            Printcolor.green("\tSetting EOS file as: {}".format(glob_eos))
            print("Initializing parallel reshape...")
        else:# click.confirm('Is the EOS fname correct? {}'.format(glob_eos), default=True):
            Printcolor.yellow("\tSetting EOS file as: {}".format(glob_eos))
            print("Initializing parallel reshape...")
        #
        assert os.path.isfile(glob_eos)
        for det in glob_detectors:
            fname = "outflow_surface_det_%d_fluxdens.asc" % det
            fpath = Paths.ppr_sims + glob_sim + '/' + fname.replace(".asc", ".h5")
            if (os.path.isfile(fpath) and glob_overwrite) or not os.path.isfile(fpath):
                if os.path.isfile(fpath): os.remove(fpath)
                Printcolor.print_colored_string(
                    ["Task:", "reshape", "detector:", "{}".format(det), "Executing..."],
                    ["blue", "green", "blue", "green", "green"])
                LOAD_RESHAPE_SAVE_PARALLEL(glob_sim, det, glob_nproc, glob_eos)
            else:
                Printcolor.print_colored_string(
                    ["Task:", "reshape", "detector:", "{}".format(det), "skipping..."],
                    ["blue", "green", "blue", "green", "blue"])
        exit(0)

    # prepare dir tree for other tasks output
    outflowed = EJECTA_PARS(glob_sim)
    assert len(glob_masks) > 0
    assert len(glob_detectors) > 0
    outdir = Paths.ppr_sims + glob_sim + '/'
    if not os.path.isdir(outdir):
        raise IOError("sim directory: {} not found".format(outdir))
    for det in glob_detectors:
        outdir_ = outdir + "outflow_{}/".format(det)
        if not os.path.isdir(outdir_):
            os.mkdir(outdir_)
        for mask in glob_masks:
            outdir__ = outdir_ + mask + '/'
            if not os.path.isdir(outdir__):
                os.mkdir(outdir__)

    # creating main object
    outflowed = EJECTA_PARS(glob_sim)

    # entire pipeline
    if len(glob_tasklist) == 1 and "all" in glob_tasklist:
        # glob_tasklist = __outflowed__["tasklist"]
        v_ns = []
        for v_n in outflowed.list_corr_v_ns:
            v_ns += v_n.split()
        outflowed_correlations(outflowed, glob_detectors, glob_masks, v_ns, glob_overwrite)
        outflowed_timecorr(outflowed, glob_detectors, glob_masks, outflowed.list_hist_v_ns, glob_overwrite)
        outflowed_historgrams(outflowed, glob_detectors, glob_masks, outflowed.list_hist_v_ns, glob_overwrite)
        outflowed_totmass(outflowed, glob_detectors, glob_masks, glob_overwrite)
        outflowed_massaverages(outflowed, glob_detectors, glob_masks, glob_overwrite)
        outflowed_ejectatau(outflowed, glob_detectors, glob_masks, glob_overwrite)
        outflowed_yields(outflowed, glob_detectors, glob_masks, glob_overwrite)
        outflowed_mkn_profile(outflowed, glob_detectors, glob_masks, glob_overwrite)
        outflowed_summary(outflowed, glob_detectors, glob_masks, glob_overwrite)
        exit(0)

    # selected tasks
    for task in glob_tasklist:
        if task == "reshape":
            pass
        elif task == "hist":
            assert len(glob_v_ns) > 0
            outflowed_historgrams(outflowed, glob_detectors, glob_masks, glob_v_ns, glob_overwrite)
        elif task == "timecorr":
            assert len(glob_v_ns) > 0
            outflowed_timecorr(outflowed, glob_detectors, glob_masks, glob_v_ns, glob_overwrite)
        elif task == "corr":
            assert len(glob_v_ns) > 0
            outflowed_correlations(outflowed, glob_detectors, glob_masks, glob_v_ns, glob_overwrite)
        elif task == "totflux":
            outflowed_totmass(outflowed, glob_detectors, glob_masks, glob_overwrite)
        elif task == "massave":
            outflowed_massaverages(outflowed, glob_detectors, glob_masks, glob_overwrite)
        elif task == "ejtau":
            outflowed_ejectatau(outflowed, glob_detectors, glob_masks, glob_overwrite)
        elif task == "yeilds":
            outflowed_yields(outflowed, glob_detectors, glob_masks, glob_overwrite)
        elif task == "mknprof":
            outflowed_mkn_profile(outflowed, glob_detectors, glob_masks, glob_overwrite)
        elif task == "summary":
            outflowed_summary(outflowed, glob_detectors, glob_masks, glob_overwrite)
        else:
            raise NameError("No method fund for task: {}".format(task))

    #
    #
    #
    # for task in glob_tasklist:
    #
    #     if task == "reshape":
    #         o_os = COMPUTE_OUTFLOW_SURFACE(glob_sim)
    #         for det in glob_detectors:
    #             Printcolor.print_colored_string(
    #                 ["Task:", task, "detector:", "{}".format(det), "Executing..."],
    #                 ["blue", "green", "blue", "green", "blue"])
    #             if not glob_eos == None: o_os.eos_fname = glob_eos
    #             o_os.save_outflow(det, glob_overwrite)
    #             Printcolor.print_colored_string(
    #                 ["Task:", task, "detector:", "{}".format(det), "DONE..."],
    #                 ["blue", "green", "blue", "green", "green"])
    #         exit(0)
    #
    #
    # for task in glob_tasklist:
    #     outflowed = EJECTA_PARS(glob_sim)
    #     assert len(glob_masks) > 0
    #     assert len(glob_detectors) > 0
    #     outdir = Paths.ppr_sims + glob_sim + '/'
    #     if not os.path.isdir(outdir):
    #         raise IOError("sim directory: {} not found".format(outdir))
    #     for det in glob_detectors:
    #         outdir_ = outdir + "outflow_{}/".format(det)
    #         if not os.path.isdir(outdir_):
    #             os.mkdir(outdir_)
    #         for mask in glob_masks:
    #             outdir__ = outdir_ + mask + '/'
    #             if not os.path.isdir(outdir__):
    #                 os.mkdir(outdir__)
    #
    #     for det in glob_detectors:
    #         for mask in glob_masks:
    #             outdir = Paths.ppr_sims + glob_sim + '/' + "outflow_{}/".format(det) + mask + '/'
    #             fpath = outdir + "mass_averages.h5"
    #             try:
    #                 if (os.path.isfile(fpath) and glob_overwrite) or not os.path.isfile(fpath):
    #                     if os.path.isfile(fpath): os.remove(fpath)
    #                     Printcolor.print_colored_string(
    #                         ["task:", task, "det:", "{}".format(det), "mask:", mask, ":", "computing"],
    #                         ["blue", "green", "blue", "green", "blue", "green", "", "green"])
    #                     # --- --- --- --- --- --- TASKS with NO v_ns  --- --- --- --- --- --- --- --- ---
    #
    #                     # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    #                 else:
    #                     Printcolor.print_colored_string(
    #                         ["task:", task, "det:", "{}".format(det), "mask:", mask, ":", "skipping"],
    #                         ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
    #             except:
    #                 Printcolor.print_colored_string(
    #                     ["task:", task, "det:", "{}".format(det), "mask:", mask, ":", "failed"],
    #                     ["blue", "green", "blue", "green", "blue", "green", "", "red"])


    # outflow.asc -> outflow.5
    #outflow = COMPUTE_OUTFLOW_SURFACE("SLy4_M13641364_M0_SR")
    #outflow.save_outflow(det=0)


    # o_ej = EJECTA_PARS("SLy4_M13641364_M0_SR")

    ''' -- testing --- '''
    # data = o_ej.get_ejecta_arr(0, "bern", "tot_flux")
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(data[0,:], data[2,:], color='black')
    # plt.savefig(Paths.plots+"test_outflow/" + "test_outflow_mass.png", dpi=128)
    # print("DONE")

    # hist_ye = o_ej.get_ejecta_arr(0, "geo", "hist Y_e")
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(hist_ye[0,:], hist_ye[1,:], color='black', drawstyle="steps")
    # ax.set_yscale("log")
    # ax.set_xlim(0., .5)
    # plt.savefig(Paths.plots+"test_outflow/" + "test_ye_hist.png", dpi=128)
    # print("DONE")

    # hist_theta = o_ej.get_ejecta_arr(0, "geo", "hist theta")
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(hist_theta[0,:], hist_theta[1,:], color='black', drawstyle="steps")
    # ax.set_yscale("log")
    # # ax.set_xlim(0., .5)
    # plt.savefig(Paths.plots+"test_outflow/" + "test_theta_hist.png", dpi=128)
    # print("DONE")

    # hist_theta = o_ej.get_ejecta_arr(0, "geo", "hist vel_inf")
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(hist_theta[0,:], hist_theta[1,:], color='black', drawstyle="steps")
    # ax.set_yscale("log")
    # # ax.set_xlim(0., .5)
    # plt.savefig(Paths.plots+"test_outflow/" + "test_vel_inf_hist.png", dpi=128)
    # print("DONE")

    # corr_ye = o_ej.get_ejecta_arr(0, "geo", "corr2d Y_e theta")
    # print(np.sum(corr_ye[1:,1:]))
    # from matplotlib.colors import LogNorm
    # norm = LogNorm(vmax=1e-1, vmin=1e-7)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # im = ax.pcolormesh(corr_ye[1:, 0], corr_ye[0, 1:], corr_ye[1:,1:].T, norm=norm, cmap="inferno")
    # plt.savefig(Paths.plots+"test_outflow/" + "test_ye_theta_corr.png", dpi=128)
    # print("DONE")

    # corr = o_ej.get_ejecta_arr(0, "geo", "corr3d Y_e entropy tau")
    # print(np.sum(corr))
    # print("DONE")


    # mass_ave = o_ej.get_ejecta_arr(0, "geo", "mass_ave Y_e")
    # print(mass_ave)
    # theta = o_ej.get_full_arr(0, "theta")
    # phi = o_ej.get_full_arr(0, "phi")
    # from matplotlib.colors import LogNorm
    # norm = Normalize(vmin=0., vmax=0.5)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # im = ax.pcolormesh(phi, theta, mass_ave, norm=norm, cmap="inferno")
    # plt.savefig(Paths.plots+"test_outflow/" + "test_ye_mass_ave.png", dpi=128)
    # print("DONE")

    # det, mask = 0, "bern & 98%geo"
    # print("Mej_tot:  {}".format(o_ej.get_ejecta_par(det, mask, "Mej_tot")))
    # print("Ye_ave:   {}".format(o_ej.get_ejecta_par(det, mask, "Ye_ave")))
    # print("s_ave:    {}".format(o_ej.get_ejecta_par(det, mask, "s_ave")))
    # print("vinf_ave: {}".format(o_ej.get_ejecta_par(det, mask, "vel_inf_ave")))
    # print("E_kin_ave:{}".format(o_ej.get_ejecta_par(det, mask, "E_kin_ave")))
    # print("theta_rms:{}".format(o_ej.get_ejecta_par(det, mask, "theta_rms")))

    # sim_nuc = o_ej.get_normed_sim_abund(0, "geo", "Asol=195")
    # sol_nuc = o_ej.get_nored_sol_abund()
    # print(sol_nuc)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(sim_nuc[:,0], sim_nuc[:,1], color='black', drawstyle="steps")
    # ax.plot(sol_nuc[:, 0], sol_nuc[:, 1], color='gray', marker=".", alpha=0.8)
    # ax.set_yscale("log")
    # ax.set_xlim(50., 200.)
    # ax.set_ylim(1e-7, 1e-1)
    # plt.savefig(Paths.plots+"test_outflow/" + "test_nucleo.png", dpi=128)
    # print("DONE")

