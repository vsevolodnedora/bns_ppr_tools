"""

    Standalone script that converts list of `outflow_surface_det_%d_fluxdens.asc`
    files from all `output-xxxx` of a simulation into a single `outflow_surface_det_%d_fluxdens.h5`
    files that is ~4 time smaller and simpler to use in Python

    Data in that file is mapped from original ascii files onto a spherical grid with nphi, ntheta
    given by `SphericalSurface` class (avoiding double counting data at phi = 2 pi)

    Data is that file is augmented by the data extracted from the Hydro EOS, computed via class `EOSTable`

    Options:
        -i Path to the root dir that contains output-xxxx/data/ with `outflow_surface_det_%d_fluxdens.asc` in each
        -o Output dir, where the resulted `outflow_surface_det_%d_fluxdens.h5` will be saved
        --eos path to the hydro `EOS.h5` file to use
        -d : int : number of the detector (see number % in the file name `outflow_surface_det_%d_fluxdens.asc`)
        -p : int : numer of processors to use for parallel work
        --oberwrite : bool : if True, the existing file will be replaced, else -- sckipped
        --maxtime : float : time in milliseconds, up to which to limit the processing. Usefull if data is corrupt
"""

from __future__ import division
import units as ut # for tmerg
from math import pi, sqrt
import os.path
import h5py
import numpy as np
from glob import glob
import re
from scipy.interpolate import RegularGridInterpolator

import os
from argparse import ArgumentParser

import multiprocessing as mp
from functools import partial
import copy


from uutils import (Printcolor, Constants)

# name of the file that will be searched, collected and processed (for a given detector 'det' )
filename = lambda det: "outflow_surface_det_%d_fluxdens.asc" % det

# data -- columns in the `outflow_surface_det_%d_fluxdens.asc` files
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
v_ns = ['it', 'time', "fluxdens", "w_lorentz", "eninf", "surface_element",
        "alp", "rho", "vel[0]", "vel[1]", "vel[2]", "Y_e", "entropy", "temperature"]
eos_v_ns = ['eps', 'press']

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

# with parallelalisation
class LOAD_RESHAPE_SAVE_PARALLEL:

    # def __init__(self, sim, det, n_proc, eosfname,
    #              indir, outdir, usemaxtime, maxtime_val):
    #
    #     # LOAD_ITTIME.__init__(self, sim, pprdir=outdir)
    #

    def __init__(self, det, flist, n_procs, maxtime, outdir, eosfpath):

        # self.iterations = itlist
        # self.timesteps = timesteplist
        assert len(flist) > 0
        self.det = det
        self.flist = flist
        self.eos_fpath = eosfpath
        self.outdirtmp = outdir + 'tmp/'

        if not os.path.isdir(self.outdirtmp):
            os.mkdir(self.outdirtmp)

        self.grid = self.get_grid()

        pool = mp.Pool(processes=int(n_procs))

        task = partial(serial_load_reshape_save, grid_object=self.grid, outdir=self.outdirtmp, maxtime=maxtime)

        try:
            pool.map(task, self.flist)
        finally:  # To make sure processes are closed in the end, even if errors happen
            pool.close()
            #pool.join()

        tmp_flist = [outdir + 'tmp/' + outfile.split('/')[-3] + ".h5" for outfile in self.flist]
        tmp_flist = sorted(tmp_flist)
        assert len(tmp_flist) == len(self.flist)

        # load reshaped data
        iterations, times, data_matrix = self.load_tmp_files(tmp_flist)

        # concatenate data into [ntimes, ntheta, nphi] arrays
        self.iterations = np.sort(iterations)
        self.times = np.sort(times)

        concatenated_data = {}
        for v_n in v_ns:
            concatenated_data[v_n] = np.stack(([data_matrix[it][v_n] for it in sorted(data_matrix.keys())]))

        # compute EOS quantities
        concatenated_data = self.add_eos_quantities(concatenated_data)

        # save data
        fname = filename(det)
        outfname = outdir + fname.replace(".asc", ".h5")
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
                    for var_name in v_ns:
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
        for v_n in eos_v_ns:
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

        tmp_vns = copy.deepcopy(v_ns)

        tmp_vns.remove("it")
        tmp_vns.remove("time")

        for v_n in (tmp_vns + eos_v_ns):
            outfile.create_dataset(v_n, data=concatenated_data[v_n], dtype=np.float32)
        outfile.close()


def serial_load_reshape_save(outflow_ascii_file, outdir, grid_object, maxtime=-1):

    data_matrix = {}

    # load ascii
    fdata = np.loadtxt(outflow_ascii_file, usecols=v_n_to_file_dic.values(), unpack=True)
    for i_v_n, v_n in enumerate(v_n_to_file_dic.keys()):
        data = np.array(fdata[i_v_n])
        data_matrix[v_n] = np.array(data)
    iterations = np.sort(np.unique(data_matrix["it"]))
    timesteps = np.sort(np.unique(data_matrix["time"]))
    assert len(iterations) == len(timesteps)
    if maxtime > -1:
        iterations = iterations[timesteps <= maxtime]
    # if maxit > -1.:
    #     iterations = iterations[iterations <= maxit]
    reshaped_data_matrix = [{} for i in range(len(iterations))]

    # extract the data and reshape to [ntheta, nphi] grid for every iteration
    for i_it, it in enumerate(iterations):
        raw_iterations = np.array(data_matrix["it"], dtype=int)
        for i_v_n, v_n in enumerate(v_n_to_file_dic.keys()):
            raw_data = np.array(data_matrix[v_n])
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

    if maxtime > -1:
        print("Done: {} [maxtime = {:.2f} output ends at {:.2f} hence {} iterations processed]"
              .format(fname, maxtime, timesteps[-1], len(iterations)))
    else:
        print("Done: {} ({} iterations processed)".format(fname, len(iterations)))

def locate_files(det, indir):

    fname = filename(det)

    if not os.path.isdir(indir):
        raise IOError("Data directory does not exist: {}".format(indir))

    flist = glob(indir + "output-????" + "/data/" + fname)

    if len(flist) == 0:
        raise IOError("No files found. Searching for: {} in {}".format(fname, indir + "output-????" + "/data/"))
    assert len(flist) > 0

    return flist

def do_reshape(
    glob_detectors,
    glob_nproc,
    glob_maxtime,
    glob_outdir,
    glob_indir,
    glob_eos,
    glob_overwrite
):

    assert len(glob_detectors) > 0

    Printcolor.yellow("\tSetting EOS file as: {}".format(glob_eos))
    print("Initializing parallel reshape...")

    assert os.path.isfile(glob_eos)

    for det in glob_detectors:
        fname = filename(det)
        fpath = glob_outdir + fname.replace(".asc", ".h5")
        if (os.path.isfile(fpath) and glob_overwrite) or not os.path.isfile(fpath):
            if os.path.isfile(fpath): os.remove(fpath)

            print("Pool procs = %d" % glob_nproc)
            flist = locate_files(det, glob_indir)
            maxtime = glob_maxtime / Constants.time_constant  # ms -> GEO (Msun)

            Printcolor.print_colored_string(
                ["Task:", "reshape", "detector:", "{}".format(det), "files:","{}".format(len(flist)),
                 "maxtime:", "{:.1f} [Msun]".format(maxtime), "Executing..."],
                ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "green"])

            LOAD_RESHAPE_SAVE_PARALLEL(det, flist, glob_nproc, maxtime, glob_outdir, glob_eos)

        else:
            Printcolor.print_colored_string(
                ["Task:", "reshape", "detector:", "{}".format(det), "skipping..."],
                ["blue", "green", "blue", "green", "blue"])


if __name__ == '__main__':
    #
    parser = ArgumentParser(description="Convert outflow surface dens file into a single .h5 file")

    parser.add_argument("-d", dest="detectors", nargs='+', required=False, default=[],
                        help="detectors to use (0, 1...)")

    parser.add_argument("-p", dest="num_proc", required=False, default=1, help="number of processes in parallel")

    parser.add_argument("--maxtime", dest="maxtime", required=False, default=-1., help="Time limiter for 'reshape' task only")

    parser.add_argument("-o", dest="outdir", required=True, default=None, help="path for output dir")

    parser.add_argument("-i", dest="indir", required=True, default=None, help="path to simulation dir")

    parser.add_argument("--overwrite", dest="overwrite", required=False, default="no", help="overwrite if exists")

    parser.add_argument("--eos", dest="eosfpath", required=True, default=None, help="Hydro EOS to use")
    # examples
    # python old_outflowed.py -s SLy4_M13641364_M0_SR -t d1hist -v Y_e vel_inf theta phi entropy -d 0 -m geo --overwrite yes
    # python old_outflowed.py -s SLy4_M13641364_M0_SR -t d1corr -v Y_e theta vel_inf theta -d 0 -m geo --overwrite yes
    #
    args = parser.parse_args()
    # glob_sim = args.sim
    glob_eos = args.eosfpath
    # glob_skynet = args.skynet
    glob_indir = args.indir
    glob_outdir = args.outdir
    # glob_tasklist = args.tasklist
    glob_overwrite = args.overwrite
    glob_detectors = np.array(args.detectors, dtype=int)
    # glob_v_ns = args.v_ns
    # glob_masks = args.masks
    glob_nproc = int(args.num_proc)
    # glob_usemaxtime = args.usemaxtime
    glob_maxtime = args.maxtime
    # check given data
    if glob_eos is None:
        # if not os.path.isfile(glob_eos):
        raise NameError("Hydro EOS file is not given: {}".format(glob_eos))


    if glob_indir is None:
        #  glob_indir = Paths.gw170817 + glob_sim + '/'
        #if not os.path.isdir(glob_indir):
        raise IOError("Default data dir not found: {}".format(glob_indir))

    if glob_outdir is None:
        # glob_outdir = Paths.ppr_sims + glob_sim + '/'
        # if not os.path.isdir(glob_outdir):
        raise IOError("Default output dir not found: {}".format(glob_outdir))

    if glob_overwrite == "no":
        glob_overwrite = False
    elif glob_overwrite == "yes":
        glob_overwrite = True
    else:
        raise NameError("for '--overwrite' option use 'yes' or 'no'. Given: {}"
                        .format(glob_overwrite))

    if len(glob_detectors) == 0:
        raise NameError("No detectors selected. Set '-d' option to 0, 1, etc")

    # checking if to use maxtime
    if glob_maxtime == -1:
        glob_usemaxtime = False
        glob_maxtime = -1
    elif re.match(r'^-?\d+(?:\.\d+)?$', glob_maxtime):
        glob_maxtime = float(glob_maxtime) # [ms]
        glob_usemaxtime = True
    else:
        raise NameError("To limit the data usage profive --maxtime in [ms] (after simulation start_. Given: {}"
                        .format(glob_maxtime))

    if glob_nproc == 0:
        raise IOError("Set up a number of processors to use. Set e.g., '-p 4' property.")
    else:
        do_reshape(
            glob_detectors,
            glob_nproc,
            glob_maxtime,
            glob_outdir,
            glob_indir,
            glob_eos,
            glob_overwrite
        )
