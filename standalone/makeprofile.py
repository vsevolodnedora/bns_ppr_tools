#!/usr/bin/env python

"""
    Converts the 3D simulation data from carpet output (for every Node, that for every
    variable name creates a list of .h5 files) into a single profile.h5 for a given iteration using `scidata`.

    options:
        -i : str : path to the simulation dir that contains `output-xxxx` directories with /data/files structure
        -o : str : path to where to dump the output
        --eos : str : file of the Hydro eos to compute additiona quantity, .e.g. enthalpy
        -t : str : either `prof` or `nuprof` -- what type of data to collect, hydro and GR data or neutrino M0 data.
        -m : str : if `times` then the code expect to bi given for what timesteps to extract profiles,
                   if `iterations` then code expects a list of iterations for which to extract profiles
        -it : list of ints : is expected if chosen `-m iterations`. Proivde the list of iteration
        -time : list of floats : is expected if chosen `-m times`. Provide list of timesteps

    NOTE:
        In order to know what timestep corresponds to what output in the simulation, i.e., what data to
        process for user required iteration or a timestep, the code loads additional files where
        there is mapping between iteration and timesteps.
        This file is `"dens.norm1.asc"` that is usually present in simulation.

"""

from __future__ import division
from glob import glob
import numpy as np
import h5py
import argparse
from math import sqrt
import gc

from scidata.utils import locate
import scidata.carpet.hdf5 as h5
from scidata import units as ut

from scipy.interpolate import RegularGridInterpolator
import os

import click
import time

from uutils import Paths


class Names(object):

    # naming conventions, -- content of the .dat.tar

    outdir = "profiles/"

    dattar = {
        'alp'        : 'ADMBASE::alp',
        'betax'      : 'ADMBASE::betax',
        'betay'      : 'ADMBASE::betay',
        'betaz'      : 'ADMBASE::betaz',
        'gxx'        : 'ADMBASE::gxx',
        'gxy'        : 'ADMBASE::gxy',
        'gxz'        : 'ADMBASE::gxz',
        'gyy'        : 'ADMBASE::gyy',
        'gyz'        : 'ADMBASE::gyz',
        'gzz'        : 'ADMBASE::gzz',
        'rho'        : 'HYDROBASE::rho',
        'vel[0]'     : 'HYDROBASE::vel[0]',
        'vel[1]'     : 'HYDROBASE::vel[1]',
        'vel[2]'     : 'HYDROBASE::vel[2]',
        'Y_e'        : 'HYDROBASE::Y_e',
        'temperature': 'HYDROBASE::temperature',
        'w_lorentz'  : 'HYDROBASE::w_lorentz',
        'volform'    : 'THC_CORE::volform',
    }

    nu_dattar = {
        'thc_M0_abs_energy':    'THC_LEAKAGEM0::thc_M0_abs_energy',
        'thc_M0_abs_nua':       'THC_LEAKAGEM0::thc_M0_abs_nua',
        'thc_M0_abs_nue':       'THC_LEAKAGEM0::thc_M0_abs_nue',
        'thc_M0_abs_number':    'THC_LEAKAGEM0::thc_M0_abs_number',
        'thc_M0_eave_nua':      'THC_LEAKAGEM0::thc_M0_eave_nua',
        'thc_M0_eave_nue':      'THC_LEAKAGEM0::thc_M0_eave_nue',
        'thc_M0_eave_nux':      'THC_LEAKAGEM0::thc_M0_eave_nux',
        'thc_M0_E_nua':         'THC_LEAKAGEM0::thc_M0_E_nua',
        'thc_M0_E_nue':         'THC_LEAKAGEM0::thc_M0_E_nue',
        'thc_M0_E_nux':         'THC_LEAKAGEM0::thc_M0_E_nux',
        'thc_M0_flux_fac':      'THC_LEAKAGEM0::thc_M0_flux_fac',
        'thc_M0_ndens_nua':     'THC_LEAKAGEM0::thc_M0_ndens_nua',
        'thc_M0_ndens_nue':     'THC_LEAKAGEM0::thc_M0_ndens_nue',
        'thc_M0_ndens_nux':     'THC_LEAKAGEM0::thc_M0_ndens_nux',
        'thc_M0_N_nua':         'THC_LEAKAGEM0::thc_M0_N_nua',
        'thc_M0_N_nue':         'THC_LEAKAGEM0::thc_M0_N_nue',
        'thc_M0_N_nux':         'THC_LEAKAGEM0::thc_M0_N_nux',

    }

    # naming conventions, -- content of the EOS
    eos = {
        'eps':     "internalEnergy",
        'press':   "pressure",
        'entropy': "entropy"
    }
    # naming conventions, -- change for the output module_profile
    out = {
        'alp':          'lapse',
        'vel[0]':       'velx',
        'vel[1]':       'vely',
        'vel[2]':       'velz',
        'volform':      'vol',
        'temperature':  'temp',
        'Y_e':          'Ye',
        'entropy':      'entr',

        'thc_M0_abs_energy':    'abs_energy',
        'thc_M0_abs_nua':       'abs_nua',
        'thc_M0_abs_nue':       'abs_nue',
        'thc_M0_abs_number':    'abs_number',
        'thc_M0_eave_nua':      'eave_nua',
        'thc_M0_eave_nue':      'eave_nue',
        'thc_M0_eave_nux':      'eave_nux',
        'thc_M0_E_nua':         'E_nua',
        'thc_M0_E_nue':         'E_nue',
        'thc_M0_E_nux':         'E_nux',
        'thc_M0_flux_fac':      'flux_fac',
        'thc_M0_ndens_nua':     'ndens_nua',
        'thc_M0_ndens_nue':     'ndens_nue',
        'thc_M0_ndens_nux':     'ndens_nux',
        'thc_M0_N_nua':         'N_nua',
        'thc_M0_N_nue':         'N_nue',
        'thc_M0_N_nux':         'N_nux',
    }

    # @staticmethod
    # def get_dattar_conent():
    #     if gen_set["content"] == "m0":
    #         return Names.nu_dattar
    #     else:
    #         return Names.dattar

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
        else:
            print("EOS interp. object already exists")

        return self.interp[prop](xi).reshape(rho.shape)


class FileWork:

    def __init__(self):
        pass

    @staticmethod
    def get_number(file_):
        return int(str(file_.split('.file_')[-1]).split('.h5')[0])

    @staticmethod
    def get_filelist(key, inpath, output, clean = True):
        # loading files for a given variable (multiple files from multiple nodes of supercomputer)
        # start_t = time.time()
        # print("\t Locating input files..."),
        fname = key + '.file_*.h5'
        fullpath = inpath + output + '/data/'
        files = locate(fname, root=fullpath, followlinks=True)
        files = sorted(files, key=FileWork.get_number)
        if len(files) == 0:
            raise ValueError("For '{}' in {} found NO files searched:{}"
                             .format(key, fullpath, fname))
        # if len(files) <= 128:
        #     print("For '{}' {}, {}files found. Too many. searched:{}"
        #           .format(key, fullpath, len(files), fname))
        # elif len(files) > 128 and len(files) <= 256:
        #     print('')
        #     print("WARNING! For '{}' {}, {}files found. Too many. searched:{}"
        #           .format(key, fullpath, len(files), fname))
        # elif len(files) > 256 and len(files) <= 512:
        #     print("Warning! For '{}' {}, {} files found. Too many. searched:{}"
        #                      .format(key, fullpath, len(files), fname))
        # elif len(files) > 512:
        #     print("Error! For '{}' {}, {} files found. Too many. searched:{}"
        #                      .format(key, fullpath, len(files), fname))
        # else:
        #     raise ValueError("Error! For '{}' {}, {} files found. Too many. \n searched:{}"
        #                      .format(key, fullpath, len(files), fname))
        # # if len(files) >
        #
        # print(" {}, for {} ".format(len(files), key)),
        # print("done! (%.2f sec)" % (time.time() - start_t))
        return files


class ExtractProfile:

    def __init__(self, it, output, inpath, outpath, eos_fpath, def_v_n ="rho", overwrite=False):
        self.it = it
        self.output = output
        self.inpath = inpath
        self.outpath = outpath
        self.description = None
        self.nlevels = 7 # has to be updated for every file
        self.eos_fpath = eos_fpath
        self.overwrite = overwrite
        # self.gen_set = gen_set

        # extract grid for a given iteration (this is time consuming, so better to do it once)


        outfname = self.outpath + str(self.it) + ".h5"

        if (not os.path.isfile(outfname)) or \
                (os.path.isfile(outfname) and self.overwrite):

            print("Extracting carpet grid for future use")
            self.dset_rho, self.grid = self.get_grid_for_it(def_v_n)
            self.nlevels =  len(self.grid.levels)
            print("\tfound {} ref.levels".format(self.nlevels))
            # for every var. name, load, create dataset, save only for a given iteration
            print("Processing available data")
            for key, val in Names.dattar.iteritems():#Names.dattar.iteritems():
                print("\tkey:'{}' val:'{}' ...".format(key, val))
                self.process_datasets_for_it(key, val)

            # interpoalte and save the EOS quantities (returning 'rho' dataset
            # if not self.gen_set["content"] == "m0":
            print("Processing EOS data...")
            self.default_data = self.inter_save_eos_vars()

            # load all variables iteration_v_n.h5 and combine them into the module_profile.h5
            print("Saving the result as a single file")
            self.load_combine_save()

            print("DONE")
            print("ITERATION {} WAS SAVED".format(str(self.it) + ".h5"))
        else:
            print("File: {} already exists. Skipping.")

    # @staticmethod
    # def get_number(file_):
    #     return int(str(file_.split('.file_')[-1]).split('.h5')[0])
    #
    # def get_filelist(self, key):
    #     # loading files for a given variable (multiple files from multiple nodes of supercomputer)
    #     start_t = time.time()
    #     print("\t Locating input files..."),
    #     fname = key + '.file_*.h5'
    #     fullpath = self.gen_set["inpath"] + self.output + '/data/'
    #     files = locate(fname, root=fullpath, followlinks=True)
    #     files = sorted(files, key=self.get_number)
    #     if len(files) == 0:
    #         raise ValueError("For '{}' in {} found NO files \n searched:{}"
    #                          .format(key, fullpath, fname))
    #     if len(files) > 128:
    #         print("WARNING! For '{}' {}, {}files found. Too many. \n searched:{}"
    #               .format(key, fullpath, len(files), fname))
    #     if len(files) > 256:
    #         raise ValueError("Error! For '{}' {}, {}files found. Too many. \n searched:{}"
    #                          .format(key, fullpath, len(files), fname))
    #
    #     print(" {}, for {} ".format(len(files), key)),
    #     print("done! (%.2f sec)" % (time.time() - start_t))
    #     return files

    def get_grid_for_it(self, key):

        files = FileWork.get_filelist(key, self.inpath, self.output)
        # files = self.get_filelist(key)

        # create a scidata dataset out of those files
        print("\t Parsing the metadata..."),
        start_t = time.time()
        dset = h5.dataset(files)
        if not self.it in dset.iterations:
            raise ValueError("Required it: {} is not in dset.iterations() {}"
                             .format(self.it, dset.iterations))
        print("done! (%.2f sec)" % (time.time() - start_t))

        # Get the grid
        print("\t Reading the grid..."),
        start_t = time.time()
        grid = dset.get_grid(iteration=self.it)
        print("done! (%.2f sec)" % (time.time() - start_t))

        return dset, grid

    def process_datasets_for_it(self, key, val):


        files = FileWork.get_filelist(key, self.inpath, self.output)
        # files = self.get_filelist(key)

        print("\t Parsing the metadata..."),
        start_t = time.time()
        dset = h5.dataset(files)
        print("done! (%.2f sec)" % (time.time() - start_t))

        if not self.it in dset.iterations:
            raise ValueError("it: {} is missing in dset for v_n: {}\n{}"
                             .format(self.it, key, dset.iterations))

        # saving data for iteration
        outfname = self.outpath + str(self.it) + '_' + key + ".h5"
        dfile = h5py.File(outfname, "w")

        if self.description is not None:
            dfile.create_dataset("description", data=np.string_(self.description))

        print("\t Saving {}...".format(outfname)),
        for rl in range(len(self.grid)):
            gname = "reflevel=%d" % rl
            dfile.create_group(gname)
            dfile[gname].attrs.create("delta", self.grid[rl].delta)
            dfile[gname].attrs.create("extent", self.grid[rl].extent())
            dfile[gname].attrs.create("iteration", self.it)
            dfile[gname].attrs.create("reflevel", rl)
            dfile[gname].attrs.create("time", dset.get_time(self.it))

            # found = False
            # for entry in dset.contents.keys():
            #     print("\tNot found {} in {}".format(val, entry.split()))
            #     if val in entry.split() \
            #             and "it={}".format(self.it) in entry.split() \
            #             and 'c=0' in entry.split():
            #         found = True
            #         print("\tFound {} -> {}".format(val, entry))
            #         break

            # if found == False:
            #     raise KeyError("Check for found failed.")
            # self.grid[rl]
            # print("\t\tdset.contents : {}".format(dset.iterations))
            data = dset.get_reflevel_data(self.grid[rl], iteration=int(self.it),
                                          variable=val, timelevel=0, dtype=np.float32)
            try:
                data = dset.get_reflevel_data(self.grid[rl], iteration=int(self.it),
                                          variable=val, timelevel=0, dtype=np.float32)
            except KeyError:
                raise KeyError("Failed to extract data from {} file \n"
                               "Data: rl: {} it: {} v_n: {}\n"
                               ""
                               .format(files[0], rl, self.it, val))
            dfile[gname].create_dataset(key, data=data)
        dfile.close()
        print("done! (%.2f sec)" % (time.time() - start_t))
        dset.close_files()
        gc.collect()

    def interpolate_save_eos_quantity(self, v_n, dset_rho, dset_temp, dset_ye, eostable):

        print("\t Insterpolating/saving {} ...".format(v_n))
        start_t = time.time()

        dfile = h5py.File(self.outpath + str(self.it) + '_' + v_n + ".h5", "w")

        if self.description is not None:
            dfile.create_dataset("description", data=np.string_(self.description))

        for rl in range(self.nlevels):
            print("\t\trl:{}".format(rl))
            print("\t\t extracting rho, temp, ye...")
            group_rho = dset_rho["reflevel={}".format(rl)]
            group_temp = dset_temp["reflevel={}".format(rl)]
            group_ye = dset_ye["reflevel={}".format(rl)]

            arr_rho = np.array(group_rho["rho"])
            arr_temp = np.array(group_temp["temperature"])
            arr_ye = np.array(group_ye["Y_e"])

            # arr_rho_ = units.conv_dens(units.cactus, units.cgs, arr_rho)
            # arr_temp_ = units.conv_temperature(units.cactus, units.cgs, arr_temp)

            # print("\t interpolating eos rl:{}".format(rl))
            print("\t\t evaluating {}".format(Names.eos[v_n]))
            data_arr = eostable.evaluate(Names.eos[v_n], arr_rho, arr_temp, arr_ye)

            print("\t\t converting units for {}".format(Names.eos[v_n]))
            if v_n == 'eps':
                data_arr = ut.conv_spec_energy(ut.cgs, ut.cactus, data_arr)
            elif v_n == 'press':
                data_arr = ut.conv_press(ut.cgs, ut.cactus, data_arr)
            elif v_n == 'entropy':
                data_arr = data_arr
            else:
                raise NameError("EOS quantity: {}".format(v_n))

            gname = "reflevel=%d" % rl
            dfile.create_group(gname)
            dfile[gname].attrs.create("delta", group_rho.attrs["delta"])
            dfile[gname].attrs.create("extent", group_rho.attrs["extent"])
            dfile[gname].attrs.create("iteration", group_rho.attrs["iteration"])
            dfile[gname].attrs.create("reflevel", rl)
            dfile[gname].attrs.create("time", group_rho.attrs["time"])

            dfile[gname].create_dataset(v_n, data=data_arr, dtype=np.float32)

            del arr_rho
            del group_temp
            del group_ye

        dfile.close()
        print("done! (%.2f sec)" % (time.time() - start_t))

        gc.collect()

    def inter_save_eos_vars(self):

        # from scivis import eostable
        o_eos = EOSTable()
        o_eos.read_table(self.eos_fpath)
        data_rho = h5py.File(self.outpath + str(self.it) + '_' + "rho" + ".h5", "r")
        data_temp = h5py.File(self.outpath + str(self.it) + '_' + "temperature" + ".h5", "r")
        data_ye = h5py.File(self.outpath + str(self.it) + '_' + "Y_e" + ".h5", "r")

        for v_n in Names.eos.keys():
            print("\t{}...".format(v_n))
            self.interpolate_save_eos_quantity(v_n, data_rho, data_temp, data_ye, o_eos)

        return data_rho

    @staticmethod
    def merge_two_dicts(x, y):
        z = x.copy()  # start with x's keys and values
        z.update(y)  # modifies z with y's keys and values & returns None
        return z

    def load_combine_save(self):

        all_in_names = self.merge_two_dicts(Names.dattar, Names.eos)

        print("\t Combining data into the module_profile {}.h5...".format(self.it)),
        start_t = time.time()

        dfile = h5py.File(self.outpath + str(self.it) + ".h5", "w")
        if self.description is not None:
            dfile.create_dataset("description", data=np.string_(self.description))
        for rl in range(self.nlevels):
            gname = "reflevel=%d" % rl
            dfile.create_group(gname)
            dfile[gname].attrs.create("delta", self.default_data["reflevel={}".format(rl)].attrs["delta"])
            dfile[gname].attrs.create("extent", self.default_data["reflevel={}".format(rl)].attrs["extent"])
            dfile[gname].attrs.create("iteration", self.default_data["reflevel={}".format(rl)].attrs["iteration"])
            dfile[gname].attrs.create("reflevel", rl)
            dfile[gname].attrs.create("time", self.default_data["reflevel={}".format(rl)].attrs["time"])
            for key, val in all_in_names.iteritems():

                # loading the input h5
                dfile__ = h5py.File(self.outpath + str(self.it) + '_' + key + ".h5")
                data = np.array(dfile__["reflevel={}".format(rl)][key])

                if key in Names.out.keys():
                    key = Names.out[key]

                dfile[gname].create_dataset(key, data=data, dtype=np.float32)

        dfile.close()
        print("done! (%.2f sec)" % (time.time() - start_t))


class ExtractNuProfile:

    def __init__(self, it, output, inpath, outpath, def_nu_v_n ="thc_M0_abs_energy", overwrite=False):

        self.it = it
        self.output = output
        self.inpath = inpath
        self.outpath = outpath
        self.description = None
        self.overwrite = overwrite

        outfname = self.outpath + str(self.it) + "nu.h5"
        if (not os.path.isfile(outfname)) or \
                (os.path.isfile(outfname) and self.overwrite):

            # get reflevel for future use
            default_dset = h5.dataset(FileWork.get_filelist(def_nu_v_n, self.inpath, self.output))

            reflevel = default_dset.get_reflevel()
            nrad = reflevel.n[0]
            ntheta = int(round(sqrt(float(reflevel.n[1] / 2))))
            nphi = 2 * ntheta
            if ntheta * nphi != reflevel.n[1]:
                raise ValueError("The leakage grid is inconsistent")

            for key, val in Names.nu_dattar.iteritems():
                print("\tProcessing key'{}' val:'{}'".format(key, val))
                files = FileWork.get_filelist(key, self.inpath, self.output)
                assert len(files)
                dset = h5.dataset(files)
                data = dset.get_reflevel_data(reflevel=reflevel, iteration=int(self.it),
                                              variable=val, timelevel=0, dtype=np.float32)
                # print(data)
                # output
                fname = self.outpath + str(self.it) + '_' + key + ".h5"
                dfile = h5py.File(fname, "w")
                # dfile.attrs.create("delta", reflevel.delta)
                # dfile.attrs.create("extent", reflevel.extent())
                dfile.attrs.create("iteration", self.it)
                dfile.attrs.create("time", default_dset.get_time(self.it))
                dfile.attrs.create("nrad", nrad)
                dfile.attrs.create("ntheta", ntheta)
                dfile.attrs.create("nphi", nphi)
                print(data.shape)


                # print('delta: {}'.format(reflevel.delta))
                # print('extent:{}'.format(reflevel.extent()))
                # print('iteration:{}'.format(self.it))
                # print('time:{}'.format(dset.get_time(self.it)))
                # print('nrad:{}'.format(nrad))
                # print('ntheta:{}'.format(ntheta))
                # print('nphi:{}'.format(nphi))
                # exit(1)

                dfile.create_dataset(key, data=data)

                dset.close_files()
                dfile.close()
                print("\tFinished key'{}' val:'{}'".format(key, val))
            # print("done! (%.2f sec)" % (time.time() - start_t))
            default_dset.close_files()

            # load extracted data and save as one file:
            all_in_names = Names.nu_dattar
            dfile = h5py.File(outfname, "w")
            for key, val in all_in_names.iteritems():
                print("\tLoading and appending {}".format(key))
                dfile__ = h5py.File(self.outpath + str(self.it) + '_' + key + ".h5")
                data = np.array(dfile__[key])
                if key in Names.out.keys():
                    key = Names.out[key]
                dfile.create_dataset(key, data=data, dtype=np.float32)
                dfile.attrs.create("iteration", self.it)
                dfile.attrs.create("time", default_dset.get_time(self.it))
                dfile.attrs.create("nrad", nrad)
                dfile.attrs.create("ntheta", ntheta)
                dfile.attrs.create("nphi", nphi)
            dfile.close()
            print("\tDONE")
        else:
            print("File: {} already exists. Skipping."
                  .format(outfname))


""" ==================================| independent output-it-time mapping methods |================================="""

def find_nearest_index(array, value):
    ''' Finds index of the value in the array that is the closest to the provided one '''
    idx = (np.abs(array - value)).argmin()
    return idx

def get_output_for_time(time, output_it_time_dic, it_time):

    it_time = np.array(it_time)

    if time > it_time[:,1].max():
        raise ValueError("time {:.3f}s beyond the simulation length ({:.3f}s)".format(time, it_time[:,1].max()))
    if time < it_time[:,1].min():
        raise ValueError("time {:.3f}s is too small, minimum is ({:.3f}s)".format(time, it_time[:,1].min()))

    closest_iteration = it_time[find_nearest_index(it_time[:,1], time), 0]

    output = ''
    for output_dir, it_time in output_it_time_dic.iteritems():
        if closest_iteration in it_time[:, 0]:
            output = output_dir

    if output == '':
        raise ValueError("output was not found")
    print("\t required time:{} found in {} output".format(time, output))

    return output

def load_one_dset_to_get_iter(time_, key, inpath, output):

    files = FileWork.get_filelist(key, inpath, output)
    # files = get_filelist(key, output_dir)

    dset = h5.dataset(files[0]) # fastest way

    dataset_iterations = dset.iterations
    dataset_times = []
    for it in dataset_iterations:
        dataset_times.append(float(dset.get_time(it)))

    print("\t Iterations {}".format(dataset_iterations))
    print("\t Times    "),
    print([("{:.3f}, ".format(i_time)) for i_time in dataset_times])
    # print(' ')

    # selecting the iteration that has the closest time to the required
    idx = find_nearest_index(np.array(dataset_times), time_ / (0.004925794970773136 * 1e-3))
    iteration = dataset_iterations[idx]
    closest_time = dataset_times[idx]

    print("\t it:{} with time:{:.3f} is the closest to required time:{:.3f}"
          .format(iteration, closest_time * 0.004925794970773136 * 1e-3, time_))

    return iteration, closest_time * 0.004925794970773136 * 1e-3

def set_it_output_map(inpath, outpath, it_time_fname = "dens.norm1.asc"):
    """
    Loads set of it_time_files that have '1:it 2:time ...' structure to get a map
    of what output-xxxx contains what iteration (and time)
    """

    output_it_time_dic = {}

    # if not os.path.isdir(gen_set["inpath"] + "profiles/"):
    #     print("creating output dir: {}".format(gen_set["inpath"] + "profiles/"))
    #     os.mkdir(gen_set["inpath"] + "profiles/")
    #
    # it_time_files = glob(gen_set["inpath"] + "output-*" + "/data/" + gen_set["it_map_file"])
    #
    # print('-' * 25 + 'LOADING it list ({})'
    #       .format(gen_set["it_map_file"]) + '-' * 25)
    # print("\t loading from: {}, {} it_time_files".format(gen_set["inpath"], len(it_time_files)))

    assert os.path.isdir(inpath)
    assert os.path.isdir(outpath)

    it_time_files = glob(inpath + "output-*" + "/data/" + it_time_fname)
    assert len(it_time_files) > 0

    print('-' * 25 + 'LOADING it list ({})'
          .format(it_time_fname) + '-' * 25)
    print("\t loading from: {}, {} it_time_files".format(inpath, len(it_time_files)))

    it_time = np.zeros(2)
    for file in it_time_files:
        o_name = file.split('/')
        o_dir = ''
        for o_part in o_name:
            if o_part.__contains__('output-'):
                o_dir = o_part
        if o_dir == '':
            raise NameError("Did not find output-xxxx in {}".format(o_name))
        it_time_i = np.loadtxt(file, usecols=(0, 1))
        it_time_i[:, 1] *= 0.004925794970773136 * 1e-3 # time is seconds
        output_it_time_dic[o_dir] = it_time_i
        it_time = np.vstack((it_time, it_time_i))
    it_time = np.delete(it_time, 0, 0)
    print('outputs:{} its:{} [{}->{}] time:[{}->{:.3f}]'.format(len(it_time_files),
                                                      len(it_time[:, 0]),
                                                      int(it_time[:, 0].min()),
                                                      int(it_time[:, 0].max()),
                                                      float(it_time[:, 1].min()),
                                                      float(it_time[:, 1].max())))
    if len(it_time[:, 0]) != len(set(it_time[:, 0])):
        print("Warning: repetitions found in the loaded iterations")
        iterations = np.unique(it_time[:, 0])
        timestpes = np.unique(it_time[:, 1])
        if not len(iterations) == len(timestpes):
            raise ValueError("Failed attmept to remove repetitions from "
                             "\t it and time lists. Wrong lengths: {} {}"
                             .format(len(iterations), len(timestpes)))
    else:
        print("\t repetitions are not found in loaded it list, continue nurmally")
        iterations = np.unique(it_time[:, 0])
        timestpes = np.unique(it_time[:, 1])

    print('-' * 30 + '------DONE-----' + '-' * 30)

    return output_it_time_dic, np.vstack((iterations, timestpes)).T


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", default='./', required=False, help="path/to/input/data/")
    parser.add_argument("-o", "--output", dest="output", default='same', required=False, help="path/to/output/dir")
    parser.add_argument("-t", "--tasklist", nargs='+', dest="tasklist", default=[], required=True, help="tasklist to perform")
    parser.add_argument("-m", "--mode", dest="mode", default="times", required=True, help="times or iterations")
    parser.add_argument("--it", dest="iterations", nargs='+', default=[], required=False, help="iterations to postprocess")
    parser.add_argument("--time", dest="times", nargs='+', default=[], required=False, help="times to postprocess [ms]")
    parser.add_argument("--eos", dest="eos", required=False, default="auto", help="Hydro EOS file to use")
    args = parser.parse_args()
    #
    glob_input_dir = args.input
    glob_output_dir = args.output
    glob_tasklist = args.tasklist
    glob_mode = args.mode
    glob_iterations =args.iterations
    glob_times = args.times
    glob_eosfpath = args.eos
    #
    assert len(glob_tasklist) > 0
    assert os.path.isdir(glob_input_dir)
    if glob_output_dir == "same": glob_output_dir = glob_input_dir
    assert os.path.isdir(glob_input_dir)
    assert os.path.isdir(glob_output_dir)
    #
    if glob_mode == "iterations":
        glob_iterations = np.array(glob_iterations, dtype=int)
        assert len(glob_iterations) > 0
    elif glob_mode == "times":
        glob_times = np.array(glob_times, dtype=float) / 1e3 # back to [s]
        assert len(glob_times) > 0
    else:
        raise NameError("mode {} is not recognized".format(glob_mode))
    #
    print("Setting output-iteration-time map")
    output_it_time_dic, it_time = set_it_output_map(glob_input_dir, glob_output_dir)
    #
    if not os.path.isdir(glob_output_dir+Names.outdir):
        os.mkdir(glob_output_dir+Names.outdir)
    #
    if glob_mode == "times":
        assert len(glob_times) > 0
        if glob_times.min() < it_time[:,1].min():
            raise ValueError("Given time: {} is below minimum in the data: {}"
                             .format(glob_times.min()*1e3, it_time[:,1].min()*1e3))
        if glob_times.max() > it_time[:,1].max():
            raise ValueError("Given time: {} is above maximum in the data: {}"
                             .format(glob_times.max()*1e3, it_time[:, 1].max()*1e3))

        print("Required times are: {}".format(glob_times))

        outputs = []
        iterations = []
        closest_times = []
        for required_time in glob_times:
            # find in what output the required time is located
            print("Locating the required output for time:{:.3f}".format(required_time))
            output = get_output_for_time(required_time, output_it_time_dic, it_time)
            outputs.append(output)
            iteration, closest_time = load_one_dset_to_get_iter(required_time, "rho", glob_input_dir, output)
            iterations.append(iteration)
            closest_times.append(closest_time)

        print("\n")
        print("    < tmin: {:.3f} tmax: {:.3f} >".format(it_time[:, 1].min(), it_time[:, 1].max()))
        print("    --------------- TASK -------------------")
        print("    t_req  |  t_aval  |  it      |  output ")
        for required_time, closest_time, iteration, output in zip(glob_times, closest_times, iterations, outputs):
            print("    {:.3f}  |  {:.3f}   |  {}  |  {}  ".format(required_time, closest_time, iteration, output))
        print("    --------------- DONE -------------------")
        print("\n")

        print("Mode: {}".format(glob_mode))
        print("Task: {}".format(glob_tasklist))

        # get the EOS file (if hydro is to be computed)
        if "prof" in glob_tasklist:
            if glob_eosfpath == 'auto':
                glob_eosfpath = Paths.get_eos_fname_from_curr_dir(glob_input_dir)
            else:
                glob_eosfpath = glob_eosfpath
            assert os.path.isfile(glob_eosfpath)
            if click.confirm('Is it right EOS: {}'.format(glob_eosfpath.split('/')[-1]), default=True):
                pass
            else:
                exit(1)

        if click.confirm('Do you wish to start?', default=True):
            print("Initializing...")

        # main loop (here, it can be parallelized)
        n = 1
        for output, iteration in zip(outputs, iterations):
            print("it:{} ({}/{})".format(iteration, n, len(iterations)))
            if "prof" in glob_tasklist:
                #
                ExtractProfile(iteration, output, glob_input_dir, glob_output_dir + Names.outdir, glob_eosfpath, "rho",
                               overwrite=False)
                #
                # try:
                #     ExtractProfile(iteration, output, glob_input_dir, glob_output_dir+Names.outdir, glob_eosfpath, "rho", overwrite=False)
                # except KeyboardInterrupt:
                #     exit(0)
                # except:
                #     print("ERROR HYDRO output:{} iteration:{}".format(output, iteration))
            if "nuprof" in glob_tasklist:
                if True:
                    ExtractNuProfile(iteration, output, glob_input_dir, glob_output_dir+Names.outdir, "thc_M0_abs_energy", overwrite=False)
                else:
                    print("ERROR NU output:{} iteration:{}".format(output, iteration))
            n=n+1
    elif glob_mode == "iterations":
        raise AttributeError("this mode is not done yet...")
    else:
        raise NameError("mode (-m {}) is not recognized".format(glob_mode))
    print("    ------------- ALL DONE -----------------     ")
    print("     remove the temporary files ( rm *_* )       ")

