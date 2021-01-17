#!/usr/bin/env python

"""
    Classes to:

    - SIM_STATUS
        - look into the simulation folder that analyze what data is available:
            For every output-0000 (output-dddd) collect what .asc files are present - hereafter `d1` data
            what .xy.h5 files are available (2D module_slices of the 3D data from the carpet grid) - hereafter `d2` data
            what .dd.h5 files are available (3D data on the carpet grid) - hereafter `d3` data
            For `profiles/3d/` what .h5 files are available. - hereafter `profiles` data
        - saves file `ittime.h5` that for every datatime (d1, d2, d3, prodiles) has 2 lists:
            - available iterations
            - available times
            This allows all other postprocessing tools to always know what data for what time steps and iteration
            is present in the simulation
        - checks of there is a `maxtime.txt` file that should limit to what timestep, the simulation data
            should be postprocessed
            File `maxtime.txt` should contain only one float in [ms] up to what to limit the `ittime.h5` lists

    - LOAD_ITTIME
        - loads the file `ittime.h5`
        - has a set of methods to get for what data type (d1, d2, d3, profiles) what iterations and corresponding
            timesteps are available.

    - PRINT_SIM_STATUS
        - loads file `ittime` via `LOAD_ITTIME` and
            print in a fancy way, for what iteration what data is available

"""

from __future__ import division

from numpy import inf
from glob import glob
import numpy as np
import os.path
import h5py
import csv
import os
import re
from scipy import interpolate
from argparse import ArgumentParser
from uutils import Paths, Printcolor, Lists, Constants, Tools

__all__ = ["SIM_STATUS", "LOAD_ITTIME", "PRINT_SIM_STATUS"]

# inside of the rootdir, where to look for prifles.h5 files
PATH_TO_PROFILES = "profiles/3d/"

# ascii file to use to map iterations to timesteps
FILE_FOR_ITTIME = "dens.norm1.asc"

# ascii file to use to map iterations to timesteps (that outputed by the `Hydro` thorn of the carpet)
FILE_FOR_ITTIME_OUTFLOW = "outflow_det_0.asc"

# list of files to check for presence in the simulation directory (d1 data)
LIST_FILES_D1_DATA = [
    "dens.norm1.asc",
    "dens_unbnd.norm1.asc",
     "H.norm2.asc",
    "mp_Psi4_l2_m2_r400.00.asc",
    "rho.maximum.asc",
    "temperature.maximum.asc",
    "outflow_det_0.asc",
    "outflow_det_1.asc",
    "outflow_det_2.asc",
    "outflow_det_3.asc",
    "outflow_surface_det_0_fluxdens.asc",
    "outflow_surface_det_1_fluxdens.asc"
]


# h5 file to use to map iterations to timesteps for 2D data (module_slices on XY and XZ)
FILE_FOR_ITTIME_D2 = "entropy.xy.h5"

# list of files to check for presence in the simulation directory (d2 data)
LIST_FILES_D2_DATA = [
    "entropy.xy.h5",
    "entropy.xz.h5",
    "dens_unbnd.xy.h5",
    "dens_unbnd.xz.h5",
    "alp.xy.h5",
    "rho.xy.h5",
    "rho.xz.h5",
    "s_phi.xy.h5",
    "s_phi.xz.h5",
    "temperature.xy.h5",
    "temperature.xz.h5",
    "Y_e.xy.h5",
    "Y_e.xz.h5"
]


# h5 file to use to map iterations to timesteps for 3D data (full carpet output, large files fro each node)
FILE_FOR_ITTIME_D3 = "Y_e.file_0.h5"

# list of files to check for presence in the simulation directory (d3 data)
LIST_FILES_D3_DATA = [
    "Y_e.file_0.h5",
    "w_lorentz.file_0.h5",
    "volform.file_0.h5",
    "vel[2].file_0.h5",
    "vel[1].file_0.h5",
    "vel[0].file_0.h5",
    "temperature.file_0.h5",
    "rho.file_0.h5",
    "gzz.file_0.h5",
    "gyz.file_0.h5",
    "gyy.file_0.h5",
    "gxz.file_0.h5",
    "gxy.file_0.h5",
    "gxx.file_0.h5",
    "betaz.file_0.h5",
    "betay.file_0.h5",
    "betax.file_0.h5"
]


# produce ititme.h5
class SIM_STATUS:

    def __init__(self, sim, indir, outdir):
        self.sim = sim
        self.debug = True

        self.simdir = indir #simdir + sim + '/'
        self.resdir = outdir #pprdir + sim + '/'

        self.profdir = self.simdir + PATH_TO_PROFILES

        self.resfile = "ittime.h5"
        #
        self.d1_ittime_file = FILE_FOR_ITTIME
        self.d1_ittime_outflow_file = FILE_FOR_ITTIME_OUTFLOW
        self.d1_flag_files = LIST_FILES_D1_DATA
        self.d2_it_file = FILE_FOR_ITTIME_D2
        self.d2_flag_files = LIST_FILES_D2_DATA
        self.d3_it_file = FILE_FOR_ITTIME_D3
        self.d3_flag_files = LIST_FILES_D3_DATA
        self.output_dics = {}
        self.missing_outputs = []
        #
        self.main()

    def count_profiles(self, fname=''):
        if not os.path.isdir(self.profdir):
            if not self.debug:
                print("Note. No profiels directory found. \nExpected: {}"
                      .format(self.profdir))
            return []
        profiles = glob(self.profdir + '*' + fname)
        if len(profiles) > 0:
            profiles = [profile.split("/")[-1] for profile in profiles]
        #
        return profiles

    def count_tars(self):
        tars = glob(self.simdir + 'output-????.tar')
        tars = [str(tar.split('/')[-1]).split('.tar')[0] for tar in tars]
        return tars

    def count_dattars(self):
        dattars = glob(self.simdir + 'output-????.dat.tar')
        dattars = [str(dattar.split('/')[-1]).split('.dat.tar')[0] for dattar in dattars]
        return dattars

    def count_output_dirs(self):
        dirs = os.listdir(self.simdir)
        output_dirs = []
        for dir_ in dirs:
            dir_ = str(dir_)
            if dir_.__contains__("output-"):
                if re.match("^[-+]?[0-9]+$", dir_.strip("output-")):
                    output_dirs.append(dir_)
        return output_dirs

    def find_max_time(self, endtimefname = "maxtime.txt"):
        #
        if os.path.isfile(self.simdir + endtimefname):
            tend = float(np.loadtxt(self.simdir + endtimefname, unpack=True))
            if tend < 1.:
                pass # [assume s]
            else:
                tend = float(tend) * Constants.time_constant * 1e-3 # [ convert GEO to s]
        else:
            tend = np.nan
        return tend # [s]

    def scan_d1_data(self, output_dir, maxtime=np.nan):

        d1data, itd1, td1 = False, [], []
        if not os.path.isdir(self.simdir + '/' + output_dir + '/data/'):
            return d1data, np.array(itd1, dtype=int), np.array(td1, dtype=float)
        #
        if not os.path.isfile(self.simdir + '/' + output_dir + '/data/' + self.d1_ittime_file):
            print("\t{} does not contain {} -> d1 data is not appended".format(output_dir, self.d1_ittime_file))
            return d1data, np.array(itd1, dtype=int), np.array(td1, dtype=float)
        #
        it_time_i = np.loadtxt(self.simdir + '/' + output_dir + '/data/' + self.d1_ittime_file, usecols=(0, 1))
        itd1 = np.array(it_time_i[:, 0], dtype=int)
        td1 = np.array(it_time_i[:, 1], dtype=float) * Constants.time_constant * 1e-3
        #
        if not np.isnan(maxtime):
            itd1 = itd1[td1 < maxtime]
            td1 = td1[td1 < maxtime]
        #
        return True, np.array(itd1, dtype=int), np.array(td1, dtype=float)

    def scan_d2_data(self, output_dir, d1it, d1times, maxtime=np.nan):

        d2data, itd2, td2 = False, [], []
        if not os.path.isdir(self.simdir + '/' + output_dir + '/data/'):
            return d2data, np.array(itd2, dtype=int), np.array(td2, dtype=float)
        #
        if not os.path.isfile(self.simdir + '/' + output_dir + '/data/' + self.d2_it_file):
            print("\t{} does not contain {} -> d2 data is not appended".format(output_dir, self.d1_ittime_file))
            return d2data, np.array(itd2, dtype=int), np.array(td2, dtype=float)
        #
        iterations = []
        dfile = h5py.File(self.simdir + '/' + output_dir + '/data/' + self.d2_it_file, "r")
        for row in dfile.iterkeys():
            for subrow in row.split():
                if subrow.__contains__("it="):
                    iterations.append(int(subrow.split("it=")[-1]))
        dfile.close()
        if len(iterations) > 0: iterations = np.array(list(sorted(set(iterations))),dtype=int)
        else: iterations = np.array(iterations, dtype=int)
        #
        assert len(d1it) == len(d1times)
        if len(d1times) == 0:
            raise ValueError("len(d1it) = 0 -> cannot compute times for d2it")
        #
        f = interpolate.interp1d(d1it, d1times, kind="slinear",fill_value="extrapolate")
        times = f(iterations)
        if not np.isnan(maxtime):
            iterations = iterations[times<maxtime]
            times = times[times<maxtime]
        #
        return True, np.array(iterations, dtype=int), np.array(times, dtype=float)

    def scan_d3_data(self, output_dir, d1it, d1times, maxtime=np.nan):

        d3data, itd3, td3 = False, [], []
        if not os.path.isdir(self.simdir + '/' + output_dir + '/data/'):
            return d3data, np.array(itd3, dtype=int), np.array(td3, dtype=float)
        #
        if not os.path.isfile(self.simdir + '/' + output_dir + '/data/' + self.d3_it_file):
            # print("\t{} does not contain {} -> d3 data is not appended".format(output_dir, self.d3_it_file))
            return d3data, np.array(itd3, dtype=int), np.array(td3, dtype=float)
        #
        iterations = []
        dfile = h5py.File(self.simdir + '/' + output_dir + '/data/' + self.d3_it_file, "r")
        for row in dfile.iterkeys():
            for subrow in row.split():
                if subrow.__contains__("it="):
                    iterations.append(int(subrow.split("it=")[-1]))
        dfile.close()
        if len(iterations) > 0: iterations = np.array(list(sorted(set(iterations))),dtype=int)
        else: iterations = np.array(iterations, dtype=int)
        #
        assert len(d1it) == len(d1times)
        if len(d1times) == 0:
            raise ValueError("len(d1it) = 0 -> cannot compute times for d3it")
        #
        f = interpolate.interp1d(d1it, d1times, kind="slinear", fill_value="extrapolate")
        times = f(iterations)
        if not np.isnan(maxtime):
            iterations = iterations[times<maxtime]
            times = times[times<maxtime]
        #
        return True, np.array(iterations, dtype=int), np.array(times, dtype=float)

    def scan_outflow_data(self, output_dir, maxtime=np.nan):

        d1data, itd1, td1 = False, [], []
        if not os.path.isdir(self.simdir + '/' + output_dir + '/data/'):
            return d1data, np.array(itd1, dtype=int), np.array(td1, dtype=float)
        #
        if not os.path.isfile(self.simdir + '/' + output_dir + '/data/' + self.d1_ittime_file):
            print("\t{} does not contain {} -> d1 data is not appended".format(output_dir, self.d1_ittime_file))
            return d1data, np.array(itd1, dtype=int), np.array(td1, dtype=float)
        #
        it_time_i = np.loadtxt(self.simdir + '/' + output_dir + '/data/' + self.d1_ittime_file, usecols=(0, 1))
        itd1 = np.array(it_time_i[:, 0], dtype=int)
        td1 = np.array(it_time_i[:, 1], dtype=float) * Constants.time_constant * 1e-3
        #
        if not np.isnan(maxtime):
            itd1 = itd1[td1 < maxtime]
            td1 = td1[td1 < maxtime]
        #
        return True, np.array(itd1, dtype=int), np.array(td1, dtype=float)

    def scan_prof_data(self, profiles, itd1, td1, extenstion=".h5", maxtime=np.nan):

        profdata, itprof, tprof = False, [], []
        if not os.path.isdir(self.profdir):
            return profdata, np.array(itprof, dtype=int), np.array(tprof, dtype=float)
        #
        if len(profiles) == 0:
            return profdata, np.array(itprof, dtype=int), np.array(tprof, dtype=float)
        #
        list_ = [int(profile.split(extenstion)[0]) for profile in profiles if
                 re.match("^[-+]?[0-9]+$", profile.split('/')[-1].split(extenstion)[0])]
        #
        iterations = np.array(np.sort(np.array(list(list_))), dtype=int)
        #
        if len(iterations) != len(profiles):
            if not self.debug:
                print("ValueError. Though {} {} profiles found, {} iterations found."
                      .format(len(profiles), extenstion, len(iterations)))
        #
        if len(iterations) == 0:
            print("\tNote, {} files in {} -> {} selected as profiles"
                  .format(len(profiles), self.profdir, len(iterations)))
        #
        f = interpolate.interp1d(itd1, td1, kind="linear", fill_value="extrapolate")
        times = f(iterations)
        if not np.isnan(maxtime):
            iterations = iterations[times < maxtime]
            times = times[times < maxtime]
        #
        return True, np.array(iterations, dtype=int), np.array(times, dtype=float)

    def save(self, output_dirs, maxtime=np.nan):

        resfile = self.resdir + self.resfile

        if not os.path.isdir(self.resdir):
            os.mkdir(self.resdir)

        if os.path.isfile(resfile):
            os.remove(resfile)
            if not self.debug:
                print("Rewriting the result file {}".format(resfile))

        dfile = h5py.File(resfile, "w")
        for output in output_dirs:
            one_output = self.output_dics[output]
            dfile.create_group(output)
            for key in one_output.keys():
                if not self.debug: print("\twriting key:{} output:{}".format(key, output))
                dfile[output].create_dataset(key, data=one_output[key])

        dfile.create_group("profiles")
        for key in self.output_dics["module_profile"].keys():
            dfile["profiles"].create_dataset(key, data=self.output_dics["module_profile"][key])

        dfile.create_group("nuprofiles")
        for key in self.output_dics["nuprofile"].keys():
            dfile["nuprofiles"].create_dataset(key, data=self.output_dics["nuprofile"][key])

        dfile.create_group("overall")
        for key in self.output_dics["overall"].keys():
            if not self.debug: print("\twriting key:{} overall".format(key))
            dfile["overall"].create_dataset(key, data=self.output_dics["overall"][key])

        dfile.attrs.create("maxtime",data=maxtime)

        dfile.close()

    def main(self):

        # d1data itd2 td2

        #
        output_tars = self.count_tars()
        output_dattars = self.count_dattars()
        output_dirs = self.count_output_dirs()
        parfiles = self.count_profiles(".h5")
        nuparfiles = self.count_profiles("nu.h5")
        #
        maxtime = self.find_max_time()

        #
        for output in output_dirs:
            self.output_dics[output] = {}
            outflowdata, itoutflow, toutflow = self.scan_outflow_data(output)
            d1data, itd1, td1 = self.scan_d1_data(output)
            d2data, itd2, td2 = self.scan_d2_data(output,itd1,td1)
            d3data, itd3, td3 = self.scan_d3_data(output, itd1, td1)
            print("\t{} [d1:{} outflow:{} d2:{} d3:{}] steps".format(output,len(toutflow),len(td1),len(td2),len(td3)))
            self.output_dics[output]["outflowdata"] = outflowdata
            self.output_dics[output]["itoutflow"] = itoutflow
            self.output_dics[output]["toutflow"] = toutflow
            self.output_dics[output]["d1data"] = d1data
            self.output_dics[output]["itd1"] = itd1
            self.output_dics[output]["td1"] = td1
            self.output_dics[output]["d2data"] = d2data
            self.output_dics[output]["itd2"] = itd2
            self.output_dics[output]["td2"] = td2
            self.output_dics[output]["d3data"] = d3data
            self.output_dics[output]["itd3"] = itd3
            self.output_dics[output]["td3"] = td3
        #
        self.output_dics["overall"] = {}
        for key in ["itd1", "td1", "itd2", "td2", "itd3", "td3", "itoutflow", "toutflow"]:
            self.output_dics["overall"][key] = np.concatenate(
                [self.output_dics[output][key] for output in output_dirs])
        #
        profdata, itprof, tprof = self.scan_prof_data(parfiles, self.output_dics["overall"]["itd1"],
                                                      self.output_dics["overall"]["td1"],".h5")
        nuprofdata, itnuprof, tnuprof = self.scan_prof_data(nuparfiles, self.output_dics["overall"]["itd1"],
                                                            self.output_dics["overall"]["td1"],"nu.h5")
        #
        self.output_dics["module_profile"] = {}
        self.output_dics["nuprofile"] = {}
        self.output_dics["module_profile"]["itprof"] = itprof
        self.output_dics["module_profile"]["tprof"] = tprof
        self.output_dics["nuprofile"]["itnuprof"] = itnuprof
        self.output_dics["nuprofile"]["tnuprof"] = tnuprof
        #
        print("\toverall {} outputs, t1d:{} outflow:{} t2d:{} t3d:{} prof:{} nuprof:{}".format(len(output_dirs),
                                                                  len(self.output_dics["overall"]["toutflow"]),
                                                                  len(self.output_dics["overall"]["td1"]),
                                                                  len(self.output_dics["overall"]["td2"]),
                                                                  len(self.output_dics["overall"]["td3"]),
                                                                  len(self.output_dics["module_profile"]["tprof"]),
                                                                  len(self.output_dics["nuprofile"]["tnuprof"])))
        #
        self.save(output_dirs, maxtime)


# get ittime.h5
class LOAD_ITTIME:

    def __init__(self, sim, pprdir):
        #
        self.pprdir = pprdir
        self.sim = sim
        self.debug = False
        self.set_use_1st_found_output_for_it = True
        self.set_limit_ittime_to_maxtime = False
        #
        fpath = pprdir+"ittime.h5" #pprdir + sim + '/' + "ittime.h5"
        if not os.path.isdir(pprdir):
            raise IOError("Directory for postprocessing does not exists.")
        #
        if not os.path.isdir(self.pprdir):
            raise IOError("\tdir for output: {}/ does not exist".format(self.pprdir))
            # os.mkdir(self.pprdir)
        #
        if not os.path.isfile(fpath):
            IOError("Log flie 'ittime.h5' does not exist. Run the module with task '-t update_status' ")
            # SIM_STATUS(sim)
            if not os.path.isfile(fpath):
                raise IOError("ittime.h5 does not exist. AFTER running SIM_STATUS(sim)")
        #
        self.dfile = h5py.File(fpath, "r")
        #
        self.maxtime = self.get_attribute("maxtime")
        #
        #### DEBUG
        # print(self.get_ittime("overall", "d1"))
        # print(self.get_ittime("overall", "d3"))
        # print(self.get_ittime("nuprofiles", "nuprof"))
        #
        # print(self.get_output_for_it(319488, "d1")) -> output-0010 (it < maxtime)
        # print(self.get_output_for_it(543232, "d1")) # -> None ( it > maxtime )
        # print(self.get_nearest_time(3e-2, "d1"))
        # print(self.get_it_for_time(3e-2, "d1"))
        # print(self.get_time_for_it(543232))

    def get_list_outputs(self):
        outputs = []
        for key in self.dfile.keys():
            if key.__contains__("output-"):
                if re.match("^[-+]?[0-9]+$", key.strip("output-")):
                    outputs.append(key)
        return outputs

    def get_attribute(self, v_n):
        try:
            return self.dfile.attrs[v_n]
        except:
            print(self.dfile.attrs.keys())

    def get_ittime(self, output="overall", d1d2d3prof='d1'):
        """
        :param output: "output-0000", or "overall" or "profiles", "nuprofiles"
        :param d1d2d3prof: d1, d2, d3, prof, nuprof
        :return:
        """

        if not output in self.dfile.keys():
            raise KeyError("key:{} not in ittime.h5 keys: \n{}".format(output, self.dfile.keys()))
        # isdata
        if not '{}data'.format(str(d1d2d3prof)) in self.dfile[output].keys():
            isdata = None
        else:
            isdata = bool(self.dfile[output]['{}data'.format(str(d1d2d3prof))])
        # iterations
        if not 'it{}'.format(str(d1d2d3prof)) in self.dfile[output].keys():
            raise KeyError(" 'it{}' is not in ittime[{}] keys ".format(d1d2d3prof, output))
        # times
        if not 't{}'.format(str(d1d2d3prof)) in self.dfile[output].keys():
            raise KeyError(" 't{}' is not in ittime[{}] keys ".format(d1d2d3prof, output))
        #
        iterations = np.array(self.dfile[output]['it{}'.format(str(d1d2d3prof))], dtype=int)
        times = np.array(self.dfile[output]['t{}'.format(str(d1d2d3prof))], dtype=float)
        #
        if self.set_limit_ittime_to_maxtime:
            iterations = iterations[times<self.maxtime]
            times = times[times<self.maxtime]
        #
        return (isdata, iterations, times)

    def get_output_for_it(self,  it, d1d2d3='d1'):

        _, allit, alltimes = self.get_ittime(output="overall", d1d2d3prof=d1d2d3)
        #
        if len(allit) == 0:
            print("\tError data for d1d2d3:{} not available".format(d1d2d3))
            return None
        #
        if it < allit.min():
            print("\tError it: {} < {} - it.min() for d1d2d3:{} ".format(it, allit.min(), d1d2d3))
            return None
        #
        if it > allit.max():
            print("\tError it: {} > {} - it.max() for d1d2d3:{} ".format(it, allit.min(), d1d2d3))
            return None
        #
        selected_outputs = []
        for output in self.get_list_outputs():
            _, iterations, _ = self.get_ittime(output=output, d1d2d3prof=d1d2d3)
            if len(iterations) > 0:
                if it >= iterations.min() and it <= iterations.max():
                    selected_outputs.append(output)
        #
        if len(selected_outputs) == 0:
            raise ValueError("no output is found for it:{} d1d2d3:{}"
                             .format(it, d1d2d3))
        #
        if len(selected_outputs) > 1:
            print("\tWarning {} outputs contain it:{}".format(selected_outputs, it))
        #
        if self.set_use_1st_found_output_for_it:
            return selected_outputs[0]
        else:
            raise ValueError("Set 'self.set_use_1st_found_output_for_it=True' to get"
                             "0th output out of many found")

    def get_nearest_time(self, time__, output="overall", d1d2d3='d1'):

        _, allit, alltimes = self.get_ittime(output=output, d1d2d3prof=d1d2d3)
        #
        if len(allit) == 0:
            print("\tError nearest time is not found for time:{} d1d2d3:{}".format(time__, d1d2d3))
            return np.nan
        #
        if time__ > alltimes.max():
            print("\tWarning time__ {} > {} - alltime.max() returning maximum".format(time__, alltimes.max()))
            return alltimes.max()
        #
        if time__ < alltimes.min():
            print("\tWarning time {} < {} - alltime.min() returning minimum".format(time__, alltimes.min()))
        #
        if time__ in alltimes: return time__
        #
        return alltimes[Tools.find_nearest_index(alltimes, time__)]

    def get_it_for_time(self, time__, output="overall", d1d2d3='d1'):

        _, allit, alltime = self.get_ittime(output=output, d1d2d3prof=d1d2d3)
        #
        if time__ in alltime:
            return int(allit[Tools.find_nearest_index(alltime, time__)])
        #
        time_ = self.get_nearest_time(time__,output=output, d1d2d3=d1d2d3)
        if not np.isnan(time_):
            return int(allit[Tools.find_nearest_index(alltime, time_)])
        else:
            return np.nan

    def get_time_for_it(self, it, output="overall", d1d2d3prof='d1', nan_if_out_of_bound=False):

        it = int(it)

        _, allit, alltime = self.get_ittime(output, d1d2d3prof)
        #
        if len(allit) == 0:
            print("\tError no time found for it:{} as len(allit[output={}][d1d2d3={}]) = {}"
                  .format(it, output, d1d2d3prof, len(allit)))
        #
        if it < allit[0]:
            print("\tWarning it:{} < {} - allit[0] for output:{} d1d2d3:{}".format(it, allit[0], output,d1d2d3prof))
            if nan_if_out_of_bound: return np.nan
        #
        if it > allit[-1]:
            print("\tWarning it:{} > {} - allit[-1] for output:{} d1d2d3:{}".format(it, allit[-1], output,d1d2d3prof))
            if nan_if_out_of_bound: return np.nan
        #
        if it in allit:
            return alltime[Tools.find_nearest_index(allit, it)]
        #
        f = interpolate.interp1d(allit, alltime, kind="linear", fill_value="extrapolate")
        t = f(it)
        return float(t)

    def get_output_for_time(self, time__, d1d2d3='d1'):

        it = self.get_it_for_time(time__, d1d2d3)
        output = self.get_output_for_it(int(it), d1d2d3)

        return output

    # unused methods

    def get_outputs_between_it1_it2(self, it1, it2, d1d2d3="d1"):
        outputs = self.get_list_outputs()
        output1 = self.get_output_for_it(it1, d1d2d3=d1d2d3)
        output2 = self.get_output_for_it(it2, d1d2d3=d1d2d3)
        res_outputs = []
        # res_outputs.append(output1)
        do_append = False
        for output in outputs:
            if output == output1:
                do_append = True
            if output == output2:
                do_append = False
            if do_append:
                res_outputs.append(output)
        res_outputs.append(output2)
        assert output1 in res_outputs
        assert output2 in res_outputs
        return res_outputs

    def get_outputs_between_t1_t2(self, t1, t2, d1d2d3="d1"):
        outputs = self.get_list_outputs()
        output1 = self.get_output_for_time(t1, d1d2d3=d1d2d3)
        output2 = self.get_output_for_time(t2, d1d2d3=d1d2d3)
        res_outputs = []
        # res_outputs.append(output1)
        do_append = False
        for output in outputs:
            if output == output1:
                do_append = True
            if output == output2:
                do_append = False
            if do_append:
                res_outputs.append(output)
        res_outputs.append(output2)
        assert output1 in res_outputs
        assert output2 in res_outputs
        return res_outputs


# show the data for a sim in the terminal
class PRINT_SIM_STATUS(LOAD_ITTIME):

    def __init__(self, sim, indir, pprdir):

        LOAD_ITTIME.__init__(self, sim, pprdir=pprdir)

        self.set_limit_ittime_to_maxtime = False

        self.sim = sim

        self.path_in_data = indir   #indir + sim + '/'
        self.prof_in_data = indir+'profiles/3d/' #indir + sim + '/profiles/3d/'
        self.path_out_data = pprdir + sim + '/'
        self.file_for_gw_time = "/data/dens.norm1.asc"
        self.file_for_ppr_time = "/collated/dens.norm1.asc"

        ''' --- '''

        tstep = 1.
        prec = 0.5

        ''' --- PRINTING ---  '''
        print('=' * 100)
        print("<<< {} >>>".format(sim))

        # assert that the ittime.h5 file is upt to date
        self.print_data_from_parfile(self.path_in_data + 'output-0001/' + 'parfile.par')

        # check if ittime.h5 exists and up to date
        isgood = self.assert_ittime()

        #
        self.print_what_output_tarbal_dattar_present(comma=False)
        print("\tAsserting output contnet:")
        self.print_assert_tarball_content()
        print("\tAsserting data availability: ")

        tstart, tend = self.get_overall_tstart_tend()
        Printcolor.green("\tOverall Data span: {:.1f} to {:.1f} [ms]"
                         .format(tstart - 1, tend - 1))
        if not np.isnan(self.maxtime):
            Printcolor.yellow("\tMaximum time is set: {:.1f} [ms]".format(self.maxtime*1.e3))

        self.print_timemarks_output(start=tstart, stop=tend, tstep=tstep, precision=0.5)
        self.print_timemarks(start=tstart, stop=tend, tstep=tstep, tmark=10., comma=False)
        self.print_ititme_status("overall", d1d2d3prof="d1", start=tstart, stop=tend, tstep=tstep, precision=prec)
        self.print_ititme_status("overall", d1d2d3prof="d2", start=tstart, stop=tend, tstep=tstep, precision=prec)
        self.print_ititme_status("overall", d1d2d3prof="d3", start=tstart, stop=tend, tstep=tstep, precision=prec)
        self.print_ititme_status("profiles", d1d2d3prof="prof", start=tstart, stop=tend, tstep=tstep, precision=prec)
        self.print_ititme_status("nuprofiles", d1d2d3prof="nuprof", start=tstart, stop=tend, tstep=tstep, precision=prec)
        self.print_prof_ittime()
        # self.print_gw_ppr_time(comma=True)
        # self.print_assert_collated_data()
        #
        # self.print_assert_outflowed_data(criterion="_0")
        # self.print_assert_outflowed_data(criterion="_0_b_w")
        # self.print_assert_outflowed_corr_data(criterion="_0")
        # self.print_assert_outflowed_corr_data(criterion="_0_b_w")
        # self.print_assert_gw_data()
        # self.print_assert_mkn_data("_0")
        # self.print_assert_mkn_data("_0_b_w")
        #
        # self.print_assert_d1_plots()
        # self.print_assert_d2_movies()

    def get_tars(self):
        tars = glob(self.path_in_data + 'output-????.tar')
        tars = [str(tar.split('/')[-1]).split('.tar')[0] for tar in tars]
        return tars

    def get_dattars(self):
        dattars = glob(self.path_in_data + 'output-????.dat.tar')
        dattars = [str(dattar.split('/')[-1]).split('.dat.tar')[0] for dattar in dattars]
        return dattars

    @staticmethod
    def get_number(output_dir):
        return int(str(output_dir.split('/')[-1]).split("output-")[-1])

    def get_outputs(self):

        dirs = os.listdir(self.path_in_data)
        output_dirs = []
        for dir_ in dirs:
            dir_ = str(dir_)
            if dir_.__contains__("output-"):
                if re.match("^[-+]?[0-9]+$", dir_.strip("output-")):
                    output_dirs.append(dir_)
        output_dirs.sort(key=self.get_number)

        return output_dirs

    def get_profiles(self, extra=''):

        # list_ = [int(module_profile.split(extenstion)[0]) for module_profile in profiles if
        #          re.match("^[-+]?[0-9]+$", module_profile.split('/')[-1].split(extenstion)[0])]
        if not os.path.isdir(self.prof_in_data):
            return []
        profiles = glob(self.prof_in_data + '*' + extra)
        # print(profiles)
        return profiles

    def get_profile_its(self, extra=".h5"):

        profiles = self.get_profiles(extra)
        #
        list_ = [int(profile.split(extra)[0]) for profile in profiles if
                 re.match("^[-+]?[0-9]+$", profile.split('/')[-1].split(extra)[0])]
        iterations = np.array(np.sort(np.array(list(list_))), dtype=int)
        #
        if len(iterations) == 0:
            return np.array([], dtype=int)
        #
        return iterations

    def assert_ittime(self):

        is_up_to_date = True
        #
        sim_dir_outputs = self.get_outputs()        # from actual sim dir
        ppr_dir_outputs = self.get_list_outputs()   # from_load_ittime
        #
        if sorted(sim_dir_outputs) == sorted(ppr_dir_outputs):
            # get last iteration from simulation
            last_source_output = list(sim_dir_outputs)[-1]
            it_time_i = np.loadtxt(self.path_in_data + last_source_output + '/' + self.file_for_gw_time, usecols=(0, 1))
            sim_it_end = int(it_time_i[-1, 0])
            sim_time_end = float(it_time_i[-1, 1]) * Constants.time_constant
            # get last iteration from simulation
            _, itd1, td1 = self.get_ittime("overall", d1d2d3prof="d1")
            ppr_it_end = itd1[-1]
            ppr_time_end = td1[-1] * 1.e3
            #
            if int(sim_it_end) == int(ppr_it_end):
                Printcolor.green("\tsim time:    {:.2f} = {:.2f} from ppr [ms] ".format(sim_time_end, ppr_time_end))
            else:
                Printcolor.red("\tsim time:    {:.2f} != {:.2f} from ppr [ms]".format(sim_time_end, ppr_time_end))
                is_up_to_date = False

        # profiles
        sim_profiles = glob(self.prof_in_data + "*.h5")
        sim_nu_profiles = glob(self.prof_in_data + "*nu.h5")
        n_sim_prof = int(len(sim_profiles) - len(sim_nu_profiles))
        n_sim_nuprof = len(sim_nu_profiles)
        #
        _, ppr_profs, _ = self.get_ittime("profiles", d1d2d3prof="prof")
        _, ppr_nu_profs, _ = self.get_ittime("nuprofiles", d1d2d3prof="nuprof")
        if n_sim_prof == len(ppr_profs):
            Printcolor.green("\tsim profs:   {:d} = {:d} ittme.h5 profs".format(n_sim_prof, len(ppr_profs)))
        else:
            Printcolor.red("\tsim profs:  {:d} != {:d} ittme.h5 profs".format(n_sim_prof, len(ppr_profs)))
            is_up_to_date = False
        #
        if n_sim_nuprof == len(ppr_nu_profs):
            Printcolor.green("\tsim nuprofs: {:d} = {:d} ittme.h5 profs".format(n_sim_nuprof, len(ppr_nu_profs)))
        else:
            Printcolor.red("\tsim nuprofs: {:d} != {:d} ittme.h5 profs".format(n_sim_nuprof, len(ppr_nu_profs)))
            is_up_to_date = False
        #
        if is_up_to_date:
            Printcolor.green("\t[ ----------------------- ]")
            Printcolor.green("\t[ ittime.h5 is up to date ]")
            Printcolor.green("\t[ ----------------------- ]")
        else:
            Printcolor.red("\t[ --------------------------- ]")
            Printcolor.red("\t[ ittime.h5 is NOT up to date ]")
            Printcolor.red("\t[ --------------------------- ]")

        return is_up_to_date

    def get_overall_tstart_tend(self):

        t1, t2 = [], []
        _, itd1, td1 = self.get_ittime("overall", d1d2d3prof="d1")
        _, itd2, td2 = self.get_ittime("overall", d1d2d3prof="d2")
        _, itd3, td3 = self.get_ittime("overall", d1d2d3prof="d3")
        _, itprof, tprof = self.get_ittime("profiles", d1d2d3prof="prof")
        #
        if len(td1) > 0:
            assert not np.isnan(td1[0]) and not np.isnan(td1[-1])
            t1.append(td1[0])
            t2.append(td1[-1])
        if len(td2) > 0:
            assert not np.isnan(td2[0]) and not np.isnan(td2[-1])
            t1.append(td2[0])
            t2.append(td2[-1])
        if len(td3) > 0:
            assert not np.isnan(td3[0]) and not np.isnan(td3[-1])
            t1.append(td3[0])
            t2.append(td3[-1])
        if len(tprof) > 0:
            assert not np.isnan(tprof[0]) and not np.isnan(tprof[-1])
            t1.append(tprof[0])
            t2.append(tprof[-1])
        #
        return np.array(t1).min() * 1e3 + 1, np.array(t2).max() * 1e3 + 1

    ''' --- '''

    def print_what_output_tarbal_dattar_present(self, comma=False):

        n_outputs = len(self.get_outputs())
        n_tars = len(self.get_tars())
        n_datatars = len(self.get_dattars())
        n_nuprofs = len(self.get_profiles("nu.h5"))
        n_profs = int(len(self.get_profiles("h5"))-n_nuprofs)

        Printcolor.blue("\toutputs: ",comma=True)
        if n_outputs == 0:
            Printcolor.red(str(n_outputs), comma=True)
        else:
            Printcolor.green(str(n_outputs), comma=True)

        Printcolor.blue("\ttars: ",comma=True)
        if n_tars == 0:
            Printcolor.green(str(n_tars), comma=True)
        else:
            Printcolor.red(str(n_tars), comma=True)

        Printcolor.blue("\tdattars: ",comma=True)
        if n_datatars == 0:
            Printcolor.green(str(n_datatars), comma=True)
        else:
            Printcolor.red(str(n_datatars), comma=True)

        Printcolor.blue("\tprofs: ",comma=True)
        if n_profs == 0:
            Printcolor.red(str(n_profs), comma=True)
        else:
            Printcolor.green(str(n_profs), comma=True)

        Printcolor.blue("\tnuprofs: ",comma=True)
        if n_nuprofs == 0:
            Printcolor.red(str(n_nuprofs), comma=True)
        else:
            Printcolor.green(str(n_nuprofs), comma=True)

        if comma:
            print(' '),
        else:
            print(' ')

    ''' --- '''

    def print_data_from_parfile(self, fpath_parfile):

        parlist_to_print = [
            "PizzaIDBase::eos_file",
            "LoreneID::lorene_bns_file",
            "EOS_Thermal_Table3d::eos_filename",
            "WeakRates::table_filename"

        ]

        if not os.path.isfile(fpath_parfile):
            Printcolor.red("\tParfile is absent")
        else:
            flines = open(fpath_parfile, "r").readlines()
            for fname in parlist_to_print:
                found = False
                for fline in flines:
                    if fline.__contains__(fname):
                        Printcolor.blue("\t{}".format(fline), comma=True)
                        found = True
                if not found:
                    Printcolor.red("\t{} not found in parfile".format(fname))

    @staticmethod
    def print_assert_content(dir, expected_files, marker1='.', marker2='x'):
        """
        If all files are found:  return "full", []
        else:                    return "partial", [missing files]
        or  :                    return "empty",   [missing files]
        :param expected_files:
        :param dir:
        :return:
        """
        status = "full"
        missing_files = []

        assert os.path.isdir(dir)
        print('['),
        for file_ in expected_files:
            if os.path.isfile(dir + file_):
                Printcolor.green(marker1, comma=True)
            else:
                Printcolor.red(marker2, comma=True)
                status = "partial"
                missing_files.append(file_)
        print(']'),
        if len(missing_files) == len(expected_files):
            status = "empty"

        return status, missing_files

    def print_assert_data_status(self, name, path, flist, comma=True):

        Printcolor.blue("\t{}: ".format(name), comma=True)
        # flist = copy.deepcopy(LOAD_FILES.list_collated_files)

        status, missing = self.print_assert_content(path, flist)

        if status == "full":
            Printcolor.green(" complete", comma=True)
        elif status == "partial":
            Printcolor.yellow(" partial, ({}) missing".format(len(missing)), comma=True)
        else:
            Printcolor.red(" absent", comma=True)

        if comma:
            print(' '),
        else:
            print(' ')

        return status, missing

    def print_assert_tarball_content(self, comma=False):

        outputs = self.get_list_outputs()
        for output in outputs:
            try:
                _, itd1, td1 = self.get_ittime(output=output, d1d2d3prof="d1")

                output = self.path_in_data + output
                assert os.path.isdir(output)
                output_n = int(str(output.split('/')[-1]).split('output-')[-1])
                n_files = len([name for name in os.listdir(output + '/data/')])
                Printcolor.blue("\toutput: {0:03d}".format(output_n), comma=True)
                Printcolor.blue("[", comma=True)
                Printcolor.green("{:.1f}".format(td1[0]*1e3), comma=True)
                # Printcolor.blue(",", comma=True)
                Printcolor.green("{:.1f}".format(td1[-1]*1e3), comma=True)
                Printcolor.blue("ms ]", comma=True)
                # print('('),

                if td1[0]*1e3 < 10. and td1[-1]*1e3 < 10.:
                    print(' '),
                elif td1[0]*1e3 < 10. or td1[-1]*1e3 < 10.:
                    print(''),
                else:
                    pass

                if n_files == 259 or n_files == 258:
                    Printcolor.green("{0:05d} files".format(n_files), comma=True)
                else:
                    Printcolor.yellow("{0:05d} files".format(n_files), comma=True)
                # print(')'),
                status, missing = self.print_assert_content(output + '/data/', Lists.tarball)
                if status == "full":
                    Printcolor.green(" complete", comma=True)
                elif status == "partial":
                    Printcolor.yellow(" partial, ({}) missing".format(missing), comma=True)
                else:
                    Printcolor.red(" absent", comma=True)
                print('')
            except KeyError:
                output_n = int(str(output.split('/')[-1]).split('output-')[-1])
                Printcolor.blue("\toutput: {0:03d}".format(output_n), comma=True)
                Printcolor.red("[", comma=True)
                Printcolor.red(" absent ", comma=True)
                Printcolor.red(" ]", comma=False)
            except IndexError:

                Printcolor.red("[", comma=True)
                Printcolor.red(" empty data ", comma=True)
                Printcolor.red(" ]", comma=False)
        if comma:
            print(' '),
        else:
            print(' ')

    def print_timemarks(self, start=0., stop=30., tstep=1., tmark=10., comma=False):

        trange = np.arange(start=start, stop=stop, step=tstep)

        Printcolor.blue("\tTimesteps {}ms   ".format(tmark, tstep), comma=True)
        print('['),
        for t in trange:
            if t % tmark == 0:
                print("{:d}".format(int(t / tmark))),
            else:
                print(' '),
        print(']'),
        if comma:
            print(' '),
        else:
            print(' ')

    def print_timemarks_output(self, start=0., stop=30., tstep=1., comma=False, precision=0.5):

        tstart = []
        tend = []
        dic_outend = {}
        for output in self.get_outputs():
            _, itd1, td1 = self.get_ittime(output=output, d1d2d3prof="d1")
            if len(itd1) > 0:
                tstart.append(td1[0] * 1e3)
                tend.append(td1[-1] * 1e3)
                dic_outend["%.3f" % (td1[-1] * 1e3)] = output.split("output-")[-1]

        for digit, letter, in zip(range(4), ['o', 'u', 't', '-']):
            print("\t         {}         ".format(letter)),
            # Printcolor.blue("\tOutputs end [ms] ", comma=True)
            # print(start, stop, tstep)
            trange = np.arange(start=start, stop=stop, step=tstep)
            print('['),
            for t in trange:
                tnear = tend[Tools.find_nearest_index(tend, t)]
                if abs(tnear - t) < precision:  # (tnear - t) >= 0
                    output = dic_outend["%.3f" % tnear]
                    numbers = []
                    for i in [0, 1, 2, 3]:
                        numbers.append(str(output[i]))

                    if digit != 3 and int(output[digit]) == 0:
                        print(' '),
                        # Printcolor.blue(output[digit], comma=True)
                    else:
                        Printcolor.blue(output[digit], comma=True)

                    # for i in range(len(numbers)-1):
                    #     if numbers[i] == "0" and numbers[i+1] != "0":
                    #         Printcolor.blue(numbers[i], comma=True)
                    #     else:
                    #         Printcolor.yellow(numbers[i], comma=True)
                    # print("%.2f"%tnear, t)
                else:
                    print(' '),
            print(']')

    def print_ititme_status(self, output, d1d2d3prof, start=0., stop=30., tstep=1., precision=0.5):

        _, itd1, td = self.get_ittime(output, d1d2d3prof=d1d2d3prof)
        td = td * 1e3  # ms
        # print(td); exit(1)
        # trange = np.arange(start=td[0], stop=td[-1], step=tstep)
        trange = np.arange(start=start, stop=stop, step=tstep)

        _name_ = '  '
        if d1d2d3prof == 'd1':
            _name_ = "D1    "
        elif d1d2d3prof == "d2":
            _name_ = "D2    "
        elif d1d2d3prof == "d3":
            _name_ = "D3    "
        elif d1d2d3prof == "prof":
            _name_ = "prof  "
        elif d1d2d3prof == "nuprof":
            _name_ = "nuprof"

        # print(td)

        if len(td) > 0:
            Printcolor.blue("\tTime {} [{}ms]".format(_name_, tstep), comma=True)
            print('['),
            for t in trange:
                tnear = td[Tools.find_nearest_index(td, t)]
                if abs(tnear - t) < precision:  # (tnear - t) >= 0
                    if not np.isnan(self.maxtime) and tnear > self.maxtime*1.e3: Printcolor.yellow('x', comma=True)
                    else: Printcolor.green('.', comma=True)
                    # print("%.2f"%tnear, t)
                else:
                    print(' '),
                    # print("%.2f"%tnear, t)

            print(']'),
            Printcolor.green("{:.1f}ms".format(td[-1]), comma=False)
        else:
            Printcolor.red("\tTime {} No Data".format(_name_), comma=False)

        # ---

        # isdi2, itd2, td2 = self.get_ittime("overall", d1d2d3prof="d2")
        # td2 = td2 * 1e3  # ms
        # trange = np.arange(start=td2[0], stop=td2[-1], step=tstep)
        #
        # Printcolor.blue("\tTime 2D [1ms]", comma=True)
        # print('['),
        # for t in trange:
        #     tnear = td2[self.find_nearest_index(td2, t)]
        #     if abs(tnear - t) < tstep:
        #         Printcolor.green('.', comma=True)
        # print(']'),
        # Printcolor.green("{:.1f}ms".format(td2[-1]), comma=False)
        #
        #
        # exit(1)
        #
        # isdi1, itd1, td = self.get_ittime("overall", d1d2d3prof="d1")
        # td = td * 1e3 # ms
        # # print(td); exit(1)
        # Printcolor.blue("\tTime 1D [1ms]", comma=True)
        # n=1
        # print('['),
        # for it, t in enumerate(td[1:]):
        #     # tcum = tcum + td[it]
        #     # print(tcum, tstart + n*tstep)
        #     if td[it] > n*tstep:
        #         Printcolor.green('.', comma=True)
        #         n = n+1
        # print(']'),
        # Printcolor.green("{:.1f}ms".format(td[-1]), comma=False)
        #
        # isd2, itd2, td2 = self.get_ittime("overall", d1d2d3prof="d2")
        # td2 = td2 * 1e3 # ms
        # # print(td); exit(1)
        # Printcolor.blue("\tTime 2D [1ms]", comma=True)
        # n=1
        # print('['),
        # for it, t in enumerate(td2[1:]):
        #     # tcum = tcum + td[it]
        #     # print(tcum, tstart + n*tstep)
        #     if td2[it] > n*tstep:
        #         Printcolor.green('.', comma=True)
        #         n = n+1
        # print(']'),
        # Printcolor.green("{:.1f}ms".format(td2[-1]), comma=False)

    def print_ititme_status_(self, tstep=1.):

        _, itd1, td1 = self.get_ittime("overall", d1d2d3prof="d1")
        td1 = td1 * 1e3  # ms
        # print(td1); exit(1)
        Printcolor.blue("\tTime 1D [1ms]", comma=True)
        n = 1
        print('['),
        for it, t in enumerate(td1[1:]):
            # tcum = tcum + td1[it]
            # print(tcum, tstart + n*tstep)
            if td1[it] > n * tstep:
                Printcolor.green('.', comma=True)
                n = n + 1
        print(']'),
        Printcolor.green("{:.1f}ms".format(td1[-1]), comma=False)

        _, itd2, td2 = self.get_ittime("overall", d1d2d3prof="d2")
        td2 = td2 * 1e3  # ms
        # print(td1); exit(1)
        Printcolor.blue("\tTime 2D [1ms]", comma=True)
        n = 1
        print('['),
        for it, t in enumerate(td2[1:]):
            # tcum = tcum + td1[it]
            # print(tcum, tstart + n*tstep)
            if td2[it] > n * tstep:
                Printcolor.green('.', comma=True)
                n = n + 1
        print(']'),
        Printcolor.green("{:.1f}ms".format(td2[-1]), comma=False)

    def print_prof_ittime(self):

        _, itprof, tprof = self.get_ittime("profiles", d1d2d3prof="prof")
        _, itnu, tnu = self.get_ittime("nuprofiles", d1d2d3prof="nuprof")

        all_it = sorted(list(set(list(itprof) + list(itprof))))

        for it in all_it:
            time_ = self.get_time_for_it(it, "profiles", "prof")
            is_prof = False
            if int(it) in np.array(itprof, dtype=int):
                is_prof = True
            is_nu = False
            if int(it) in np.array(itnu, dtype=int):
                is_nu = True

            if not np.isnan(self.maxtime) and time_ > self.maxtime:
                goodcolor="yellow"
            else:
                goodcolor="green"

            Printcolor.print_colored_string(
                ["\tit", str(it), "[", "{:.1f}".format(time_ * 1e3), "ms]"],
                ["blue", goodcolor, "blue", goodcolor, "blue"], comma=True
            )

            print("["),

            if is_prof:
                Printcolor.print_colored_string(["prof"],[goodcolor],comma=True)
            else: Printcolor.red("prof", comma=True)

            if is_nu:Printcolor.print_colored_string(["nuprof"],[goodcolor],comma=True)
            else: Printcolor.red("nuprof", comma=True)

            print("]")




    # def print_assert_outflowed_data(self, criterion):
    #
    #     flist = copy.deepcopy(LOAD_FILES.list_outflowed_files)
    #     if not criterion.__contains__("_b"):
    #         # if the criterion is not Bernoulli
    #         flist.remove("hist_vel_inf_bern.dat")
    #         flist.remove("ejecta_profile_bern.dat")
    #
    #     outflow_status, outflow_missing = \
    #         self.__assert_content(Paths.ppr_sims + self.sim + "/outflow{}/".format(criterion),
    #                               flist)
    #
    #     return outflow_status, outflow_missing