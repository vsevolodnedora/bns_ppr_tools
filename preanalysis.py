from __future__ import division
from sys import path
path.append('modules/')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
from numpy import inf
from glob import glob
import numpy as np
import os.path
import h5py
import csv
import os
import re
from argparse import ArgumentParser
from utils import Paths, Printcolor, Lists


""" ==============================================| SETTINGS |======================================================="""

__preanalysis__ = {
    "name": "preanalysis",
    "tasklist": ["update_status", "collate", "print_status", "init_data"],
    "files": ["ittime.h5", "init_data.csv", "parfile.par"]
}

""" ================================================================================================================="""


class SIM_STATUS:

    def __init__(self, sim, clean=True, save=True):
        self.sim = sim
        self.clean = clean

        self.simdir = Paths.gw170817 + sim + '/' #"/data1/numrel/WhiskyTHC/Backup/2018/GW170817/" + sim +'/'
        self.resdir = Paths.ppr_sims + sim + '/'
        self.profdir = self.simdir + "profiles/3d/"
        self.resfile = "ittime.h5"

        self.d1_ittime_file= "dens.norm1.asc"
        self.d1_ittime_outflow_file = "outflow_det_0.asc"
        self.d1_flag_files = ["dens.norm1.asc",
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
                              "outflow_surface_det_1_fluxdens.asc"]

        self.d2_it_file = "entropy.xy.h5"
        self.d2_flag_files = ["entropy.xy.h5",
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
                              "Y_e.xz.h5"]

        self.d3_it_file = "Y_e.file_0.h5"
        self.d3_flag_files = ["Y_e.file_0.h5",
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

        self.output_dics = {}
        '''
            {
            'output-0000':{
                          'int':0, 'dir':True, 'dat':False,'dattar':True, 
                          'd1data': True,  'it1d':[1234, 1245, ...], 't1d': [0.001, 0.002, 0.003],
                          'd2data': True,  'it2d':[1234, 1245, ...], 't2d': [0.001, 0.002, 0.003],
                          'd3data': True,  'it3d':[1234, 1255, ...], 't3d': [0.001, ...],
                          }
            }
        '''
        self.overall = {}
        '''
            {
                          'd1data': True,  'it1d':[1234, 1245, ...], 't1d': [0.001, 0.002, 0.003],
                          'd2data': True,  'it2d':[1234, 1245, ...], 't2d': [0.001, 0.002, 0.003],
                          'd3data': True,  'it3d':[1234, 1255, ...], 't3d': [0.001, ...],
            }
        '''

        self.profiles = {}

        self.nuprofiles = {}
        '''
            {'profdata':True, 'itprofs':[1345, ...],    'tprofs':[1345, ...]} 
        '''

        self.missing_outputs = []
        self.outputs_with_missing_ittime_file = []

        self.dattars = self.get_dattars()
        self.tars = self.get_tars()
        self.outputs = self.get_outputs()

        # initilize the dics
        for output in self.outputs:
            self.output_dics[str(output)] = {}
        print(self.output_dics.keys())

        for output in self.outputs:
            #
            _int = int(output.split("output-")[-1])
            _dir = True
            if output in self.tars:
                _tar = True
            else:
                _tar = False
            if output in self.dattars:
                _dattar = True
            else:
                _dattar = False

            self.output_dics[output]['int'] = _int
            self.output_dics[output]['dir'] = _dir
            self.output_dics[output]['tar'] = _tar
            self.output_dics[output]['dattar'] = _dattar

        # --- D1 --- #

        alld1iterations = []
        alld1timesteps = []
        for output in self.outputs:
            isd1data, d1iterations, d1timesteps = \
                self.scan_d1_data(output, self.d1_ittime_file)
            alld1iterations.append(d1iterations)
            alld1timesteps.append(d1timesteps)
            self.output_dics[output]['d1data'] = isd1data
            self.output_dics[output]['itd1'] = d1iterations
            self.output_dics[output]['td1'] = d1timesteps
        assert len(alld1timesteps) == len(alld1iterations)
        # assert not np.isnan(np.sum(alld1timesteps)) and not np.isnan(np.sum(alld1iterations))

        if len(alld1timesteps) > 0:
            alld1iterations = np.sort(np.unique(np.concatenate(alld1iterations)))
            alld1timesteps = np.sort(np.unique(np.concatenate(alld1timesteps)))
            assert len(alld1timesteps) == len(alld1iterations)
            assert not np.isnan(np.sum(alld1timesteps)) and not np.isnan(np.sum(alld1iterations))
            self.overall["d1data"] = True
            self.overall["itd1"] = alld1iterations
            self.overall["td1"] = alld1timesteps
            print(alld1timesteps)
        else:
            self.overall["d1data"] = False
            self.overall["itd1"] = np.empty(0,)
            self.overall["t1d"] = np.empty(0,)

        # --- D1 outflow --- #

        alld1outflowiterations = []
        alld1outflowtimesteps = []
        for output in self.outputs:
            isd1outflowdata, d1outflowiterations, d1outflowtimesteps = \
                self.scan_d1_data(output, self.d1_ittime_outflow_file)
            alld1outflowiterations.append(d1outflowiterations)
            alld1outflowtimesteps.append(d1outflowtimesteps)
            self.output_dics[output]['outflowdata'] = isd1outflowdata
            self.output_dics[output]['itoutflow'] = d1outflowiterations
            self.output_dics[output]['toutflow'] = d1outflowtimesteps
        assert len(alld1outflowtimesteps) == len(alld1outflowiterations)
        # assert not np.isnan(np.sum(alld1timesteps)) and not np.isnan(np.sum(alld1iterations))

        if len(alld1timesteps) > 0:
            alld1outflowiterations = np.sort(np.unique(np.concatenate(alld1outflowiterations)))
            alld1outflowtimesteps = np.sort(np.unique(np.concatenate(alld1outflowtimesteps)))
            assert len(alld1outflowtimesteps) == len(alld1outflowiterations)
            assert not np.isnan(np.sum(alld1outflowtimesteps)) and not np.isnan(np.sum(alld1outflowiterations))
            self.overall["outflowdata"] = True
            self.overall["itoutflow"] = alld1outflowiterations
            self.overall["toutflow"] = alld1outflowtimesteps
            print(alld1outflowtimesteps)
        else:
            self.overall["outflowdata"] = False
            self.overall["itoutflow"] = np.empty(0,)
            self.overall["toutflow"] = np.empty(0,)


        # --- D2 --- #

        alld2iterations = []
        alld2timesteps = []
        for output in self.outputs:
            isd2data, d2iterations, d2timesteps = \
                self.scan_d2_data(output)
            alld2iterations.append(d2iterations)
            alld2timesteps.append(d2timesteps)
            self.output_dics[output]['d2data'] = isd2data
            self.output_dics[output]['itd2'] = d2iterations
            self.output_dics[output]['td2'] = d2timesteps
        assert len(alld2timesteps) == len(alld2iterations)
        # assert not np.isnan(np.sum(alld2timesteps)) and not np.isnan(np.sum(alld2iterations))

        if len(alld2timesteps) > 0:
            alld2iterations = np.sort(np.unique(np.concatenate(alld2iterations)))
            alld2timesteps = np.sort(np.unique(np.concatenate(alld2timesteps)))
            assert len(alld2timesteps) == len(alld2iterations)
            assert not np.isnan(np.sum(alld2timesteps)) and not np.isnan(np.sum(alld2iterations))
            self.overall["d2data"] = True
            self.overall["itd2"] = alld2iterations
            self.overall["td2"] = alld2timesteps
        else:
            self.overall["d2data"] = False
            self.overall["itd2"] = np.empty(0,)
            self.overall["t2d"] = np.empty(0,)

        # --- D3 --- #

        alld3iterations = []
        alld3timesteps = []
        for output in self.outputs:
            isd3data, d3iterations, d3timesteps = \
                self.scan_d3_data(output)
            alld3iterations.append(d3iterations)
            alld3timesteps.append(d3timesteps)
            self.output_dics[output]['d3data'] = isd3data
            self.output_dics[output]['itd3']   = d3iterations
            self.output_dics[output]['td3']    = d3timesteps
        assert len(alld3timesteps) == len(alld3iterations)
        # assert not np.isnan(np.sum(alld3timesteps)) and not np.isnan(np.sum(alld3iterations))
        if len(alld3timesteps) > 0:
            alld3iterations = np.sort(np.unique(np.concatenate(alld3iterations)))
            alld3timesteps = np.sort(np.unique(np.concatenate(alld3timesteps)))
            assert len(alld3timesteps) == len(alld3iterations)
            assert not np.isnan(np.sum(alld3timesteps)) and not np.isnan(np.sum(alld3iterations))
            self.overall["d3data"] = True
            self.overall["itd3"] = alld3iterations
            self.overall["td3"] = alld3timesteps
        else:
            self.overall["d3data"] = False
            self.overall["itd3"] = np.empty(0,)
            self.overall["t3d"] = np.empty(0,)

        # --- profs --- #

        isprofdata, profiterations, proftimesteps = \
            self.scan_profs_data(fname=".h5")
        assert len(profiterations) == len(proftimesteps)
        assert not np.isnan(np.sum(profiterations)) and not np.isnan(np.sum(proftimesteps))

        self.profiles['profdata'] = isprofdata
        self.profiles['itprof']  = profiterations
        self.profiles['tprof']   = proftimesteps

        # --- nu profs --- #

        isnuprofdata, nuprofiterations, nuproftimesteps = \
            self.scan_profs_data(fname="nu.h5")
        assert len(nuprofiterations) == len(nuproftimesteps)
        assert not np.isnan(np.sum(nuprofiterations)) and not np.isnan(np.sum(nuproftimesteps))

        self.nuprofiles['nuprofdata'] = isnuprofdata
        self.nuprofiles['itnuprof']  = nuprofiterations
        self.nuprofiles['tnuprof']   = nuproftimesteps

        print("\t{}".format(self.sim))
        print("\toutputs : {}".format(len(self.outputs)))
        print("\ttars    : {}".format(len(self.tars)))
        print("\tdattars : {}".format(len(self.dattars)))
        print("\tprofs   : {}".format(len(proftimesteps)))
        print("\tnu profs: {}".format(len(nuproftimesteps)))
        # for outout_dir in self.outputs_dirs:
            # _name = str(outout_dir.split('/')[-1])
            # print(self.outputs[_name])
        if save:
            resfile = self.resdir + self.resfile
            self.save(resfile)
            print("\tsaved {}".format(resfile))

    @staticmethod
    def find_nearest_index(array, value):
        ''' Finds index of the value in the array that is the closest to the provided one '''
        idx = (np.abs(array - value)).argmin()
        return idx

    def get_tars(self):
        tars = glob(self.simdir + 'output-????.tar')
        tars = [str(tar.split('/')[-1]).split('.tar')[0] for tar in tars]
        return tars

    def get_dattars(self):
        dattars = glob(self.simdir + 'output-????.dat.tar')
        dattars = [str(dattar.split('/')[-1]).split('.dat.tar')[0] for dattar in dattars]
        return dattars

    def get_outputdirs(self):
        dirs = os.listdir(self.simdir)
        output_dirs = []
        for dir_ in dirs:
            if str(dir_).__contains__("output-") and \
                    not str(dir_).__contains__('.tar') and \
                    not str(dir_).__contains__('.dat.tar'):
                output_dirs.append(dir_)

        return output_dirs

    def get_outputs(self):
        return [str(output_dir.split('/')[-1]) for output_dir in self.get_outputdirs()]

    def get_next_output(self, output):
        if not output in self.outputs:
            raise NameError("output: {} not in the list of outputs: {}"
                            .format(output, self.outputs))
        if output == self.outputs[-1]:
            raise NameError("output: {} is the last output in the list. No more."
                            .format(output))
        return self.outputs[self.outputs.index(output)+1]

    def get_profiles(self, fname=''):
        if not os.path.isdir(self.profdir):
            if not self.clean:
                print("Note. No profiels directory found. \nExpected: {}"
                      .format(self.profdir))
            return []
        profiles = glob(self.profdir + '*' + fname)
        return profiles

    def scan_d1_data(self, output_dir, ittime_file):

        if output_dir.__contains__('__'):
            # error outputs
            return 0, np.empty(0, ), np.empty(0, )

        missing_files = []
        for flag_file in self.d1_flag_files:
            if os.path.isfile(self.simdir + '/' + output_dir + '/data/' + flag_file):
                pass
            else:
                missing_files.append(flag_file)
        if len(missing_files) == 0:
            pass
        elif not self.clean:
            print("Warning. Missing d1 files: {}\nin output: {} \n({})".format(missing_files, output_dir,
                                                                               self.simdir + '/' + output_dir + '/data/'))
        else:
            pass
        if os.path.isfile(self.simdir + '/' + output_dir + '/data/' + ittime_file):
            d1data = 1
            it_time_i = np.loadtxt(self.simdir + '/' + output_dir + '/data/' + ittime_file, usecols=(0, 1))
            it_time_i[:, 1] *= 0.004925794970773136 * 1e-3 # ms
            iterations = np.array(it_time_i[:, 0], dtype=int)
            timesteps  = np.array(it_time_i[:, 1], dtype=float)
        else:
            # print("error. no {} file".format(self.simdir + '/' + output_dir + '/data/' + self.d1_ittime_file))
            d1data = 0
            iterations = np.empty(0, )
            timesteps  = np.empty(0, )


        assert len(iterations) == len(timesteps)
        return d1data, iterations, timesteps

    def scan_d2_data(self, output):
        missing_files = []
        for flag_file in self.d2_flag_files:
            if os.path.isfile(self.simdir + '/' + output + '/data/' + flag_file):
                pass
            else:
                missing_files.append(flag_file)
        if len(missing_files) == 0:
            pass
        elif not self.clean:
            print("Warning. Missing d2 files: {}\nin output: {} ({})".format(missing_files, output,
                                                                             self.simdir + '/' + output + '/data/'))
        else:
            pass
        if os.path.isfile(self.simdir + '/' + output + '/data/' + self.d2_it_file):
            d2data = 1
            dfile = h5py.File(self.simdir + '/' + output + '/data/' + self.d2_it_file, "r")
            iterations = []
            for row in dfile:
                for row__ in row.split():
                    if str(row__).__contains__('it='):
                        iterations.append(int(str(row__).split('it=')[-1]))
            if len(iterations) != 0:
                pass
            elif not self.clean:
                print("Warning. No iterations found for output:\n{}".format(output))
            iterations = np.unique(iterations)
            d1iterations = self.overall["itd1"]
            d1times = self.overall["td1"]
            if len(d1iterations) > 0 and len(d1times) > 0:
                listd1iterations = list(d1iterations)
                timesteps = []
                for it in iterations:

                    if not int(it) in d1iterations:
                        if not self.clean:
                            print("Warning d2 data. it:{} is not in the itd1 list"
                                  .format(it))

                        if it > d1iterations.max():
                            print("Warning: d2 it:{} is above d1.max():{}"
                                  .format(it, d1iterations.max()))
                            _t_ = self.linear_fit(it, d1iterations[0], d1iterations[-1], d1times[0], d1times[-1])
                        elif it < d1iterations.min():
                            print("Warning: d2 it:{} is below d1.max():{}"
                                  .format(it, d1iterations.max()))
                            _t_ = self.linear_fit(it, d1iterations[0], d1iterations[-1], d1times[0], d1times[-1])
                        else:
                            from scipy import interpolate
                            _t_ = interpolate.interp1d(d1iterations, d1times, bounds_error=False)(it)

                        #
                        #
                        #
                        # from scipy import interpolate
                        # _t_ = interpolate.interp1d(d1iterations, d1times, bounds_error=False)(it)
                        timesteps.append(_t_)
                    else:
                        timesteps.append(d1times[listd1iterations.index(int(it))])
                    #
                    # if it in listd1iterations:
                    #     timesteps.append(d1times[listd1iterations.index(int(it))])
                    # else:
                    #     raise ValueError("it:{} from 2d data is not in total 1d list of iterations\n"
                    #                          "nearest is {}. Boundaries are:{} {}"
                    #                          .format(it,
                    #                                  listd1iterations[self.find_nearest_index(
                    #                                      np.array(listd1iterations), it)],
                    #                                  listd1iterations[0], listd1iterations[-1]))

                if len(timesteps) == len(iterations):
                    pass
                elif not self.clean:
                    print("Warning. N(it){} != N(times){} for d2 data in \n{}"
                          .format(len(iterations), len(timesteps), output))
                timesteps = np.array(timesteps, dtype=float)
            else:
                if not self.clean:
                    print("Error. Given d1 iterations ({}) and times ({}) do not match or empty. in\n{}"
                          .format(len(d1iterations), len(d1times), output))
                timesteps = np.empty(0, )
        else:
            d2data = 0
            iterations = np.empty(0, )
            timesteps = np.empty(0, )
            if not self.clean:
                print("Note. No 2D data found in output:\n{}".format(output))

        return d2data, iterations, timesteps

    def scan_d3_data(self, output):
        missing_files = []
        for flag_file in self.d3_flag_files:
            if os.path.isfile(self.simdir + '/' + output + '/data/' + flag_file):
                pass
            else:
                missing_files.append(flag_file)
        if len(missing_files) == 0:
            pass
        elif not self.clean:
            print("Warning. Missing d3 files: {}\nin output: {}".format(missing_files, output))
        else:
            pass
        if os.path.isfile(self.simdir + '/' + output + '/data/' + self.d3_it_file):
            d3data = 1
            dfile = h5py.File(self.simdir + '/' + output + '/data/' + self.d3_it_file, "r")
            iterations = []
            for row in dfile:
                for row__ in row.split():
                    if str(row__).__contains__('it='):
                        iterations.append(int(str(row__).split('it=')[-1]))
            if len(iterations) != 0:
                pass
            elif not self.clean:
                print("Warning. No iterations found for output:\n{}".format(output))
            iterations = np.unique(iterations)
            d1iterations = self.overall["itd1"]
            d1times = self.overall["td1"]
            if len(d1iterations) > 0 and len(d1times) > 0:
                listd1iterations = list(d1iterations)
                timesteps = []
                for it in iterations:
                    timesteps.append(d1times[listd1iterations.index(int(it))])
                if len(timesteps) == len(iterations):
                    pass
                elif not self.clean:
                    print("Warning. N(it){} != N(times){} for d2 data in \n{}"
                          .format(len(iterations), len(timesteps), output))
                timesteps = np.array(timesteps, dtype=float)
            else:
                if not self.clean:
                    print("Error. Given d1 iterations ({}) and times ({}) do not match or empty. in\n{}"
                          .format(len(d1iterations), len(d1times), output))
                timesteps = np.empty(0, )
        else:
            d3data = 0
            iterations = np.empty(0, )
            timesteps = np.empty(0, )
            if not self.clean:
                print("Note. No 3D data found in output:\n{}".format(output))

        # if d3data:
        #     print(output_dir); exit(0)
        return d3data, iterations, timesteps

    @staticmethod
    def linear_fit(it, it1=1, it2=1.4, t1=5., t2=10.):

        k = (it2 - it1) / (t2 - t1)
        b = it2 - (k * t2)

        return (it - b) / k

    def scan_profs_data(self, fname='.h5'):

        profiles = self.get_profiles(fname=fname)

        if len(profiles) > 0:
            import re
            iterations = np.array(np.sort(np.array(list([int(profile.split('/')[-1].split(fname)[0])
                                                              for profile in profiles
                                                              if re.match("^[-+]?[0-9]+$",
                                                                          profile.split('/')[-1].split(
                                                                              fname
                                                                          )[0])]))),
                                                                dtype=int)
            if len(iterations) != len(profiles):
                if not self.clean:
                    print("ValueError. Though {} {} profiles found, {} iterations found."
                          .format(len(profiles), fname, len(iterations)))
                #return 0, np.empty(0,), np.empty(0,)

            d1iterations = self.overall["itd1"]
            d1times = self.overall["td1"]
            iterations = np.unique(iterations)
            listd1iterations = list(d1iterations)
            times = []
            for it in iterations:
                if not int(it) in d1iterations:
                    if not self.clean:
                        print("Warning {} prof. it:{} is not in the itd1 list"
                              .format(fname, it))

                    if it > d1iterations.max():
                        print("Warning: prof it:{} is above d1.max():{}"
                              .format(it, d1iterations.max()))
                        _t_ = self.linear_fit(it, d1iterations[0], d1iterations[-1], d1times[0], d1times[-1])
                    elif it < d1iterations.min():
                        print("Warning: prof it:{} is below d1.max():{}"
                              .format(it, d1iterations.max()))
                        _t_ = self.linear_fit(it, d1iterations[0], d1iterations[-1], d1times[0], d1times[-1])
                    else:
                        from scipy import interpolate
                        _t_ = interpolate.interp1d(d1iterations, d1times, bounds_error=False)(it)

                    assert not np.isnan(_t_)
                    times.append(_t_)
                else:
                    times.append(d1times[listd1iterations.index(int(it))])
            times = np.array(times, dtype=float)
            return 1, iterations, times



        else:
            if not self.clean:
                print("Note. No {} profiles found in dir:\n{}".format(fname, self.profdir))
            return 0, np.empty(0,), np.empty(0,)

    def save(self, resfile):

        if not os.path.isdir(self.resdir):
            os.mkdir(self.resdir)

        if os.path.isfile(resfile):
            os.remove(resfile)
            if not self.clean:
                print("Rewriting the result file {}".format(resfile))

        dfile = h5py.File(resfile, "w")
        for output in self.outputs:
            one_output = self.output_dics[output]
            dfile.create_group(output)
            for key in one_output.keys():
                if not self.clean: print("\twriting key:{} output:{}".format(key, output))
                dfile[output].create_dataset(key, data=one_output[key])

        dfile.create_group("profiles")
        for key in self.profiles.keys():
            dfile["profiles"].create_dataset(key, data=self.profiles[key])

        dfile.create_group("nuprofiles")
        for key in self.nuprofiles.keys():
            dfile["nuprofiles"].create_dataset(key, data=self.nuprofiles[key])

        dfile.create_group("overall")
        for key in self.overall.keys():
            if not self.clean: print("\twriting key:{} overall".format(key))
            dfile["overall"].create_dataset(key, data=self.overall[key])

        dfile.close()


class LOAD_ITTIME:

    def __init__(self, sim):

        if not os.path.isfile(Paths.ppr_sims + sim + '/' + "ittime.h5"):
            # from analyze import SIM_STATUS
            print("\tno ittime.h5 found. Creating...")
            SIM_STATUS(sim, save=True, clean=True)

        self.set_use_selected_output_if_many_found = True
        self.clean = True
        self.sim = sim

        self.ittime_fname = Paths.ppr_sims + self.sim + '/ittime.h5'

        self.dfile = h5py.File(self.ittime_fname, "r")

        self.exclusion_list = ["nuprofiles", "profiles", "overall"]

        if not self.clean:
            print("loaded file:\n{}\n contains:")
            for v_n in self.dfile:
                print(v_n)

    @staticmethod
    def find_nearest_index(array, value):
        ''' Finds index of the value in the array that is the closest to the provided one '''
        idx = (np.abs(array - value)).argmin()
        return idx

    def get_list_outputs(self):
        return [str(output) for output in self.dfile.keys() if not output in self.exclusion_list]

    def get_ittime(self, output="overall", d1d2d3prof='d1'):
        """
        :param output: "output-0000", or "overall" or "profiles", "nuprofiles"
        :param d1d2d3prof: d1, d2, d3, prof, nuprof
        :return:
        """
        # print(self.dfile[output].keys())
        # print(output, d1d2d3prof)
        # assert output in self.dfile.keys()
        # assert '{}data'.format(str(d1d2d3prof)) in self.dfile[output].keys()
        # assert 'it{}'.format(str(d1d2d3prof)) in self.dfile[output].keys()
        # assert 't{}'.format(str(d1d2d3prof)) in self.dfile[output].keys()
        return bool(np.array(self.dfile[output]['{}data'.format(str(d1d2d3prof))], dtype=int)), \
               np.array(self.dfile[output]['it{}'.format(str(d1d2d3prof))], dtype=int), \
               np.array(self.dfile[output]['t{}'.format(str(d1d2d3prof))], dtype=float)

    def get_output_for_it(self, it, d1d2d3='d1'):
        isdata, allit, alltimes = self.get_ittime(output="overall", d1d2d3prof=d1d2d3)
        if not isdata:
            raise ValueError("data for d1d2d3:{} not available".format(d1d2d3))
        if it < allit[0] or it > allit[-1]:
            raise ValueError("it:{} is below min:{} or above max:{} in d1d2d3:{}"
                             .format(it, allit[0], allit[-1], d1d2d3))
        if not it in allit:
            raise ValueError("it:{} is not in alliterations:{} for d1d2d3:{}"
                             .format(it, allit, d1d2d3))
        required_outputs = []
        for key in self.dfile:
            if key not in self.exclusion_list:
                output = key
                isdata, outputiterations, outputtimesteps = \
                    self.get_ittime(output, d1d2d3)
                if isdata:
                    if int(it) in outputiterations:
                        required_outputs.append(output)
        if len(required_outputs) == 0:
            raise ValueError("no output is found for it:{} d1d2d3:{}"
                             .format(it, d1d2d3))
        elif len(required_outputs) > 1:
            if not self.clean:
                print("Warning. it:{} is found in multiple outputs:{} for d1d2d3:{}"
                      .format(it, required_outputs, d1d2d3))
            if self.set_use_selected_output_if_many_found:
                return required_outputs[0]
            else:
                raise ValueError("Set 'self.set_use_selected_output_if_many_found=True' to get"
                                 "0th output out of many found")
        else:
            return required_outputs[0]

    def get_nearest_time(self, time__, d1d2d3='d1'):

        isdata, allit, alltimes = self.get_ittime(output="overall", d1d2d3prof=d1d2d3)
        if not isdata:
            raise ValueError("data for d1d2d3:{} not available".format(d1d2d3))
        if time__ < alltimes[0] or time__ > alltimes[-1]:
            raise ValueError("time:{} is below min:{} or above max:{} in d1d2d3:{}"
                             .format(time__, alltimes[0], alltimes[-1], d1d2d3))
        if time__ in alltimes:
            time_ = time__
        else:
            time_ = alltimes[self.find_nearest_index(alltimes, time__)]
            if not self.clean:
                print("nearest time to {}, is {}, selected for d1d2d3:{}"
                      .format(time__, time_, d1d2d3))

        return time_

    def get_it_for_time(self, time__, d1d2d3='d1'):
        time_ = self.get_nearest_time(time__, d1d2d3)
        isdata, allit, alltimes = self.get_ittime(output="overall", d1d2d3prof=d1d2d3)
        if isdata:
            return int(allit[self.find_nearest_index(alltimes, time_)])
        else:
            raise ValueError("no data available for d1d2d3:{}".format(d1d2d3))

    def get_time_for_it(self, it, d1d2d3prof='d1'):

        if d1d2d3prof == "prof":
            isdata, allit, alltimes = self.get_ittime(output="profiles", d1d2d3prof=d1d2d3prof)
        else:
            isdata, allit, alltimes = self.get_ittime(output="overall", d1d2d3prof=d1d2d3prof)
        if not isdata:
            raise ValueError("data for d1d2d3:{} not available".format(d1d2d3prof))
        if it < allit[0] or it > allit[-1]:
            raise ValueError("it:{} is below min:{} or above max:{} in d1d2d3:{}"
                             .format(it, allit[0], allit[-1], d1d2d3prof))
        if not it in allit:
            print("\tWarning it:{} is not in the list of it for d1d2d3: {}".format(it, d1d2d3prof))

            from scipy import interpolate
            f = interpolate.interp1d(allit, alltimes, kind="linear")
            t = f(it)
            return float(t)
            # raise ValueError("it:{} is not in alliterations:{} for d1d2d3:{}"
            #                  .format(it, allit, d1d2d3prof))

        if isdata:
            return float(alltimes[self.find_nearest_index(allit, it)])
        else:
            raise ValueError("no data available for d1d2d3:{}".format(d1d2d3prof))

    def get_output_for_time(self, time__, d1d2d3='d1'):

        it = self.get_it_for_time(time__, d1d2d3)
        output = self.get_output_for_it(int(it), d1d2d3)

        return output

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


class PRINT_SIM_STATUS(LOAD_ITTIME):

    def __init__(self, sim):

        LOAD_ITTIME.__init__(self, sim)

        self.sim = sim

        self.path_in_data = Paths.gw170817 + sim + '/'
        self.prof_in_data = Paths.gw170817 + sim + '/profiles/3d/'
        self.path_out_data = Paths.ppr_sims + sim + '/'
        self.file_for_gw_time = "/data/dens.norm1.asc"
        self.file_for_ppr_time = "/collated/dens.norm1.asc"

        ''' --- '''

        tstart = 0.
        tend = 130.
        tstep = 1.
        prec = 0.5

        ''' --- PRINTING ---  '''
        print('=' * 100)
        print("<<< {} >>>".format(sim))
        # assert that the ittime.h5 file is upt to date

        self.print_data_from_parfile(self.path_in_data + 'output-0001/' + 'parfile.par')

        # check if ittime.h5 exists and up to date
        isgood = self.assert_ittime()
        if not isgood:
            # from preanalysis import SIM_STATUS
            SIM_STATUS(sim, save=True, clean=True)
            Printcolor.green("\tittime.h5 is updated")

        self.print_what_output_tarbal_dattar_present(comma=False)
        print("\tAsserting output contnet:")
        self.print_assert_tarball_content()
        print("\tAsserting data availability: ")

        tstart, tend = self.get_overall_tstart_tend()
        Printcolor.green("\tOverall Data span: {:.1f} to {:.1f} [ms]"
                         .format(tstart - 1, tend - 1))

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

    def get_tars(self):
        tars = glob(self.path_in_data + 'output-????.tar')
        tars = [str(tar.split('/')[-1]).split('.tar')[0] for tar in tars]
        return tars

    def get_dattars(self):
        dattars = glob(self.path_in_data + 'output-????.dat.tar')
        dattars = [str(dattar.split('/')[-1]).split('.dat.tar')[0] for dattar in dattars]
        return dattars

    def get_outputdirs(self):

        def get_number(output_dir):
            return int(str(output_dir.split('/')[-1]).split("output-")[-1])

        dirs = os.listdir(self.path_in_data)
        output_dirs = []
        for dir_ in dirs:
            if str(dir_).__contains__("output-") and \
                    not str(dir_).__contains__('.tar') and \
                    not str(dir_).__contains__('.dat.tar'):
                output_dirs.append(dir_)
        output_dirs.sort(key=get_number)
        return output_dirs

    def get_outputs(self):
        return [str(output_dir.split('/')[-1]) for output_dir in self.get_outputdirs()]

    def get_profiles(self, extra=''):
        if not os.path.isdir(self.prof_in_data):
            return []
        profiles = glob(self.prof_in_data + "*{}.h5".format(extra))
        # print(profiles)
        return profiles

    def get_profile_its(self, extra=""):

        profiles = glob(self.prof_in_data + "*.h5")
        fnames = []
        for profile in profiles:
            fname = str(profile.split('/')[-1]).split('.h5')[0]
            if extra != "":
                if str(fname).__contains__(extra):
                    fnames.append(fname.replace(extra,''))
                # else:
                #     print(fname, extra)
            else:
                fnames.append(fname)
        #
        if len(fnames) == 0:
            return np.empty(0,)
        #
        list_iterations = np.array(
            np.sort(np.array(list([int(itdir) for itdir in fnames if re.match("^[-+]?[0-9]+$", itdir)]))))
        return list_iterations

    def assert_ittime(self):

        new_output_dirs = self.get_outputdirs()
        new_outputs = [str(output) for output in new_output_dirs]
        old_outputs = self.get_list_outputs()

        if sorted(old_outputs) == sorted(new_outputs):
            last_output = list(new_output_dirs)[-1]
            it_time_i = np.loadtxt(self.path_in_data + last_output + self.file_for_gw_time, usecols=(0, 1))
            new_it_end = int(it_time_i[-1, 0])
            _, itd1, _ = self.get_ittime("overall", d1d2d3prof="d1")
            old_it_end = itd1[-1]

            if int(new_it_end) == int(old_it_end):
                is_up_to_data = True

                new_profiles = glob(self.prof_in_data + "*.h5")
                _, itprofs, _ = self.get_ittime("profiles", d1d2d3prof="prof")
                if len(new_profiles) == len(itprofs):
                    Printcolor.green("\tittime.h5 is up to date")
                else:
                    Printcolor.red("\tittime.h5 is NOT up to date: profiles (old{:d} != new{:d})"
                                   .format(len(itprofs), len(new_profiles)))
            else:
                is_up_to_data = False
                Printcolor.red("\tittime.h5 is NOT up to date: d1 iterations (old{:d} != new{:d})"
                               .format(old_it_end, new_it_end))
        else:
            Printcolor.red("\tittime.h5 is NOT up to date: outputs: (old{} != new{})"
                           .format(old_outputs[-1], new_outputs[-1]))
            return False

        return is_up_to_data

    def get_overall_tstart_tend(self):

        t1, t2 = [], []
        isd1, itd1, td1 = self.get_ittime("overall", d1d2d3prof="d1")
        isd2, itd2, td2 = self.get_ittime("overall", d1d2d3prof="d2")
        isd3, itd3, td3 = self.get_ittime("overall", d1d2d3prof="d3")
        isprof, itprof, tprof = self.get_ittime("profiles", d1d2d3prof="prof")



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



        return np.array(t1).min() * 1e3 + 1, np.array(t2).max() * 1e3 + 1

    ''' --- '''

    def print_what_output_tarbal_dattar_present(self, comma=False):

        n_outputs = len(self.get_outputs())
        n_tars = len(self.get_tars())
        n_datatars = len(self.get_dattars())
        n_profs = len(self.get_profiles())

        print("\toutputs: "),
        if n_outputs == 0:
            Printcolor.red(str(n_outputs), comma=True)
        else:
            Printcolor.green(str(n_outputs), comma=True)

        print("\ttars: "),
        if n_tars == 0:
            Printcolor.green(str(n_tars), comma=True)
        else:
            Printcolor.red(str(n_tars), comma=True)

        print("\tdattars: "),
        if n_datatars == 0:
            Printcolor.green(str(n_datatars), comma=True)
        else:
            Printcolor.red(str(n_datatars), comma=True)

        print("\tprofiles: "),
        if n_profs == 0:
            Printcolor.red(str(n_profs), comma=True)
        else:
            Printcolor.green(str(n_profs), comma=True)

        if comma:
            print(' '),
        else:
            print(' ')

    ''' --- '''

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

        outputs = self.get_outputdirs()
        for output in outputs:

            isd1, itd1, td1 = self.get_ittime(output=output, d1d2d3prof="d1")

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
                Printcolor.yellow(" partial, ({}) missing".format(len(missing)), comma=True)
            else:
                Printcolor.red(" absent", comma=True)
            print('')

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
            isdata, itd1, td1 = self.get_ittime(output=output, d1d2d3prof="d1")
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
                tnear = tend[self.find_nearest_index(tend, t)]
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

        isdi1, itd1, td = self.get_ittime(output, d1d2d3prof=d1d2d3prof)
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
                tnear = td[self.find_nearest_index(td, t)]
                if abs(tnear - t) < precision:  # (tnear - t) >= 0
                    Printcolor.green('.', comma=True)
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

        isdi1, itd1, td1 = self.get_ittime("overall", d1d2d3prof="d1")
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

        isd2, itd2, td2 = self.get_ittime("overall", d1d2d3prof="d2")
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

        _, itprof, tprof = self.get_ittime("profiles", "prof")
        _, itnu, tnu = self.get_ittime("nuprofiles", "nuprof")

        all_it = sorted(list(set(list(itprof) + list(itprof))))



        for it in all_it:
            time_ = self.get_time_for_it(it, "prof")
            is_prof = False
            if int(it) in np.array(itprof, dtype=int):
                is_prof = True
            is_nu = False
            if int(it) in np.array(itnu, dtype=int):
                is_nu = True

            Printcolor.print_colored_string(
                ["\tit", str(it), "[", "{:.1f}".format(time_*1e3), "ms]"],["blue", "green", "blue", "green", "blue"], comma=True
            )

            print("["),

            if is_prof: Printcolor.green("prof", comma=True)
            else: Printcolor.red("prof", comma=True)

            if is_nu:Printcolor.green("nuprof", comma=True)
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


class INIT_DATA:

    def __init__(self, sim, clean=True, lor_archive_fpath=None, tov_fpath=None):

        self.lor_archive_fpath = lor_archive_fpath
        self.sim = sim
        self.clean = clean
        # ---
        self.is_init_data_available = True
        if not os.path.isdir(Paths.ppr_sims+sim):
            if not self.clean:
                print("Dir {} does not exist -- creating".format(Paths.ppr_sims+sim))
                os.mkdir(Paths.ppr_sims+sim)
        if not os.path.isfile(Paths.ppr_sims+sim+"parfile.par"):
            if not self.clean:
                print("parfile.par is absent in ppr dir: {} -- trying to copy"
                      .format(Paths.ppr_sims+sim))
            os.system("cp {} {}".format(Paths.gw170817 + sim + "/output-0003/parfile.par",
                                  Paths.ppr_sims + sim + "/"))
            if not os.path.isfile(Paths.ppr_sims + sim + "/" + "parfile.par"):
                if not self.clean:
                    print("Error. copyting of {} TO {} has failed."
                          .format(Paths.gw170817 + sim + "/output-0003/parfile.par",
                                  Paths.ppr_sims + sim + "/"))

        self.par_dic = {}

        self.init_data_dir = Paths.ppr_sims+sim+'/initial_data/'
        if not os.path.isdir(self.init_data_dir):
            run, pizza_fname, hydro_fname, weak_fname = self.copy_extract_initial_data()
            self.par_dic["run"] = run
            self.par_dic["pizza_eos"] = pizza_fname
            self.par_dic["hydro_eos"] = hydro_fname
            self.par_dic["weak_eos"] = weak_fname
        if self.is_init_data_available:
            assert os.path.isdir(self.init_data_dir)
            # Printcolor.blue("\tinitial_data directory is found.")
            # ---

            self.list_expected_eos = [
                "SFHo", "SLy4", "DD2", "BLh", "LS220", "BHB", "BHBlp"
            ]

            self.list_expected_resolutions = [
                "HR", "LR", "SR", "VLR"
            ]

            self.list_expected_viscosities = [
                "LK", "L50", "L25", "L5"
            ]

            self.list_expected_neutrinos = [
                "M0", "M1"
            ]

            self.list_expected_initial_data = [
                "R01", "R02", "R03", "R04", "R05", "R04_corot"
            ]

            self.init_data_dir = Paths.ppr_sims+sim+'/initial_data/'
            assert os.path.isdir(self.init_data_dir)

            self.get_pars_from_sim_name()
            Printcolor.blue("\tSelf data from the sim name is parsed")
            self.load_calcul_extract_pars()
            Printcolor.blue("\tData from calcul.d os parsed")

            if tov_fpath != None:
                tov_fname = tov_fpath
            elif self.par_dic["EOS"] == 'SFHo':
                tov_fname = Paths.TOVs + 'SFHo_sequence.txt' #'SFHo_love.dat'
            elif self.par_dic["EOS"] == 'DD2':
                tov_fname = Paths.TOVs + 'DD2_sequence.txt' #'DD2_love.dat'
            elif self.par_dic["EOS"] == 'LS220':
                tov_fname = Paths.TOVs + 'LS200_sequence.txt' #'LS220_love.dat'
            elif self.par_dic["EOS"] == 'SLy4':
                tov_fname = Paths.TOVs + 'SLy_sequence.txt' #'SLy4_love.dat'
            elif self.par_dic["EOS"] == 'BHBlp' or self.par_dic["EOS"] == 'BHB':
                tov_fname = Paths.TOVs + 'BHBlp_love.dat'
            elif self.par_dic["EOS"] == "BLh":
                tov_fname = Paths.TOVs + 'BLh_sequence.txt' #"BLh_love.dat"
            else:
                raise NameError("\tTOV sequences are not found for EOS:{} ".format(self.par_dic["EOS"]))
            self.load_tov_extract_pars(tov_fname)
            Printcolor.blue("\tData from TOV is parsed")

            self.save_as_csv("init_data.csv")
            Printcolor.blue("\tinitial data is saved")
        else:
            Printcolor.red("\tError. Initial data was not computed")

    def get_fname_for_init_data_from_parfile(self):

        if not os.path.isfile(Paths.ppr_sims + self.sim + '/parfile.par'):
            print("Error. parfile in ppr directory is not found. ")

        initial_data = ""
        pizza_eos_fname = ""
        hydro_eos_fname = ""
        weak_eos_fname = ""
        lines = open(Paths.ppr_sims + self.sim + '/parfile.par', "r").readlines()
        for line in lines:

            if line.__contains__("PizzaIDBase::eos_file"):
                pizza_eos_fname = line.split()[-1]

            if line.__contains__("LoreneID::lorene_bns_file"):
                initial_data = line

            if line.__contains__("EOS_Thermal_Table3d::eos_filename"):
                hydro_eos_fname = line.split()[-1]

            if line.__contains__("WeakRates::table_filename"):
                weak_eos_fname = line.split()[-1]

            if not "" in [initial_data, pizza_eos_fname, hydro_eos_fname, weak_eos_fname]:
                break

        assert initial_data != ""
        #
        run = initial_data.split("/")[-3]
        if not run.__contains__("R"):
            if str(initial_data.split("/")[-2]).__contains__("R05"):
                Printcolor.yellow("\tWrong path of initial data. Using R05 for initial_data:'\n\t{}".format(initial_data))
                run = "R05"
            else:
                raise ValueError("found 'run':{} does not contain 'R'. in initial_data:{}".format(run, initial_data))
        #
        initial_data_dirname = initial_data.split("/")[-2]
        #
        pizza_fname = str(pizza_eos_fname.split("/")[-1])
        pizza_fname = pizza_fname[:-1]
        #
        hydro_fname = str(hydro_eos_fname[1:-1])
        #
        weak_fname = str(weak_eos_fname.split("/")[-1])
        weak_fname = weak_fname[:-1]

        # print(weak_fname)
        # print(hydro_fname)
        # print(pizza_fname)
        # exit(1)

        return run, initial_data_dirname, pizza_fname, hydro_fname, weak_fname

    def copy_extract_initial_data(self, finaldir='initial_data'):

        # if not os.path.isdir(Paths.ppr_sims + self.sim + '/' + finaldir):

        if self.lor_archive_fpath == None:

            run, dirnmame, pizza_fname, hydro_fname, weak_fname = self.get_fname_for_init_data_from_parfile()
            # print(run)
            if not os.path.isdir(Paths.lorene + run + '/'):
                self.is_init_data_available = False
                Printcolor.red("\tError. Init. data source dir is not found: {}"
                               .format(Paths.lorene + run + '/'))

            elif not os.path.isfile(Paths.lorene + run + '/' + dirnmame + '.tar.gz'):
                self.is_init_data_available = False
                Printcolor.red("\tError. Init. data archive is not found: {}"
                               .format(Paths.lorene + run + '/' + dirnmame + '.tar.gz'))

            else:
                self.is_init_data_available = True
                if not os.path.isdir(Paths.ppr_sims + self.sim + '/' + finaldir + '/'):
                    os.mkdir(Paths.ppr_sims + self.sim + '/' + finaldir + '/')

                # Andrea's fucking new approach
                if run == "R05":
                    os.system("tar -xzf {} --directory {}".format(
                        Paths.lorene + run + '/' + dirnmame + ".tar.gz",
                        Paths.ppr_sims + self.sim + "/" + finaldir + '/'
                    ))
                else:
                    os.system("rm -r {}".format(Paths.ppr_sims + self.sim + '/' + finaldir + '/'))
                    os.system("tar -xzf {} --directory {}".format(
                        Paths.lorene + run + '/' + dirnmame + ".tar.gz",
                        Paths.ppr_sims + self.sim + "/"
                    ))
                    os.system("mv {} {}".format(
                        Paths.ppr_sims + self.sim + '/' + dirnmame,
                        Paths.ppr_sims + self.sim + '/' + finaldir
                    ))

                Printcolor.blue("Initial data {} ({}) has been extracted and moved"
                                .format(dirnmame, run))
        else:
            raise IOError("Method to override the lor. data path is not written yet... ")

        return run, pizza_fname, hydro_fname, weak_fname

        # else:
        #     Printcolor.blue("\tinitial_data directory is found.")

    def get_pars_from_sim_name(self):

        parts = self.sim.split("_")
        # eos
        eos = parts[0]
        if not eos in self.list_expected_eos:
            print("Error in reading EOS from sim name "
                  "({} is not in the expectation list {})"
                  .format(eos, self.list_expected_eos))
            eos = ""
        self.par_dic["EOS"] = eos
        # m1m2
        m1m2 = parts[1]
        if m1m2[0] != 'M':
            print("Warning. m1m2 is not [1] component of name. Using [2] (run:{})".format(self.sim))
            # print("Warning. m1m2 is not [1] component of name. Using [2] (run:{})".format(run["name"]))
            m1m2 = parts[2]
        else:
            m1m2 = ''.join(m1m2[1:])
        try:
            m1 = float(''.join(m1m2[:4])) / 1000
            m2 = float(''.join(m1m2[4:])) / 1000
            if m1 < m2:
                _m1 = m1
                _m2 = m2
                m1 = _m2
                m2 = _m1
        except:
            print("Error in extracting m1m2 from sim name"
                  "({} is not separated into floats)"
                  .format(m1m2))
            m1 = 0.
            m2 = 0.
        self.par_dic["M1"] = m1
        self.par_dic["M2"] = m2

        # resolution
        resolution = []
        for part in parts:
            if part in self.list_expected_resolutions:
                resolution.append(part)
        if len(resolution) != 1:
            print("Error in getting resolution from simulation name"
                      "({} is not recognized)".format(resolution))
            resolution = [""]
        self.par_dic["res"] = resolution[0]

        # viscosity
        viscosity = []
        for part in parts:
            if part in self.list_expected_viscosities:
                viscosity.append(part)
        if len(viscosity) != 1:
            print("Note viscosity from simulation name is not extracted")
            viscosity = [""]
        self.par_dic["vis"] = viscosity[0]

        # q
        try:
            self.par_dic["q"] = float(self.par_dic["M1"]) / float(self.par_dic["M2"])
        except:
            print("Error in computing 'q' = m1/m2")
            self.par_dic["q"] = 0.

    def load_calcul_extract_pars(self, fname="calcul.d"):
        assert os.path.isfile(self.init_data_dir+fname)
        lines = open(self.init_data_dir+fname).readlines()
        # data_dic = {}
        for line in lines:

            # if not self.clean:
            #     print("\t\t{}".format(line))

            if line.__contains__("Baryon mass required for star 1"):
                self.par_dic["Mb1"] = float(line.split()[0]) # Msun

            if line.__contains__("Baryon mass required for star 2"):
                try:
                    self.par_dic["Mb2"] = float(line.split()[0]) # Msun
                except ValueError:
                    try:
                        self.par_dic["Mb2"] = float(line.split()[0][:5])
                    except ValueError:
                        try:
                            self.par_dic["Mb2"] = float(line.split()[0][:4])
                        except ValueError:
                            try:
                                self.par_dic["Mb2"] = float(line.split()[0][:3])
                            except:
                                raise ValueError("failed to extract Mb2")


            if line.__contains__("Omega") and line.__contains__("Orbital frequency"):
                self.par_dic["Omega"] = float(line.split()[2]) # rad/s

            if line.__contains__("Omega") and line.__contains__("Orbital frequency"):
                self.par_dic["Orbital freq"] = float(line.split()[8]) # Hz

            if line.__contains__("Coordinate separation"):
                self.par_dic["CoordSep"] = float(line.split()[3]) # rm

            if line.__contains__("1/2 ADM mass"):
                self.par_dic["MADM"] = 2*float(line.split()[4]) # Msun

            if line.__contains__("Total angular momentum"):
                self.par_dic["JADM"] = float(line.split()[4]) # [GMsun^2/c]

        if float(self.par_dic["Mb1"]) < float(self.par_dic["Mb2"]):
            _m1 = self.par_dic["Mb1"]
            _m2 = self.par_dic["Mb2"]
            self.par_dic["Mb1"] = _m2
            self.par_dic["Mb2"] = _m1

        # print(data_dic)
        self.par_dic["Mb"] = float(self.par_dic["Mb1"]) + float(self.par_dic["Mb2"])
        self.par_dic["f0"] = float(self.par_dic["Omega"]) / (2. * np.pi)

        # return data_dic

    def load_tov_extract_pars(self, tov_fpath):

        assert os.path.isfile(tov_fpath)

        from scipy import interpolate

        # tov_dic = {}
        tov_table = np.loadtxt(tov_fpath)

        m_grav = tov_table[:, 1]
        m_bary = tov_table[:, 2]
        r = tov_table[:, 3]
        comp = tov_table[:, 4]  # compactness
        kl = tov_table[:, 5]
        lamb = tov_table[:, 6]  # lam

        idx = np.argmax(m_grav)

        m_grav = m_grav[:idx]
        m_bary = m_bary[:idx]
        r = r[:idx]
        comp = comp[:idx]
        kl = kl[:idx]
        lamb = lamb[:idx]

        interp_grav_bary = interpolate.interp1d(m_bary, m_grav, kind='cubic')
        interp_lamb_bary = interpolate.interp1d(m_bary, lamb, kind='cubic')
        interp_comp_bary = interpolate.interp1d(m_bary, comp, kind='cubic')
        interp_k_bary = interpolate.interp1d(m_bary, kl, kind='cubic')
        interp_r_bary = interpolate.interp1d(m_bary, r, kind='cubic')

        if self.par_dic["Mb1"] != '':
            self.par_dic["lam21"] = float(interp_lamb_bary(float(self.par_dic["Mb1"])))  # lam21
            self.par_dic["Mg1"] = float(interp_grav_bary(float(self.par_dic["Mb1"])))
            self.par_dic["C1"] = float(interp_comp_bary(float(self.par_dic["Mb1"])))  # C1
            self.par_dic["k21"] = float(interp_k_bary(float(self.par_dic["Mb1"])))
            self.par_dic["R1"] = float(interp_r_bary(float(self.par_dic["Mb1"])))
            # run["R1"] = run["M1"] / run["C1"]

        if self.par_dic["Mb2"] != '':
            self.par_dic["lam22"] = float(interp_lamb_bary(float(self.par_dic["Mb2"])))  # lam22
            self.par_dic["Mg2"] = float(interp_grav_bary(float(self.par_dic["Mb2"])))
            self.par_dic["C2"] = float(interp_comp_bary(float(self.par_dic["Mb2"])))  # C2
            self.par_dic["k22"] = float(interp_k_bary(float(self.par_dic["Mb2"])))
            self.par_dic["R2"] = float(interp_r_bary(float(self.par_dic["Mb2"])))
            # run["R2"] = run["M2"] / run["C2"]

        if self.par_dic["Mg1"] != '' and self.par_dic["Mg2"] != '':
            mg1 = float(self.par_dic["Mg1"])
            mg2 = float(self.par_dic["Mg2"])
            mg_tot = mg1 + mg2
            k21 = float(self.par_dic["k21"])
            k22 = float(self.par_dic["k22"])
            c1 = float(self.par_dic["C1"])
            c2 = float(self.par_dic["C2"])
            lam1 = float(self.par_dic["lam21"])
            lam2 = float(self.par_dic["lam22"])

            kappa21 = 2 * ((mg1 / mg_tot) ** 5) * (mg2 / mg1) * (k21 / (c1 ** 5))

            kappa22 = 2 * ((mg2 / mg_tot) ** 5) * (mg1 / mg2) * (k22 / (c2 ** 5))

            self.par_dic["k2T"] = kappa21 + kappa22

            tmp1 = (mg1 + (12 * mg2)) * (mg1 ** 4) * lam1
            tmp2 = (mg2 + (12 * mg1)) * (mg2 ** 4) * lam2
            self.par_dic["Lambda"] = (16. / 13.) * (tmp1 + tmp2) / (mg_tot ** 5.)

    def save_as_csv(self, fname):
        # import csv
        w = csv.writer(open(Paths.ppr_sims+self.sim+'/'+fname, "w"))
        for key, val in self.par_dic.items():
            w.writerow([key, val])


class LOAD_INIT_DATA:

    def __init__(self, sim):


        self.list_v_ns = ["f0", "JADM", "k21", "k2T", "EOS", "M1", "M2",
                          "CorrdSep", "k22", "res", "vis", "MADM", "C2", "C1",
                          "Omega", "Mb1", "Mb2", "R1", "R2", "Mb", "Lambda",
                          "lam21","lam22", "q","Mg2", "Mg1", "Orbital freq",
                          "run", "weak_eos", "hydro_eos", "pizza_eos"]
        self.sim = sim
        self.par_dic = {}
        self.fname = "init_data.csv"
        self.load_csv(self.fname)

    def load_csv(self, fname):
        # import csv
        if not os.path.isfile(Paths.ppr_sims+self.sim+'/'+fname):
            print("Error: initial data is not extracted for: {}".format(self.sim))
        # reader = csv.DictReader(open(Paths.ppr_sims+self.sim+'/'+fname))
        # for row in reader:
        #     print(row)
        with open(Paths.ppr_sims+self.sim+'/'+fname, 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                if len(row):
                    self.par_dic[row[0]] = row[1]
                # print(row)

        # print(self.par_dic.keys())

    def get_par(self, v_n):
        if not v_n in self.par_dic.keys():
            print("Error. v_n:{} sim:{} is not in init_data.keys()\n{}"
                  .format(v_n, self.sim, self.par_dic))
        if not v_n in self.list_v_ns:
            raise NameError("v_n:{} sim:{} not in self.list_v_ns[] {} \n\nUpdate the list."
                            .format(v_n, self.sim, self.list_v_ns))

        par = self.par_dic[v_n]
        try:
            return float(par)
        except:
            return par


class COLLATE_DATA(LOAD_ITTIME):

    def __init__(self, sim):

        LOAD_ITTIME.__init__(self, sim)

        self.all_fnames = Lists.collate_list
        self.all_outputs = self.get_list_outputs()
        self.outdir = Paths.ppr_sims+'/'+sim+'/collated/'
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)

        self.tmax = inf         # Maximum time to include (default: inf)
        self.epsilon = 1e-15    # Precision used in comparing timestamps
        self.tidx = 1           # Index of the time column, from 1 (default: 1)

        self.collate(glob_overwrite)

    def __collate(self, list_of_files, fname, comment, include_comments=True):

        ofile = open(self.outdir+fname, 'w')

        told = None
        for fpath in list_of_files:
            for dline in open(fpath, 'r'):
                skip = False
                for c in comment:
                    if dline[:len(c)] == c:
                        if include_comments:
                            ofile.write(dline)
                        skip = True
                        break
                if len(dline.split()) == 0:
                    skip = True
                if skip:
                    continue

                tnew = float(dline.split()[self.tidx - 1])
                if tnew > self.tmax:
                    break
                if told is None or tnew > told * (1 + self.epsilon):
                    ofile.write(dline)
                    told = tnew

        ofile.close()

    def collate(self, rewrite=False):
        for fname in self.all_fnames:
            output_files = []
            for output in self.all_outputs:
                fpath = Paths.gw170817+self.sim+'/'+output+'/data/'+fname
                if os.path.isfile(fpath):
                    output_files.append(fpath)
                else:
                    Printcolor.yellow("\tFile not found: {}".format(fpath))
            # assert len(output_files) > 0
            if len(output_files) > 0:
                fpath = self.outdir + fname
                try:
                    if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                        if os.path.isfile(fpath): os.remove(fpath)
                        Printcolor.print_colored_string(
                            ["Task:", "collate", "file:", "{}".format(fname),":", "Executing..."],
                            ["blue", "green", "blue", "green","", "green"])
                        # -------------------------------------------------
                        self.__collate(output_files, fname, ['#'], True)
                        # -------------------------------------------------
                    else:
                        Printcolor.print_colored_string(
                            ["Task:", "colate", "file:", "{}".format(fname),":", "skipping..."],
                            ["blue", "green", "blue", "green","", "blue"])
                except KeyboardInterrupt:
                    exit(1)
                except:
                    Printcolor.print_colored_string(
                        ["Task:", "colate", "file:", "{}".format(fname),":", "failed..."],
                        ["blue", "green", "blue", "green","", "red"])
            else:
                Printcolor.print_colored_string(
                    ["Task:", "colate", "file:", "{}".format(fname), ":", "no files found..."],
                    ["blue", "green", "blue", "green", "", "red"])



if __name__ == '__main__':

    parser = ArgumentParser(description="postprocessing pipeline")
    parser.add_argument("-s", dest="sim", required=True, help="name of the simulation dir")
    parser.add_argument("-t", dest="tasklist", nargs='+', required=False, default=[], help="list of tasks to to")
    #
    parser.add_argument("-o", dest="outdir", required=False, default=Paths.ppr_sims, help="path for output dir")
    parser.add_argument("-i", dest="simdir", required=False, default=Paths.gw170817, help="path to simulation dir")
    parser.add_argument("--overwrite", dest="overwrite", required=False, default="no", help="overwrite if exists")
    #
    parser.add_argument("--lorene", dest="lorene", required=False, default=None,
                        help="path to lorene .tar.gz arxive")
    parser.add_argument("--tov", dest="tov", required=False, default=None, help="path to TOVs (EOS_love.dat) file")
    #


    #
    # parser.add_argument("--v_n", dest="v_n", required=False, default='no', help="variable (or group) name")
    # parser.add_argument("--rl", dest="rl", required=False, default=-1, help="reflevel")
    # parser.add_argument("--it", dest="it", required=False, default=-1, help="iteration")
    # parser.add_argument('--times', nargs='+', help='Timesteps to use', required=False)
    # parser.add_argument("--sym", dest="symmetry", required=False, default=None, help="symmetry (like 'pi')")
    # parser.add_argument("--crits", dest="criteria", nargs='+', required=False, default=[],
    #                     help="criteria to use (like _0 ...)")

    args = parser.parse_args()
    glob_sim = args.sim
    glob_simdir = args.simdir
    glob_outdir = args.outdir
    glob_tasklist = args.tasklist
    glob_overwrite = args.overwrite
    glob_lorene = args.lorene
    glob_tov = args.tov


    # check given data

    if not os.path.isdir(glob_simdir + glob_sim):
        raise NameError("simulation dir: {} does not exist in rootpath: {} "
                        .format(glob_sim, glob_simdir))
    if len(glob_tasklist) == 0:
        raise NameError("tasklist is empty. Set what tasks to perform with '-t' option")
    else:
        for task in glob_tasklist:
            if task not in __preanalysis__["tasklist"]:
                raise NameError("task: {} is not among available ones: {}"
                                .format(task, __preanalysis__["tasklist"]))
    if glob_overwrite == "no":  glob_overwrite = False
    elif glob_overwrite == "yes": glob_overwrite = True
    else: raise NameError("for '--overwrite' option use 'yes' or 'no'. Given: {}"
                          .format(glob_overwrite))
    glob_outdir_sim = Paths.ppr_sims + glob_sim
    if not os.path.isdir(glob_outdir_sim):
        os.mkdir(glob_outdir_sim)
    if glob_lorene != None:
        if not os.path.isfile(glob_lorene):
            raise NameError("Given lorene fpath: {} is not avialable"
                            .format(glob_lorene))
    if glob_tov != None:
        if not os.path.isfile(glob_tov):
            raise NameError("Given TOV fpath: {} is not avialable"
                            .format(glob_tov))

    # set globals
    Paths.gw170817 = glob_simdir
    Paths.ppr_sims = glob_outdir

    # do tasks



    for task in glob_tasklist:
        if task == "update_status":
            Printcolor.blue("Task:'{}' Executing...".format(task))
            statis = SIM_STATUS(glob_sim, clean=True, save=True)
            Printcolor.blue("Task:'{}' DONE...".format(task))
        elif task == "collate":
            COLLATE_DATA(glob_sim)
        elif task == "print_status":
            Printcolor.blue("Task:'{}' Executing...".format(task))
            statis = PRINT_SIM_STATUS(glob_sim)
            Printcolor.blue("Task:'{}' DONE...".format(task))
        elif task == "init_data":
            Printcolor.blue("Task:'{}' Executing...".format(task))
            statis = INIT_DATA(glob_sim, clean=True, lor_archive_fpath=glob_lorene)
            Printcolor.blue("Task:'{}' DONE...".format(task))
        else:
            raise NameError("No method fund for task: {}".format(task))

    # self = SIM_SELF_PARS("LS220_M13641364_M0_LK_SR")

    # INIT_DATA("DD2_M13641364_M0_LK_SR_R04")
    # l = LOAD_INIT_DATA("DD2_M13641364_M0_LK_SR_R04")
    # print(l.get_par("Lambda"))
    # print(self.param_dic["initial_data_run"])
    # print(self.param_dic["initial_data_fname"])
