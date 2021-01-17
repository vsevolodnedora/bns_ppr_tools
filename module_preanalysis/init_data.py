#!/usr/bin/env python

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

list_expected_eos = [
    "SFHo",
    "SLy4",
    "DD2",
    "BLh",
    "LS220",
    "BHB",
    "BHBlp"
]

list_expected_resolutions = [
    "HR",
    "LR",
    "SR",
    "VLR"
]

list_expected_viscosities = [
    "LK",
    "L50",
    "L25",
    "L5"
]

list_expected_neutrinos = [
    "M0",
    "M1"
]

list_expected_initial_data = [
    "R01",
    "R02",
    "R03",
    "R04",
    "R05",
    "R04_corot"
]

list_tov_seq = {
    "SFHo": "SFHo_sequence.txt",
    "LS220":"LS220_sequence.txt",
    "DD2": "DD2_sequence.txt",
    "BLh": "BLh_sequence.txt",
    "SLy4": "SLy4_sequence.txt",
    "BHBlp": "BHBlp_love.dat"
}

# lorene TOV data
class INIT_DATA:

    def __init__(self, sim, indir=None, outdir=None):

        self.sim = sim
        self.par_dic = {}
        # ---
        self.extract_parameters_from_sim_name()
        # ---
        self.in_simdir = indir   #simdir + sim + '/'
        self.ppr_dir = outdir #pprdir + sim + '/'

        assert os.path.isdir(indir)
        assert os.path.isdir(outdir)

        # locate or transfer parfile
        if not os.path.isfile(self.ppr_dir + "parfile.par"):
            # find parfile:
            listdirs = os.listdir(self.in_simdir)
            for dir_ in listdirs:
                print("searching for parfile in {}".format(self.in_simdir+dir_ + '/'))
                if dir_.__contains__("output-"):
                    if os.path.isfile(self.in_simdir+dir_ + '/' + 'parfile.par'):
                        os.system("cp {} {}".format(self.in_simdir + dir_ + '/' +'parfile.par', self.ppr_dir))
                        print("\tparfile is copied from {}".format(self.in_simdir + dir_ + '/'))
                        break
        else:
            print("\tparfile is already collected")
        if not os.path.isfile(self.ppr_dir + "parfile.par"):
            raise IOError("parfile is neither found nor copied from source.")
        # ---
        initial_data_line = self.extract_parameters_from_parfile()
        # ---

        if not os.path.isdir(self.ppr_dir + "initial_data/") or \
            not os.path.isfile(self.ppr_dir + "initial_data/" + "calcul.d") or \
            not os.path.isfile(self.ppr_dir + "initial_data/" + "resu.d"):
            # make a dir for the lorene data
            if not os.path.isdir(self.ppr_dir + "initial_data/"):
                os.mkdir(self.ppr_dir + "initial_data/")
            # find extract and copy lorene files
            archive_fpath = self.find_untar_move_lorene_files(initial_data_line)
            self.extract_lorene_archive(archive_fpath, self.ppr_dir + "initial_data/")
            # check again
            if not os.path.isdir(self.ppr_dir + "initial_data/") or \
                    not os.path.isfile(self.ppr_dir + "initial_data/" + "calcul.d") or \
                    not os.path.isfile(self.ppr_dir + "initial_data/" + "resu.d"):
                raise IOError("Failed to extract, copy lorene data: {} \ninto {}"
                              .format(archive_fpath, self.ppr_dir + "initial_data/"))
        else:
            pass

        # get masses, lambdas, etc
        self.extract_parameters_from_calculd(self.ppr_dir + "initial_data/" + "calcul.d")
        #
        tov_fname = list_tov_seq[self.par_dic["EOS"]]
        self.extract_parameters_from_tov_sequences(Paths.TOVs + tov_fname)
        #
        self.save_as_csv(self.ppr_dir + "init_data.csv")


    # get the files

    def extract_parameters_from_parfile(self):

        initial_data = ""
        pizza_eos_fname = ""
        hydro_eos_fname = ""
        weak_eos_fname = ""
        #
        lines = open(self.ppr_dir + 'parfile.par', "r").readlines()
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
        self.par_dic["hydro_eos"] = str(hydro_eos_fname[1:-1])
        self.par_dic["pizza_eos"] = str(pizza_eos_fname.split("/")[-1])[:-1]
        self.par_dic["weak_eos"]  = str(weak_eos_fname.split("/")[-1])[:-1]
        #
        return initial_data

        #
        # #
        # run = initial_data.split("/")[-3]
        # initial_data_archive_name = initial_data.split("/")[-2]
        # if not run.__contains__("R"):
        #     if str(initial_data.split("/")[-2]).__contains__("R05"):
        #         Printcolor.yellow(
        #             "\tWrong path of initial data. Using R05 for initial_data:'\n\t{}".format(initial_data))
        #         run = "R05"
        #         initial_data_archive_name = initial_data.split("/")[-2]
        #     else:
        #         for n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        #             _run = "R0{:d}".format(n)
        #             if os.path.isdir(Paths.lorene + _run + '/'):
        #                 _masses = self.sim.split('_')[1]
        #                 assert _masses.__contains__("M")
        #                 _masses.replace('M', '')
        #                 _lorene = Paths.lorene + _run + '/'
        #                 onlyfiles = [f for f in os.listdir(_lorene) if os.path.isfile(os.path.join(_lorene, f))]
        #                 assert len(onlyfiles) > 0
        #                 for onefile in onlyfiles:
        #                     if onefile.__contains__(_masses):
        #                         initial_data_archive_name = onefile.split('.')[0]
        #                         run = _run
        #                         break
        #         if run == initial_data.split("/")[-3]:
        #             Printcolor.yellow("Filed to extract 'run': from: {}".format(initial_data))
        #             Printcolor.yellow("Manual overwrite required")
        #             manual = raw_input("set run (e.g. R01): ")
        #             if str(manual) == "":
        #                 raise NameError("Filed to extract 'run': from: {}".format(initial_data))
        #             else:
        #                 Printcolor.yellow("Setting Run manually to: {}".format(manual))
        #                 run = str(manual)
                # raise ValueError("found 'run':{} does not contain 'R'. in initial_data:{}".format(run, initial_data))
        #

        #
        # pizza_fname = str(pizza_eos_fname.split("/")[-1])
        #
        # pizza_fname = pizza_fname[:-1]
        # #
        # hydro_fname = str(hydro_eos_fname[1:-1])
        # #
        # weak_fname = str(weak_eos_fname.split("/")[-1])
        # weak_fname = weak_fname[:-1]

    def find_untar_move_lorene_files(self, line_from_parfile):
        #
        run = ""
        lorene_archive_fpath = ""
        # if line cotains the run R01 - R05
        for expected_run in list_expected_initial_data:
            if line_from_parfile.__contains__(expected_run):
                run = expected_run
                print("found run: {} in the line: {}".format(run, line_from_parfile))
                break
        # if this run is found, check if there an archive with matching mass. If not, check ALL runs for this archive
        if run != "":
            _masses = self.sim.split('_')[1]
            _lorene = Paths.lorene + run + '/'
            onlyfiles = [f for f in os.listdir(_lorene) if os.path.isfile(os.path.join(_lorene, f))]
            for onefile in onlyfiles:
                if onefile.__contains__(_masses):
                    lorene_archive_fpath = Paths.lorene + run + '/' + onefile#.split('.')[0]
                    print("found file {} in run: {}".format(lorene_archive_fpath, run))
                    break
            if lorene_archive_fpath == "":
                print("failed to find lorene archive for run: {} in {}"
                      .format(run, _lorene))
            else:
                if not os.path.isfile(lorene_archive_fpath):
                    print("file does not exist: {} Continue searching...".format(lorene_archive_fpath))
                    lorene_archive_fpath = ""
        else:
            print("failed to find run (R0?) in {} . Trying to check ALL the list...".format(line_from_parfile))
            for __run in list_expected_initial_data:
                print("checking {}".format(__run))
                _lorene = Paths.lorene + __run + '/'
                onlyfiles = [f for f in os.listdir(_lorene) if os.path.isfile(os.path.join(_lorene, f))]
                assert len(onlyfiles) > 0
                _masses = self.sim.split('_')[1]
                for onefile in onlyfiles:
                    if onefile.__contains__(_masses):
                        lorene_archive_fpath = Paths.lorene + __run + '/' + onefile#.split('.')[0]
                        run = __run
                        print("found file {} in run: {}".format(lorene_archive_fpath, run))
                        break
        # if the archive is found -- return; if NOT or if does not exist: ask user
        if run != "" and lorene_archive_fpath != "":
            if os.path.isfile(lorene_archive_fpath):
                self.par_dic["run"] = run
                return lorene_archive_fpath
            else:
                print("run: {} is found, but file does not exist: {} "
                      .format(run, lorene_archive_fpath))
        else:
            print("failed to find run '{}' or/and archive name: '{}' ".format(run, lorene_archive_fpath))
        # get run from the user, showing him the line
        manual = raw_input("set run (e.g. R01): ")
        if not manual in list_expected_initial_data:
            print("Note: given run: {} is not in the list of runs:\n\t{}"
                  .format(manual, list_expected_initial_data))
        run = manual
        # get the archive name from the user
        manual = raw_input("archive name (e.g. SLy_1264_R45.tar.gz): ")
        if not os.path.isfile(Paths.lorene + run + '/' + manual):
            print("Error: given run {} + archive name {} -> file does not exists: {}"
                  .format(run, manual, Paths.lorene + run + '/' + manual))
            raise IOError("file not found:{}".format(Paths.lorene + run + '/' + manual))
        lorene_archive_fpath = Paths.lorene + run + '/' + manual
        self.par_dic["run"] = run
        return lorene_archive_fpath

    def extract_lorene_archive(self, archive_fpath, new_dir_fpath):
        #
        assert os.path.isdir(new_dir_fpath)
        assert os.path.isfile(archive_fpath)
        #
        run = self.par_dic["run"]
        if run == "R05":
            # andrea's fucking approach
            os.system("tar -xzf {} --directory {}".format(archive_fpath, new_dir_fpath))
        else:
            tmp = archive_fpath.split('/')[-1]
            tmp = tmp.split('.')[0]
            # os.mkdir(new_dir_fpath + 'tmp/')
            os.system("tar -xzf {} --directory {}".format(archive_fpath, new_dir_fpath))
            os.system("mv {} {}".format(new_dir_fpath + tmp + '/*', new_dir_fpath))
            os.rmdir(new_dir_fpath + tmp + '/')

    # extract data

    def extract_parameters_from_sim_name(self):

        parts = self.sim.split("_")
        # eos
        eos = parts[0]
        if not eos in list_expected_eos:
            print("Error in reading EOS from sim name "
                  "({} is not in the expectation list {})"
                  .format(eos, list_expected_eos))
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
            if part in list_expected_resolutions:
                resolution.append(part)
        if len(resolution) != 1:
            print("Error in getting resolution from simulation name"
                      "({} is not recognized)".format(resolution))
            resolution = [""]
        self.par_dic["res"] = resolution[0]

        # viscosity
        viscosity = []
        for part in parts:
            if part in list_expected_viscosities:
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

    def extract_parameters_from_calculd(self, fpath):

        # print fpath; exit(1)

        assert os.path.isfile(fpath)
        lines = open(fpath).readlines()
        # data_dic = {}
        grav_masses = []
        for line in lines:
            if line.__contains__("Gravitational mass :"):
                strval = ''.join(line.split("Gravitational mass :")[-1])
                val = float(strval.split()[-2])
                grav_masses.append(val)
        if len(grav_masses) != 2:
            print("Error! len(gravmasses)!=2")
            raise ValueError("Error! len(gravmasses)!=2")

        # self.par_dic["Mg1"] = np.min(np.array(grav_masses))
        # self.par_dic["Mg2"] = np.max(np.array(grav_masses))

        # baryonic masses
        bar_masses = [0, 0]
        for line in lines:

            # if not self.clean:
            # print("\t\t{}".format(line))

            if line.__contains__("Baryon mass required for star 1"):
                try:
                    bar_masses[0] = float(line.split()[0])  # Msun
                except ValueError:
                    try:
                        bar_masses[0] = float(line.split()[0][:5])
                    except ValueError:
                        try:
                            bar_masses[0] = float(line.split()[0][:4])
                        except ValueError:
                            try:
                                bar_masses[0] = float(line.split()[0][:3])
                            except:
                                raise ValueError("failed to extract Mb2")

                # self.par_dic["Mb1"] = float(line.split()[0])  # Msun

            if line.__contains__("Baryon mass required for star 2"):
                try:
                    bar_masses[1] = float(line.split()[0])  # Msun
                except ValueError:
                    try:
                        bar_masses[1] = float(line.split()[0][:5])
                    except ValueError:
                        try:
                            bar_masses[1] = float(line.split()[0][:4])
                        except ValueError:
                            try:
                                bar_masses[1] = float(line.split()[0][:3])
                            except:
                                raise ValueError("failed to extract Mb2")

            if line.__contains__("Omega") and line.__contains__("Orbital frequency"):
                self.par_dic["Omega"] = float(line.split()[2])  # rad/s

            if line.__contains__("Omega") and line.__contains__("Orbital frequency"):
                self.par_dic["Orbital freq"] = float(line.split()[8])  # Hz

            if line.__contains__("Coordinate separation"):
                self.par_dic["CoordSep"] = float(line.split()[3])  # rm

            if line.__contains__("1/2 ADM mass"):
                self.par_dic["MADM"] = 2 * float(line.split()[4])  # Msun

            if line.__contains__("Total angular momentum"):
                self.par_dic["JADM"] = float(line.split()[4])  # [GMsun^2/c]

        #
        self.par_dic["Mb1"] = np.max(np.array(bar_masses))
        self.par_dic["Mb2"] = np.min(np.array(bar_masses))

        # if float(self.par_dic["Mb1"]) < float(self.par_dic["Mb2"]):
        #     _m1 = self.par_dic["Mb1"]
        #     _m2 = self.par_dic["Mb2"]
        #     self.par_dic["Mb1"] = _m2
        #     self.par_dic["Mb2"] = _m1



        # print(data_dic)
        self.par_dic["Mb"] = float(self.par_dic["Mb1"]) + float(self.par_dic["Mb2"])
        self.par_dic["f0"] = float(self.par_dic["Omega"]) / (2. * np.pi)

    def extract_parameters_from_tov_sequences(self, tov_fpath):

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

        interp_grav_bary = interpolate.interp1d(m_bary, m_grav, kind='linear')
        interp_lamb_bary = interpolate.interp1d(m_bary, lamb, kind='linear')
        interp_comp_bary = interpolate.interp1d(m_bary, comp, kind='linear')
        interp_k_bary = interpolate.interp1d(m_bary, kl, kind='linear')
        interp_r_bary = interpolate.interp1d(m_bary, r, kind='linear')

        if self.par_dic["Mb1"] != '':
            self.par_dic["lam21"] = float(interp_lamb_bary(float(self.par_dic["Mb1"])))  # lam21
            self.par_dic["Mg1"] = float(interp_grav_bary(float(self.par_dic["Mb1"]))) # -> from lorene
            self.par_dic["C1"] = float(interp_comp_bary(float(self.par_dic["Mb1"])))  # C1
            self.par_dic["k21"] = float(interp_k_bary(float(self.par_dic["Mb1"])))
            self.par_dic["R1"] = float(interp_r_bary(float(self.par_dic["Mb1"])))
            # run["R1"] = run["M1"] / run["C1"]

        if self.par_dic["Mb2"] != '':
            self.par_dic["lam22"] = float(interp_lamb_bary(float(self.par_dic["Mb2"])))  # lam22
            self.par_dic["Mg2"] = float(interp_grav_bary(float(self.par_dic["Mb2"]))) # -> from lorene
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

    # saving

    def save_as_csv(self, fpath):
        # import csv
        w = csv.writer(open(fpath, "w"))
        for key, val in self.par_dic.items():
            w.writerow([key, val])

# get init_data.csv
class LOAD_INIT_DATA:

    def __init__(self, sim, pprdir=None):


        self.list_v_ns = ["f0", "JADM", "k21", "k2T", "EOS", "M1", "M2",
                          "CorrdSep", "k22", "res", "vis", "MADM", "C2", "C1",
                          "Omega", "Mb1", "Mb2", "R1", "R2", "Mb", "Lambda",
                          "lam21","lam22", "q","Mg2", "Mg1", "Orbital freq",
                          "run", "weak_eos", "hydro_eos", "pizza_eos"]
        self.sim = sim
        self.ppr_dir = pprdir
        self.par_dic = {}
        self.fname = "init_data.csv"
        self.load_csv(self.fname)

    def load_csv(self, fname):
        # import csv
        if not os.path.isfile(self.ppr_dir+fname):
            print("Error: initial data is not extracted for: {}".format(self.ppr_dir+fname))
        # reader = csv.DictReader(open(Paths.ppr_sims+self.sim+'/'+fname))
        # for row in reader:
        #     print(row)
        with open(self.ppr_dir+fname, 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                if len(row):
                    self.par_dic[row[0]] = row[1]
                # print(row)

        # print(self.par_dic.keys())

    def get_par(self, v_n):
        if not v_n in self.par_dic.keys():
            print("\tError. v_n:{} sim:{} is not in init_data.keys()\n\t{}"
                  .format(v_n, self.sim, self.par_dic))
        if not v_n in self.list_v_ns:
            raise NameError("v_n:{} sim:{} not in self.list_v_ns[] {} \n\nUpdate the list."
                            .format(v_n, self.sim, self.list_v_ns))

        # if v_n == "Mb":
        #     return float(self.get_par("Mb1") + self.get_par("Mb2"))
        # print(v_n, self.sim, self.par_dic.keys(), '\n')
        par = self.par_dic[v_n]
        try:
            return float(par)
        except:
            return par

