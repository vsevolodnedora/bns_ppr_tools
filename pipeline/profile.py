###################################################################################
#                                                                                 #
# This is a comprehensive set of analysis methods and tools                       #
# for the standart output of the Neutron Star Merger simulations                  #
# done with WhiskyTHC code.                                                       #
#                                                   #
###################################################################################
from __future__ import division

import click
import h5py
import sys
import numpy as np
import os
import re
from argparse import ArgumentParser

from uutils import Printcolor

import config as Paths

from module_preanalysis.it_time import LOAD_ITTIME

from uutils import Tools, Labels

from module_profile.profile_grids import (POLAR_GRID, CARTESIAN_GRID, CYLINDRICAL_GRID)
from module_profile.profile_methods import (MASK_STORE, MAINMETHODS_STORE, INTMETHODS_STORE, get_time_for_it)
from module_profile.profile_results import (LOAD_DENSITY_MODES, LOAD_RES_CORR)
from module_profile.profile_slice_methods import (MAINMETHODS_STORE_SLICE)
from hist_bins import (get_hist_bins, get_corr_dic, get_reflev_borders)
from plotting.plotting_methods import PLOT_MANY_TASKS


__tasklist__ = ["all", "corr", "hist", "slice", "mass", "densmode", "vtk", "densmode",
                "densmodeint", "mjenclosed",
                "plotall", "plotcorr", "plotslicecorr", "plothist", "plotslice", "plotmass", "slicecorr",
                "plotdensmode", "plotcenterofmass", "plotdensmodephase"
                ]

__masks__ = ["disk", "remnant", "total"]

__d3slicesvns__ = ["x", "y", "z", "rho", "w_lorentz", "vol", "press", "entr", "eps", "lapse", "velx", "vely", "velz",
                    "gxx", "gxy", "gxz", "gyy", "gyz", "gzz", "betax", "betay", "betaz", 'temp', 'Ye'] + \
                  ["u_0", "density",  "enthalpy", "vphi", "vr", "dens_unb_geo", "dens_unb_bern", "dens_unb_garch",
                    "ang_mom", "ang_mom_flux", "theta", "r", "phi", "hu_0"]

__d3corrs__ = ["rho_r", "rho_Ye", "r_Ye", "temp_Ye", "rho_temp", "rho_theta", "velz_theta", "rho_ang_mom", "velz_Ye",
               "rho_ang_mom_flux", "rho_dens_unb_bern", "ang_mom_flux_theta",
               "ang_mom_flux_dens_unb_bern", "inv_ang_mom_flux_dens_unb_bern",
               "velz_dens_unb_bern", "Ye_dens_unb_bern", "theta_dens_unb_bern",
               "hu_0_ang_mom", "hu_0_ang_mom_flux", "hu_0_Ye", "hu_0_temp", "hu_0_entr", "Ye_entr" #"hu_0_pressure"
               ]

__d2corrs__ = [ "Q_eff_nua_u_0", "Q_eff_nua_hu_0", "Q_eff_nua_dens_unb_bern",
               "Q_eff_nua_over_density_hu_0", "Q_eff_nua_over_density_theta", "Q_eff_nua_over_density_Ye",
               "Q_eff_nua_Ye", "velz_Ye"]

__d3histvns__      = ["r", "theta", "Ye", "entr", "temp", "velz", "rho", "dens_unb_bern", "press"]

__d3slicesplanes__ = ["xy", "xz"]

# __d3diskmass__ = "disk_mass.txt"
__d3remnantmass__ = "remnant_mass.txt"
__d3intmjfname__ = "MJ_encl.txt"
__d3densitymodesfame__ = "density_modes_lap15.h5"
__center_of_mass_plotname__ = "center_of_mass.png"

__d3sliceplotvns__ = ["Ye", "velx", "rho", "ang_mom_flux","ang_mom","dens_unb_garch",
                      "dens_unb_bern","dens_unb_geo","vr","vphi","enthalpy",
                        "density","temp","velz","vely","lapse","entr","eps","press","vol","w_lorentz",
                      "Q_eff_nua", "Q_eff_nue", "Q_eff_nux"]
__d3sliceplotrls__ = [0, 1, 2, 3, 4, 5, 6]

__rootoutdir__ = "profiles/"




# tools
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

# for iteration
def d3_mass_for_it(it, d3corrclass, mask, outdir, rewrite=False):
    # disk

    fpath = outdir + "mass.txt"

    def task(fpath):
        mass = d3corrclass.get_total_mass(it, multiplier=2., mask_v_n=mask)
        np.savetxt(fname=fpath, X=np.array([mass]))
        return mass

    if Paths.debug:
        task(fpath)
    else:
        try:
            if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                if os.path.isfile(fpath): os.remove(fpath)
                print_colored_string(["task:", "disk_mass", "it:", "{}".format(it), "mask:", mask, ":", "computing"],
                                     ["blue", "green", "blue", "green", "blue", "green", "", "green"])
                mass = task(fpath)
                if mass == 0.:  Printcolor.yellow("\tComputed disk mass = 0.")
        except IOError:
            print_colored_string(["task:", "disk_mass", "it:", "{}".format(it), "mask:", mask, ":", "IOError"],
                             ["blue", "green", "blue", "green", "blue", "green", "", "red"])
        except:
            print_colored_string(["task:", "disk_mass", "it:", "{}".format(it), "mask:", mask, ":", "Error"],
                             ["blue", "green", "blue", "green", "blue", "green", "", "red"])
#
# def d3_remnant_mass_for_it(it, d3corrclass, outdir, rewrite=False):
#     fpath = outdir + __d3remnantmass__
#     try:
#         if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
#             if os.path.isfile(fpath): os.remove(fpath)
#             print_colored_string(["task:", "remnant_mass", "it:", "{}".format(it), ":", "computing"],
#                                  ["blue", "green", "blue", "green", "", "green"])
#             mass = d3corrclass.get_total_mass(it, multiplier=2., mask_v_n="remnant")
#             np.savetxt(fname=fpath, X=np.array([mass]))
#             if mass == 0.:
#                 Printcolor.yellow("\tComputed remnant mass = 0.")
#         else:
#             print_colored_string(["task:", "remnant_mass", "it:", "{}".format(it), ":", "skipping"],
#                                  ["blue", "green", "blue", "green", "", "blue"])
#     except KeyboardInterrupt:
#         exit(1)
#     except IOError:
#         print_colored_string(["task:", "remnant_mass", "it:", "{}".format(it), ":", "IOError"],
#                          ["blue", "green", "blue", "green", "", "red"])
#     except:
#         print_colored_string(["task:", "remnant_mass", "it:", "{}".format(it), ":", "Error"],
#                          ["blue", "green", "blue", "green", "", "red"])
def d3_hist_for_it(it, d3corrclass, mask, glob_v_ns, outdir, rewrite=False):

    selected_vn1vn2s = select_string(glob_v_ns, __d3histvns__)

    def task(v_n, hist_dic, it, mask, fpath):
        edges_weights = d3corrclass.get_histogram(it, hist_dic, mask=mask, multiplier=2.)
        np.savetxt(fname=fpath, X=edges_weights, header="# {}   mass".format(v_n))

    for v_n in selected_vn1vn2s:

        hist_dic = {"v_n": v_n, "edges": get_hist_bins(mask, v_n)}

        # chose a dic
        # if v_n == "r":
        #     hist_dic = d3corrclass.hist_task_dic_r
        # elif v_n == "theta":
        #     hist_dic = d3corrclass.hist_task_dic_theta
        # elif v_n == "Ye":
        #     hist_dic = d3corrclass.hist_task_dic_ye
        # elif v_n == "velz":
        #     hist_dic = d3corrclass.hist_task_dic_velz
        # elif v_n == "temp":
        #     hist_dic = d3corrclass.hist_task_dic_temp
        # elif v_n == "rho" and mask == "disk":
        #     hist_dic = d3corrclass.hist_task_dic_rho_d
        # elif v_n == "rho" and mask == "remnant":
        #     hist_dic = d3corrclass.hist_task_dic_rho_r
        # elif v_n == "dens_unb_bern":
        #     hist_dic = d3corrclass.hist_task_dens_unb_bern
        # elif v_n == "press":
        #     hist_dic = d3corrclass.hist_task_pressure
        # elif v_n == "entr" and mask == "disk":
        #     hist_dic = d3corrclass.hist_task_dic_entropy_d
        # elif v_n == "entr" and mask == "remnant":
        #     hist_dic = d3corrclass.hist_task_dic_entropy_r
        # else:
        #     raise NameError("hist v_n:{} is not recognized".format(v_n))
        #             pressure = d3corrclass.get_prof_arr(it, 3, v_n)
        #             print(pressure)
        #             print(pressure.min(), pressure.max())
        #             exit(1)
        fpath = outdir + "hist_{}.dat".format(v_n)
        #

        if Paths.debug:
            task(v_n, hist_dic, it, mask, fpath)
        else:
            if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                if os.path.isfile(fpath): os.remove(fpath)
                print_colored_string(["task:", "hist", "it:", "{}".format(it), "mask", mask, "v_n:", v_n, ":", "computing"],
                                     ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "green"])
                try:
                    task(v_n, hist_dic, it, mask, fpath)
                except KeyboardInterrupt:
                    exit(1)
                except IOError:
                    print_colored_string(["task:", "hist", "it:", "{}".format(it),"mask", mask, "v_n:", v_n, ":", "IOError"],
                                         ["blue", "green", "blue", "green", "blue", "green","blue", "green", "", "red"])
                except:
                    print_colored_string(["task:", "hist", "it:", "{}".format(it),"mask", mask, "v_n:", v_n, ":", "Error"],
                                         ["blue", "green", "blue", "green", "blue", "green","blue", "green", "", "red"])
            else:
                print_colored_string(["task:", "hist", "it:", "{}".format(it), "mask", mask, "v_n:", v_n, ":", "skipping"],
                                     ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "blue"])

def d3_corr_for_it(it, d3corrclass, mask, glob_v_ns, outdir, rewrite=False):

    selected_vn1vn2s = select_string(glob_v_ns, __d3corrs__)

    def task(it, corr_task_dic, mask, fpath):
        edges, mass = d3corrclass.get_correlation(it, corr_task_dic, mask, multiplier=2.)
        dfile = h5py.File(fpath, "w")
        dfile.create_dataset("mass", data=mass, dtype=np.float32)
        for dic, edge in zip(corr_task_dic, edges):
            dfile.create_dataset("{}".format(dic["v_n"]), data=edge)
        dfile.close()

    for v_ns in selected_vn1vn2s:
        # chose a dictionary

        corr_task_dic = get_corr_dic(mask, v_ns)

        # if v_ns == "rho_r":
        #     corr_task_dic = d3corrclass.corr_task_dic_rho_r
        # elif v_ns == "r_Ye":
        #     corr_task_dic = d3corrclass.corr_task_dic_r_ye
        # elif v_ns == "rho_Ye":
        #     corr_task_dic = d3corrclass.corr_task_dic_rho_ye
        # elif v_ns == "Ye_entr":
        #     corr_task_dic = d3corrclass.corr_task_dic_ye_entr
        # elif v_ns == "temp_Ye":
        #     corr_task_dic = d3corrclass.corr_task_dic_temp_ye
        # elif v_ns == "velz_Ye":
        #     corr_task_dic = d3corrclass.corr_task_dic_velz_ye
        # elif v_ns == "rho_theta":
        #     corr_task_dic = d3corrclass.corr_task_dic_rho_theta
        # elif v_ns == "velz_theta":
        #     corr_task_dic = d3corrclass.corr_task_dic_velz_theta
        # elif v_ns == "rho_temp":
        #     corr_task_dic = d3corrclass.corr_task_dic_rho_temp
        # elif v_ns == "rho_ang_mom":
        #     corr_task_dic = d3corrclass.corr_task_dic_rho_ang_mom
        # elif v_ns == "rho_ang_mom_flux":
        #     corr_task_dic = d3corrclass.corr_task_dic_rho_ang_mom_flux
        # elif v_ns == "Ye_dens_unb_bern":
        #     corr_task_dic = d3corrclass.corr_task_dic_ye_dens_unb_bern
        # elif v_ns == "rho_dens_unb_bern":
        #     corr_task_dic = d3corrclass.corr_task_dic_rho_dens_unb_bern
        # elif v_ns == "velz_dens_unb_bern":
        #     corr_task_dic = d3corrclass.corr_task_dic_velz_dens_unb_bern
        # elif v_ns == "theta_dens_unb_bern":
        #     corr_task_dic = d3corrclass.corr_task_dic_theta_dens_unb_bern
        # elif v_ns == "ang_mom_flux_theta":
        #     corr_task_dic = d3corrclass.corr_task_dic_ang_mom_flux_theta
        # elif v_ns == "ang_mom_flux_dens_unb_bern":
        #     corr_task_dic = d3corrclass.corr_task_dic_ang_mom_flux_dens_unb_bern
        # elif v_ns == "inv_ang_mom_flux_dens_unb_bern":
        #     corr_task_dic = d3corrclass.corr_task_dic_inv_ang_mom_flux_dens_unb_bern
        # elif v_ns == "hu_0_ang_mom":
        #     corr_task_dic = d3corrclass.corr_task_dic_hu_0_ang_mom
        # elif v_ns == "hu_0_ang_mom_flux":
        #     corr_task_dic = d3corrclass.corr_task_dic_hu_0_ang_mom_flux
        # elif v_ns == "hu_0_Ye":
        #     corr_task_dic = d3corrclass.corr_task_dic_hu_0_ye
        # elif v_ns == "hu_0_temp":
        #     corr_task_dic = d3corrclass.corr_task_dic_hu_0_temp
        # elif v_ns == "hu_0_entr":
        #     corr_task_dic = d3corrclass.corr_task_dic_hu_0_entr
        # else:
        #     raise NameError("unknown task for correlation computation: {}"
        #                     .format(v_ns))

        fpath = outdir + "corr_{}.h5".format(v_ns)

        if Paths.debug:
            task(it, corr_task_dic, mask, fpath)
        else:
            try:
                if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                    if os.path.isfile(fpath): os.remove(fpath)
                    print_colored_string(["task:", "corr", "it:", "{}".format(it), "mask:", mask, "v_ns:", v_ns, ":", "computing"],
                                         ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "green"])
                    task(it, corr_task_dic, mask, fpath)
                else:
                    print_colored_string(["task:", "corr", "it:", "{}".format(it), "mask:", mask, "v_ns:", v_ns, ":", "skipping"],
                                         ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "blue"])
            except IOError:
                print_colored_string(["task:", "corr", "it:", "{}".format(it), "mask:", mask, "v_ns:", v_ns, ":", "IOError"],
                                     ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])
            except ValueError:
                print_colored_string(["task:", "corr", "it:", "{}".format(it), "mask:", mask, "v_ns:", v_ns, ":", "ValueError"],
                                     ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])
            except KeyboardInterrupt:
                exit(1)
            except:
                print_colored_string(["task:", "corr", "it:", "{}".format(it), "mask:", mask, "v_ns:", v_ns, ":", "faild"],
                                     ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])

def d3_to_d2_slice_for_it(it, d3corrclass, glob_planes, outdir, rewrite=False):

    selected_planes = select_string(glob_planes, __d3slicesplanes__)

    def task(it, plane, fpath):
        d3corrclass.make_save_prof_slice(it, plane, __d3slicesvns__, fpath)

    for plane in selected_planes:
        fpath = outdir + "profile" + '.' + plane + ".h5"

        if Paths.debug:
            task(it, plane, fpath)
        else:
            try:#if True: #try:
                if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                    if os.path.isfile(fpath): os.remove(fpath)
                    print_colored_string(["task:", "prof slice", "it:", "{}".format(it), "plane:", plane, ":", "computing"],
                                         ["blue", "green", "blue", "green", "blue", "green", "", "green"])
                    task(it, plane, fpath)
                else:
                    print_colored_string(["task:", "prof slice", "it:", "{}".format(it), "plane:", plane, ":", "skipping"],
                                         ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
            except ValueError:
                print_colored_string(["task:", "prof slice", "it:", "{}".format(it), "plane:", plane, ":", "ValueError"],
                                     ["blue", "green", "blue", "green", "blue", "green", "", "red"])
            except IOError:
                print_colored_string(["task:", "prof slice", "it:", "{}".format(it), "plane:", plane, ":", "IOError"],
                                     ["blue", "green", "blue", "green", "blue", "green", "", "red"])
            except:
                print_colored_string(["task:", "prof slice", "it:", "{}".format(it), "plane:", plane, ":", "failed"],
                                     ["blue", "green", "blue", "green", "blue", "green", "", "red"])



def d2_slice_corr_for_it(it, d3slice, glob_v_ns, glob_masks, plane, outdir, rewrite):

    selected_vn1vn2s = select_string(glob_v_ns, __d2corrs__)

    def task(it, corr_task_dic, mask, fpath):
        edges, mass = d3slice.get_correlation(it, corr_task_dic, mask, multiplier=2.)
        dfile = h5py.File(fpath, "w")
        dfile.create_dataset("mass", data=mass, dtype=np.float32)
        for dic, edge in zip(corr_task_dic, edges):
            dfile.create_dataset("{}".format(dic["v_n"]), data=edge)
        dfile.close()

    for mask in glob_masks:
        if not os.path.isdir(outdir + mask + '/'):
            os.mkdir(outdir + mask + '/')
        outdir_ = outdir + mask + '/'
        #__outdir = _outdir + 'slicecorr_{}/'.format(plane)
        if not os.path.isdir(outdir_):
            os.mkdir(outdir_)
        outdir__ = outdir_ + 'slicecorr_{}/'.format(plane)
        if not os.path.isdir(outdir__):
            os.mkdir(outdir__)
        for v_ns in selected_vn1vn2s:
            #

            corr_task_dic = get_corr_dic(mask, v_ns)

            # if v_ns == "Q_eff_nua_dens_unb_bern":
            #     #corr_task_dic = d3slice.corr_task_dic_q_eff_nua_dens_unb_bern
            #     corr_task_dic = [
            #         {"v_n": "Q_eff_nua", "edges": get_hist_bins(mask, 'Q_eff_nua')},
            #         {"v_n": "dens_unb_bern", "edges": get_hist_bins(mask, 'dens_unb_bern')}
            #     ]
            # elif v_ns == "Q_eff_nua_over_density_hu_0":
            #     # corr_task_dic = d3slice.corr_task_dic_q_eff_nua_over_D_hu_0
            #     corr_task_dic = [
            #         {"v_n": "Q_eff_nua_over_density", "edges": get_hist_bins(mask, 'Q_eff_nua_over_density')},
            #         {"v_n": "hu_0", "edges": get_hist_bins(mask, 'hu_0')}
            #     ]
            # elif v_ns == "Q_eff_nua_over_density_theta":
            #     # corr_task_dic = d3slice.corr_task_dic_q_eff_nua_over_D_theta
            #     corr_task_dic = [
            #         {"v_n": "Q_eff_nua_over_density", "edges": get_hist_bins(mask, 'Q_eff_nua_over_density')},
            #         {"v_n": "theta", "edges": get_hist_bins(mask, 'theta')}
            #     ]
            # elif v_ns == "Q_eff_nua_over_density_Ye":
            #     # corr_task_dic = d3slice.corr_task_dic_q_eff_nua_over_D_Ye
            #     corr_task_dic = [
            #         {"v_n": "Q_eff_nua_over_density", "edges": get_hist_bins(mask, 'Q_eff_nua_over_density')},
            #         {"v_n": "Ye", "edges": get_hist_bins(mask, 'Ye')}
            #     ]
            # elif v_ns == "Q_eff_nua_u_0":
            #     # corr_task_dic = d3slice.corr_task_dic_q_eff_nua_u_0
            #     corr_task_dic = [
            #         {"v_n": "Q_eff_nua", "edges": get_hist_bins(mask, 'Q_eff_nua')},
            #         {"v_n": "u_0", "edges": get_hist_bins(mask, 'u_0')}
            #     ]
            # elif v_ns == "Q_eff_nua_Ye":
            #     # corr_task_dic = d3slice.corr_task_dic_q_eff_nua_ye
            #     corr_task_dic = [
            #         {"v_n": "Q_eff_nua", "edges": get_hist_bins(mask, 'Q_eff_nua')},
            #         {"v_n": "Ye", "edges": get_hist_bins(mask, 'Ye')}
            #     ]
            # elif v_ns == "velz_Ye":
            #     # corr_task_dic = d3slice.corr_task_dic_velz_ye
            #     corr_task_dic = [
            #         {"v_n": "velz", "edges": get_hist_bins(mask, 'velz')},  # in c
            #         {"v_n": "Ye", "edges": get_hist_bins(mask, 'Ye')}
            #     ]
            # elif v_ns == "Q_eff_nua_hu_0":
            #     # corr_task_dic = d3slice.corr_task_dic_q_eff_nua_hu_0
            #     corr_task_dic = [
            #         {"v_n": "Q_eff_nua", "edges": get_hist_bins(mask, 'Q_eff_nua')},
            #         {"v_n": "hu_0", "edges": get_hist_bins(mask, 'hu_0')}
            #     ]
            # else:
            #     raise NameError("unknown task for correlation computation: {}".format(v_ns))

            fpath = outdir__ + "corr_{}.h5".format(plane, v_ns)

            if Paths.debug:
                task(it, corr_task_dic, mask, fpath)
            else:
                try:
                    if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                        if os.path.isfile(fpath): os.remove(fpath)
                        print_colored_string(
                            ["task:", "slicecorr", "it:", "{}".format(it), "plane:", plane, "mask", mask, "v_ns:", v_ns,
                             ":", "computing"],
                            ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "blue", "green", "",
                             "green"])
                        task(it, corr_task_dic, mask, fpath)
                    else:
                        print_colored_string(
                            ["task:", "slicecorr", "it:", "{}".format(it), "plane:", plane, "v_ns:", v_ns, ":", "skipping"],
                            ["blue", "green", "blue", "green", "blue", "green" "blue", "green", "", "blue"])
                except IOError:
                    print_colored_string(
                        ["task:", "slicecorr", "it:", "{}".format(it), "plane:", plane, "mask", mask, "v_ns:", v_ns, ":",
                         "IOError"],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])
                except NameError:
                    print_colored_string(
                        ["task:", "slicecorr", "it:", "{}".format(it), "plane:", plane, "mask", mask, "v_ns:", v_ns, ":",
                         "NameError"],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])
                except ValueError:
                    print_colored_string(
                        ["task:", "slicecorr", "it:", "{}".format(it), "plane:", plane, "mask", mask, "v_ns:", v_ns, ":",
                         "ValueError"],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])
                except KeyboardInterrupt:
                    exit(1)
                except:
                    print_colored_string(
                        ["task:", "slicecorr", "it:", "{}".format(it), "plane:", plane, "mask", mask, "v_ns:", v_ns, ":",
                         "failed"],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])

def d3_dens_modes(d3corrclass, outdir, rewrite=False):
    fpath = outdir + "density_modes_lap15.h5"
    rl = 3
    mmax = 8
    Printcolor.yellow("\tNote: that for density mode computation, masks (lapse etc) are NOT used")

    def task(rl, mmax, fpath, nshells=100):
        times, iterations, xcs, ycs, modes, rs, mmodes = \
            d3corrclass.get_dens_modes_for_rl(rl=rl, mmax=mmax, nshells=nshells)
        dfile = h5py.File(fpath, "w")
        dfile.create_dataset("times", data=times)  # times that actually used
        dfile.create_dataset("iterations", data=iterations)  # iterations for these times
        dfile.create_dataset("xc", data=xcs)  # x coordinate of the center of mass
        dfile.create_dataset("yc", data=ycs)  # y coordinate of the center of mass
        dfile.create_dataset("rs", data=rs)  # central radii of the shells
        for m in range(mmax + 1):
            group = dfile.create_group("m=%d" % m)
            group["int_phi"] = np.array(mmodes[m])  # NOT USED (suppose to be data for every 'R' in disk and NS)
            group["int_phi_r"] = np.array(modes[m]).flatten()  # integrated over 'R' data
        dfile.close()

    if Paths.debug:
        task(rl, mmax, fpath, nshells=100)
    else:
        try:
            if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                if os.path.isfile(fpath): os.remove(fpath)
                print_colored_string(["task:", "dens modes", "rl:", str(rl), "mmax:", str(mmax), ":", "computing"],
                                     ["blue", "green", "blue", "green", "blue", "green", "", "green"])
                task(rl, mmax, fpath, nshells=100)
            else:
                print_colored_string(["task:", "dens modes", "rl:", str(rl), "mmax:", str(mmax), ":", "skipping"],
                                     ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
        except KeyboardInterrupt:
            exit(1)
        except IOError:
            print_colored_string(["task:", "dens modes", "rl:", str(rl), "mmax:", str(mmax), ":", "IOError"],
                                 ["blue", "green", "blue", "green", "blue", "green", "", "red"])
        except:
            print_colored_string(["task:", "dens modes", "rl:", str(rl), "mmax:", str(mmax), ":", "failed"],
                                 ["blue", "green", "blue", "green", "blue", "green", "", "red"])

def d3_dens_modes_int(d3intclass, outdir, rewrite=False):
    """
    Density modes from interpolated onto cylindrical grid data
    """

    fpath = outdir + "density_modes_int_lap15.h5"
    rl = 3
    mmax = 8
    Printcolor.blue("\tNote: that for density mode computation, masks (lapse etc) are NOT used")
    Printcolor.yellow("\tNote: that in this task, grid interpolation takes long time")

    def task(mmax, fpath, masklapse=0.15):
        times, iterations, rcs, phics, modes, r_pol, modes_r = \
            d3intclass.compute_density_modes(mmode=mmax, masklapse=masklapse)
        dfile = h5py.File(fpath, "w")
        dfile.create_dataset("times", data=times)  # times that actually used
        dfile.create_dataset("iterations", data=iterations)  # iterations for these times
        dfile.create_dataset("r_pol", data=r_pol)  # iterations for these times
        dfile.create_dataset("rcs", data=rcs)  # x coordinate of the center of mass
        dfile.create_dataset("phics", data=phics)  # y coordinate of the center of mass
        for m in range(mmax + 1):
            group = dfile.create_group("m=%d" % m)
            group["int_phi"] = np.array(
                modes_r[m]).flatten()  # NOT USED (suppose to be data for every 'R' in disk and NS)
            group["int_phi_r"] = np.array(modes[m]).flatten()  # integrated over 'R' data
        dfile.close()

    if Paths.debug:
        task(mmax, fpath, masklapse=0.15)
    else:
        try:
            if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                if os.path.isfile(fpath): os.remove(fpath)
                print_colored_string(["task:", "dens modes", "rl:", str(rl), "mmax:", str(mmax), ":", "computing"],
                                     ["blue", "green", "blue", "green", "blue", "green", "", "green"])
                task(mmax, fpath, masklapse=0.15)
            else:
                print_colored_string(["task:", "dens modes", "rl:", str(rl), "mmax:", str(mmax), ":", "skipping"],
                                     ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
        except KeyboardInterrupt:
            exit(1)
        except IOError:
            print_colored_string(["task:", "dens modes", "rl:", str(rl), "mmax:", str(mmax), ":", "IOError"],
                                 ["blue", "green", "blue", "green", "blue", "green", "", "red"])
        except:
            print_colored_string(["task:", "dens modes", "rl:", str(rl), "mmax:", str(mmax), ":", "failed"],
                                 ["blue", "green", "blue", "green", "blue", "green", "", "red"])

def d3_int_data_to_vtk(d3intclass, glob_its, glob_v_ns, outdir, rewrite=False):
    private_dir = "vtk/"

    selected_v_ns = select_string(glob_v_ns, __d3slicesvns__)

    try:
        from evtk.hl import gridToVTK
    except ImportError:
        print("Failed: 'from evtk.hl import gridToVTK' ")
        try:
            import pyevtk
            from pyevtk.hl import gridToVTK
        except:
            print("Failed: 'import pyevtk' or 'from pyevtk.hl import gridToVTK' ")
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


def d3_interpolate_mjenclosed(d3intclass, glob_its, glob_masks, outdir, rewrite=False):
    # getting cylindrical grid [same for any iteration)
    dphi_cyl = d3intclass.new_grid.get_int_grid("dphi_cyl")
    dr_cyl = d3intclass.new_grid.get_int_grid("dr_cyl")
    dz_cyl = d3intclass.new_grid.get_int_grid("dz_cyl")
    r_cyl = d3intclass.new_grid.get_int_grid("r_cyl")
    #



    for it in glob_its:
        sys.stdout.flush()
        _outdir = outdir + str(it) + '/'
        if not os.path.isdir(_outdir):
            os.mkdir(_outdir)
        #
        for mask in glob_masks:
            __outdir = _outdir + mask + '/'
            if not os.path.isdir(__outdir):
                os.mkdir(__outdir)
            #
            fpath = __outdir + __d3intmjfname__
            #
            if True:
                #
                rho = d3intclass.get_int(it, "rho")     # [rho_NS, rho_ATM]
                lapse = d3intclass.get_int(it, "lapse") # [lapse_BH, dummy1]
                if mask == "disk":
                    rho_lims = MASK_STORE.disk_mask_setup["rho"]
                    lapse_lims = MASK_STORE.disk_mask_setup["lapse"]
                    rho_mask = (rho > rho_lims[0]) & (rho < rho_lims[1])
                    lapse_mask = lapse > lapse_lims[0] # > BH
                elif mask == "remnant":
                    rho_lims = MASK_STORE.disk_mask_setup["rho"]
                    lapse_lims = MASK_STORE.disk_mask_setup["lapse"]
                    rho_mask = rho > rho_lims[1]
                    lapse_mask = lapse > lapse_lims[0] # > BH
                else:
                    raise NameError("No method for mask: {}".format(mask))
                #
                tot_mask = rho_mask & lapse_mask
                #
                if np.sum(tot_mask.astype(int)) ==0 :
                    print_colored_string(["task:", "MJ_encl", "it:", "{}".format(it), "mask:", mask, ":", "Mask=0"],
                                         ["blue", "green", "blue", "green", "blue", "green", "", "red"])

                if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                    if os.path.isfile(fpath): os.remove(fpath)
                    print_colored_string(["task:", "MJ_encl", "it:", "{}".format(it), "mask:", mask, ":", "computing"],
                                         ["blue", "green", "blue", "green", "blue", "green", "", "green"])
                    #
                    dens_cyl = d3intclass.get_int(it, "density")
                    ang_mom_cyl = d3intclass.get_int(it, "ang_mom")
                    ang_mom_flux_cyl = d3intclass.get_int(it, "ang_mom_flux")
                    #
                    dens_cyl[~tot_mask] = 0.
                    ang_mom_cyl[~tot_mask] = 0.
                    ang_mom_flux_cyl[~tot_mask] = 0.
                    #
                    I_rc = 2 * np.sum(dens_cyl * r_cyl ** 2 * dz_cyl * dphi_cyl, axis=(1, 2))
                    D_rc = 2 * np.sum(dens_cyl * dz_cyl * dphi_cyl, axis=(1, 2)) # integrate over phi,z
                    J_rc = 2 * np.sum(ang_mom_cyl * dz_cyl * dphi_cyl, axis=(1, 2)) # integrate over phi,z
                    Jf_rc= 2 * np.sum(ang_mom_flux_cyl * dz_cyl * dphi_cyl, axis=(1, 2))
                    #
                    ofile = open(fpath, "w")
                    ofile.write("# 1:rcyl 2:drcyl 3:M 4:J 5:Jf 6:I\n")
                    for i in range(r_cyl.shape[0]):
                        ofile.write("{} {} {} {} {} {}\n".format(r_cyl[i, 0, 0], dr_cyl[i, 0, 0],
                                                                 D_rc[i], J_rc[i], Jf_rc[i], I_rc[i]))
                    ofile.close()
                    #
                    d3intclass.delete_for_it(it=it, except_v_ns=[], rm_masks=True, rm_comp=True, rm_prof=False)
                    sys.stdout.flush()
                    #
                else:
                    print_colored_string(["task:", "MJ_encl", "it:", "{}".format(it), "mask:", mask, ":", "skipping"],
                                         ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
        # except KeyboardInterrupt:
        #     exit(1)
        # except IOError:
        #     print_colored_string(["task:", "MJ_encl", "it:", "{}".format(it), ":", "IOError"],
        #                          ["blue", "green", "blue", "green", "", "red"])
        # except:
        #     print_colored_string(["task:", "MJ_encl", "it:", "{}".format(it), ":", "failed"],
        #                          ["blue", "green", "blue", "green", "", "red"])




def plot_d3_prof_slices(d3class, glob_its, glob_v_ns, resdir, figdir='module_slices/', rewritefigs=False):


    iterations = select_number(glob_its, d3class.list_iterations)
    v_ns = select_string(glob_v_ns, __d3sliceplotvns__, for_all="all")

    # tmerg = d1class.get_par("tmerger_gw")
    i = 1
    for it in iterations:
        for rl in __d3sliceplotrls__:
            for v_n in v_ns:
                # --- Getting XZ data ---
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
                # --- Getting XY data ---
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
                elif v_n == 'entr':
                    def_dic_xy['v_n'] = 'entropy'
                    def_dic_xy['vmin'] = 0.
                    def_dic_xy['vmax'] = 100.
                    def_dic_xy['norm'] = None
                    def_dic_xy['cmap'] = 'inferno'

                    def_dic_xz['v_n'] = 'entropy'
                    def_dic_xz['vmin'] = 0.
                    def_dic_xz['vmax'] = 100.
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
                    = get_reflev_borders(rl)
                def_dic_xz["xmin"], def_dic_xz["xmax"], _, _, def_dic_xz["ymin"], def_dic_xz["ymax"] \
                    = get_reflev_borders(rl)

                """ --- --- --- """
                datafpath = resdir

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
                if not os.path.isfile(datafpath + str(it) + '/' + "module_profile.xy.h5") \
                        or not os.path.isfile(datafpath + str(it) + '/' + "module_profile.xz.h5"):
                    Printcolor.yellow(
                        "Required data ia missing: {}".format(datafpath + str(it) + '/' + "module_profile.xy(or yz).h5"))
                    continue
                fpath = datafpath + str(it) + '/' + figdir + figname
                t = d3class.get_time_for_it(it, "profiles", "prof")
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

def plot_d3_corr(d3histclass, glob_its, glob_ts, glob_v_ns, glob_masks, resdir, rewrite=False):

    def task(default_dic, it, vn1vn2, v_n_x, v_n_y, outfpath):
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

        # -------------------------------
        # tr = (t - tmerg) * 1e3  # ms
        # t = d3histclass.get_time_for_it(it, output="profiles", d1d2d3prof="prof")
        default_dic["it"] = it
        default_dic["title"]["text"] = r'$t:{:.1f}$ [ms]'.format(float(t * 1e3))
        o_plot.set_plot_dics.append(default_dic)

        o_plot.main()
        o_plot.set_plot_dics = []
        o_plot.figure.clear()


    iterations = select_number(glob_its, d3histclass.list_iterations)
    v_ns = select_string(glob_v_ns, __d3corrs__, for_all="all")

    for it in iterations:
        t = get_time_for_it(it, glob_its, glob_ts)
        for mask in glob_masks:
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
                elif vn1vn2 == "r_Ye":
                    v_n_x = 'r'
                    v_n_y = 'Ye'
                    # default_dic['v_n_x'] = 'rho'
                    # default_dic['v_n_y'] = 'Ye'
                    default_dic['xmin'] = 0
                    default_dic['xmax'] = 100
                    default_dic['xscale'] = None
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
                elif vn1vn2 == "Ye_entr":
                    v_n_x = 'Ye'
                    v_n_y = 'entr'
                    # default_dic['v_n_x'] = 'temp'
                    # default_dic['v_n_y'] = 'Ye'
                    default_dic['ymin'] = 0
                    default_dic['ymax'] = 50
                    default_dic['xmin'] = 0.01
                    default_dic['xmax'] = 0.5
                    default_dic['yscale'] = None
                    default_dic['xscale'] = None
                elif vn1vn2 == "rho_temp":
                    v_n_x = 'rho'
                    v_n_y = 'temp'
                    # default_dic['v_n_x'] = 'rho'
                    # default_dic['v_n_y'] = 'theta'
                    default_dic['xmin'] = 1e-9
                    default_dic['xmax'] = 2e-5
                    default_dic['ymin'] = 1e-2
                    default_dic['ymax'] = 1e2
                    #default_dic['yscale'] = None
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
                elif vn1vn2 == "hu_0_ang_mom":
                    v_n_x = 'hu_0'
                    v_n_y = 'ang_mom'
                    default_dic["xscale"] = None
                    default_dic['xmin'] = -1.2
                    default_dic['xmax'] = -0.8
                    default_dic['ymin'] = 1e-9
                    default_dic['ymax'] = 1e-3
                elif vn1vn2 == "hu_0_ang_mom_flux":
                    v_n_x = 'hu_0'
                    v_n_y = 'ang_mom_flux'
                    default_dic["xscale"] = None
                    default_dic['xmin'] = -1.2
                    default_dic['xmax'] = -0.8
                    default_dic['ymin'] = 1e-11
                    default_dic['ymax'] = 1e-7
                elif vn1vn2 == "hu_0_Ye":
                    v_n_x = 'hu_0'
                    v_n_y = 'Ye'
                    default_dic["xscale"] = None
                    default_dic['xmin'] = -1.2
                    default_dic['xmax'] = -0.8
                    default_dic['ymin'] = 0.01
                    default_dic['ymax'] = 0.5
                    default_dic['yscale'] = None
                elif vn1vn2 == "hu_0_entr":
                    v_n_x = 'hu_0'
                    v_n_y = 'entr'
                    default_dic["xscale"] = None
                    default_dic['xmin'] = -1.2
                    default_dic['xmax'] = -0.8
                    default_dic['ymin'] = 0.
                    default_dic['ymax'] = 0.80
                    default_dic['yscale'] = None
                elif vn1vn2 == "hu_0_temp":
                    v_n_x = 'hu_0'
                    v_n_y = 'temp'
                    default_dic["xscale"] = None
                    default_dic['xmin'] = -1.2
                    default_dic['xmax'] = -0.8
                    default_dic['ymin'] = 1e-1
                    default_dic['ymax'] = 1e2
                else:
                    raise NameError("vn1vn2:{} is not recognized"
                                    .format(vn1vn2))
                outfpath = resdir + str(it) + '/' + mask + "/corr_plots/"
                if not os.path.isdir(outfpath):
                    os.mkdir(outfpath)
                fpath = outfpath + "{}.png".format(vn1vn2)

                if Paths.debug:
                    task(default_dic, it, vn1vn2, v_n_x, v_n_y, outfpath)
                else:
                    try:
                        if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                            if os.path.isfile(fpath): os.remove(fpath)
                            print_colored_string(["task:", "plot corr", "it:", "{}".format(it), "v_ns:", vn1vn2, ":", "computing"],
                                                 ["blue", "green", "blue", "green", "blue", "green", "", "green"])

                            task(default_dic, it, vn1vn2, v_n_x, v_n_y, outfpath)
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

def plot_d2_slice_corr(d3histclass, glob_its, glob_ts, glob_v_ns, glob_planes, glob_masks, resdir, rewrite=False):


    iterations = select_number(glob_its, d3histclass.list_iterations)
    v_ns = select_string(glob_v_ns, __d2corrs__, for_all="all")
    planes = select_string(glob_planes, __d3slicesplanes__, for_all="all")

    for it in iterations:
        t = get_time_for_it(it, glob_its, glob_ts)
        for plane in planes:
            for mask in glob_masks:
            #
                if mask == "None" or mask == None or mask == "":
                    d3histclass.set_corr_fname_intro = "{}_corr_".format(plane)
                    outfpath = resdir + str(it) + "/corr_plots/"
                else:
                    d3histclass.set_corr_fname_intro = "{}/{}_corr_".format(mask, plane)
                    outfpath = resdir + str(it) + '/' + mask + "/corr_plots/"
                #
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

                    if vn1vn2 == "Q_eff_nua_dens_unb_bern":
                        v_n_x = 'Q_eff_nua'
                        v_n_y = 'dens_unb_bern'
                        default_dic['xmin'] = 1e-15
                        default_dic['xmax'] = 1e-10
                        default_dic['ymin'] = 1e-10
                        default_dic['ymax'] = 1e-8
                    elif vn1vn2 == "Q_eff_nua_Ye":
                        v_n_x = 'Q_eff_nua'
                        v_n_y = 'Ye'
                        default_dic['xmin'] = 1e-15
                        default_dic['xmax'] = 1e-10
                        default_dic['ymin'] = 0.01
                        default_dic['ymax'] = 0.5
                        default_dic['yscale'] = None
                    elif vn1vn2 == "velz_Ye":
                        v_n_x = 'velz'
                        v_n_y = 'Ye'
                        default_dic['xmin'] = -.5
                        default_dic['xmax'] = .5
                        default_dic['ymin'] = 0.01
                        default_dic['ymax'] = 0.5
                        default_dic['yscale'] = None
                        default_dic['xscale'] = None
                    elif vn1vn2 == "Q_eff_nua_u_0": # Q_eff_nua_hu_0
                        v_n_x = 'Q_eff_nua'
                        v_n_y = 'u_0'
                        default_dic['xmin'] = 1e-15
                        default_dic['xmax'] = 1e-10
                        default_dic['ymin'] = -0.95
                        default_dic['ymax'] = 1.05
                        default_dic['yscale'] = None #
                    elif vn1vn2 == "Q_eff_nua_hu_0":
                        v_n_x = 'Q_eff_nua'
                        v_n_y = 'hu_0'
                        default_dic['xmin'] = 1e-15
                        default_dic['xmax'] = 1e-10
                        default_dic['ymin'] = -0.95
                        default_dic['ymax'] = 1.05
                        default_dic['yscale'] = None
                    elif vn1vn2 == "Q_eff_nua_over_density_hu_0":
                        v_n_x = 'Q_eff_nua_over_density'
                        v_n_y = 'hu_0'
                        default_dic['xmin'] = 1e-4
                        default_dic['xmax'] = 1e-8
                        default_dic['ymin'] = -0.95
                        default_dic['ymax'] = 1.05
                        default_dic['yscale'] = None
                    elif vn1vn2 == "Q_eff_nua_over_density_theta":
                        v_n_x = 'Q_eff_nua_over_density'
                        v_n_y = 'theta'
                        default_dic['xmin'] = 1e-4
                        default_dic['xmax'] = 1e-8
                        default_dic['ymin'] = 0
                        default_dic['ymax'] = np.pi
                        default_dic['yscale'] = None
                    elif vn1vn2 == "Q_eff_nua_over_density_Ye":
                        v_n_x = 'Q_eff_nua_over_density'
                        v_n_y = 'Ye'
                        default_dic['xmin'] = 1e-4
                        default_dic['xmax'] = 1e-8
                        default_dic['ymin'] = 0
                        default_dic['ymax'] = 0.5
                        default_dic['yscale'] = None
                    else:
                        raise NameError("vn1vn2:{} is not recognized"
                                        .format(vn1vn2))

                    if not os.path.isdir(outfpath):
                        os.mkdir(outfpath)
                    fpath = outfpath + "{}_{}.png".format(plane, vn1vn2)
                    try:
                        if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                            if os.path.isfile(fpath): os.remove(fpath)
                            print_colored_string(["task:", "plot slice corr", "it:", "{}".format(it),"plane", plane, "mask", mask, "v_ns:", vn1vn2, ":", "computing"],
                                                 ["blue", "green", "blue", "green", "blue", "green", "blue", "green","blue", "green", "", "green"])

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
                            # t = d3histclass.get_time_for_it(it, output="profiles", d1d2d3prof="prof")
                            default_dic["it"] = it
                            default_dic["title"]["text"] = r'$t:{:.1f}$ [ms]'.format(float(t*1e3))
                            o_plot.set_plot_dics.append(default_dic)

                            o_plot.main()
                            o_plot.set_plot_dics = []
                            o_plot.figure.clear()
                            #-------------------------------
                        else:
                            print_colored_string(["task:", "plot slice corr", "it:", "{}".format(it), "plane", plane,"mask", mask, "v_ns:", vn1vn2, ":", "skipping"],
                                                 ["blue", "green", "blue", "green", "blue", "green","blue", "green", "", "blue"])
                    except IOError:
                        print_colored_string(["task:", "plot slice corr", "it:", "{}".format(it), "plane", plane,"mask", mask, "v_ns:", vn1vn2, ":", "missing file"],
                                             ["blue", "green", "blue", "green", "blue", "green", "blue", "green","blue", "green", "", "red"])
                    except KeyboardInterrupt:
                        exit(1)
                    except:
                        print_colored_string(["task:", "plot slice corr", "it:", "{}".format(it), "plane", plane,"mask", mask, "v_ns:", vn1vn2, ":", "failed"],
                                             ["blue", "green", "blue", "green", "blue", "green", "blue", "green","blue", "green", "", "red"])
                    default_dic = {}

def plot_d3_hist(d3histclass, glob_its, glob_ts, glob_v_ns, glob_masks, resdir, rewrite=False):

    iterations = select_number(glob_its, d3histclass.list_iterations)
    v_ns = select_string(glob_v_ns, __d3histvns__, for_all="all")

    for mask in glob_masks:
        for it in iterations:
            for v_n in v_ns:

                fpath = resdir + str(it) + '/' + mask + "/hist_{}.dat".format(v_n)
                # print(data)
                default_dic = {
                    'task': 'hist1d', 'ptype': 'cartesian',
                    'position': (1, 1),
                    'data': None, 'normalize': False,
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

                if v_n == "r" and mask == "disk":
                    default_dic['v_n_x'] = 'r'
                    default_dic['xlabel'] = 'cylindrical radius'
                    default_dic['xmin'] = 10.
                    default_dic['xmax'] = 50.
                elif v_n == "r" and mask == "remnant":
                    default_dic['v_n_x'] = 'r'
                    default_dic['xlabel'] = 'cylindrical radius'
                    default_dic['xmin'] = 0.
                    default_dic['xmax'] = 25.
                elif v_n == "theta":
                    default_dic['v_n_x'] = 'theta'
                    default_dic['xlabel'] = 'angle from binary plane'
                    default_dic['xmin'] = 0
                    default_dic['xmax'] = 90.
                elif v_n == "entr" and mask == "disk":
                    default_dic['v_n_x'] = 'entropy'
                    default_dic['xlabel'] = 'entropy'
                    default_dic['xmin'] = 0
                    default_dic['xmax'] = 150.
                elif v_n == "entr" and mask == "remnant":
                    default_dic['v_n_x'] = 'entropy'
                    default_dic['xlabel'] = 'entropy'
                    default_dic['xmin'] = 0
                    default_dic['xmax'] = 25.
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
                elif v_n == "rho" and mask == "disk":
                    default_dic['v_n_x'] = "rho"
                    default_dic["xlabel"] = "rho"
                    default_dic['xmin'] = 1e-10
                    default_dic['xmax'] = 1e-6
                    default_dic['xscale'] = "log"
                elif v_n == "rho" and mask == "remnant":
                    default_dic['v_n_x'] = "rho"
                    default_dic["xlabel"] = "rho"
                    default_dic['xmin'] = 1e-6
                    default_dic['xmax'] = 1e-2
                    default_dic['xscale'] = "log"
                elif v_n == "dens_unb_bern":
                    default_dic['v_n_x'] = "temp"
                    default_dic["xlabel"] = "temp"
                    default_dic['xmin'] = 1e-10
                    default_dic['xmax'] = 1e-6
                    default_dic['xscale'] = "log"
                elif v_n == "press" and mask == "disk":
                    default_dic['v_n_x'] = "press"
                    default_dic["xlabel"] = "press"
                    default_dic['xmin'] = 1e-13
                    default_dic['xmax'] = 1e-5
                    default_dic['xscale'] = "log"
                elif v_n == "press" and mask == "remnant":
                    default_dic['v_n_x'] = "press"
                    default_dic["xlabel"] = "press"
                    default_dic['xmin'] = 1e-8
                    default_dic['xmax'] = 1e-1
                    default_dic['xscale'] = "log"
                else:
                    raise NameError("hist v_n:{} is not recognized".format(v_n))

                outfpath = resdir + str(it) + '/' + mask +  "/hist_plots/"
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

                figpath = outfpath + "{}.png".format(v_n)

                try:
                    if (os.path.isfile(figpath) and rewrite) or not os.path.isfile(figpath):
                        if os.path.isfile(figpath): os.remove(figpath)
                        print_colored_string(["task:", "plot hist", "it:", "{}".format(it), "mask", mask, "v_ns:", v_n, ":", "computing"],
                                             ["blue", "green", "blue", "green", "blue", "green","blue", "green", "", "green"])
                        #-------------------------------
                        data = np.loadtxt(fpath, unpack=False)
                        default_dic["it"] = it
                        default_dic["data"] = data
                        o_plot.set_plot_dics.append(default_dic)

                        o_plot.main()
                        o_plot.set_plot_dics = []
                        o_plot.figure.clear()
                        #-------------------------------
                    else:
                        print_colored_string(["task:", "plot hist", "it:", "{}".format(it),"mask", mask, "v_ns:", v_n, ":", "skipping"],
                                             ["blue", "green", "blue", "green", "blue", "green","blue", "green", "", "blue"])
                except IOError:
                    print_colored_string(["task:", "plot hist", "it:", "{}".format(it),"mask", mask, "v_ns:", v_n, ":", "missing file"],
                                         ["blue", "green", "blue", "green", "blue", "green","blue", "green", "", "red"])
                except KeyboardInterrupt:
                    exit(1)
                except:
                    print_colored_string(["task:", "plot hist", "it:", "{}".format(it),"mask", mask, "v_ns:", v_n, ":", "failed"],
                                         ["blue", "green", "blue", "green", "blue", "green","blue", "green", "", "red"])

def plot_center_of_mass(dmclass, resdir, rewrite=False):

    plotfname = __center_of_mass_plotname__
    path = resdir
    # fpath = path + fname
    dmclass.gen_set['fname'] = path + __d3densitymodesfame__  # "density_modes_lap15.h5"
    #
    fpath = path + __center_of_mass_plotname__
    # plot the data
    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = path
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = __center_of_mass_plotname__
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = False
    o_plot.gen_set["subplots_adjust_h"] = 0.2
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []
    #
    xc = dmclass.get_grid("xc")
    yc = dmclass.get_grid("yc")
    #
    xc, yc = Tools.x_y_z_sort(xc, yc)
    #
    plot_dic = {
        'task': 'line', 'ptype': 'cartesian',
        'xarr': xc, 'yarr': yc,
        'position': (1, 1),
        'v_n_x': 'times', 'v_n_y': 'int_phi_r abs',
        'ls': '-', 'color': 'black', 'lw': 0.7, 'ds': 'default', 'alpha': 1.0,
        'label': None, 'ylabel': r'$y$ [GEO]', 'xlabel': r"$x$ [GEO]",
        'xmin': -8, 'xmax': 8, 'ymin': -8, 'ymax': 8,
        'xscale': None, 'yscale': None,
        'fancyticks': True, 'minorticks': True,
        'legend': {'loc': 'upper right', 'ncol': 2, 'fontsize': 10, 'shadow': False, 'framealpha': 0.5,
                   'borderaxespad': 0.0},
        'fontsize': 14,
        'labelsize': 14,
        'title': {'text': "Center of mass", 'fontsize': 14},
        'mark_end':{'marker':'x', 'ms':5, 'color':'red', 'alpha':0.7, 'label':'end'},
        'mark_beginning': {'marker': 's', 'ms': 5, 'color': 'blue', 'alpha': 0.7, 'label': 'beginning'},
        'axvline':{'x':0, 'linestyle':'dashed', 'color':'gray', 'linewidth':0.5},
        'axhline': {'y': 0, 'linestyle': 'dashed', 'color': 'gray', 'linewidth': 0.5}
    }
    #
    try:
        if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
            if os.path.isfile(fpath): os.remove(fpath)
            print_colored_string(["task:", "plot dens modes", "fname:", plotfname, "mmodes:", "[1,2]", ":", "computing"],
                                 ["blue", "green", "blue", "green", "blue", "green", "", "green"])
            o_plot.set_plot_dics.append(plot_dic)
            #
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

def plot_density_modes_phase(dmclass, resdir, rewrite=False):
    plotfname = __d3densitymodesfame__.replace(".h5", "phase.png")
    path = resdir
    # fpath = path + fname
    dmclass.gen_set['fname'] = path + __d3densitymodesfame__  # "density_modes_lap15.h5"
    #
    fpath = path + plotfname
    #
    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = path
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = plotfname
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = False
    o_plot.set_plot_dics = []
    #
    iterations = dmclass.get_grid("iterations")
    times = dmclass.get_grid("times")
    assert len(iterations) == len(times)
    #
    colors = ["blue", "red", "green"]
    lss = ["-", "--", ":"]
    # piece of code to get three equally spaced timesteps
    req_times = np.linspace(times.min(), times.max(), num=3)
    assert len(req_times) == 3
    avail_times, avail_its = [], []
    for t in req_times:
        idx = Tools.find_nearest_index(times, t)
        avail_times.append(times[idx])
        avail_its.append(iterations[idx])
    avail_times = np.array(avail_times, dtype=float) * 1e3 # ms
    avail_its = np.array(avail_its, dtype=int)
    #
    for it, t, color, ls in zip(avail_its, avail_times, colors, lss):
        try:
            if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                if os.path.isfile(fpath): os.remove(fpath)
                print_colored_string(["task:", "plot dens modes phase", "it:",str(it), "t:", "{:.1f}".format(t),
                                      "fname:", plotfname, "mmodes:", "[1]", ":", "computing"],
                                     ["blue", "green", "blue", "green", "blue", "green", "blue",
                                      "green",  "blue", "green", "", "green"])
                #
                r = dmclass.get_grid_for_it(it, "rs")
                complex_mag_0 = dmclass.get_data_for_it(it, mode=0, v_n="int_phi")
                complex_mag_1 = dmclass.get_data_for_it(it, mode=1, v_n="int_phi")
                complex_mag = complex_mag_1 / complex_mag_0
                phis = np.angle(complex_mag)#) #
                x, y = Tools.pol2cart(r, phis)
                #
                plot_dic = {
                    'task': 'line', 'ptype': 'cartesian',
                    'xarr': x, 'yarr': y,
                    'position': (1, 1),
                    'v_n_x': 'times', 'v_n_y': 'int_phi_r abs',
                    'ls': ls, 'color': color, 'lw': 0.7, 'ds': 'default', 'alpha': 1.0,
                    'label': "t:{:.1f}".format(t), 'ylabel': r'$y$ [GEO]', 'xlabel': r"$x$ [GEO]",
                    'xmin': -80, 'xmax': 80, 'ymin': -80, 'ymax': 80,
                    'xscale': None, 'yscale': None,
                    'fancyticks': True, 'minorticks': True,
                    'legend': {'loc': 'upper right', 'ncol': 1, 'fontsize': 10, 'shadow': False, 'framealpha': 0.5,
                               'borderaxespad': 0.0},
                    'fontsize': 14,
                    'labelsize': 14,
                    'title': {'text': "Phase of $m=1$ density mode", 'fontsize': 14},
                    # 'mark_end': {'marker': 'x', 'ms': 5, 'color': color, 'alpha': 0.7, 'label': 'end'},
                    # 'mark_beginning': {'marker': 's', 'ms': 5, 'color': color, 'alpha': 0.7, 'label': 'beginning'},
                    # 'axvline': {'x': 0, 'linestyle': 'dashed', 'color': 'gray', 'linewidth': 0.5},
                    # 'axhline': {'y': 0, 'linestyle': 'dashed', 'color': 'gray', 'linewidth': 0.5}
                }
                if it == avail_its[-1]:
                    plot_dic['axvline'] = {'x': 0, 'linestyle': 'dashed', 'color': 'gray', 'linewidth': 0.5}
                    plot_dic['axhline'] = {'y': 0, 'linestyle': 'dashed', 'color': 'gray', 'linewidth': 0.5}
                o_plot.set_plot_dics.append(plot_dic)
                #
                o_plot.main()
            else:
                print_colored_string(["task:", "plot dens modes phase", "it:", str(it), "t:", "{:.1f}".format(t),
                                      "fname:", plotfname, "mmodes:", "[1]", ":", "skipping"],
                                     ["blue", "green", "blue", "green", "blue", "green", "blue",
                                      "green", "blue", "green", "", "blue"])
        except IOError:
            print_colored_string(["task:", "plot dens modes phase", "it:", str(it), "t:", "{:.1f}".format(t),
                                  "fname:", plotfname, "mmodes:", "[1]", ":", "missing file"],
                                 ["blue", "green", "blue", "green", "blue", "green", "blue",
                                  "green", "blue", "green", "", "red"])
        except KeyboardInterrupt:
            exit(1)
        except:
            print_colored_string(["task:", "plot dens modes phase", "it:", str(it), "t:", "{:.1f}".format(t),
                                  "fname:", plotfname, "mmodes:", "[1]", ":", "failed"],
                                 ["blue", "green", "blue", "green", "blue", "green", "blue",
                                  "green", "blue", "green", "", "red"])

def plot_density_modes(dmclass, resdir, rewrite=False):
    plotfname = __d3densitymodesfame__.replace(".h5", ".png")
    path = resdir
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

    # o_plot.set_plot_dics.append(densmode_m0)

    try:
        if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
            if os.path.isfile(fpath): os.remove(fpath)
            print_colored_string(["task:", "plot dens modes", "fname:", plotfname, "mmodes:", "[1,2]", ":", "computing"],
                                 ["blue", "green", "blue", "green", "blue", "green", "", "green"])
            #
            mags = dmclass.get_data(1, "int_phi_r")
            times = dmclass.get_grid("times")
            densmode_m1 = {
                'task': 'line', 'ptype': 'cartesian',
                'xarr': times * 1e3, 'yarr': mags,
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
                'task': 'line', 'ptype': 'cartesian',
                'xarr': times * 1e3, 'yarr': mags,
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

            #
            o_plot.set_plot_dics.append(densmode_m1)
            o_plot.set_plot_dics.append(densmode_m2)

            o_plot.main()
        else:
            print_colored_string(["task:", "plot dens modes", "fname:", plotfname, "mmodes:", "[1,2]", ":", "skipping"],
                                 ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
    except IOError:
        print_colored_string(["task:", "plot dens modes", "fname:", plotfname, "mmodes:", "[1,2]", ":", "missing input efile"],
                             ["blue", "green", "blue", "green", "blue", "green", "", "red"])
    except KeyboardInterrupt:
        exit(1)
    except:
        print_colored_string(["task:", "plot dens modes", "fname:", plotfname, "mmodes:", "[1,2]", ":", "failed"],
                             ["blue", "green", "blue", "green", "blue", "green", "", "red"])



# broken
def plot_mass(d3class, masks, resdir, rewrite=False):
    #
    # path = Paths.ppr_sims + d3class.sim + '/' + __rootoutdir__
    #

    # fname = __d3diskmass__.replace(".txt",".png")
    # figname = __d3diskmass__.replace(".txt",".png")
    parfilepath = resdir
    # fpath = parfilepath + mask + '/'

    for mask in masks:
        try:
            if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                if os.path.isfile(fpath): os.remove(fpath)
                print_colored_string(["task:", "plotmass", ":", "saving/plotting"],
                                     ["blue", "green", "", "green"])
                #
                list_iterations = get_list_iterations_from_res_3d(resdir)
                #
                it_arr =   []
                time_arr = []
                data_arr = []
                for it in list_iterations:
                    fpath = parfilepath + str(int(it)) + '/' + mask + "mass.txt"
                    time_ = d3class.get_time_for_it(it, "profiles", "prof")
                    time_arr.append(time_)
                    it_arr.append(it)
                    if os.path.isfile(fpath):
                        data_ = np.float(np.loadtxt(fpath, unpack=True))
                        data_arr.append(data_)
                    else:
                        data_arr.append(np.nan)
                #
                it_arr = np.array(it_arr, dtype=int)
                time_arr = np.array(time_arr, dtype=float)
                data_arr = np.array(data_arr, dtype=float)
                #
                if len(it_arr) > 0:
                    x = np.vstack((it_arr, time_arr, data_arr)).T
                    np.savetxt(parfilepath+__d3diskmass__, x, header="1:it 2:time[s] 3:mass[Msun]", fmt='%i %0.5f %0.5f')
                else:
                    Printcolor.yellow("No disk mass found")
                #
                if len(it_arr) > 0:

                    time_arr = time_arr * 1e3

                    o_plot = PLOT_MANY_TASKS()
                    o_plot.gen_set["figdir"] = parfilepath
                    o_plot.gen_set["type"] = "cartesian"
                    o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
                    o_plot.gen_set["figname"] = __d3diskmass__.replace(".txt",".png")
                    o_plot.gen_set["sharex"] = False
                    o_plot.gen_set["sharey"] = False
                    o_plot.gen_set["subplots_adjust_h"] = 0.2
                    o_plot.gen_set["subplots_adjust_w"] = 0.0
                    o_plot.set_plot_dics = []

                    # plot
                    plot_dic = {
                        'task': 'line', 'ptype': 'cartesian',
                        'xarr': time_arr, 'yarr': data_arr,
                        'position': (1, 1),
                        'v_n_x': 'times', 'v_n_y': 'int_phi_r abs',
                        'marker': '.', 'color': 'black', 'ms': 5., 'alpha': 1.0, #'ds': 'default',
                        'label': None, 'ylabel': r'$M_{\rm{disk}}$ [$M_{\odot}$]', 'xlabel': r"$t$ [ms]",
                        'xmin': -5., 'xmax': time_arr.max(), 'ymin': 0, 'ymax': 0.5,
                        'xscale': None, 'yscale': None,
                        'fancyticks': True, 'minorticks': True,
                        'legend': {'loc': 'upper right', 'ncol': 2, 'fontsize': 10, 'shadow': False, 'framealpha': 0.5,
                                   'borderaxespad': 0.0},
                        'fontsize': 14,
                        'labelsize': 14,
                        'title': {'text': "Disk Mass Evolution", 'fontsize': 14},
                        # 'mark_end': {'marker': 'x', 'ms': 5, 'color': 'red', 'alpha': 0.7, 'label': 'end'},
                        # 'mark_beginning': {'marker': 's', 'ms': 5, 'color': 'blue', 'alpha': 0.7, 'label': 'beginning'},
                        # 'axvline': {'x': 0, 'linestyle': 'dashed', 'color': 'gray', 'linewidth': 0.5},
                        # 'axhline': {'y': 0, 'linestyle': 'dashed', 'color': 'gray', 'linewidth': 0.5}
                    }

                    o_plot.set_plot_dics.append(plot_dic)

                    o_plot.main()
            else:
                print_colored_string(["task:", "plotmass", ":", "skipping"],
                                     ["blue", "green", "", "blue"])
        except IOError:
            print_colored_string(["task:", "plotmass", ":", "IOError"],
                                 ["blue", "green", "", "red"])
        except KeyboardInterrupt:
            exit(1)
        except:
            print_colored_string(["task:", "plotmass", ":", "failed"],
                                 ["blue", "green", "", "red"])




# _, itnuprofs, timenuprofs = self.get_ittime("profiles", "prof")
# _, itnuprofs, timenuprofs = self.get_ittime("nuprofiles", "nuprof")
#
# fpath = self.profpath + str(it) + self.nuprof_name + ".h5"
#
#
# # for corr plot
# self.list_iterations = Paths.get_list_iterations_from_res_3d(resdir)
# listdir = self.set_rootdir + str(it)
#
# fpath = resdir + "density_modes.h5",
#
# path = self.set_rootdir + str(it) + '/'
# fname = "profile" + '.' + plane + ".h5"
# fpath = path + fname


def get_list_iterations_from_res_3d(prodfir):
    """
    Checks the /res_3d/ for 12345 folders, (iterations) retunrs their sorted list
    :param sim:
    :return:
    """

    if not os.path.isdir(prodfir):
        raise IOError("no {} directory found".format(prodfir))

    itdirs = os.listdir(prodfir)

    if len(itdirs) == 0:
        raise NameError("No iteration-folders found in the {}".format(prodfir))

    # this is a f*cking masterpiece of programming)))
    list_iterations = np.array(
        np.sort(np.array(list([int(itdir) for itdir in itdirs if re.match("^[-+]?[0-9]+$", itdir)]))))
    if len(list_iterations) == 0:
        raise ValueError("Error extracting the iterations")

    return list(list_iterations)


def compute_methods_with_interpolation(
        glob_outdir,
        glob_fpaths,
        glob_its,
        glob_times,
        glob_tasklist,
        glob_masks,
        glob_symmetry,
        glob_overwrite
):
    outdir = glob_outdir
    outdir += __rootoutdir__
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # methods that required inteprolation [No masks used!]
    if "mjenclosed" in glob_tasklist:
        new_type = {'type': 'cyl', 'n_r': 75, 'n_phi': 64, 'n_z': 100}
        o_grid = CYLINDRICAL_GRID(grid_info=new_type)
        o_d3int = INTMETHODS_STORE(grid_object=o_grid, flist=glob_fpaths,
                                   itlist=glob_its, timesteplist=glob_times, symmetry=glob_symmetry)
        d3_interpolate_mjenclosed(o_d3int, glob_its, glob_masks, outdir, rewrite=False)

    if "vtk" in glob_tasklist:
        o_grid = CARTESIAN_GRID()
        o_d3int = INTMETHODS_STORE(grid_object=o_grid, flist=glob_fpaths,
                                   itlist=glob_its, timesteplist=glob_times, symmetry=glob_symmetry)

        d3_int_data_to_vtk(o_d3int, glob_its, glob_v_ns, outdir, rewrite=False)

        for it in glob_its:
            sys.stdout.flush()  # it, v_n_s, outdir, overwrite=False, private_dir="vtk"
            o_d3int.save_vtk_file(it, glob_v_ns, outdir=outdir, overwrite=False, private_dir="vtk/")
            sys.stdout.flush()

    if "densmodeint" in glob_tasklist:
        o_grid = POLAR_GRID()
        o_d3int = INTMETHODS_STORE(grid_object=o_grid, flist=glob_fpaths,
                                   itlist=glob_its, timesteplist=glob_times, symmetry=glob_symmetry)
        o_d3int.enforce_xy_grid = True
        d3_dens_modes_int(o_d3int, outdir=outdir, rewrite=glob_overwrite)

def compute_methods_with_original_data(
        glob_outdir,
        glob_fpaths,
        glob_its,
        glob_times,
        glob_tasklist,
        glob_masks,
        glob_symmetry,
        glob_overwrite
):

    outdir = glob_outdir
    outdir += __rootoutdir__
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # methods that do not require interplation [Use masks for reflevels and lapse]
    d3corr_class = MAINMETHODS_STORE(flist=glob_fpaths, itlist=glob_its,
                                     timesteplist=glob_its, symmetry=glob_symmetry)
    # d3corr_class.update_storage_lists(new_iterations=glob_its, new_times=glob_times) # remove corrupt
    # d3corr_class.mask_setup = {'rm_rl': True, # REMOVE previouse ref. level from the next
    #                             'rho': [6.e4 / 6.176e+17, 1.e13 / 6.176e+17],  # REMOVE atmo and NS
    #                             'lapse': [0.15, 1.]}  # remove apparent horizon

    # tasks for each iteration
    for it in glob_its:
        _outdir = outdir + str(it) + '/'
        if not os.path.isdir(_outdir):
            os.mkdir(_outdir)
        for task in glob_tasklist:
            # if task in ["all", "plotall", "densmode"]:   pass

            if task == "slice": d3_to_d2_slice_for_it(it, d3corr_class, glob_planes, _outdir, rewrite=glob_overwrite)

            for mask in glob_masks:

                __outdir = _outdir + mask + '/'
                if not os.path.isdir(__outdir):
                    os.mkdir(__outdir)

                if task == "corr":  d3_corr_for_it(it, d3corr_class, mask, glob_v_ns, __outdir, rewrite=glob_overwrite)
                if task == "hist":  d3_hist_for_it(it, d3corr_class, mask, glob_v_ns, __outdir, rewrite=glob_overwrite)
                if task == "mass": d3_mass_for_it(it, d3corr_class, mask, __outdir, rewrite=glob_overwrite)
                # d3_remnant_mass_for_it(it, d3corr_class, outdir, rewrite=glob_overwrite)
            # else:
            #     raise NameError("d3 method is not recognized: {}".format(task))
        d3corr_class.delete_for_it(it=it, except_v_ns=[], rm_masks=True, rm_comp=True, rm_prof=False)
        sys.stdout.flush()
        print("\n")

    # methods that require all iterations loaded
    if "densmode" in glob_tasklist:
        if len(glob_its) < 2:
            raise ValueError("For density model computation at least two iterstions are needed")
        d3_dens_modes(d3corr_class, outdir=outdir, rewrite=glob_overwrite)

    # summary plot of values in every iteration
    if "plotmass" in glob_tasklist:
        pass
        # plot_disk_mass(d3corr_class, resdir=outdir, rewrite=glob_overwrite)


def compute_methods_with_processed_data(
        glob_outdir,
        glob_fpaths,
        glob_its,
        glob_times,
        glob_tasklist,
        glob_masks,
        glob_symmetry,
        glob_overwrite
):

    ottime = LOAD_ITTIME(glob_sim, glob_outdir)
    _, _its, _times = ottime.get_ittime("profiles", "prof")

    outdir = glob_outdir
    outdir += __rootoutdir__
    if not os.path.isdir(outdir):
        raise IOError("Directory with processed profile data does not exists: {}".format(outdir))

    processed_iterations = np.array(get_list_iterations_from_res_3d(outdir), dtype=int)
    if len(processed_iterations) == 0:
        raise IOError("Directory with processed profile data is empty -- not subfolders with names 12345 found in {}"
                      .format(outdir))

    processed_timesteps = get_time_for_it(processed_iterations, _its, _times)

    dirpaths = [outdir + str(int(it)) + '/' for it in processed_iterations]

    # profiles_xy_paths = [dirpath + "profile.xy.h5" for dirpath in dirpaths]
    # profiles_xz_paths = [dirpath + "profile.xz.h5" for dirpath in dirpaths]

    # xy_slices = MAINMETHODS_STORE_SLICE(flist=profiles_xy_paths, itlist=processed_iterations, timesteplist=processed_timesteps)
    # xz_slices = MAINMETHODS_STORE_SLICE(flist=profiles_xz_paths, itlist=processed_iterations, timesteplist=processed_timesteps)



    # tasks that rely on the previos outputs
    for it in glob_its:

        if not it in processed_iterations:
            raise IOError("Iteration: {} has not been processed".format(it))

        _outdir = outdir + str(it) + '/'
        if not os.path.isdir(_outdir):
            os.mkdir(_outdir)
        for task in glob_tasklist:
            if task == "slicecorr":
                for plane in glob_planes:

                    profiles_xy_paths = [dirpath + "profile.{}.h5".format(plane) for dirpath in dirpaths]
                    o_slices = MAINMETHODS_STORE_SLICE(flist=profiles_xy_paths, itlist=processed_iterations,
                                                                                timesteplist=processed_timesteps)
                    d2_slice_corr_for_it(it, o_slices, glob_v_ns, glob_masks, plane, _outdir, rewrite=glob_overwrite)
                    sys.stdout.flush()


    d3_corr = LOAD_RES_CORR(dirlist=dirpaths, itlist=processed_iterations, timesteplist=processed_timesteps)
    dm_class = LOAD_DENSITY_MODES(fpath=outdir + __d3densitymodesfame__)


    # plotting tasks
    for task in glob_tasklist:
        if task.__contains__("plot"):
            print("Error, Plotting methods are not implemented")
            continue
            # if task in ["all", "plotall", "densmode"]:  pass
            if task == "plotcorr":          plot_d3_corr(d3_corr, processed_iterations, processed_timesteps,
                                                         glob_v_ns, glob_masks, outdir, rewrite=glob_overwrite)
            if task == "plotslicecorr":     plot_d2_slice_corr(d3_corr, processed_iterations, processed_timesteps,
                                                               glob_v_ns, glob_planes, glob_masks, outdir,
                                                               rewrite=glob_overwrite)
            if task == "plotslice":         plot_d3_prof_slices(d3class, glob_its, glob_v_ns, resdir, figdir='module_slices/', rewritefigs=False)
            if task == "plothist":          plot_d3_hist(d3_corr, rewrite=glob_overwrite)
            if task == "plotdensmode":      plot_density_modes(dm_class, rewrite=glob_overwrite)
            if task == "plotcenterofmass":  plot_center_of_mass(dm_class, rewrite=glob_overwrite)
            if task == "plotdensmodephase": plot_density_modes_phase(dm_class, rewrite=glob_overwrite)
            sys.stdout.flush()
            # else:
            #     raise NameError("glob_task for plotting is not recognized: {}"
            #                     .format(task))


if __name__ == '__main__':
    #
    parser = ArgumentParser(description="postprocessing pipeline")
    parser.add_argument("-s", dest="sim", required=True, help="task to perform")
    parser.add_argument("-t", dest="tasklist", required=False, nargs='+', default=[], help="tasks to perform")
    parser.add_argument('--mask', dest="mask", required=False, nargs='+', default=[],
                        help="Mask data for specific analysis. 'disk' is default ")
    parser.add_argument("--v_n", dest="v_ns", required=False, nargs='+', default=[], help="variable (or group) name")
    parser.add_argument("--rl", dest="reflevels", required=False, nargs='+', default=[], help="reflevels")
    parser.add_argument("--it", dest="iterations", required=False, nargs='+', default=[], help="iterations")
    parser.add_argument('--time', dest="times", required=False, nargs='+', default=[], help='Timesteps')
    parser.add_argument('--plane', dest="plane", required=False, nargs='+', default=[], help='Plane: xy,xz,yz for slice analysis')
    #
    parser.add_argument("-o", dest="outdir", required=False, default=None, help="path for output dir")
    parser.add_argument("-i", dest="indir", required=False, default=None, help="path to simulation dir")
    parser.add_argument("--overwrite", dest="overwrite", required=False, default="no", help="overwrite if exists")
    parser.add_argument("--maxtime", dest="maxtime", required=False, default=-1,
                        help=" limit the postprocessing to this vale of time [ms] ")
    #
    parser.add_argument("--sym", dest="symmetry", required=False, default=None, help="symmetry (like 'pi')")
    # Info/checks
    args = parser.parse_args()
    glob_tasklist = args.tasklist
    glob_sim = args.sim
    glob_indir = args.indir
    glob_outdir = args.outdir
    glob_v_ns = args.v_ns
    glob_rls = args.reflevels
    glob_its = args.iterations
    glob_times = args.times
    glob_planes = args.plane
    glob_symmetry = args.symmetry
    glob_overwrite = args.overwrite
    glob_masks = args.mask
    # simdir = Paths.gw170817 + glob_sim + '/'
    # resdir = Paths.ppr_sims + glob_sim + '/'
    # glob_usemaxtime = args.usemaxtime
    glob_maxtime = args.maxtime

    # check given data
    if glob_symmetry != None:
        if not click.confirm("Selected symmetry: {} Is it correct?".format(glob_symmetry),
                             default=True, show_default=True):
            exit(1)

    # check mask
    if len(glob_masks) == 0:
        glob_masks = ["disk"]
    elif len(glob_masks) == 1 and "all" in glob_masks:
        glob_masks = __masks__
    else:
        for mask in glob_masks:
            if not mask in __masks__:
                raise NameError("mask: {} is not recognized. Use: \n{}".format(mask, __masks__))
    # TODO Implement mask for every method, make clear that fr interpolation cases it is not used.
    #  See 'd2_slice_corr_for_it' for example

    # check plane


    # assert that paths to data and to output are valid
    if glob_indir is None:
        glob_indir = Paths.default_data_dir + glob_sim + '/' + Paths.default_profile_dic#"profiles/3d/"
        if not os.path.isdir(glob_indir):
            raise IOError("Default path for profiles is not valid: {}".format(glob_indir))
    if not os.path.isdir(glob_indir):
        raise IOError("Path for profiles is not valid: {}".format(glob_indir))
    if glob_outdir is None:
        glob_outdir = Paths.default_ppr_dir + glob_sim + '/'
        if not os.path.isdir(glob_outdir):
            raise IOError("Default output path is not valid: {}".format(glob_outdir))
    if not os.path.isdir(glob_outdir):
        raise IOError("Output path is not valid: {}".format(glob_outdir))

    # checking if to use maxtime
    if glob_maxtime == -1:
        glob_usemaxtime = False
        glob_maxtime = -1
        # print(glob_outdir + "maxtime.txt")
        if os.path.isfile(glob_outdir + "maxtime.txt"):
            glob_maxtime = float(np.loadtxt(glob_outdir + "maxtime.txt"))
            glob_usemaxtime = True
            Printcolor.print_colored_string(
                ["Note: maxtime.txt found. Setting value:","{:.1f}".format(glob_maxtime), "[ms]"],
                ["yellow", "green", "yellow"]
            )
    elif re.match(r'^-?\d+(?:\.\d+)?$', glob_maxtime):
        glob_maxtime = float(glob_maxtime)  # [ms]
        glob_usemaxtime = True
    else:
        raise NameError("To limit the data usage profive --maxtime in [ms] (after simulation start_. Given: {}"
                        .format(glob_maxtime))
    # if glob_maxtime == -1:
    #     glob_usemaxtime = False
    #     glob_maxtime = -1
    # elif re.match(r'^-?\d+(?:\.\d+)?$', glob_maxtime):
    #     glob_maxtime = float(glob_maxtime)  # [ms]
    #     glob_usemaxtime = True
    # else:
    #     raise NameError("To limit the data usage profive --maxtime in [ms] (after simulation start_. Given: {}"
    #                     .format(glob_maxtime))

    # check if tasks are set properly
    if len(glob_tasklist) == 0:
        raise NameError("tasklist is empty. Set what tasks to perform with '-t' option")
    elif len(glob_tasklist) == 1 and "all" in glob_tasklist:
        glob_tasklist = __tasklist__
        glob_tasklist.remove("vtk")
        Printcolor.print_colored_string(["Set", "All", "tasks"],
                                        ["blue", "green", "blue"])
    else:
        for task in glob_tasklist:
            if not task in __tasklist__:
                raise NameError("task: {} is not among available ones: {}"
                                .format(task, __tasklist__))

    # assert planes
    if len(glob_planes) == 0:
        for t in glob_tasklist:
            if t.__contains__('slice'):
                raise IOError("Task with slice '{}' requires --plane parameters".format(t))
    elif len(glob_planes) == 1 and "all" in glob_planes:
        glob_planes = __d3slicesplanes__
    elif len(glob_planes) > 1:
        for plane in glob_planes:
            if not plane in __d3slicesplanes__:
                raise NameError("plane:{} is not in the list of the __d3slicesplanes__:{}"
                                .format(plane, __d3slicesplanes__))

    # check if there any profiles to use
    ittime = LOAD_ITTIME(glob_sim, pprdir=glob_outdir)
    _, itprof, tprof = ittime.get_ittime("profiles", d1d2d3prof="prof")
    #
    if len(itprof) == 0:
        Printcolor.red("No profiles found. Please, extract profiles for {} "
                         "and save them in /sim_dir/{} and/or update ittime.h5"
                       .format(glob_sim, Paths.default_profile_dic))
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
        Printcolor.print_colored_string(["Tasked with All", "{}".format(len(itprof)), "iterations to postprocess"],
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
                _glob_times = np.append(_glob_times, ittime.get_time_for_it(it, "profiles", "prof"))
            else:
                raise ValueError("For given iteraton:{} module_profile is not found (in ittime.h5)"
                                 .format(it))
        glob_its = _glob_its
        glob_times = _glob_times
        assert len(glob_its) > 0
    elif len(glob_its) == 0 and len(glob_times) > 0:
        glob_times = np.array(glob_times, dtype=float) / 1e3 # back to [s]
        _glob_its = []
        _glob_times = []
        for t in glob_times:
            idx = Tools.find_nearest_index(tprof, t)
            _t =  tprof[idx]
            _it = ittime.get_it_for_time(_t, output="overall", d1d2d3="d1")
            _glob_its = np.append(_glob_its, _it)
            _glob_times = np.append(_glob_times, _t)
        glob_its = np.unique(_glob_its)
        glob_times = np.unique(_glob_times)
        assert len(glob_its) > 0
        assert len(glob_times) == len(glob_its)
    else:
        raise IOError("Input iterations (--it) or times (--time) are not recognized")

    assert len(glob_times) == len(glob_its)
    # get maximum available iteration
    regected_its = []
    regected_times = []
    if glob_usemaxtime and (~np.isnan(glob_maxtime) or ~np.isnan(ittime.maxtime)):
        # use maxtime, just chose which
        if np.isnan(glob_maxtime) and not np.isnan(ittime.maxtime):
            maxtime = ittime.maxtime
        elif not np.isnan(glob_maxtime) and not np.isnan(ittime.maxtime):
            maxtime = glob_maxtime
            Printcolor.yellow("\tOverwriting ittime maxtime:{:.1f}ms with {:.1f}ms"
                              .format(ittime.maxtime * 1.e3, glob_maxtime * 1.e3))
        elif np.isnan(glob_maxtime) and np.isnan(ittime.maxtime):
            maxtime = tprof[-1]
        else:
            maxtime = glob_maxtime
        #
        regected_times = glob_times[glob_times > maxtime]
        regected_its = glob_its[glob_times > maxtime]
        glob_its = glob_its[glob_times <= maxtime]
        glob_times = glob_times[glob_times <= maxtime]
        #
        if len(glob_times) == 0:
            Printcolor.print_colored_string(["Max. it set:", "{}".format(glob_its[-1]), "out of",
                                             "{}".format(itprof[-1]), "leaving", str(len(glob_its)), "its"],
                                            ["yellow", "green", "yellow", "green", "blue", "red", "blue"])
        else:
            Printcolor.print_colored_string(["Max. it set:", "{}".format(glob_its[-1]), "out of",
                                             "{}".format(itprof[-1]), "leaving", str(len(glob_its)), "its"],
                                            ["yellow", "green", "yellow", "green", "blue", "green", "blue"])
    if len(regected_its) > 0:
        Printcolor.print_colored_string(["Ignore --it (beyond maxtime)  ", "{}".format(regected_its)],
                                        ["red", "red"])
        Printcolor.print_colored_string(["Ignore --time (beyond maxtime)", "{}".format(regected_times * 1.e3, fmt=".1f")],
                                        ["red", "red"])

    # if source parfie.h5 is corrupt remove corresponding iteration fromthe list
    _corrupt_it = []
    _corrput_times = []
    _noncorrupt_it = []
    _non_corrupttimes = []
    for it, t in zip(glob_its, glob_times):
        fname = glob_indir + str(int(it)) + ".h5"
        assert os.path.isdir(glob_indir)
        assert os.path.isfile(glob_indir + str(int(it)) + ".h5")
        _is_readable = h5py.is_hdf5(fname)
        if not _is_readable:
            _corrupt_it.append(it)
            _corrput_times.append(t)
        else:
            _noncorrupt_it.append(it)
            _non_corrupttimes.append(t)
    #
    _corrupt_it = np.array(_corrupt_it, dtype=int)
    _corrput_times = np.array(_corrput_times, dtype=float)
    glob_its = np.array(_noncorrupt_it, dtype=int)
    glob_times = np.array(_non_corrupttimes, dtype=float)
    glob_fpaths = [glob_indir + str(int(it)) + ".h5" for it in glob_its]

    if len(_corrupt_it) > 0:
        Printcolor.print_colored_string(["Ignore --it (corrput h5)  ", "{}".format(_corrupt_it)], ["red", "red"])
        Printcolor.print_colored_string(["Ignore --time (corrupt h5)", "{}".format(_corrput_times * 1e3, fmt=".1f")],
                                        ["red", "red"])
    if len(glob_its) == 0:
        Printcolor.print_colored_string(["Set --it (avial)    ", "{}".format(glob_its)],["blue","red"])
        Printcolor.print_colored_string(["Set --time (avail)  ", "{}".format(glob_times*1e3, fmt=".1f")], ["blue", "red"])
    else:
        Printcolor.print_colored_string(["Set --it (avial)    ", "{}".format(glob_its)], ["blue", "green"])
        Printcolor.print_colored_string(["Set --time (avail)  ", "{}".format(glob_times * 1e3, fmt=".1f")],
                                        ["blue", "green"])
    #
    if glob_overwrite == "no":
        glob_overwrite = False
    elif glob_overwrite == "yes":
        glob_overwrite = True
    else:
        raise NameError("for '--overwrite' option use 'yes' or 'no'. Given: {}"
                        .format(glob_overwrite))

    # exit(0)

    # tasks
    if len(glob_its) > 0:
        compute_methods_with_interpolation(
            glob_outdir,
            glob_fpaths,
            glob_its,
            glob_times,
            glob_tasklist,
            glob_masks,
            glob_symmetry,
            glob_overwrite
        )
        compute_methods_with_original_data(
            glob_outdir,
            glob_fpaths,
            glob_its,
            glob_times,
            glob_tasklist,
            glob_masks,
            glob_symmetry,
            glob_overwrite
        )
        compute_methods_with_processed_data(
            glob_outdir,
            glob_fpaths,
            glob_its,
            glob_times,
            glob_tasklist,
            glob_masks,
            glob_symmetry,
            glob_overwrite
        )
        Printcolor.blue("Done.")
    else:
        Printcolor.yellow("No iterations set.")
