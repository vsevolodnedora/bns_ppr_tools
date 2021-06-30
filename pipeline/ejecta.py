"""
    pass
"""

from __future__ import division
import numpy as np
import re
import h5py
import os
from argparse import ArgumentParser
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

__tasklist__ = ["reshape", "all", "hist", "timecorr", "corr", "totflux",
                 "massave", "ejtau", "yields", "thetaprof", "summary"]

__masks__ = ["geo", "geo_v06", "bern_geoend", "Y_e04_geoend", "theta60_geoend"]

from standalone.outflowsurfdens_asciiToH5 import do_reshape
from uutils import (Printcolor, Labels, Limits, Constants)
from module_ejecta.ejecta_methods import (EJECTA_PARS)
from plotting.plotting_methods import PLOT_MANY_TASKS
import config as Paths

filename = lambda det: "outflow_surface_det_%d_fluxdens.h5" % det

def outflowed_historgrams(o_outflow, pprdir, det, masks, v_ns, rewrite=False):

    # exit(1)

    # creating histograms

    def task(mask, v_n, outdir):
        # print(np.min(o_outflow.get_full_arr("rho")),np.max(o_outflow.get_full_arr("rho")))
        hist = o_outflow.get_ejecta_arr(mask, "hist {}".format(v_n))
        # print(hist)
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
            'xmin': None, 'xamx': None, 'ymin': 1e-4, 'ymax': 1e0,
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

    for mask in masks:
        outdir = pprdir + "outflow_{}/".format(det) + mask + '/'
        for v_n in v_ns:
            fpath = outdir + "/hist_{}.dat".format(v_n)
            if Paths.debug:
                task(mask, v_n, outdir)
            else:
                try:
                    if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                        if os.path.isfile(fpath): os.remove(fpath)
                        Printcolor.print_colored_string(
                            ["task:", "d1hist", "det:", "{}".format(det), "mask:", mask, "v_n:", v_n, ":", "computing"],
                            ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "green"])
                        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                        task(mask, v_n, outdir)
                        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                    else:
                        Printcolor.print_colored_string(
                            ["task:", "d1hist", "det:", "{}".format(det), "mask:", mask, "v_n:", v_n, ":", "skipping"],
                            ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "blue"])
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

def outflowed_correlations(o_outflow, pprdir, det, masks, v_ns, rewrite=False):

    def task(mask, v_n1, v_n2, fpath, outdir):
        corr = o_outflow.get_ejecta_arr(mask, "corr2d {} {}".format(v_n1, v_n2))
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
            'v_n_x': v_n1, 'v_n_y': v_n2, 'v_n': 'mass', 'normalize': True,
            'cbar': {
                'location': 'right .03 .0', 'label': Labels.labels("mass"),  # 'fmt': '%.1f',
                'labelsize': 14, 'fontsize': 14},
            'cmap': 'inferno_r', 'set_under': 'white', 'set_over': 'black',
            'xlabel': Labels.labels(v_n1), 'ylabel': Labels.labels(v_n2),
            'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None, 'vmin': 1e-4, 'vmax': 1e-1,
            'xscale': "linear", 'yscale': "linear", 'norm': 'log',
            'mask_below': None, 'mask_above': None,
            'title': {},  # {"text": o_corr_data.sim.replace('_', '\_'), 'fontsize': 14},
            'fancyticks': True,
            'minorticks': True,
            'sharex': False,  # removes angular citkscitks
            'sharey': False,
            'fontsize': 14,
            'labelsize': 14
        }
        corr_dic2 = Limits.in_dic(corr_dic2)

        corr_dic2["axhline"] = {"y": 60, "linestyle": "-", "linewidth": 0.5, "color": "black"}
        corr_dic2["axvline"] = {"x": 0.4, "linestyle": "-", "linewidth": 0.5, "color": "black"}

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

    assert len(v_ns) % 2 == 0

    for mask in masks:
        outdir = pprdir + "outflow_{}/".format(det) + mask + '/'
        for v_n1, v_n2 in zip(v_ns[::2], v_ns[1::2]):
            fpath = outdir + "corr_{}_{}.h5".format(v_n1, v_n2)
            if Paths.debug:
                task(mask, v_n1, v_n2, fpath, outdir)
            else:
                try:
                    if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                        if os.path.isfile(fpath): os.remove(fpath)
                        Printcolor.print_colored_string(
                            ["task:", "d1corr", "det:", "{}".format(det), "mask:", mask, "v_n:", "{}_{}".format(v_n1, v_n2),
                             ":", "computing"],
                            ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "green"])
                        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                        task(mask, v_n1, v_n2, fpath, outdir)
                        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                    else:
                        Printcolor.print_colored_string(
                            ["task:", "d1corr", "det:", "{}".format(det), "mask:", mask, "v_n:", "{}_{}".format(v_n1, v_n2),
                             ":", "skipping"],
                            ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "blue"])
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
                        ["task:", "d1corr", "det:", "{}".format(det), "mask:", mask, "v_n:", "{}_{}".format(v_n1, v_n2),
                         ":", "failed"],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])

def outflowed_timecorr(o_outflow, pprdir, det, masks, v_ns, rewrite=False):

    # assert len(v_ns) % 2 == 0

    def task(mask, v_n, fpath, outdir):
        table = o_outflow.get_ejecta_arr(mask, "timecorr {}".format(v_n))
        table[0, 1:] *= Constants.time_constant
        timearr = table[0, 1:]
        yarr = table[1:, 0]
        zarr = table[1:, 1:]

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
            'v_n_x': "time", 'v_n_y': v_n, 'v_n': 'mass', 'normalize': True,
            'cbar': {
                'location': 'right .03 .0', 'label': Labels.labels("mass"),  # 'fmt': '%.1f',
                'labelsize': 14, 'fontsize': 14},
            'cmap': 'inferno',
            'xlabel': Labels.labels("time"), 'ylabel': Labels.labels(v_n),
            'xmin': timearr[0], 'xmax': timearr[-1], 'ymin': None, 'ymax': None, 'vmin': 1e-4, 'vmax': 1e-1,
            'xscale': "linear", 'yscale': "linear", 'norm': 'log',
            'mask_below': None, 'mask_above': None,
            'title': {},  # {"text": o_corr_data.sim.replace('_', '\_'), 'fontsize': 14},
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

    for mask in masks:
        outdir = pprdir + "outflow_{}/".format(det) + mask + '/'
        for v_n in v_ns:
            fpath = outdir + "timecorr_{}.h5".format(v_n)
            if Paths.debug:
                task(mask, v_n, fpath, outdir)
            else:
                try:
                    if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                        if os.path.isfile(fpath): os.remove(fpath)
                        Printcolor.print_colored_string(
                            ["task:", "timecorr", "det:", "{}".format(det), "mask:", mask, "v_n:", "{}".format(v_n), ":",
                             "computing"],
                            ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "green"])
                        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                        task(mask, v_n, fpath, outdir)
                        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                    else:
                        Printcolor.print_colored_string(
                            ["task:", "timecorr", "det:", "{}".format(det), "mask:", mask, "v_n:", "{}".format(v_n), ":",
                             "skipping"],
                            ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "blue"])
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
                        ["task:", "timecorr", "det:", "{}".format(det), "mask:", mask, "v_n:", "{}".format(v_n), ":",
                         "failed"],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"])

def outflowed_totmass(o_outflow, pprdir, det, masks, rewrite=False):

    def task(mask, fpath, outdir):
        data = o_outflow.get_ejecta_arr(mask, "tot_flux")
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
            'xarr': data[:, 0] * 1e3, 'yarr': data[:, 2] * 1e2,
            'v_n_x': "time", 'v_n_y': "mass",
            'color': "black", 'ls': '-', 'lw': 0.8, 'ds': 'default', 'alpha': 1.0,
            'ymin': 0, 'ymax': 3.0, 'xmin': np.array(data[:, 0] * 1e3).min(),
            'xmax': np.array(data[:, 0] * 1e3).max(),
            'xlabel': Labels.labels("time"), 'ylabel': Labels.labels("ejmass"),
            'label': None, 'yscale': 'linear',
            'fancyticks': True, 'minorticks': True,
            'fontsize': 14,
            'labelsize': 14,
            'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
        }

        o_plot.set_plot_dics.append(plot_dic)
        o_plot.main()

    for mask in masks:
        outdir = pprdir + "outflow_{}/".format(det) + mask + '/'
        fpath = outdir + "total_flux.dat"
        if Paths.debug:
            task(mask, fpath, outdir)
        else:
            try:
                if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                    if os.path.isfile(fpath): os.remove(fpath)
                    Printcolor.print_colored_string(
                        ["task:", "mass flux", "det:", "{}".format(det), "mask:", mask, ":", "computing"],
                        ["blue", "green", "blue", "green", "blue", "green", "", "green"])
                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                    task(mask, fpath, outdir)
                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                else:
                    Printcolor.print_colored_string(
                        ["task:", "mass flux", "det:", "{}".format(det), "mask:", mask, ":", "skipping"],
                        ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
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

def outflowed_massaverages(o_outflow, pprdir, det, masks, rewrite=False):

    def task(mask, fpath):
        v_ns = ["fluxdens", "w_lorentz", "eninf", "surface_element", "rho", "Y_e", "entropy", "temperature"]
        dfile = h5py.File(fpath, "w")
        for v_n in v_ns:
            arr = o_outflow.get_ejecta_arr(mask, "mass_ave Y_e")
            # print(arr.shape)
            dfile.create_dataset(v_n, data=arr)
        # print("end")
        dfile.create_dataset("theta", data=o_outflow.get_full_arr("theta"))
        dfile.create_dataset("phi", data=o_outflow.get_full_arr("phi"))
        dfile.close()

    for mask in masks:
        outdir = pprdir + "outflow_{}/".format(det) + mask + '/'
        fpath = outdir + "mass_averages.h5"
        if Paths.debug:
            task(mask, fpath)
        else:
            try:
                if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                    if os.path.isfile(fpath): os.remove(fpath)
                    Printcolor.print_colored_string(
                        ["task:", "mass averages", "det:", "{}".format(det), "mask:", mask, ":", "computing"],
                        ["blue", "green", "blue", "green", "blue", "green", "", "green"])
                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                    task(mask, fpath)
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

def outflowed_ejectatau(o_outflow, pprdir, det, masks, rewrite=False):

    def task(mask, fpath):
        arr = o_outflow.get_ejecta_arr(mask, "corr3d Y_e entropy tau")
        ye, entropy, tau = arr[1:, 0, 0], arr[0, 1:, 0], arr[0, 0, 1:]
        mass = arr[1:, 1:, 1:]

        assert ye.min() > 0. and ye.max() < 0.51
        assert entropy.min() > 0. and entropy.max() < 201.

        dfile = h5py.File(fpath, "w")
        dfile.create_dataset("Y_e", data=ye)
        dfile.create_dataset("entropy", data=entropy)
        dfile.create_dataset("tau", data=tau)
        dfile.create_dataset("mass", data=mass)
        dfile.close()

    for mask in masks:
        outdir = pprdir + "outflow_{}/".format(det) + mask + '/'
        fpath = outdir + "module_ejecta.h5"
        if Paths.debug:
            task(mask, fpath)
        else:
            try:
                if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                    if os.path.isfile(fpath): os.remove(fpath)
                    Printcolor.print_colored_string(
                        ["task:", "module_ejecta tau", "det:", "{}".format(det), "mask:", mask, ":", "computing"],
                        ["blue", "green", "blue", "green", "blue", "green", "", "green"])
                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                    task(mask, fpath)
                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                else:
                    Printcolor.print_colored_string(
                        ["task:", "module_ejecta tau", "det:", "{}".format(det), "mask:", mask, ":", "skipping"],
                        ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
            except KeyboardInterrupt:
                Printcolor.red("Forced termination... done")
                exit(1)
            except ValueError:
                Printcolor.print_colored_string(
                    ["task:", "module_ejecta tau", "det:", "{}".format(det), "mask:", mask, ":", "ValueError"],
                    ["blue", "green", "blue", "green", "blue", "green", "", "red"])
            except:
                Printcolor.print_colored_string(
                    ["task:", "module_ejecta tau", "det:", "{}".format(det), "mask:", mask, ":", "failed"],
                    ["blue", "green", "blue", "green", "blue", "green", "", "red"])

def outflowed_yields(o_outflow, pprdir, det, masks, rewrite=False):

    def task(mask, fpath, outdir):
        yields = o_outflow.get_nucleo_arr(mask, "yields")
        a = o_outflow.get_nucleo_arr(mask, "A")
        z = o_outflow.get_nucleo_arr(mask, "Z")
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

        sim_nuc = o_outflow.get_normed_sim_abund(mask, "Asol=195")
        sol_nuc = o_outflow.get_nored_sol_abund("sum")
        sim_nucleo = {
            'task': 'line', 'ptype': 'cartesian',
            'position': (1, 1),
            'xarr': sim_nuc[:, 0], 'yarr': sim_nuc[:, 1],
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
            'xarr': sol_nuc[:, 0], 'yarr': sol_nuc[:, 1],
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

    for mask in masks:
        outdir = pprdir + "outflow_{}/".format(det) + mask + '/'
        fpath = outdir + "yields.h5"
        if Paths.debug:
            task(mask, fpath, outdir)
        else:
            try:
                if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                    if os.path.isfile(fpath): os.remove(fpath)
                    Printcolor.print_colored_string(
                        ["task:", "module_ejecta nucleo", "det:", "{}".format(det), "mask:", mask, ":", "computing"],
                        ["blue", "green", "blue", "green", "blue", "green", "", "green"])
                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                    task(mask, fpath, outdir)
                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                else:
                    Printcolor.print_colored_string(
                        ["task:", "module_ejecta nucleo", "det:", "{}".format(det), "mask:", mask, ":", "skipping"],
                        ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
            except KeyboardInterrupt:
                Printcolor.red("Forced termination... done")
                exit(1)
            except ValueError:
                Printcolor.print_colored_string(
                    ["task:", "module_ejecta nucleo", "det:", "{}".format(det), "mask:", mask, ":", "ValueError"],
                    ["blue", "green", "blue", "green", "blue", "green", "", "red"])
            except NameError:
                Printcolor.print_colored_string(
                    ["task:", "module_ejecta nucleo", "det:", "{}".format(det), "mask:", mask, ":", "NameError"],
                    ["blue", "green", "blue", "green", "blue", "green", "", "red"])
            except:
                Printcolor.print_colored_string(
                    ["task:", "module_ejecta nucleo", "det:", "{}".format(det), "mask:", mask, ":", "failed"],
                    ["blue", "green", "blue", "green", "blue", "green", "", "red"])

def outflowed_mkn_profile(o_outflow, pprdir, det, masks, rewrite=False):

    def task(mask, fpath, outdir):
        corr_ye_theta = o_outflow.get_ejecta_arr(mask, "corr2d Y_e theta")
        corr_vel_inf_theta = o_outflow.get_ejecta_arr(mask, "corr2d vel_inf theta")

        # print(corr_vel_inf_theta[0, 1:]) # velocity
        # print(corr_vel_inf_theta[1:, 0])  # theta
        assert corr_vel_inf_theta[0, 1:].min() > 0. and corr_vel_inf_theta[0, 1:].max() < 1.
        assert corr_vel_inf_theta[1:, 0].min() > 0. and corr_vel_inf_theta[1:, 0].max() < 3.14
        assert corr_ye_theta[0, 1:].min() > 0.035 and corr_ye_theta[0, 1:].max() < 0.55

        vel_v = np.array(corr_vel_inf_theta[0, 1:])
        thf = np.array(corr_vel_inf_theta[1:, 0])
        vel_M = np.array(corr_vel_inf_theta[1:, 1:]).T

        ye_ye = np.array(corr_ye_theta[0, 1:])
        ye_M = np.array(corr_ye_theta[1:, 1:]).T

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
            'xarr': M_of_th * 1e3, 'yarr': 90. - (thf / np.pi * 180.),
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
            'legend': {'loc': 'best', 'ncol': 1, 'fontsize': 14}
        }
        o_plot.set_plot_dics.append(ye_dic)
        o_plot.main()

    for mask in masks:
        outdir = pprdir + "outflow_{}/".format(det) + mask + '/'
        fpath = outdir + "ejecta_profile.dat"
        if Paths.debug:
            task(mask, fpath, outdir)
        else:
            try:
                if (os.path.isfile(fpath) and rewrite) or not os.path.isfile(fpath):
                    if os.path.isfile(fpath): os.remove(fpath)
                    Printcolor.print_colored_string(
                        ["task:", "module_ejecta nucleo", "det:", "{}".format(det), "mask:", mask, ":", "computing"],
                        ["blue", "green", "blue", "green", "blue", "green", "", "green"])
                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                    task(mask, fpath, outdir)
                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                else:
                    Printcolor.print_colored_string(
                        ["task:", "module_ejecta nucleo", "det:", "{}".format(det), "mask:", mask, ":", "skipping"],
                        ["blue", "green", "blue", "green", "blue", "green", "", "blue"])
            except KeyboardInterrupt:
                Printcolor.red("Forced termination... done")
                exit(1)
            except ValueError:
                Printcolor.print_colored_string(
                    ["task:", "module_ejecta nucleo", "det:", "{}".format(det), "mask:", mask, ":", "failed"],
                    ["blue", "green", "blue", "green", "blue", "green", "", "red"])
            except:
                Printcolor.print_colored_string(
                    ["task:", "module_ejecta nucleo", "det:", "{}".format(det), "mask:", mask, ":", "failed"],
                    ["blue", "green", "blue", "green", "blue", "green", "", "red"])

def outflowed_summary(o_outflow, pprdir, det, masks, rewrite=False):

    def task(outdir, outfpath):

        # total flux
        fpath = outdir + "total_flux.dat"
        if os.path.isfile(fpath):
            data = np.array(np.loadtxt(fpath))
            mass_arr = data[:, 2]
            time_arr = data[:, 0]
            total_ej_mass = float(mass_arr[-1])
            time_end = float(time_arr[-1])  # * Constants.time_constant * 1e-3 # s
        else:
            total_ej_mass = np.nan
            time_end = np.nan
            Printcolor.red("Missing: {}".format(fpath))

        # Y_e ave
        v_n = "Y_e"
        fpath = outdir + "hist_{}.dat".format(v_n)
        if os.path.isfile(fpath):
            hist = np.array(np.loadtxt(fpath))
            ye_ave = o_outflow.compute_ave_ye(np.sum(hist[:, 1]), hist)
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
            f1.write("# module_ejecta properties for det:{} mask:{} \n".format(det, mask))
            f1.write("m_ej      {:.5f} [M_sun]  total ejected mass \n".format(total_ej_mass))
            f1.write("<Y_e>     {:.3f}            mass-averaged electron fraction \n".format(ye_ave))
            f1.write("<s>       {:.3f} [k_b]      mass-averaged entropy \n".format(s_ave))
            f1.write("<v_inf>   {:.3f} [c]        mass-averaged terminal velocity \n".format(vel_inf_ave))
            f1.write("<E_kin>   {:.3f} [c^2]      mass-averaged terminal kinetical energy \n".format(e_kin_ave))
            f1.write(
                "theta_rms {:.2f} [degrees]  root mean squared angle of the module_ejecta (2 planes) \n".format(
                    2. * theta_rms))
            f1.write("time_end  {:.3f} [s]        end data time \n".format(time_end))

    for mask in masks:
        outdir = pprdir + "outflow_{}/".format(det) + mask + '/'
        #
        outfpath = outdir + "summary.txt"
        if Paths.debug:
            task(outdir, outfpath)
        else:
            try:
                if (os.path.isfile(outfpath) and rewrite) or not os.path.isfile(outfpath):
                    if os.path.isfile(outfpath): os.remove(outfpath)
                    Printcolor.print_colored_string(
                        ["task:", "summary", "det:", "{}".format(det), "mask:", mask, ":", "computing"],
                        ["blue", "green", "blue", "green", "blue", "green", "", "green"])
                    task(outdir, outfpath)
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


def do_folder_tree(glob_outdir):
    # prepare dir tree for other tasks output
    assert len(glob_masks) > 0
    assert len(glob_detectors) > 0
    # outdir = Paths.ppr_sims + glob_sim + '/'
    if not os.path.isdir(glob_outdir):
        raise IOError("Target directory: {} not found".format(glob_outdir))
    for det in glob_detectors:
        outdir_ = glob_outdir + "outflow_{}/".format(det)
        if not os.path.isdir(outdir_):
            os.mkdir(outdir_)
        for mask in glob_masks:
            outdir__ = outdir_ + mask + '/'
            if not os.path.isdir(outdir__):
                os.mkdir(outdir__)

def do_full_analysis():

    for det in glob_detectors:

        fname = glob_outdir + filename(det)

        if not os.path.isfile(fname):
            print("Error. Analysis not possible. File not found {}".format(fname))
            Printcolor.print_colored_string(
                ["task:", "all", "det:", "{}".format(det), ':', "failed"],
                ["blue", "green", "blue", "green", "", "red"])
            continue

        outflowed = EJECTA_PARS(fname=fname, skynetdir=glob_skynet, add_mask=None)

        v_ns = []
        for v_n in outflowed.list_corr_v_ns:
            v_ns += v_n.split()
        outflowed_correlations(outflowed, glob_outdir, det, glob_masks, v_ns, glob_overwrite)
        outflowed_timecorr(outflowed, glob_outdir, det, glob_masks, outflowed.list_hist_v_ns, glob_overwrite)
        outflowed_historgrams(outflowed, glob_outdir, det, glob_masks, outflowed.list_hist_v_ns, glob_overwrite)
        outflowed_totmass(outflowed, glob_outdir, det, glob_masks, glob_overwrite)
        outflowed_massaverages(outflowed, glob_outdir, det, glob_masks, glob_overwrite)
        outflowed_ejectatau(outflowed, glob_outdir, det, glob_masks, glob_overwrite)
        outflowed_yields(outflowed, glob_outdir, det, glob_masks, glob_overwrite)
        outflowed_mkn_profile(outflowed, glob_outdir, det, glob_masks, glob_overwrite)
        outflowed_summary(outflowed, glob_outdir, det, glob_masks, glob_overwrite)


def do_selected_tasks():
    for det in glob_detectors:

        fname = glob_outdir + filename(det)
        outflowed = EJECTA_PARS(fname=fname, skynetdir=glob_skynet, add_mask=None)

        for task in glob_tasklist:

            if task == "reshape":
                pass
            elif task == "hist":
                assert len(glob_v_ns) > 0
                outflowed_historgrams(outflowed, glob_outdir, det, glob_masks, glob_v_ns, glob_overwrite)
            elif task == "timecorr":
                assert len(glob_v_ns) > 0
                outflowed_timecorr(outflowed, glob_outdir, det, glob_masks, glob_v_ns, glob_overwrite)
            elif task == "corr":
                assert len(glob_v_ns) > 0
                outflowed_correlations(outflowed, glob_outdir, det, glob_masks, glob_v_ns, glob_overwrite)
            elif task == "totflux":
                outflowed_totmass(outflowed, glob_outdir, det, glob_masks, glob_overwrite)
            elif task == "massave":
                outflowed_massaverages(outflowed, glob_outdir, det, glob_masks, glob_overwrite)
            elif task == "ejtau":
                outflowed_ejectatau(outflowed, glob_outdir, det, glob_masks, glob_overwrite)
            elif task == "yields":
                outflowed_yields(outflowed, glob_outdir, det, glob_masks, glob_overwrite)
            elif task == "thetaprof":
                outflowed_mkn_profile(outflowed, glob_outdir, det, glob_masks, glob_overwrite)
            elif task == "summary":
                outflowed_summary(outflowed, glob_outdir, det, glob_masks, glob_overwrite)
            else:
                raise NameError("No method fund for task: {}".format(task))


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
    # parser.add_argument("--usemaxtime", dest="usemaxtime", required=False, default="no",
    #                     help=" auto/no to use ittime.h5 set value. Or set a float [ms] to overwrite ")
    parser.add_argument("--maxtime", dest="maxtime", required=False, default=-1.,
                        help="Time limiter for 'reshape' task only")
    parser.add_argument("-o", dest="outdir", required=False, default=None, help="path for output dir")
    parser.add_argument("-i", dest="indir", required=False, default=None, help="path to simulation dir")
    parser.add_argument("--overwrite", dest="overwrite", required=False, default="no", help="overwrite if exists")
    #
    parser.add_argument("--eos", dest="eosfpath", required=False, default=None, help="Hydro EOS to use")
    parser.add_argument("--skynet", dest="skynet", required=False, default=None, help="Path to skynet directory")
    # examples
    # python old_outflowed.py -s SLy4_M13641364_M0_SR -t d1hist -v Y_e vel_inf theta phi entropy -d 0 -m geo --overwrite yes
    # python old_outflowed.py -s SLy4_M13641364_M0_SR -t d1corr -v Y_e theta vel_inf theta -d 0 -m geo --overwrite yes
    #
    args = parser.parse_args()
    glob_sim = args.sim
    glob_eos = args.eosfpath
    glob_skynet = args.skynet
    glob_indir = args.indir
    glob_outdir = args.outdir
    glob_tasklist = args.tasklist
    glob_overwrite = args.overwrite
    glob_detectors = np.array(args.detectors, dtype=int)
    glob_v_ns = args.v_ns
    glob_masks = args.masks
    glob_nproc = int(args.num_proc)
    # glob_usemaxtime = args.usemaxtime
    glob_maxtime = args.maxtime
    # check given data

    #
    if glob_indir is None:
        glob_indir = Paths.default_data_dir + glob_sim + '/'
        if not os.path.isdir(glob_indir):
            raise IOError("Default data dir not found: {}".format(glob_indir))

    if glob_outdir is None:
        glob_outdir = Paths.default_ppr_dir + glob_sim + '/'
        if not os.path.isdir(glob_outdir):
            raise IOError("Default output dir not found: {}".format(glob_outdir))

    # if not os.path.isdir(glob_simdir + glob_sim):
    #     raise NameError("simulation dir: {} does not exist in rootpath: {} "
    #                     .format(glob_sim, glob_simdir))
    if len(glob_tasklist) == 0:
        raise NameError("tasklist is empty. Set what tasks to perform with '-t' option")
    else:
        for task in glob_tasklist:
            if task not in __tasklist__:
                raise NameError("task: {} is not among available ones: {}"
                                .format(task, __tasklist__))

    if glob_overwrite == "no":
        glob_overwrite = False
    elif glob_overwrite == "yes":
        glob_overwrite = True
    else:
        raise NameError("for '--overwrite' option use 'yes' or 'no'. Given: {}"
                        .format(glob_overwrite))

    # glob_outdir_sim = Paths.ppr_sims + glob_sim
    # if not os.path.isdir(glob_outdir_sim):
    #     os.mkdir(glob_outdir_sim)
    if len(glob_detectors) == 0:
        raise NameError("No detectors selected. Set '-d' option to 0, 1, etc")

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

    # exit(1)

    # do tasks
    if ("reshape" in glob_tasklist) and (len(glob_tasklist) == 1):

        if glob_eos is None:
            glob_eos = Paths.get_eos_fname_from_curr_dir(glob_sim)
            if not os.path.isfile(glob_eos):
                raise IOError("Default eos file for a simulation {} not found : {}".format(glob_sim, glob_eos))

        if not os.path.isfile(glob_eos):
            raise IOError("Given eos file not found : {}".format(glob_eos))

        do_reshape(
            glob_detectors,
            glob_nproc,
            glob_maxtime,
            glob_outdir,
            glob_indir,
            glob_eos,
            glob_overwrite
        )

        exit(0)

    if "all" in glob_masks and len(glob_masks) == 1:
        glob_masks = __masks__

    # check path to skynet files (needed for nucleo analysis)
    if glob_skynet is None:
        glob_skynet = Paths.skynet
        if not os.path.isdir(glob_skynet):
            raise IOError("Default path to skynet folder is not valid: {}".format(glob_skynet))
    if not os.path.isdir(glob_skynet):
        raise IOError("Given path to skynet folder is not valid: {}".format(glob_skynet))

    # prepare dir tree for other tasks output
    do_folder_tree(glob_outdir)

    # pipeline
    if len(glob_tasklist) == 1 and ("all" in glob_tasklist):
        do_full_analysis()
    else:
        do_selected_tasks()
    # # selected tasks
    # do_selected_tasks(outflowed)
    #
    # for task in glob_tasklist:
    #     if task == "reshape":
    #         pass
    #     elif task == "hist":
    #         assert len(glob_v_ns) > 0
    #         outflowed_historgrams(outflowed, glob_detectors, glob_masks, glob_v_ns, glob_overwrite)
    #     elif task == "timecorr":
    #         assert len(glob_v_ns) > 0
    #         outflowed_timecorr(outflowed, glob_detectors, glob_masks, glob_v_ns, glob_overwrite)
    #     elif task == "corr":
    #         assert len(glob_v_ns) > 0
    #         outflowed_correlations(outflowed, glob_detectors, glob_masks, glob_v_ns, glob_overwrite)
    #     elif task == "totflux":
    #         outflowed_totmass(outflowed, glob_detectors, glob_masks, glob_overwrite)
    #     elif task == "massave":
    #         outflowed_massaverages(outflowed, glob_detectors, glob_masks, glob_overwrite)
    #     elif task == "ejtau":
    #         outflowed_ejectatau(outflowed, glob_detectors, glob_masks, glob_overwrite)
    #     elif task == "yeilds":
    #         outflowed_yields(outflowed, glob_detectors, glob_masks, glob_overwrite)
    #     elif task == "mknprof":
    #         outflowed_mkn_profile(outflowed, glob_detectors, glob_masks, glob_overwrite)
    #     elif task == "summary":
    #         outflowed_summary(outflowed, glob_detectors, glob_masks, glob_overwrite)
    #     else:
    #         raise NameError("No method fund for task: {}".format(task))

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

    # ''' -- testing --- '''
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