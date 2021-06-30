from __future__ import division
from sys import path
path.append('modules/')
import os.path
import click
import h5py
from argparse import ArgumentParser
from math import pi, log10
import sys
from scidata.utils import locate
import scidata.carpet.hdf5 as h5
from scidata.carpet.interp import Interpolator
import numpy as np

from glob import glob

from plotting.plotting_methods import PLOT_MANY_TASKS

from uutils import Printcolor, REFLEVEL_LIMITS

import config as Paths

from module_slices.slices_methods import COMPUTE_STORE

from module_slices.add_q_r_t_to_prof_xyxz import add_q_r_t_to_prof_xyxz

from module_slices.slices_dens_modes import compute_density_modes


__movie__ = "ffmpeg -framerate 10 -pattern_type glob -i '{}*.png' -s:v 1280x720 " \
            "-c:v libx264 -module_profile:v high -crf 20 -pix_fmt yuv420p {}"

def __plot_data_for_a_slice(o_slice, v_n, it, t, rl, outdir):

    # ---
    data_arr = o_slice.get_data_rl(it, "xz", rl, v_n)
    x_arr    = o_slice.get_grid_v_n_rl(it, "xz", rl,  "x")
    z_arr    = o_slice.get_grid_v_n_rl(it, "xz", rl, "z")
    def_dic_xz = {'task': 'colormesh',
                  'ptype': 'cartesian', 'aspect': 1.,
                  'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                  'position': (1, 1),  # 'title': '[{:.1f} ms]'.format(time_),
                  'cbar': {'location': 'right .03 -0.125', 'label': r'$\rho$ [geo]',  # 'fmt': '%.1e',
                           'labelsize': 14,
                           'fontsize': 14},
                  'v_n_x': 'x', 'v_n_y': 'z', 'v_n': 'rho',
                  'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None, 'vmin': 1e-10, 'vmax': 1e-4,
                  'fill_vmin': False,  # fills the x < vmin with vmin
                  'xscale': None, 'yscale': None,
                  'mask': None, 'cmap': 'inferno_r', 'norm': "log",  # 'inferno_r'
                  'fancyticks': True,
                  'title': {"text": r'${}$ [ms]'.format(0), 'fontsize': 14},
                  'sharex': True,  # removes angular citkscitks
                  'fontsize': 14,
                  'labelsize': 14
                  }

    data_arr = o_slice.get_data_rl(it, "xy", rl, v_n)
    x_arr = o_slice.get_grid_v_n_rl(it, "xy", rl, "x")
    y_arr = o_slice.get_grid_v_n_rl(it, "xy", rl,  "y")
    def_dic_xy = {'task': 'colormesh',
                  'ptype': 'cartesian', 'aspect': 1.,
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



    # setting scales and limits for data
    if v_n == "rho":
        def_dic_xz['v_n'] = 'rho'
        def_dic_xz['vmin'] = 1e-10
        def_dic_xz['vmax'] = 1e-4
        def_dic_xz['cbar']['label'] = r'$\rho$ [geo]'
        def_dic_xz['cmap'] = 'Greys_r'

        def_dic_xy['v_n'] = 'rho'
        def_dic_xy['vmin'] = 1e-10
        def_dic_xy['vmax'] = 1e-4
        def_dic_xy['cmap'] = 'Greys_r'
    elif v_n == "dens_unbnd":
        def_dic_xz['v_n'] = 'rho'
        def_dic_xz['vmin'] = 1e-13
        def_dic_xz['vmax'] = 1e-6
        def_dic_xz['cbar']['label'] = r'$D_{\rm{unb}}$ [geo]'

        def_dic_xy['v_n'] = 'rho'
        def_dic_xy['vmin'] = 1e-13
        def_dic_xy['vmax'] = 1e-6
    elif v_n == "Y_e":
        def_dic_xz['v_n'] = 'Y_e'
        def_dic_xz['vmin'] = 0.05
        def_dic_xz['vmax'] = 0.5
        def_dic_xz['cbar']['label'] = r'$Y_e$ [geo]'
        def_dic_xz['norm'] = "linear"
        def_dic_xz['cmap'] = 'inferno'

        def_dic_xy['v_n'] = 'Y_e'
        def_dic_xy['vmin'] = 0.05
        def_dic_xy['vmax'] = 0.5
        def_dic_xy['norm'] = "linear"
        def_dic_xy['cmap'] = 'inferno'
    elif v_n == "temp" or v_n == "temperature":
        def_dic_xz['v_n'] = 'temperature'
        def_dic_xz['vmin'] = 1e-2
        def_dic_xz['vmax'] = 1e2
        def_dic_xz['cbar']['label'] = r'$Temperature$ [geo]'

        def_dic_xy['v_n'] = 'temperature'
        def_dic_xy['vmin'] = 1e-2
        def_dic_xy['vmax'] = 1e2
    elif v_n == 'entropy' or v_n == "s_phi":
        def_dic_xz['v_n'] = 'entropy'
        def_dic_xz['vmin'] = 1e-1
        def_dic_xz['vmax'] = 1e2
        def_dic_xz['cbar']['label'] = r'$Entropy$ [geo]'

        def_dic_xy['v_n'] = 'entropy'
        def_dic_xy['vmin'] = 1e-1
        def_dic_xy['vmax'] = 1e2
    elif v_n == "Q_eff_nua":
        def_dic_xz['v_n'] = 'Q_eff_nua'
        def_dic_xz['vmin'] = 1e-18
        def_dic_xz['vmax'] = 1e-14
        def_dic_xz['cbar']['label'] = r'$Q_eff_nua$ [geo]'.replace('_', '\_')

        def_dic_xy['v_n'] = 'Q_eff_nua'
        def_dic_xy['vmin'] = 1e-18
        def_dic_xy['vmax'] = 1e-14
    elif v_n == "Q_eff_nue":
        def_dic_xz['v_n'] = 'Q_eff_nue'
        def_dic_xz['vmin'] = 1e-18
        def_dic_xz['vmax'] = 1e-14
        def_dic_xz['cbar']['label'] = r'$Q_eff_nue$ [geo]'.replace('_', '\_')

        def_dic_xy['v_n'] = 'Q_eff_nue'
        def_dic_xy['vmin'] = 1e-18
        def_dic_xy['vmax'] = 1e-14
    elif v_n == "Q_eff_nux":
        def_dic_xz['v_n'] = 'Q_eff_nux'
        def_dic_xz['vmin'] = 1e-18
        def_dic_xz['vmax'] = 1e-14
        def_dic_xz['cbar']['label'] = r'$Q_eff_nux$ [geo]'.replace('_', '\_')

        def_dic_xy['v_n'] = 'Q_eff_nux'
        def_dic_xy['vmin'] = 1e-18
        def_dic_xy['vmax'] = 1e-14
    elif v_n == "R_eff_nua":
        def_dic_xz['v_n'] = 'R_eff_nua'
        def_dic_xz['vmin'] = 1e-9
        def_dic_xz['vmax'] = 1e-5
        def_dic_xz['cbar']['label'] = r'$R_eff_nua$ [geo]'.replace('_', '\_')

        def_dic_xy['v_n'] = 'R_eff_nue'
        def_dic_xy['vmin'] = 1e-9
        def_dic_xy['vmax'] = 1e-5
    elif v_n == "R_eff_nue":
        def_dic_xz['v_n'] = 'R_eff_nue'
        def_dic_xz['vmin'] = 1e-9
        def_dic_xz['vmax'] = 1e-5
        def_dic_xz['cbar']['label'] = r'$R_eff_nue$ [geo]'.replace('_', '\_')

        def_dic_xy['v_n'] = 'R_eff_nue'
        def_dic_xy['vmin'] = 1e-9
        def_dic_xy['vmax'] = 1e-5
    elif v_n == "R_eff_nux":
        def_dic_xz['v_n'] = 'R_eff_nux'
        def_dic_xz['vmin'] = 1e-9
        def_dic_xz['vmax'] = 1e-5
        def_dic_xz['cbar']['label'] = r'$R_eff_nux$ [geo]'.replace('_', '\_')

        def_dic_xy['v_n'] = 'R_eff_nux'
        def_dic_xy['vmin'] = 1e-9
        def_dic_xy['vmax'] = 1e-5
    elif v_n == "optd_0_nua":
        def_dic_xz['v_n'] = 'optd_0_nua'
        def_dic_xz['vmin'] = 1e-5
        def_dic_xz['vmax'] = 1e-2
        def_dic_xz['cbar']['label'] = r'$optd_0_nua$ [geo]'.replace('_', '\_')
        # def_dic_xz['norm'] = "linear"
        def_dic_xz['cmap'] = 'inferno'

        def_dic_xy['v_n'] = 'optd_0_nua'
        def_dic_xy['vmin'] = 1e-5
        def_dic_xy['vmax'] = 1e-1
        # def_dic_xy['norm'] = "linear"
        def_dic_xy['cmap'] = 'inferno'
    elif v_n == "optd_0_nue":
        def_dic_xz['v_n'] = 'optd_0_nue'
        def_dic_xz['vmin'] = 1e-5
        def_dic_xz['vmax'] = 1e-2
        def_dic_xz['cbar']['label'] = r'$optd_0_nue$ [geo]'.replace('_', '\_')
        # def_dic_xz['norm'] = "linear"
        def_dic_xz['cmap'] = 'inferno'

        def_dic_xy['v_n'] = 'optd_0_nue'
        def_dic_xy['vmin'] = 1e-5
        def_dic_xy['vmax'] = 1e-1
        # def_dic_xy['norm'] = "linear"
        def_dic_xy['cmap'] = 'inferno'
    else: raise NameError("v_n:{} not recognized".format(v_n))

    #
    contour_dic_xy = {
        'task': 'contour',
        'ptype': 'cartesian', 'aspect': 1.,
        'xarr': x_arr, "yarr": y_arr, "zarr": data_arr, 'levels': [1.e13 / 6.176e+17],
        'position': (2, 1),  # 'title': '[{:.1f} ms]'.format(time_),
        'colors':['black'], 'lss':["-"], 'lws':[1.],
        'v_n_x': 'x', 'v_n_y': 'y', 'v_n': 'rho',
        'xscale': None, 'yscale': None,
        'fancyticks': True,
        'sharex': False,  # removes angular citkscitks
        'fontsize': 14,
        'labelsize': 14}


    # setting boundaries for plots
    xmin, xmax, ymin, ymax, zmin, zmax = REFLEVEL_LIMITS.get(rl)
    def_dic_xy['xmin'], def_dic_xy['xmax'] = xmin, xmax
    def_dic_xy['ymin'], def_dic_xy['ymax'] = ymin, ymax
    def_dic_xz['xmin'], def_dic_xz['xmax'] = xmin, xmax
    def_dic_xz['ymin'], def_dic_xz['ymax'] = zmin, zmax

    if not os.path.isdir(outdir):
        raise IOError("Outdir does not exists".format(outdir))

    # plotting


    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = outdir
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["figsize"] = (4.2, 8.0)  # <->, |] # to match hists with (8.5, 2.7)
    o_plot.gen_set["figname"] = "{0:07d}.png".format(int(it))
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = False
    o_plot.gen_set["subplots_adjust_h"] = -0.35
    o_plot.gen_set["subplots_adjust_w"] = 0.2
    o_plot.gen_set['style'] = 'dark_background'
    o_plot.set_plot_dics = []

    def_dic_xz["it"] = int(it)
    def_dic_xz["title"]["text"] = r'$t:{:.1f}ms$'.format(float(t * 1e3))
    o_plot.set_plot_dics.append(def_dic_xz)

    def_dic_xy["it"] = int(it)
    o_plot.set_plot_dics.append(def_dic_xy)

    if v_n == "rho":
        o_plot.set_plot_dics.append(contour_dic_xy)

    # plot reflevel boundaries
    for rl in range(o_slice.nlevels):
        try:
            x_arr = o_slice.get_grid_v_n_rl(it, "xy", rl, "x")
            y_arr = o_slice.get_grid_v_n_rl(it, "xy", rl, "y")
            x_b = [x_arr.min(), x_arr.max()]
            y_b = [y_arr.min(), y_arr.max()]
            #
            for x_b_line, y_b_line in zip([[x_b[0], x_b[-1]], [x_b[0], x_b[0]], [x_b[0], x_b[-1]], [x_b[-1], x_b[-1]]],
                                          [[y_b[0], y_b[0]], [y_b[0], y_b[-1]], [y_b[-1], y_b[-1]], [y_b[-1], y_b[0]]]):
                #
                contour_dic_xy = {
                    'task': 'line',
                    'ptype': 'cartesian', 'aspect': 1.,
                    'xarr': x_b_line, "yarr": y_b_line,
                    'position': (2, 1),  # 'title': '[{:.1f} ms]'.format(time_),
                    'color': 'cyan', 'ls': "-", 'lw': 1., 'alpha': 1., 'ds': 'default',
                    'v_n_x': 'x', 'v_n_y': 'y', 'v_n': 'rho',
                    'xscale': None, 'yscale': None,
                    'fancyticks': True,
                    'sharex': False,  # removes angular citkscitks
                    'fontsize': 14,
                    'labelsize': 14}
                o_plot.set_plot_dics.append(contour_dic_xy)
            #
            x_arr = o_slice.get_grid_v_n_rl(it, "xz", rl, "x")
            z_arr = o_slice.get_grid_v_n_rl(it, "xz", rl, "z")
            x_b = [x_arr.min(), x_arr.max()]
            z_b = [z_arr.min(), z_arr.max()]
            #
            for x_b_line, z_b_line in zip([[x_b[0], x_b[-1]], [x_b[0], x_b[0]], [x_b[0], x_b[-1]], [x_b[-1], x_b[-1]]],
                                          [[z_b[0], z_b[0]], [z_b[0], z_b[-1]], [z_b[-1], z_b[-1]], [z_b[-1], z_b[0]]]):
                #
                contour_dic_xz = {
                    'task': 'line',
                    'ptype': 'cartesian', 'aspect': 1.,
                    'xarr': x_b_line, "yarr": z_b_line,
                    'position': (1, 1),  # 'title': '[{:.1f} ms]'.format(time_),
                    'color': 'cyan', 'ls': "-", 'lw': 1., 'alpha': 1., 'ds': 'default',
                    'v_n_x': 'x', 'v_n_y': 'y', 'v_n': 'rho',
                    'xscale': None, 'yscale': None,
                    'fancyticks': True,
                    'sharex': False,  # removes angular citkscitks
                    'fontsize': 14,
                    'labelsize': 14}
                o_plot.set_plot_dics.append(contour_dic_xz)
        except IndexError:
            Printcolor.print_colored_string(["it:", str(it), "rl:", str(rl), "IndexError"],
                                            ["blue", "green", "blue", "green", "red"])

    o_plot.main()
    o_plot.set_plot_dics = []

    # plotfpath = outdir + "{0:07d}.png".format(int(it))
    # if True:
    #     if (os.path.isfile(plotfpath) and rewrite) or not os.path.isfile(plotfpath):
    #         if os.path.isfile(plotfpath): os.remove(plotfpath)
    #         Printcolor.print_colored_string(
    #         ["task:", "plot slice", "t:", "{:.1f} [ms] ({:d}/{:d})".format(t*1e3, i, len(list_times)),
    #          "rl:", "{}".format(rl), "v_n:", v_n, ':', "plotting"],
    #         ["blue",  "green",     "blue", "green",   "blue", "green",      "blue", "green", "", "green"]
    #         )
    #         # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    #
    #         def_dic_xz["it"] = int(it)
    #         def_dic_xz["title"]["text"] = r'$t:{:.1f}ms$'.format(float(t*1e3))
    #         o_plot.set_plot_dics.append(def_dic_xz)
    #
    #         def_dic_xy["it"] = int(it)
    #         o_plot.set_plot_dics.append(def_dic_xy)
    #
    #         o_plot.main()
    #         o_plot.set_plot_dics = []
    #
    #         # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    #     else:
    #         Printcolor.print_colored_string(
    #             ["task:", "plot slice", "t:", "{:.1f} [ms] ({:d}/{:d})".format(t * 1e3, i, len(list_times)), "rl:",
    #              "{}".format(rl), "v_n:", v_n, ':', "skipping"],
    #             ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "blue"]
    #         )
    #
    # # except KeyboardInterrupt:
    # #     exit(1)
    # else:
    #     Printcolor.print_colored_string(
    #         ["task:", "plot slice", "t:", "{:.1f} [ms] ({:d}/{:d})".format(t * 1e3, i, len(list_times)), "rl:",
    #          "{}".format(rl), "v_n:", v_n, ':', "failed"],
    #         ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"]
    #     )

def plot_selected_data(o_slice, v_ns, times, rls, rootdir, rewrite=False):

    _, d2it, d2t = o_slice.get_ittime("overall", d1d2d3prof="d2")
    if len(d2it) == 0:
        raise ValueError("No d2 data found in ittime.h5")

    for t in times:
        if t > d2t.max():
            raise ValueError("given t:{} is above max time available:{}"
                             .format(t, d2t.max()))
        if t < d2t.min():
            raise ValueError("given t:{} is below min time available:{}"
                             .format(t, d2t.min()))

    i = 1
    for t in times:
        nearest_time = o_slice.get_nearest_time(t, d1d2d3="d2")
        it = o_slice.get_it_for_time(nearest_time, d1d2d3="d2")
        for v_n in v_ns:
            outdir_ = rootdir + v_n + '/'
            if not os.path.isdir(outdir_):
                os.mkdir(outdir_)
            for rl in rls:
                outdir__ = outdir_ + str("rl_{:d}".format(rl)) + '/'
                if not os.path.isdir(outdir__):
                    os.mkdir(outdir__)
                plotfpath = outdir__ + "{0:07d}.png".format(int(it))
                if True:
                    if (os.path.isfile(plotfpath) and rewrite) or not os.path.isfile(plotfpath):
                        if os.path.isfile(plotfpath): os.remove(plotfpath)
                        Printcolor.print_colored_string(
                            ["task:", "plot slice", "t:", "{:.1f} [ms] ({:d}/{:d})".format(t * 1e3, i, len(times)),
                             "rl:", "{}".format(rl), "v_n:", v_n, ':', "plotting"],
                            ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "green"]
                        )
                        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                        __plot_data_for_a_slice(o_slice, v_n, it, t, rl, outdir__)
                        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                    else:
                        Printcolor.print_colored_string(
                            ["task:", "plot slice", "t:", "{:.1f} [ms] ({:d}/{:d})".format(t * 1e3, i, len(times)),
                             "rl:",
                             "{}".format(rl), "v_n:", v_n, ':', "skipping"],
                            ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "blue"]
                        )
                # except KeyboardInterrupt:
                #     exit(1)
                # except:
                #     Printcolor.print_colored_string(
                #         ["task:", "plot slice", "t:", "{:.1f} [ms] ({:d}/{:d})".format(t * 1e3, i, len(times)),
                #          "rl:",
                #          "{}".format(rl), "v_n:", v_n, ':', "failed"],
                #         ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"]
                #     )
        sys.stdout.flush()
        i += 1

def make_movie(v_ns, rls, rootdir, rewrite=False):

    rewrite = True

    for v_n in v_ns:
        outdir_ = rootdir + v_n + '/'
        if not os.path.isdir(outdir_):
            os.mkdir(outdir_)
        for rl in rls:
            outdir__ = outdir_ + str("rl_{:d}".format(rl)) + '/'
            if not os.path.isdir(outdir__):
                os.mkdir(outdir__)
            fname = "{}_rl{}.mp4".format(v_n, rl)
            moviefath = outdir__ + fname
            nfiles = len(glob(outdir__))
            if nfiles < 1:
                Printcolor.red("No plots found to make a movie in: {}".format(outdir__))
                break
            try:
                if (os.path.isfile(moviefath) and rewrite) or not os.path.isfile(moviefath):
                    if os.path.isfile(moviefath): os.remove(moviefath)
                    Printcolor.print_colored_string(
                        ["task:", "movie slice", "N files", "{:d}".format(nfiles),
                         "rl:", "{}".format(rl), "v_n:", v_n, ':', "plotting"],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "green"]
                    )
                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                    # ffmpeg -framerate 10 -pattern_type glob -i "*.png" -s:v 1280x720 -c:v libx264 -module_profile:v high -crf 20 -pix_fmt yuv420p dt.mp4

                    os.system(__movie__.format(outdir__, outdir__ + fname))
                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                else:
                    Printcolor.print_colored_string(
                        ["task:", "movie slice", "N files", "{:d}".format(nfiles),
                         "rl:",
                         "{}".format(rl), "v_n:", v_n, ':', "skipping"],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "blue"]
                    )
            except KeyboardInterrupt:
                exit(1)
            except:
                Printcolor.print_colored_string(
                    ["task:", "plot slice", "N files", "{:d}".format(nfiles),
                     "rl:",
                     "{}".format(rl), "v_n:", v_n, ':', "failed"],
                    ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "red"]
                )


__tasklist__ = ["plot", "movie", "addm0", "dm"]
__reflevels__ = [0, 1, 2, 3, 4, 5, 6]
__outdirname__ = "module_slices"
__planes__ = ["xy", "xz"]

def do_tasks(glob_v_ns):

    for task in glob_tasklist:
        # do tasks one by one
        if task == "plot":
            assert len(glob_v_ns) > 0
            assert len(glob_times) > 0
            assert len(glob_reflevels) > 0
            outdir = glob_outdir + __outdirname__ + '/'
            if not os.path.isdir(outdir):
                os.mkdir(outdir)
            outdir += 'plots/'
            if not os.path.isdir(outdir):
                os.mkdir(outdir)

            plot_selected_data(o_slice, glob_v_ns, glob_times, glob_reflevels, outdir, rewrite=glob_overwrite)

        if task == "movie":

            assert len(glob_v_ns) > 0
            assert len(glob_times) > 0
            assert len(glob_reflevels) > 0
            outdir = glob_outdir + __outdirname__ + '/'
            if not os.path.isdir(outdir):
                os.mkdir(outdir)
            outdir += 'movie/'
            if not os.path.isdir(outdir):
                os.mkdir(outdir)

            plot_selected_data(o_slice, glob_v_ns, glob_times, glob_reflevels, outdir, rewrite=glob_overwrite)

            assert len(glob_v_ns) > 0
            assert len(glob_reflevels) > 0
            outdir = glob_outdir + __outdirname__ + '/'
            if not os.path.isdir(outdir):
                os.mkdir(outdir)
            outdir += 'movie/'

            make_movie(glob_v_ns, glob_reflevels, outdir, rewrite=glob_overwrite)

        if task == "addm0":
            if len(glob_v_ns) == len(o_slice.list_v_ns):
                glob_v_ns = o_slice.list_neut_v_ns
            print glob_it

            add_q_r_t_to_prof_xyxz(
                v_ns=glob_v_ns,
                rls=glob_reflevels,
                planes=glob_planes,
                iterations=glob_it,
                sim=glob_sim,
                indir=glob_indir,
                pprdir=glob_outdir,
                path_to_sliced_profiles=glob_profxyxz_path,
                overwrite=glob_overwrite
            )

        if task == "dm":
            outdir = Paths.default_ppr_dir + glob_sim + '/' + __outdirname__ + '/'

            compute_density_modes(o_slice, glob_reflevels, outdir, rewrite=glob_overwrite)

if __name__ == '__main__':

    parser = ArgumentParser(description="postprocessing pipeline")
    parser.add_argument("-s", dest="sim", required=True, help="name of the simulation dir")
    parser.add_argument("-t", dest="tasklist", nargs='+', required=False, default=[], help="tasks to perform")
    #
    parser.add_argument("--v_n", dest="v_ns", nargs='+', required=False, default=[], help="variable names to compute")
    parser.add_argument("--time", dest="times", nargs='+', required=False, default=[], help="times to iterate over [ms]")
    parser.add_argument("--it", dest="it", nargs='+', required=False, default=[],
                        help="iterations to use ")
    parser.add_argument("--rl", dest="reflevels", nargs='+', required=False, default=[], help="reflevels to use")
    parser.add_argument('--plane', dest="plane", required=False, nargs='+', default=[], help='Plane: xy,xz,yz for slice analysis')
    #
    parser.add_argument("-o", dest="outdir", required=False, default=None, help="path for output dir")
    parser.add_argument("-i", dest="indir", required=False, default=None, help="path to simulation dir")
    parser.add_argument("-p", dest="path_to_profs", required=False, default=None, help="path to 3D profiles")
    parser.add_argument("--overwrite", dest="overwrite", required=False, default="no", help="overwrite if exists")
    #
    args = parser.parse_args()
    glob_sim = args.sim
    glob_indir = args.indir
    glob_outdir = args.outdir
    glob_tasklist = args.tasklist
    glob_overwrite = args.overwrite
    glob_v_ns = args.v_ns
    glob_times =args.times
    glob_it = args.it
    glob_reflevels = args.reflevels
    glob_planes = args.plane
    #
    glob_profxyxz_path = args.path_to_profs#Paths.ppr_sims+glob_sim+'/profiles/'
    #
    if glob_indir is None:
        glob_indir = Paths.default_data_dir + glob_sim + '/'
        if not os.path.isdir(glob_indir):
            raise IOError("Default path to simulation data is not valid: {}".format(glob_indir))
    if not os.path.isdir(glob_indir):
        raise IOError("Path to simulation data is not valid: {}".format(glob_indir))

    if glob_outdir is None:
        glob_outdir = Paths.default_ppr_dir + glob_sim + '/'
        if not os.path.isdir(glob_indir):
            raise IOError("Default path to postprocessed data is not valid: {}".format(glob_outdir))
    if not os.path.isdir(glob_indir):
        raise IOError("Path to postprocessed data is not valid: {}".format(glob_outdir))


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

    # check plane
    if len(glob_planes) == 0:
        raise IOError("Option --plane unfilled")
    elif len(glob_planes) == 1 and "all" in glob_planes:
        glob_planes = __planes__
    elif len(glob_planes) > 1:
        for plane in glob_planes:
            if not plane in __planes__:
                raise NameError("plane:{} is not in the list of the __d3slicesplanes__:{}"
                                .format(plane, __planes__))

    # set globals
    # Paths.gw170817 = glob_simdir
    # Paths.ppr_sims = glob_outdir


    if len(glob_tasklist) == 1 and "all" in glob_tasklist:
        # do all tasksk
        pass


    o_slice = COMPUTE_STORE(glob_sim, indir=glob_indir, pprdir=glob_outdir)

    # deal with iterations and timesteps -- available as well as required by user
    do_all_iterations = False
    if len(glob_it) == 0 and len(glob_times) == 0:
        raise IOError("please specify timesteps to use '--time' or iterations '--it' ")
    elif len(glob_it) != 0 and len(glob_times) != 0:
        raise IOError("please specify Either timesteps to use '--time' or iterations '--it' (not both)")
    elif len(glob_times) == 0 and len(glob_it) == 1 and "all" in glob_it:
        do_all_iterations = True
        glob_times = o_slice.times
        glob_it = o_slice.iterations
    elif len(glob_it) == 0 and len(glob_times) == 1 and "all" in glob_times:
        do_all_iterations = True
        glob_times = o_slice.times
        glob_it = o_slice.iterations
    elif len(glob_it) > 0 and not "all" in glob_it and len(glob_times) == 0:
        glob_it = np.array(glob_it, dtype=int) # array of iterations
        glob_times = []
        for it in glob_it:
            glob_times.append(o_slice.get_time_for_it(it, "overall", "d2"))
        glob_times = np.array(glob_times, dtype=float)
    elif len(glob_times) > 0 and not "all" in glob_times and len(glob_it) == 0:
        glob_times = np.array(glob_times, dtype=float) / 1e3  # back to seconds
    else:
        raise IOError("input times and iterations are not recognized: --time {} --it {}"
                      .format(glob_times, glob_it))

    # deal with reflevels -- availble as well as required by user
    do_all_reflevels = False
    if len(glob_reflevels) == 1 and "all" in glob_reflevels:
        glob_reflevels = __reflevels__
        do_all_reflevels = True
    else:
        glob_reflevels = np.array(glob_reflevels, dtype=int)

    # deal with variable names -- available as well as required by user
    do_all_v_ns = False
    if len(glob_v_ns) == 1 and "all" in glob_v_ns:
        glob_v_ns=o_slice.list_v_ns
        do_all_v_ns = True
    else:
        pass

    # summarize what is avaialble and what is requried
    if do_all_v_ns or do_all_iterations or do_all_reflevels:
        Printcolor.yellow("Selected all", comma=True)
        if do_all_iterations:
            Printcolor.print_colored_string(["timesteps", "({})".format(len(glob_times))],
                                            ["blue", "green"], comma=True)
        if do_all_v_ns: Printcolor.print_colored_string(["v_ns", "({})".format(len(glob_v_ns))],
                                                        ["blue", "green"], comma=True)
        if do_all_reflevels: Printcolor.print_colored_string(["reflevels", "({})".format(len(glob_reflevels))],
                                                             ["blue", "green"], comma=True)
        Printcolor.yellow("this might take time.")
        # if not click.confirm(text="Confirm?",default=True,show_default=True):
        #     exit(0)

    # perform tasks
    do_tasks(glob_v_ns)