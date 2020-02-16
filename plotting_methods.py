from __future__ import division
from sys import path
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import AutoMinorLocator, FixedLocator, NullFormatter, \
    MultipleLocator
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


class BASIC_PARTS():

    def __init__(self):
        pass

    def set_plot_title(self, ax, plot_dic):
        if "title" in plot_dic.keys():
            tdic = plot_dic["title"]
            if len(tdic.keys()) > 0:
                # ax.title.set_text(r'{}'.format(tdic["text"]), fontsize=tdic["fontsize"])
                ax.set_title(r'{}'.format(tdic["text"]), fontsize=tdic["fontsize"])

            # if plot_dic["title"] != '' and plot_dic["title"] != None:
            #
            #     title = plot_dic["title"]
            #
            #     # data = plot_dic['data']
            #     #
            #     # if plot_dic["title"] == 'it':
            #     #     title = plot_dic["it"]
            #     # elif plot_dic["title"] == 'time [s]' or \
            #     #     plot_dic["title"] == 'time':
            #     #     title = "%.3f" % data.get_time(plot_dic["it"]) + " [s]"
            #     # elif plot_dic["title"] == 'time [ms]':
            #     #     title = "%.1f" % (data.get_time(plot_dic["it"]) * 1000) + " [ms]"
            #     # else:
            #     #     title = plot_dic["title"]
            #     ax.title.set_text(r'{}'.format(title))

    def set_min_max_scale(self, ax, dic, n_col, n_row):
        # ax = self.sbplot_matrix[n_col][n_row]
        if dic["ptype"] == "cartesian":
            if "xscale" in dic.keys():
                if dic["xscale"] == 'log':
                    #print("n_col:{} n_row:{} setting xscale: {}".format(n_col, n_row, dic["xscale"])),
                    ax.set_xscale("log")
                    #print(" getting xscale: {}".format(ax.get_xscale()))
                elif dic["xscale"] == 'linear':
                    #print("n_col:{} n_row:{} setting yscale: {}".format(n_col, n_row, dic["yscale"])),
                    ax.set_xscale("linear")
                    #print(" getting yscale: {}".format(ax.get_yscale()))
                else:
                    ax.set_xscale("linear")
                    # print("xscale '{}' is not recognized".format(dic["xscale"]))

            if "yscale" in dic.keys():
                if dic["yscale"] == 'log':
                    ax.set_yscale("log")
                elif dic["yscale"] == 'linear':
                    ax.set_yscale("linear")
                else:
                    ax.set_yscale("linear")
                    # print("yscale '{}' is not recognized".format(dic["yscale"]))

            if "xmin" in dic.keys() and "xmax" in dic.keys():
                if dic["xmin"] != None and dic["xmax"] != None:
                    # print(dic['xmin'], dic['xmin']); exit(1)
                    #print("n_col:{} n_row:{} setting xlim:[{} {}]".format(n_col, n_row, dic["xmin"], dic["xmax"])),
                    ax.set_xlim(float(dic["xmin"]), float(dic["xmax"]))
                    #print("getting xlim:[{}, {}]".format(n_col, n_row, ax.get_xlim()[0], ax.get_xlim()[1]))
            if "ymin" in dic.keys() and "ymax" in dic.keys():
                if dic["ymin"] != None and dic["ymax"] != None:
                    #print("n_col:{} n_row:{} setting ylim:[{} {}]".format(n_col, n_row, dic["ymin"], dic["ymax"])),
                    ax.set_ylim(float(dic["ymin"]), float(dic["ymax"]))
                    #print("getting ylim:[{} {}]".format(n_col, n_row, ax.get_ylim()[0], ax.get_ylim()[1]))

        elif dic["ptype"] == "polar":

            if "phimin" in dic.keys() and "phimax" in dic.keys():
                if dic["phimin"] != None and dic["phimax"] != None:
                    ax.set_philim(dic["phimin"], dic["phimax"])

            if "rmin" in dic.keys() and "rmax" in dic.keys():
                if dic["rmin"] != None and dic["rmax"] != None:
                    ax.set_rlim(dic["rmin"], dic["rmax"])

            if "xscale" in dic.keys():
                if dic["xscale"] == 'log':
                    raise NameError("log scale is not available for x in polar")

            if "yscale" in dic.keys():
                if dic["yscale"] == 'log':
                    raise NameError("log scale is not available for y in polar")
        else:
            raise NameError("Unknown 'ptype' of the plot: {}".format(dic["ptype"]))

        return ax

    def set_xy_labels(self, ax, dic):

        if dic["ptype"] == "cartesian":

            # if not 'fontsize' in dic.keys():
            #     raise NameError("no 'fontsize' in dic: {}".format(dic))

            if "xlabel" in dic.keys():
                if dic["xlabel"] != None:
                    ax.set_xlabel(dic["xlabel"], fontsize=dic['fontsize'])
            elif "v_n_x" in dic.keys():
                if dic["v_n_x"] != "None":
                    ax.set_xlabel(dic["v_n_x"].replace('_', '\_'), fontsize=dic['fontsize'])
            else:
                print("Waning. Neither v_n_x nor xlabel are set in the dic")

            if "ylabel" in dic.keys():
                if dic["ylabel"] != None:
                    ax.set_ylabel(dic["ylabel"], fontsize=dic['fontsize'])
            elif "v_n_y" in dic.keys():
                if dic["v_n_y"] != "None":
                    ax.set_ylabel(dic["v_n_y"].replace('_', '\_'), fontsize=dic['fontsize'])
            else:
                print("Waning. Neither v_n_x nor xlabel are set in the dic")

        elif dic["ptype"] == "polar":
            pass
        else:
            raise NameError("Unknown 'ptype' of the plot: {}".format(dic["ptype"]))

    def set_legend(self, ax, dic):
        if "legend" in dic.keys():
            ldic = dic["legend"]
            if len(ldic.keys()) > 0:
                # print("legend")
                if 'shadow' in ldic.keys() and ldic['shadow']:
                    shadow = True
                else:
                    shadow = False
                #
                if 'framealpha' in ldic.keys() and ldic['framealpha'] != None:
                    framealpha = ldic['framealpha']
                else:
                    framealpha = 0.7
                #
                if 'borderaxespad' in ldic.keys() and ldic['borderaxespad'] != None:
                    borderaxespad = ldic['borderaxespad']
                else:
                    borderaxespad = 0.5
                #
                if 'borderayespad' in ldic.keys() and ldic['borderayespad'] != None:
                    borderayespad = ldic['borderayespad']
                else:
                    borderayespad = 0.5
                #
                if 'bbox_to_anchor' in ldic.keys():
                    legend1 = ax.legend(fancybox=True, bbox_to_anchor=ldic['bbox_to_anchor'],  # (1.0, 0.3),
                              loc=ldic['loc'], shadow=shadow, ncol=ldic['ncol'], fontsize=ldic['fontsize'],
                              framealpha=framealpha, borderaxespad=borderaxespad)#, borderayespad=borderayespad)
                else:
                    legend1 = ax.legend(fancybox=True,
                              loc=ldic['loc'], shadow=shadow, ncol=ldic['ncol'], fontsize=ldic['fontsize'],
                              framealpha=framealpha, borderaxespad=borderaxespad)#, borderayespad=borderayespad)

        if "legend" in dic.keys() and "legend2" in dic.keys():
            ldic2 = dic["legend2"]
            if len(ldic.keys()) > 0:
                # print("legend")
                if 'shadow' in ldic2.keys() and ldic2['shadow']:
                    shadow = True
                else:
                    shadow = False
                #
                if 'framealpha' in ldic2.keys() and ldic2['framealpha'] != None:
                    framealpha = ldic2['framealpha']
                else:
                    framealpha = 0.7
                #
                if 'borderaxespad' in ldic2.keys() and ldic2['borderaxespad'] != None:
                    borderaxespad = ldic2['borderaxespad']
                else:
                    borderaxespad = 0.5
                #
                if 'bbox_to_anchor' in ldic2.keys():
                    ax.legend(fancybox=True, bbox_to_anchor=ldic2['bbox_to_anchor'],  # (1.0, 0.3),
                              loc=ldic2['loc'], shadow=shadow, ncol=ldic2['ncol'], fontsize=ldic2['fontsize'],
                              framealpha=framealpha, borderaxespad=borderaxespad)
                    ax.add_artist(legend1)
                else:
                    ax.legend(fancybox=True,
                              loc=ldic2['loc'], shadow=shadow, ncol=ldic2['ncol'], fontsize=ldic2['fontsize'],
                              framealpha=framealpha, borderaxespad=borderaxespad)
                    ax.add_artist(legend1)

    def remover_some_ticks(self, ax, dic):

        if "rmxlbls" in dic.keys():
            if dic["rmxlbls"]:
                ax.set_xticklabels([])
                ax.axes.xaxis.set_ticklabels([])

        if "rmylbls" in dic.keys():
            if dic["rmylbls"]:
                ax.set_yticklabels([])
                ax.axes.yaxis.set_ticklabels([])

    def plot_text(self, ax, dic):
        """

            'textold':{'coords':(0.8, 0.8), 'text':"DD2", 'color':'red', 'fs':16}

        :param ax:
        :param dic:
        :return:
        """
        # exit(1)
        if 'textold' in dic.keys() and dic['textold'] != None and dic['textold'] != {}:
            coords = dic['textold']['coords']
            text = dic['textold']['text']
            color = dic['textold']['color']
            fs =    dic['textold']['fs']
            # exit(1)
            ax.text(coords[0], coords[1], text, color=color, fontsize=fs, transform=ax.transAxes)

    def plot_text2(self, ax, dic):
        """'text':{'x':0.5, 'y':0.5, 'text':'my_text', 'fs':12, 'color':'black',
                   'horizontalalignment':'center', 'transform':True}"""
        # print("---------")
        xcorr = dic['x']
        ycorr = dic['y']
        text = dic['text']
        fs = dic['fs']
        color=dic['color']
        horal = dic['horizontalalignment']
        if 'transform' in dic.keys() and dic['transform']:
            ax.text(xcorr, ycorr, text, color=color, fontsize=fs, horizontalalignment=horal, transform=ax.transAxes)
        else:
            ax.text(xcorr, ycorr, text, color=color, fontsize=fs, horizontalalignment=horal)
        return 0

    @staticmethod
    def plot_colormesh(ax, dic, x_arr, y_arr, z_arr):

        """

        int_ang_mom_flux_dic = {
            'ptype': 'polar',
            'v_n_x': 'phi_cyl', 'v_n_y': 'r_cyl', 'v_n': 'ang_mom_flux',
            'phimin': None, 'phimax': None, 'rmin': 0, 'rmax': 50, 'vmin': 1e-8, 'vmax': 1e-5,
            'mask_below': None, 'mask_above': None, 'cmap': 'RdBu_r', 'norm': "log"
        }

        corr_dic_ang_mom_flux_dens_unb_bern = {
            'ptype': 'cartesian',
            'v_n_x': 'dens_unb_bern', 'v_n_y': 'ang_mom_flux', 'v_n': 'mass',
            'xmin': 1e-11, 'xmax': 1e-7, 'ymin': 1e-11, 'ymax': 1e-7, 'vmin': 1e-7, 'vmax': None,
            'xscale': 'log', 'yscale': 'log',
            'mask_below': None, 'mask_above': None, 'cmap': 'inferno_r', 'norm': 'log'
        }

        :param ax:
        :param dic:
        :param x_arr:
        :param y_arr:
        :param z_arr:
        :return:
        """


        if "mask" in dic.keys() and dic["mask"] != None:
            if dic["mask"] == "negative":
                z_arr = np.ma.masked_array(z_arr, z_arr < 0) # [z_arr < 0] = np.nan#
                # print(z_arr)
            elif dic["mask"] == "positive":
                z_arr = -1 * np.ma.masked_array(z_arr, z_arr > 0)
            elif dic["mask"] == "x>0":
                z_arr = np.ma.masked_array(z_arr, x_arr > 0)
            elif dic["mask"].__contains__("x>"):
                val = float(str(dic["mask"]).split("x>")[-1])
                z_arr = np.ma.masked_array(z_arr, x_arr > val)
            elif dic["mask"] == "x<0":
                z_arr = np.ma.masked_array(z_arr, x_arr < 0)
            elif dic["mask"].__contains__("x<"):
                val = float(str(dic["mask"]).split("x<")[-1])
                z_arr = np.ma.masked_array(z_arr, x_arr < val)
            elif dic["mask"].__contains__("z>"):
                val = float(dic["mask"].split("z>")[-1])
                # assert val < z_arr.max()
                z_arr = np.ma.masked_array(z_arr, z_arr > val)
            elif dic["mask"].__contains__("z<"):
                val = float(dic["mask"].split("z<")[-1])
                z_arr = np.ma.masked_array(z_arr, z_arr < val)
            else:
                raise NameError("mask:{} is not recognized. (For plotting)".format(dic["mask"]))
        if "mask_below" in dic.keys():
            z_arr = np.ma.masked_array(z_arr, z_arr < dic["mask_below"])
        elif "mask_above" in dic.keys():
            z_arr = np.ma.masked_array(z_arr, z_arr > dic["mask_below"])

        vmin = dic["vmin"]
        vmax = dic["vmax"]

        if vmin == None:
            assert not np.isnan(z_arr.min())
            assert not np.isinf(z_arr.min())
            vmin = z_arr.min()
            if dic["norm"] == "log":
                vmin = z_arr.flatten()[np.where(z_arr > 0, z_arr, np.inf).argmin()]
        if vmax == None:
            assert not np.isnan(z_arr.max())
            assert not np.isinf(z_arr.max())
            vmax = z_arr.max()
            if dic["norm"] == "log":
                vmax = z_arr.flatten()[np.where(z_arr < 0, z_arr, -np.inf).argmax()]

        if dic["norm"] == "norm" or dic["norm"] == "linear" or dic["norm"] == None:
            norm = Normalize(vmin=vmin, vmax=vmax)
        elif dic["norm"] == "log":
            assert vmin > 0
            assert vmax > 0
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            raise NameError("unrecognized norm: {} in task {}"
                            .format(dic["norm"], dic["v_n"]))
        #
        im = ax.pcolormesh(x_arr, y_arr, z_arr, norm=norm, cmap=dic["cmap"])#, vmin=dic["vmin"], vmax=dic["vmax"])
        im.set_rasterized(True)
        if 'set_under' in dic.keys() and dic['set_under'] != None:
            im.cmap.set_over(dic['set_under'])
        if 'set_over' in dic.keys() and dic['set_over'] != None:
            im.cmap.set_over(dic['set_over'])
        #
        return im

    @staticmethod
    def mscatter(x, y, ax=None, m=None, **kw):
        import matplotlib.markers as mmarkers
        if not ax: ax = plt.gca()
        sc = ax.scatter(x, y, **kw)
        if (m is not None) and (len(m) == len(x)):
            paths = []
            for marker in m:
                if isinstance(marker, mmarkers.MarkerStyle):
                    marker_obj = marker
                else:
                    marker_obj = mmarkers.MarkerStyle(marker)
                path = marker_obj.get_path().transformed(
                    marker_obj.get_transform())
                paths.append(path)
            sc.set_paths(paths)
        return sc

    @staticmethod
    def plot_scatter(ax, dic, x_arr, y_arr, z_arr):
        vmin = dic["vmin"]
        vmax = dic["vmax"]
        if vmin == None:
            assert not np.isnan(z_arr.min())
            assert not np.isinf(z_arr.min())
            vmin = z_arr.min()
            if dic["norm"] == "log":
                vmin = z_arr.flatten()[np.where(z_arr > 0, z_arr, np.inf).argmin()]
        if vmax == None:
            assert not np.isnan(z_arr.max())
            assert not np.isinf(z_arr.max())
            vmax = z_arr.max()
            if dic["norm"] == "log":
                vmax = z_arr.flatten()[np.where(z_arr < 0, z_arr, -np.inf).argmax()]
        cm = plt.cm.get_cmap(dic['cmap'])
        #
        if dic["norm"] == "norm" or dic["norm"] == "linear" or dic["norm"] == None:
            norm = Normalize(vmin=vmin, vmax=vmax)
        elif dic["norm"] == "log":
            # exit(1)
            assert vmin > 0
            assert vmax > 0
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            raise NameError("unrecognized norm: {} in task {}"
                            .format(dic["norm"], dic["v_n"]))
        #
        if 'markers' in dic.keys() and dic['markers'] != None:
            # sc = BASIC_PARTS.mscatter(x_arr, y_arr, ax=ax, c=z_arr, norm=norm, s=dic['ms'], cmap=cm, m=dic["marker"], label=dic['label'], alpha=dic['alpha'], edgecolors=dic["edgecolors"])
            if "label" in dic.keys() and dic['label'] != None:
                sc = BASIC_PARTS.mscatter(x_arr, y_arr, ax=ax, c=z_arr, norm=norm, s=dic['ms'], cmap=cm,
                                          m=dic["markers"], label=dic['label'],
                                          alpha=dic['alpha'])  # , edgecolors=dic["edgecolors"])
            else:
                sc = BASIC_PARTS.mscatter(x_arr, y_arr, ax=ax, c=z_arr, norm=norm, s=dic['ms'], cmap=cm,
                                          m=dic["markers"], alpha=dic['alpha'])  # , edgecolors=dic["edgecolors"])
        elif "marker" in dic.keys() and dic["marker"] != None:
            if "label" in dic.keys() and dic['label'] != None:
                sc = ax.scatter(x_arr, y_arr, c=z_arr, norm=norm, s=dic['ms'], marker=dic["marker"], cmap=cm,
                                alpha=dic['alpha'], label=dic['label'],
                                edgecolors=dic["edgecolors"])  # edgecolors="black"
            else:
                sc = ax.scatter(x_arr, y_arr, c=z_arr, norm=norm, s=dic['ms'], marker=dic["marker"], cmap=cm,
                                alpha=dic['alpha'], edgecolors=dic["edgecolors"])
        else:
            if "label" in dic.keys() and dic['label'] != None:
                sc = ax.scatter(x_arr, y_arr, c=z_arr, norm=norm, s=dic['ms'], cmap=cm, label=dic['label'],
                                alpha=dic['alpha'], edgecolors=dic["edgecolors"])
            else:
                sc = ax.scatter(x_arr, y_arr, c=z_arr, norm=norm, s=dic['ms'], cmap=cm, alpha=dic['alpha'],
                                edgecolors=dic["edgecolors"])
        return sc







        if "edgecolors" in dic.keys() and dic["edgecolors"] != None:
            if 'markers' in dic.keys() and dic['markers'] != None:
                # sc = BASIC_PARTS.mscatter(x_arr, y_arr, ax=ax, c=z_arr, norm=norm, s=dic['ms'], cmap=cm, m=dic["marker"], label=dic['label'], alpha=dic['alpha'], edgecolors=dic["edgecolors"])
                if "label" in dic.keys() and dic['label'] != None:
                    sc = BASIC_PARTS.mscatter(x_arr, y_arr, ax=ax, c=z_arr, norm=norm, s=dic['ms'], cmap=cm, m=dic["marker"], label=dic['label'], alpha=dic['alpha'])#, edgecolors=dic["edgecolors"])
                else:
                    sc = BASIC_PARTS.mscatter(x_arr, y_arr, ax=ax, c=z_arr, norm=norm, s=dic['ms'], cmap=cm, m=dic["marker"], alpha=dic['alpha'])#, edgecolors=dic["edgecolors"])
            elif "marker" in dic.keys() and dic["marker"] != None:
                if "label" in dic.keys() and dic['label'] != None:
                    sc = ax.scatter(x_arr, y_arr, c=z_arr, norm=norm, s=dic['ms'], marker=dic["marker"], cmap=cm, alpha=dic['alpha'], label=dic['label'], edgecolors=dic["edgecolors"]) # edgecolors="black"
                else:
                    sc = ax.scatter(x_arr, y_arr, c=z_arr, norm=norm, s=dic['ms'], marker=dic["marker"], cmap=cm, alpha=dic['alpha'], edgecolors=dic["edgecolors"])
            else:
                if "label" in dic.keys() and dic['label'] != None:
                    sc = ax.scatter(x_arr, y_arr, c=z_arr, norm=norm, s=dic['ms'], cmap=cm, label=dic['label'], alpha=dic['alpha'], edgecolors=dic["edgecolors"])
                else:
                    sc = ax.scatter(x_arr, y_arr, c=z_arr, norm=norm, s=dic['ms'], cmap=cm, alpha=dic['alpha'], edgecolors=dic["edgecolors"])
        else:
            if "marker" in dic.keys() and dic["marker"] != None:
                if "label" in dic.keys() and dic['label'] != None:
                    sc = ax.scatter(x_arr, y_arr, c=z_arr, norm=norm, s=dic['ms'], cmap=cm, marker=dic["marker"],
                                    label=dic['label'], alpha=dic['alpha'])  # edgecolors="black"
                else:
                    sc = ax.scatter(x_arr, y_arr, c=z_arr, norm=norm, s=dic['ms'], marker=dic["marker"], cmap=cm,
                                    alpha=dic['alpha'])
            else:
                if "label" in dic.keys() and dic['label'] != None:
                    sc = ax.scatter(x_arr, y_arr, c=z_arr, norm=norm, s=dic['ms'], cmap=cm, label=dic['label'],
                                    alpha=dic['alpha'])
                else:
                    sc = ax.scatter(x_arr, y_arr, c=z_arr, norm=norm, s=dic['ms'], cmap=cm, alpha=dic['alpha'])
        return sc

    @staticmethod
    def plot_countour(ax, dic, x_arr, y_arr, z_arr):

        # cp = ax.contour(x_arr, y_arr, z_arr, colors=dic['colors'], levels=dic['levels'],
        #                 linestyles=dic['lss'], linewidths=['lws'])
        cp = ax.contour(x_arr, y_arr, z_arr, colors=dic['colors'], levels=dic['levels'],
                        linestyles="-", linewidths=1.)
        # ax.clabel(cp, inline=True, fontsize=10)

    def fill_arr_with_vmin(self, arr, dic):

        if 'fill_vmin' in dic.keys():
            if dic['fill_vmin']:
                if 'vmin' in dic.keys():
                    if dic['vmin'] != None:
                        arr = np.maximum(arr, dic['vmin'])
        return arr

    def add_fancy_to_ax(self, ax, dic):

        if 'fancyticks' in dic.keys():
            if dic['fancyticks']:
                ax.tick_params(
                    axis='both', which='both', labelleft=True,
                    labelright=False, tick1On=True, tick2On=True,
                    labelsize=int(dic['labelsize']),
                    direction='in',
                    bottom=True, top=True, left=True, right=True
                )

        if 'yaxiscolor' in dic.keys() and dic['yaxiscolor'] != None:
            ldic = dic['yaxiscolor']
            if 'bottom' in ldic: ax.spines['bottom'].set_color(ldic['bottom'])
            if 'top' in ldic: ax.spines['top'].set_color(ldic['top'])
            if 'right' in ldic: ax.spines['right'].set_color(ldic['right'])
            if 'left' in ldic: ax.spines['left'].set_color(ldic['left'])

        if 'xaxiscolor' in dic.keys() and dic['xaxiscolor'] != None:
            ax.xaxis.label.set_color(dic['xaxiscolor'])

        # if 'yaxiscolor' in dic.keys() and dic['yaxiscolor'] != None:
        #     # ax.spines['bottom'].set_color(dic['yaxiscolor'])
        #     # ax.spines['top'].set_color(dic['yaxiscolor'])
        #     ax.spines['right'].set_color(dic['yaxiscolor'])
        #     ax.spines['left'].set_color(dic['yaxiscolor'])
        #     ax.yaxis.label.set_color(dic['yaxiscolor'])

        if "tick_params" in dic.keys() and dic["tick_params"] != {}:
            ax.tick_params(**dic["tick_params"])


        #
        #
        # if 'yaxiscolor' in dic.keys() and dic['yaxiscolor'] != {}:
        #     ldic = dic['yaxiscolor']
        #     ax.spines['right'].set_color(ldic['right'])
        #     ax.spines['left'].set_color(ldic['left'])
        #     ax.yaxis.label.set_color(ldic['label'])
        # if 'ytickcolor' in dic.keys() and dic['ytickcolor'] != {}:
        #     ldic = dic['ytickcolor']
        #     if 'left' in ldic.keys():
        #         ax.tick_params(axis='y', left=True, colors=ldic['left'])
        #     if 'right' in ldic.keys():
        #         ax.tick_params(axis='y', right=True, colors = ldic['right'])
        # if 'yminortickcolor' in dic.keys() and dic['yminortickcolor'] != {}:
        #     ldic = dic['yminortickcolor']
        #     if 'left' in ldic.keys():
        #         ax.tick_params(axis='y', which='minor', left=True, colors=ldic['left'])
        #     if 'right' in ldic.keys():
        #         ax.tick_params(axis='y', which='minor', right=True, colors=ldic['right'])


        #
        # if 'yaxiscolor' in dic.keys() and dic['yaxiscolor'] != {}:
        #     ldic = dic['yaxiscolor']
        #     assert 'right' in ldic.keys()
        #     assert 'left' in ldic.keys()
        #     assert 'tick' in ldic.keys()
        #     assert 'label' in ldic.keys()
        #     ax.spines['right'].set_color(ldic['right'])
        #     ax.spines['left'].set_color(ldic['left'])
        #     # ax.tick_params(axis='x', colors='red')
        #     ax.tick_params(axis='y', colors=ldic['tick'])
        #     ax.tick_params(axis='y', which='minor', colors=ldic['tick'])
        #     ax.tick_params(axis='y', which='minor', colors=ldic['tick'])
        #     ax.yaxis.label.set_color(ldic['label'])

        if 'minorticks' in dic.keys():
            if dic["minorticks"]:
                ax.minorticks_on()

        if 'xticks' in dic.keys() and dic['xticks'] != None:
            ax.set_xticks(dic['xticks'])

        if 'xmajorticks' in dic.keys() and len(dic['xmajorticks']) > 0:
            ax.xaxis.set_major_locator(FixedLocator(dic['xmajorticks']))
        if 'xminorticks' in dic.keys() and len(dic['xminorticks']) > 0:
            ax.xaxis.set_minor_locator(FixedLocator(dic['xminorticks']))
        if 'xmajorlabels' in dic.keys() and len(dic['xmajorlabels']) > 0:
            ax.set_xticklabels(dic['xmajorlabels'])

            # xmajorticks = np.arange(5) * 90. / 4.
            # xminorticks = np.arange(17) * 90. / 16
            # xmajorlabels = [r"$0^\circ$", r"$22.5^\circ$", r"$45^\circ$",
            #                 r"$67.5^\circ$", r"$90^\circ$"]
            # ax.xaxis.set_major_locator(FixedLocator(xmajorticks))
            # ax.xaxis.set_minor_locator(FixedLocator(xminorticks))
            # ax.set_xticklabels(xmajorlabels)

        # if dic['task'] == 'outflow corr':
        #     if dic['v_n_x'] == 'theta':
        #         xmajorticks = np.arange(5) * 90. / 4.
        #         xminorticks = np.arange(17) * 90. / 16
        #         xmajorlabels = [r"$0^\circ$", r"$22.5^\circ$", r"$45^\circ$",
        #                         r"$67.5^\circ$", r"$90^\circ$"]
        #         ax.xaxis.set_major_locator(FixedLocator(xmajorticks))
        #         ax.xaxis.set_minor_locator(FixedLocator(xminorticks))
        #         ax.set_xticklabels(xmajorlabels)
        #     if dic['v_n_y'] == 'theta':
        #         ymajorticks = np.arange(5) * 90. / 4.
        #         yminorticks = np.arange(17) * 90. / 16
        #         ymajorlabels = [r"$0^\circ$", r"$22.5^\circ$", r"$45^\circ$",
        #                         r"$67.5^\circ$", r"$90^\circ$"]
        #         ax.yaxis.set_major_locator(FixedLocator(ymajorticks))
        #         ax.yaxis.set_minor_locator(FixedLocator(yminorticks))
        #         ax.set_yticklabels(ymajorlabels)
        #
        #     if dic['v_n_x'] == 'vel_inf' or dic['v_n_x'] == 'vel_inf_bern':
        #         if 'sharey' in dic.keys() and dic['sharey']:
        #             ax.set_xticks(np.arange(dic['xmin'], dic['xmax'], .1))
        #         else:
        #             ax.set_xticks(np.arange(dic['xmin'], dic['xmax'], .1))
        #
        #     if dic['v_n_y'] == 'vel_inf' or dic['v_n_y'] == 'vel_inf_bern':
        #         if 'sharex' in dic.keys() and dic['sharex']:
        #             ax.set_yticks(np.arange(0, 1.0, .2))
        #         else:
        #             ax.set_yticks(np.arange(0, 1.0, .2))
        #
        #     if dic['v_n_x'] == 'ye':
        #         if 'sharey' in dic.keys() and dic['sharey']:
        #             ax.set_xticks(np.arange(0.1, 0.5, .1))
        #         else:
        #             ax.xaxis.set_major_locator(MultipleLocator(0.1))
        #             ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        #
        #     if dic['v_n_y'] == 'ye':
        #         if 'sharex' in dic.keys() and dic['sharex']:
        #             ax.set_yticks(np.arange(0.1, 0.5, .1))
        #         else:
        #             ax.yaxis.set_major_locator(MultipleLocator(0.1))
        #             ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        #
        # if 'v_n_x' in dic.keys() and 'v_n_y' in dic.keys():
        #     if (dic['v_n_x'] == 'hist_theta' and dic['v_n_y'] == 'hist_theta_m'):
        #         xmajorticks = np.arange(5) * 90. / 4.
        #         xminorticks = np.arange(17) * 90. / 16
        #         xmajorlabels = [r"$0^\circ$", r"$22.5^\circ$", r"$45^\circ$",
        #                         r"$67.5^\circ$", r"$90^\circ$"]
        #         ax.xaxis.set_major_locator(FixedLocator(xmajorticks))
        #         ax.xaxis.set_minor_locator(FixedLocator(xminorticks))
        #         ax.set_xticklabels(xmajorlabels)
        #
        #     if dic['v_n_x'] == 'hist_ye' and dic['v_n_y'] == 'hist_ye_m':
        #         if 'sharey' in dic.keys() and dic['sharey']:
        #             ax.set_xticks(np.arange(0.1, 0.6, .1))
        #         else:
        #             ax.xaxis.set_major_locator(MultipleLocator(0.1))
        #             ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        #
        #     if dic['v_n_x'] == 'hist_vel_inf' and dic['v_n_y'] == 'hist_vel_inf_m':
        #         if 'sharey' in dic.keys() and dic['sharey']:
        #             ax.set_xticks(np.arange(0.2, 1.2, .2))
        #         else:
        #             ax.set_xticks(np.arange(0, 1.2, .2))
        #
        #     if dic['v_n_x'] == 'A' and dic['v_n_y'] == 'Y_final':
        #         ax.set_xticks(np.arange(50, 200, 50))
        #
        if 'sharey' in dic.keys():
            if bool(dic['sharey']):
                # pass
                # ax.set_yticklabels([])
                # ax.axes.yaxis.set_ticklabels([])
                ax.set_ylabel('')
                ax.tick_params(labelleft=False)

        if 'sharex' in dic.keys():
            if bool(dic['sharex']):
                ax.set_xticklabels([])
                ax.axes.xaxis.set_ticklabels([])
                ax.set_xlabel('')
                ax.tick_params(labelbottom=False)
        #
        # if 'centerx' in dic.keys():
        #     if dic['centerx']:
        #         ax.spines['bottom'].set_position('center')
        #         ax.xaxis.set_ticks_position('bottom')
        #
        # if 'centery' in dic.keys():
        #     if dic['centery']:
        #         ax.spines['left'].set_position('center')
        #         ax.spines['right'].set_color('none')
        if 'invert_x' in dic.keys():
            if dic['invert_x']:
                ax.axes.invert_xaxis()
        #
        if 'invert_y' in dic.keys():
            if dic['invert_y']:
                print("revertin!")
                ax = plt.gca()
                ax.set_ylim(ax.get_ylim()[::-1])
                ax.invert_yaxis()
                ax.axes.invert_yaxis()
                plt.gca().invert_yaxis()

    def plot_generic_vertical_line(self, ax, dic):

        ax.axvline(**dic["axvline"])

        # value = dic['value']
        #
        # if 'label' in dic.keys():
        #     if dic['label'] == None:
        #         ax.axvline(x=value, linestyle=dic['ls'], color=dic['color'], linewidth=dic['lw'])
        #     else:
        #         # exit(1)
        #         ax.axvline(x=value, linestyle=dic['ls'], color=dic['color'], linewidth=dic['lw'], label=dic['label'])
        # else:
        #     ax.axvline(x=value, linestyle=dic['ls'], color=dic['color'], linewidth=dic['lw'])

        return 0

    def plot_generic_horisontal_line(self, ax, dic):

        ax.axhline(**dic["axhline"])

        # value = dic['value']
        #
        # if 'label' in dic.keys():
        #     if dic['label'] == None:
        #         ax.axvline(x=value, linestyle=dic['ls'], color=dic['color'], linewidth=dic['lw'])
        #     else:
        #         # exit(1)
        #         ax.axvline(x=value, linestyle=dic['ls'], color=dic['color'], linewidth=dic['lw'], label=dic['label'])
        # else:
        #     ax.axvline(x=value, linestyle=dic['ls'], color=dic['color'], linewidth=dic['lw'])

        return 0

    def plot_generic_line(self, ax, dic, x_arr, y_arr):

        # color, lw, alpha, label,
        if 'marker' in dic.keys():
            if 'label' in dic.keys():
                ax.plot(x_arr, y_arr, dic['marker'],  color=dic['color'], markersize=dic['ms'], alpha=dic['alpha'], label=dic['label'])
            else:
                ax.plot(x_arr, y_arr, dic['marker'],  color=dic['color'], markersize=dic['ms'], alpha=dic['alpha'])

        elif 'ls' in dic.keys():
            if 'label' in dic.keys():
                ax.plot(x_arr, y_arr, ls=dic['ls'], lw=dic['lw'], color=dic['color'],  drawstyle=dic['ds'], alpha=dic['alpha'], label=dic['label'])
            else:
                ax.plot(x_arr, y_arr, ls=dic['ls'], lw=dic['lw'], color=dic['color'],  drawstyle=dic['ds'], alpha=dic['alpha'])

        else:
            raise NameError("Use 'ls' or 'marker' to plot")

        if 'mark_beginning' in dic.keys():
            if dic['mark_beginning']:
                dic_mark = dic['mark_beginning']
                self.plot_generic_line(ax, dic_mark, x_arr[0], y_arr[0])

        if 'mark_end' in dic.keys():
            if dic['mark_end']:
                dic_mark = dic['mark_end']
                self.plot_generic_line(ax, dic_mark, x_arr[-1], y_arr[-1])

        if 'marker' in dic.keys() and 'arrow' in dic.keys():
            if dic['arrow'] != None:
                if dic['arrow'] == 'up':
                    for x, y in zip([x_arr], [y_arr]):
                        ax.annotate('', xy=(x, y), xytext=(0, 20), textcoords='offset points',
                                    arrowprops=dict(arrowstyle="<|-"))

        return 0

    def plot_generic_errorbar(self, ax, dic, x_arr, y_arr, yerr):


        if 'label' in dic.keys() and dic['label'] != None:
            ax.errorbar(x_arr, y_arr, yerr=yerr, fmt=dic['marker'], color=dic['color'], markersize=dic['ms'],
                        alpha=dic['alpha'], label=dic['label'])
        else:
            ax.errorbar(x_arr, y_arr, yerr=yerr, fmt=dic['marker'], color=dic['color'], markersize=dic['ms'],
                        alpha=dic['alpha'])

        #
        #
        # if 'alpha' in dic.keys():
        #     if 'marker' in dic.keys():
        #         if 'label' in dic.keys():
        #             print("alpha marker label")
        #             ax.errorbar(x_arr, y_arr, yerr=yerr, fmt=dic['marker'], color=dic['color'], markersize=dic['ms'], alpha=dic['alpha'], label=dic['label'])
        #         else:
        #             print("alpha marker")
        #             ax.errorbar(x_arr, y_arr, yerr=yerr, fmt=dic['marker'], color=dic['color'], markersize=dic['ms'], alpha=dic['alpha'])
        #
        # else:
        #     if 'marker' in dic.keys():
        #         if 'label' in dic.keys():
        #             print("marker label")
        #             ax.errorbar(x_arr, y_arr, yerr=yerr, fmt=dic['marker'], color=dic['color'], markersize=dic['ms'], label=dic['label'])
        #             print("marker")
        #         else:
        #             ax.errorbar(x_arr, y_arr, yerr=yerr, fmt=dic['marker'], color=dic['color'], markersize=dic['ms'])
        #     else:
        #         raise NameError("what am I: marker or a line?")

    def plot_generic_band(self, ax, dic, x_arr, y1_arr, y2_arr):

        assert len(y1_arr) == len(x_arr)
        assert len(y2_arr) == len(x_arr)

        if dic["label"] == None:
            ax.fill_between(x_arr, y1_arr, y2_arr, alpha=dic['alpha'], color=dic['color'])
        else:
            ax.fill_between(x_arr, y1_arr, y2_arr, alpha=dic['alpha'], color=dic['color'], label=dic['label'])

    def treat_time_acis(self, x_arr, dic):

        if '-t' in dic.keys():
            # print("tmerger: {}".format(dic['-t']))
            x_arr = x_arr - float(dic['-t'])

        if '+t' in dic.keys():
            # print("tmerger: {}".format(dic['-t']))
            x_arr = x_arr + float(dic['+t'])

        # print("x[0]-tmrg:{}".format(x_arr[0]))
        if 'xunits' in dic.keys():
            if dic['xunits'] == 's':
                pass
            elif dic['xunits'] == 'ms':
                x_arr = x_arr * 1e3
            else:
                raise NameError("x_units {} not recognized".format(dic['xunits']))

        return x_arr

    def treat_mass_acis(self, y_arr, dic):

        if 'yunits' in dic.keys():
            if dic['yunits'] == '1e-2Msun':
                return y_arr * 1e2

    def modify_arr(self, arr, dic):

        if 'ymod' in dic.keys() and dic['ymod'] != None:
            string = dic['ymod']
            if string[0] == '*':
                # multiply
                value = float(string.split(' ')[-1])
                arr = arr * value
                return arr


class PLOT_TASK(BASIC_PARTS):

    def __init__(self):
        BASIC_PARTS.__init__(self)

    # def plot_2d_generic_colormesh(self, ax, dic):
    #
    #     if dic['dtype'] == "corr":
    #
    #         o_data = dic["data"]
    #         if 'it' in dic.keys():
    #             table = o_data.get_res_corr(dic["it"], dic["v_n_x"], dic["v_n_y"])
    #         elif 'time' in dic.keys():
    #             t = self.treat_time_acis(dic['time'], dic)
    #             it = o_data.get_it(t)
    #             table = o_data.get_res_corr(it, dic["v_n_x"], dic["v_n_y"])
    #         else:
    #             raise NameError("set it or time for data to be loaded")
    #         table = np.array(table)
    #         x_arr = np.array(table[0, 1:])  # * 6.176269145886162e+17
    #         y_arr = np.array(table[1:, 0])
    #         z_arr = np.array(table[1:, 1:])
    #
    #         z_arr = z_arr / np.sum(z_arr)
    #         z_arr = np.maximum(z_arr, z_arr.min())
    #
    #         if dic["v_n_x"] == "theta":
    #             x_arr = 90 - (x_arr * 180 / np.pi)
    #         if dic["v_n_y"] == "theta":
    #             y_arr = 90 - (y_arr * 180 / np.pi)
    #             # print(y_arr)
    #             # exit(1)
    #
    #
    #         im = self.plot_colormesh(ax, dic, x_arr, y_arr, z_arr)
    #
    #
    #
    #     elif dic['dtype'] == "int":
    #
    #         o_data = dic["data"]
    #
    #         data = np.array(o_data.get_int(dic["it"], dic["plane"], dic["v_n"]))
    #         phi = np.array(o_data.get_new_grid(dic["plane"], dic["v_n_x"]))
    #         r = np.array(o_data.get_new_grid(dic["plane"], dic["v_n_y"]))
    #
    #         # print(data)
    #         print("min:{} max:{}".format(data.min(), data.max()))
    #         # data = np.maximum(data, 9e-15)
    #         data[np.isnan(data)] = 1e-16
    #
    #
    #         im = self.plot_colormesh(ax, dic, phi, r, data)
    #
    #
    #     else:
    #         raise NameError("plot type dic['dtype'] is not recognised (given: {})".format(dic["dtype"]))
    #
    #     return im
    #
    # def plot_2d_mkn(self, ax, dic):
    #
    #
    #     data = dic['data']
    #
    #     tarr, atrarr, magarr = data.get_table(band=dic['band'], files_name_gen=dic['files'])
    #
    #
    #
    #
    #     # print("tarr: {}".format(tarr))
    #     print("atr:  {}".format(atrarr))
    #     # print("mags: {}".format(magarr))
    #
    #
    #     im = self.plot_colormesh(ax, dic, tarr, magarr, atrarr * 1e2)
    #     return im
    #
    # def plot_2d_projection(self, ax, dic):
    #
    #     """
    #
    #     dic = {
    #         'dtype': 'corr',
    #         'it': it, 'v_n_x': 'dens_unb_bern', 'v_n_y': 'ang_mom_flux', 'v_n': 'mass',
    #     }
    #
    #     :param ax:
    #     :param dic:
    #     :return:
    #     """
    #
    #
    #     if dic['dtype'] == 'int':
    #
    #         o_data = dic["data"]
    #
    #         if 'it' in dic.keys():
    #             x_arr, y_arr, z_arr = o_data.get_modified_2d_data(
    #                 dic["it"], dic["v_n_x"], dic["v_n_y"], dic["v_n"], dic["mod"]
    #             )
    #         elif 'time' in dic.keys():
    #             t = self.treat_time_acis(dic['time'], dic)
    #             it = o_data.get_it(t)
    #             x_arr, y_arr, z_arr = o_data.get_modified_2d_data(
    #                 it, dic["v_n_x"], dic["v_n_y"], dic["v_n"], dic["mod"]
    #             )
    #         else:
    #             raise NameError("specify it or time to load data")
    #         #
    #         #
    #         # y_arr = o_data.get_grid_data(dic["it"], dic["v_n_x"])  # phi
    #         # x_arr = o_data.get_grid_data(dic["it"], dic["v_n_y"])  # r
    #         # z_arr =
    #         #
    #         #
    #         # if 'mod' in dic.keys() and dic['mod'] != None:
    #         #     if dic['mod'] == 'integ_over_z':
    #         #         x_arr = np.array(x_arr[:, 0, 0])
    #         #         y_arr = np.array(y_arr[0, :, 0])
    #         #         z_arr = o_data.get_integ_over_z(dic['it'], dic['v_n'])
    #         #     elif dic['mod'] == 'integ_over_z int':
    #         #         x_arr = np.array(x_arr[:, 0, 0]) # r
    #         #         y_arr = np.array(y_arr[0, :, 0]) # phi
    #         #         z_arr = o_data.get_integ_over_z(dic['it'], dic['v_n'])
    #         #         # print(np.rad2deg(y_arr)); exit(1)
    #         #         print(x_arr[-1], y_arr[-1])
    #         #         print(x_arr.shape, y_arr.shape, z_arr.shape)
    #         #         y_arr = np.append(y_arr, 2*np.pi)
    #         #         z_arr = np.vstack((z_arr.T, z_arr[:, -1])).T
    #         #         y_arr = np.ins(y_arr, 2 * np.pi)
    #         #         z_arr = np.vstack((z_arr[:, 0], z_arr.T)).T
    #         #         print(x_arr.shape, y_arr.shape, z_arr.shape)
    #         #         # from scipy import interpolate
    #         #         # grid_x = np.linspace(0.0, 2.0 * np.pi, 360)
    #         #         # grid_y = np.linspace(dic['rmin'], dic['rmax'], 200)
    #         #         # print(x_arr.shape, y_arr.shape, z_arr.shape)
    #         #         # X, Y = np.meshgrid(x_arr, y_arr)
    #         #         # f_ = interpolate.interp2d(X, Y, z_arr)
    #         #         # z_arr = f_(grid_x, grid_y)
    #         #     else:
    #         #         raise NameError("Unknown 'mod' parameter:{} ".format(dic['mod']))
    #         # else:
    #         #     x_arr = np.array(x_arr[:, 0, 0])
    #         #     y_arr = np.array(y_arr[0, :, 0])
    #         #     z_arr = o_data.get_int_data(dic["it"], dic["v_n"])
    #         #     z_arr = np.array(z_arr[:, :, 0])  # take a slice
    #         #
    #         #
    #         #
    #         #
    #         #
    #         #
    #         # # z_arr = self.fill_arr_with_vmin(z_arr, dic)
    #         #
    #         # # print(x_arr.shape, y_arr.shape, z_arr.shape)
    #         # # print(y_arr)
    #
    #         im = self.plot_colormesh(ax, dic, y_arr, x_arr, z_arr)  # phi, r, data
    #
    #     elif dic['dtype'] == 'dm':
    #
    #         o_data = dic['data']
    #         im = 0
    #         iterations = o_data.get_grid('iterations')
    #         r_cyl = o_data.get_grid(dic['v_n_y'])
    #
    #         if dic['v_n'] == 'int_phi':
    #             int_phi2d = o_data.get_data(dic['mode'], dic['v_n'])
    #             int_phi2d_for_it = int_phi2d[int(np.where(iterations == dic['it'])[0]), :]
    #             x_arr = np.angle(int_phi2d_for_it)[r_cyl < dic['rmax']]
    #             y_arr = r_cyl[r_cyl < dic['rmax']]
    #
    #             if 'int' in dic.keys():
    #                 if dic['int'] == 'spline':
    #                     from scipy import interpolate
    #                     y_grid = np.mgrid[y_arr[0]:y_arr[-1]:1000j]
    #                     x_grid = interpolate.interp1d(y_arr, x_arr, kind='linear')(y_grid)
    #                     ax.plot(x_grid, y_grid, dic['ls'], color=dic['color'])
    #             else:
    #                 ax.plot(x_arr, y_arr, dic['ls'], color=dic['color'])
    #
    #         elif dic['v_n'] == 'int_phi_r':
    #             int_phi_r1d = o_data.get_data(dic['mode'], dic['v_n'])
    #             int_phi_r1d_for_it = int_phi_r1d[iterations == dic['it']]
    #             phi = np.zeros(r_cyl.shape)
    #             phi.fill(float(np.angle(int_phi_r1d_for_it)))
    #             ax.plot(phi[r_cyl < dic['rmax']], r_cyl[r_cyl < dic['rmax']], dic['ls'], color=dic['color'])
    #
    #         else:
    #             raise NameError("dic['v_n'] is not recognized. Use 'int_phi' or 'int_phi_r' ")
    #
    #     else:
    #         raise NameError("plot type dic['dtype'] is not recognised (given: {})".format(dic["dtype"]))
    #
    #     return im
    #
    # def plot_density_mode_line(self, ax, dic):
    #
    #
    #     if dic['dtype'] == 'dm':
    #         o_data = dic['data']
    #         im = 0
    #         x_arr = o_data.get_grid(dic['v_n_x'])
    #         # print("x[0]:{}".format(x_arr[0]))
    #
    #         x_arr = self.treat_time_acis(x_arr, dic)
    #
    #         # print("x[0]-tmrg * 1e3:{}".format(x_arr[0]))
    #
    #         if dic['v_n_y'] == 'int_phi_r abs':
    #             int_phi_r1d = o_data.get_data(dic['mode'], 'int_phi_r')
    #             y_arr = np.abs(int_phi_r1d)
    #             if 'norm_to_m' in dic.keys():
    #                 # print('Normalizing')
    #                 norm_int_phi_r1d = o_data.get_data(dic['norm_to_m'], 'int_phi_r')
    #                 # print(norm_int_phi_r1d); exit(1)
    #                 y_arr = y_arr / abs(norm_int_phi_r1d)[0]
    #
    #         elif dic['v_n_y'] == 'int_phi_r phase':
    #             int_phi_r1d = o_data.get_data(dic['mode'], 'int_phi_r')
    #             y_arr = np.unwrap(np.angle(int_phi_r1d))
    #             if 'norm_to_m' in dic.keys() and dic['norm_to_m'] != None:
    #                 raise NameError("cannot normalize phase of the mode")
    #         else:
    #             raise NameError("v_n_y {} is not recognized".format(dic['v_n_y']))
    #
    #
    #
    #         self.plot_generic_line(ax, dic, x_arr, y_arr)
    #
    #         # if dic['label'] == 'mode':
    #         #     self.plot_generic_line(ax, dic, x_arr, y_arr)
    #
    #             # ax.plot(x_arr, y_arr, ls=dic['ls'], lw=dic['lw'], color=dic['color'],
    #             #         label='m:{}'.format(dic['mode']))
    #         # elif dic['label'] == 'sim':
    #             # sim = o_data.sim
    #             # ax.plot(x_arr, y_arr, ls=dic['ls'], lw=dic['lw'], color=dic['color'],
    #             #                 label='{}'.format(sim.replace('_', '\_')))
    #         # else:
    #         #     ax.plot(x_arr, y_arr, ls=dic['ls'], lw=dic['lw'], color=dic['color'])
    #         del(x_arr)
    #     else:
    #         raise NameError("plot type dic['dtype'] is not recognised (given: {})".format(dic["dtype"]))
    #
    # def plot_tcoll_vert_line(self, ax, dic):
    #
    #     o_data = dic['data']
    #     try:
    #         value = o_data.get_par("tcoll_gw")
    #         value = self.treat_time_acis(value, dic)
    #
    #         # print(value); exit(1)
    #         if 'label' in dic.keys():
    #             if dic['label'] != None:
    #                 ax.axvline(x=value, linestyle=dic['ls'], color=dic['color'], linewidth=dic['lw'])
    #             else:
    #                 ax.axvline(x=value, linestyle=dic['ls'], color=dic['color'], linewidth=dic['lw'], label=dic['label'])
    #         else:
    #             ax.axvline(x=value, linestyle=dic['ls'], color=dic['color'], linewidth=dic['lw'])
    #     except:
    #         print("Warning! tcoll failed to be plotted")
    #
    # def plot_histogram_1d(self, ax, dic):
    #
    #     o_data = dic['data']
    #     if dic['v_n_x'] == 'hist_theta' and dic['v_n_y'] == 'hist_theta_m':
    #         tht = o_data.get_arr('hist_theta', dic['criterion'])
    #         M = o_data.get_arr('hist_theta_m', dic['criterion'])
    #
    #         if dic['norm']: M /= np.sum(M)
    #
    #         # ax.step(90. - (tht / np.pi * 180.), M, color=dic['color'], where='mid', label=dic['label'])
    #         # ax.plot(90. - (tht / np.pi * 180.), M, color=dic['color'], ls=dic['ls'], drawstyle=dic['ds'], label=dic['label'])
    #         self.plot_generic_line(ax, dic, 90. - (tht / np.pi * 180.), M)
    #         dtht = tht[1] - tht[0]
    #
    #
    #         ax.set_xlim(xmin=0 - dtht / np.pi * 180, xmax=90.)
    #         # xmajorticks = np.arange(5) * 90. / 4.
    #         # xminorticks = np.arange(17) * 90. / 16
    #         # xmajorlabels = [r"$0^\circ$", r"$22.5^\circ$", r"$45^\circ$",
    #         #                 r"$67.5^\circ$", r"$90^\circ$"]
    #         # ax.xaxis.set_major_locator(FixedLocator(xmajorticks))
    #         # ax.xaxis.set_minor_locator(FixedLocator(xminorticks))
    #         # ax.set_xticklabels(xmajorlabels)
    #         #
    #         # ax.set_xlabel(r"Angle from orbital plane")
    #
    #     elif dic['v_n_x'] == 'hist_ye' and dic['v_n_y'] == 'hist_ye_m':
    #         o_data = dic['data']
    #         ye = o_data.get_arr('hist_ye', dic['criterion'])
    #         M = o_data.get_arr('hist_ye_m', dic['criterion'])
    #
    #         if dic['norm']: M /= np.sum(M)
    #
    #         # ax.step(ye, M, color=dic['color'], where='mid', label=dic['label'])
    #
    #         # ax.plot(ye, M, color=dic['color'], ls=dic['ls'], drawstyle=dic['ds'], label=dic['label'])
    #         self.plot_generic_line(ax, dic, ye, M)
    #
    #     elif dic['v_n_x'] == 'hist_vel_inf' and dic['v_n_y'] == 'hist_vel_inf_m':
    #
    #         o_data = dic['data']
    #         vel_inf = o_data.get_arr('hist_vel_inf', dic['criterion'])
    #         M = o_data.get_arr('hist_vel_inf_m', dic['criterion'])
    #         # print(M)
    #         if dic['norm']: M /= np.sum(M)
    #
    #         # print(M)
    #         # ax.step(ye, M, color=dic['color'], where='mid', label=dic['label'])
    #         # ax.plot(vel_inf, M, color=dic['color'], ls=dic['ls'], drawstyle=dic['ds'], label=dic['label'])
    #         self.plot_generic_line(ax, dic, vel_inf, M)
    #
    # def plot_ejecta_profile(self, ax, dic):
    #
    #     o_data = dic['data']
    #
    #     if 'extrapolation' in dic.keys():
    #         dic_ext = dic['extrapolation']
    #         x_arr, y_arr = o_data.get_extrapolated_arr(dic['v_n_x'], dic['v_n_y'], dic['criterion'],
    #                                                    dic_ext['method'], dic_ext['depth'],
    #                                                    dic_ext['x_left'], dic_ext['x_right'],
    #                                                    dic_ext['x_start'], dic_ext['x_stop'])
    #
    #         # print(y_arr)
    #     else:
    #         x_arr = o_data.get_arr(dic['v_n_x'], dic['criterion'])
    #         y_arr = o_data.get_arr(dic['v_n_y'], dic['criterion'])
    #
    #
    #     if 'ymod' in dic.keys() and dic['ymod'] != None:
    #         y_arr = self.modify_arr(y_arr, dic)
    #
    #     x_arr = self.treat_time_acis(x_arr, dic)
    #     y_arr = self.treat_mass_acis(y_arr, dic)
    #
    #     self.plot_generic_line(ax, dic, x_arr, y_arr)
    #
    #     # ax.plot(x_arr, y_arr, ls = dic['ls'], color=dic['color'], lw = dic['lw'])
    #
    # def plot_nucleo_yeilds_line(self, ax, dic):
    #
    #     o_data = dic['data']
    #     if dic['v_n_x'] in o_data.list_sims_v_ns:
    #         x_arr = o_data.get_normalized_sim_data(dic['v_n_x'], criterion=dic['criterion'], method=dic['method'])
    #     elif dic['v_n_x'] in o_data.list_sol_v_ns:
    #         x_arr = o_data.get_nored_sol_abund(dic['v_n_x'], method=dic['method'])
    #     else:
    #         raise NameError("v_n_x:{} is not in available v_ns lists"
    #                         .format(dic["v_n_x"]))
    #
    #     o_data = dic['data']
    #     if dic['v_n_y'] in o_data.list_sims_v_ns:
    #         y_arr = o_data.get_normalized_sim_data(dic['v_n_y'], criterion=dic['criterion'], method=dic['method'])
    #     elif dic['v_n_y'] in o_data.list_sol_v_ns:
    #         y_arr = o_data.get_nored_sol_abund(dic['v_n_y'], method=dic['method'])
    #     else:
    #         raise NameError("v_n_y:{} is not in available v_ns lists"
    #                         .format(dic["v_n_y"]))
    #
    #     self.plot_generic_line(ax, dic, x_arr, y_arr)
    #
    # def plot_mkn_lightcurve(self, ax, dic):
    #     print("color {}".format(dic['color']))
    #     data = dic['data']
    #     m_time, m_min, m_max = data.get_model_min_max(dic['band'], fname=dic['fname'])
    #     if dic["label"] == None:
    #         ax.fill_between(m_time, m_min, m_max, alpha=dic['alpha'], color=dic['color'])
    #     else:
    #         ax.fill_between(m_time, m_min, m_max, alpha=dic['alpha'], color=dic['color'], label=dic['label'])
    #
    # def plot_mkn_obs_data(self, ax, dic):
    #
    #     data = dic['data']
    #     data_list = data.get_obs_data(dic["band"], fname="AT2017gfo.h5")
    #
    #     for i_, arr in enumerate(data_list):
    #         if dic["label"] == None:
    #             self.plot_generic_errorbar(ax, dic, arr[:, 0], arr[:, 1], yerr=arr[:, 2])
    #         else:
    #             if i_ == 0:
    #                 self.plot_generic_errorbar(ax, dic, arr[:, 0], arr[:, 1], yerr=arr[:, 2])
    #             else:
    #                 dic['label'] = None
    #                 self.plot_generic_errorbar(ax, dic, arr[:, 0], arr[:, 1], yerr=arr[:, 2])
    #
    # def plot_mkn_mismatch(self, ax, dic):
    #
    #     data = dic['data']
    #     times, min_mismatch, max_mismatch = data.get_mismatch(dic['band'], dic['fname'])
    #
    #     self.plot_generic_line(ax, dic, times, min_mismatch)
    #
    # def plot_mkn_model_middle_line(self, ax, dic):
    #
    #     data = dic['data']
    #
    #     times, mags = data.get_model_median(dic['band'], dic['fname'])
    #
    #     self.plot_generic_line(ax, dic, times, mags)
    #
    # def plot_ejecta_band_2_objects(self, ax, dic):
    #
    #     o_data1 = dic["data1"]
    #     o_data2 = dic["data2"]
    #
    #     tmerg1 = o_data1.get_par("tmerger_gw")
    #     time_arr1 = o_data1.get_arr(dic["v_n_x"], dic["criterion1"])
    #     time_arr1 = time_arr1 - tmerg1
    #     mass_flux_arr1 = o_data1.get_arr(dic["v_n_y"], dic["criterion1"])
    #     mass_flux_arr1 = self.treat_mass_acis(mass_flux_arr1, dic)
    #
    #     tmerg2 = o_data2.get_par("tmerger_gw")
    #     time_arr2 = o_data2.get_arr(dic["v_n_x"], dic["criterion2"])
    #     time_arr2 = time_arr2 - tmerg2
    #     mass_flux_arr2 = o_data2.get_arr(dic["v_n_y"], dic["criterion2"])
    #     mass_flux_arr2 = self.treat_mass_acis(mass_flux_arr2, dic)
    #
    #     from scipy import interpolate
    #
    #     print(time_arr1[-1], time_arr2[-1])
    #
    #     if time_arr2[-1] > time_arr1[-1]:
    #
    #         mass_flux_arr2 = interpolate.interp1d(time_arr2, mass_flux_arr2, kind='linear', bounds_error=False)(time_arr1)
    #         # print(len(time_arr1))
    #         time_arr1 = self.treat_time_acis(time_arr1, dic)
    #         self.plot_generic_band(ax, dic, time_arr1, mass_flux_arr1, mass_flux_arr2)
    #     else:
    #         #  time_arr2[-1] < time_arr1[-1]
    #         mass_flux_arr1 = interpolate.interp1d(time_arr1, mass_flux_arr1, kind='linear', bounds_error=False)(time_arr2)
    #         time_arr2 = self.treat_time_acis(time_arr2, dic)
    #         self.plot_generic_band(ax, dic, time_arr2, mass_flux_arr1, mass_flux_arr2)
    #
    #
    #     return 0
    #
    #
    #
    # def plot_summed_correlation_with_time(self, ax, dic):
    #
    #     data = dic['data']
    #     times = []
    #     total_masses = []
    #     for it in data.list_iterations:
    #         try:
    #             table = data.get_res_corr(int(it), dic['v_n_x'], dic['v_n_y'])
    #             time_ = data.get_time(int(it))
    #             table = np.array(table)
    #             x_arr = table[0, 1:]  # * 6.176269145886162e+17
    #             y_arr = table[1:, 0]
    #             z_arr = table[1:, 1:]
    #             total_mass = np.sum(z_arr)
    #             times.append(time_)
    #             total_masses.append(total_mass)
    #         except IOError:
    #             print("Warning: data for it:{} not found".format(it))
    #
    #     times, total_masses = zip(*sorted(zip(times, total_masses)))
    #     # times, total_masses = x_y_z_sort(times, total_masses)
    #
    #     total_masses = self.treat_mass_acis(np.array(total_masses), dic)
    #     times = self.treat_time_acis(np.array(times), dic)
    #
    #     # self.plot_generic_band(ax, dic, times, total_masses)
    #
    #     self.plot_generic_line(ax, dic, times, total_masses)
    #
    # def plot_summed_correlation_with_time_band(self, ax, dic):
    #
    #     data1 = dic['data1']
    #     data2 = dic['data2']
    #     times = []
    #     total_masses1 = []
    #     total_masses2 = []
    #     for it in data1.list_iterations:
    #         try:
    #             table1 = data1.get_res_corr(int(it), dic['v_n_x1'], dic['v_n_y1'])
    #             time_ = data1.get_time(int(it))
    #             table1 = np.array(table1)
    #             x_arr1 = table1[0, 1:]  # * 6.176269145886162e+17
    #             y_arr1 = table1[1:, 0]
    #             z_arr1 = table1[1:, 1:]
    #             total_mass1 = np.sum(z_arr1)
    #             times.append(time_)
    #             total_masses1.append(total_mass1)
    #
    #             table2 = data2.get_res_corr(int(it), dic['v_n_x2'], dic['v_n_y2'])
    #             # time_ = data2.get_time(int(it))
    #             table2 = np.array(table2)
    #             x_arr2 = table2[0, 1:]  # * 6.176269145886162e+17
    #             y_arr2 = table2[1:, 0]
    #             z_arr2 = table2[1:, 1:]
    #             total_mass2 = np.sum(z_arr2)
    #             # times.append(time_)
    #             total_masses2.append(total_mass2)
    #
    #         except IOError:
    #             print("Warning: data for it:{} not found".format(it))
    #
    #     times, total_masses1 = zip(*sorted(zip(times, total_masses1)))
    #     times, total_masses2 = zip(*sorted(zip(times, total_masses2)))
    #     # times, total_masses = x_y_z_sort(times, total_masses)
    #
    #     total_masses1 = self.treat_mass_acis(np.array(total_masses1), dic)
    #     total_masses2 = self.treat_mass_acis(np.array(total_masses2), dic)
    #     times = self.treat_time_acis(np.array(times), dic)
    #
    #     self.plot_generic_band(ax, dic, times, total_masses1, total_masses2)
    #
    #
    # def plot_outflowed_correlation(self, ax, dic):
    #
    #     data = dic['data']
    #
    #     x_arr, y_arr, mass = data.get_corr_x_y_mass(dic['v_n_x'], dic['v_n_y'], dic['criterion'])
    #
    #
    #     if dic['v_n_x'] == 'theta':
    #         # ax.set_xlim(0, 90)
    #         x_arr = 90 - (180 * x_arr / np.pi)
    #     if dic['v_n_y'] == 'theta':
    #         # ax.set_ylim(0, 90)
    #         y_arr = 90 - (180 * y_arr / np.pi)
    #
    #     if 'normalize' in dic.keys() and dic['normalize']:
    #         mass = mass / np.sum(mass)
    #         mass = np.maximum(mass, 1e-15)  # WHAT'S THAT?
    #
    #     # print(mass)
    #
    #     return self.plot_colormesh(ax, dic, x_arr, y_arr, mass)
    #
    #
    # def plot_2d_movie_plot_xy(self, ax, dic):
    #
    #     o_data = dic["data"]
    #     x_arr, y_arr, z_arr = o_data.get_modified_2d_data(
    #         dic["it"], dic['plane'], dic["v_n_x"], dic["v_n_y"], dic["v_n"], dic["mod"]
    #     )
    #
    #     im = self.plot_colormesh(ax, dic, y_arr, x_arr, z_arr)  # phi, r, data
    #     return im
    #
    # def plot_2d_movie_plot_xz(self, ax, dic):
    #
    #     o_data = dic["data"]
    #     phi_arr = o_data.get_int_grid(dic['plane'], dic["v_n_x"])
    #     z_arr = o_data.get_int_grid(dic['plane'], dic["v_n_y"])
    #     data_arr = o_data.get_int_data(dic["it"], dic['plane'], dic["v_n"])
    #
    #     phi_arr = (phi_arr[:, 0] * 180 / np.pi) - 180
    #     z_arr = z_arr[0, :]
    #
    #     print(data_arr)
    #
    #     # print(phi_arr)
    #     #
    #     print(z_arr)
    #     #
    #     # print(data_arr.shape)
    #
    #
    #     im = self.plot_colormesh(ax, dic, phi_arr, z_arr, data_arr.T)  # phi, r, data
    #     return im
    #
    # def plot_d2_slice_for_rl(self, ax, dic):
    #
    #     data = dic['data']
    #
    #     if dic['dtype'] == '2d rl':
    #         data_arr = data.get_data_rl(dic['it'], dic['plane'], dic['rl'], dic['v_n'])
    #         x_arr = data.get_grid_v_n_rl(dic['it'], dic['plane'], dic['rl'], "x")
    #         if dic['plane'] == 'xy':
    #             yz_arr = data.get_grid_v_n_rl(dic['it'], dic['plane'], dic['rl'], "y")
    #         elif dic['plane'] == 'xz':
    #             yz_arr = data.get_grid_v_n_rl(dic['it'], dic['plane'], dic['rl'], "z")
    #         else:
    #             raise NameError("unrecognized plane:{}".format(dic['plane']))
    #         im = self.plot_colormesh(ax, dic, x_arr, yz_arr, data_arr)  # phi, r, data
    #         return im
    #     elif dic['dtype'] == '3d rl':
    #         data = dic['data']
    #         data_arr = data.get_data(dic['it'], dic['rl'], dic['plane'], dic['v_n'])
    #         x_arr = data.get_data(dic['it'], dic['rl'], dic['plane'], "x")
    #         if dic['plane'] == 'xy':
    #             yz_arr = data.get_data(dic['it'], dic['rl'], dic['plane'], "y")
    #         elif dic['plane'] == 'xz':
    #             yz_arr = data.get_data(dic['it'], dic['rl'], dic['plane'], "z")
    #         else:
    #             raise NameError("unrecognized plane:{}".format(dic['plane']))
    #
    #         # print(data_arr);
    #         # # exit(1)
    #         # print(x_arr);
    #         # # exit(1)
    #         # print(yz_arr);
    #         # exit(1)
    #
    #         im = self.plot_colormesh(ax, dic, x_arr, yz_arr, data_arr)  # phi, r, data
    #         return im
    #     else:
    #         raise NameError("dic['dtype']={} is not recognized".format(dic['dtype']))
    #
    # def plot_task(self, ax, dic):
    #
    #     if dic["task"] == '2d projection':
    #         return self.plot_2d_projection(ax, dic)
    #     elif dic["task"] == '2d colormesh':
    #         return self.plot_2d_generic_colormesh(ax, dic)
    #     elif dic["task"] == 'mkn 2d':
    #         return self.plot_2d_mkn(ax, dic)
    #     elif dic["task"] == 'line':
    #         return self.plot_generic_line(ax, dic, np.array(dic['xarr']), np.array(dic['yarr']))
    #     elif dic['task'] == 'marker':
    #         return self.plot_generic_line(ax, dic, np.array(dic['x']), np.array(dic['y']))
    #     elif dic['task'] == 'vertline':
    #         return self.plot_generic_vertical_line(ax, dic)
    #     elif dic['task'] == 'horline':
    #         return self.plot_generic_horisontal_line(ax, dic)
    #     elif dic['task'] == 'hist1d':
    #         return self.plot_histogram_1d(ax, dic)
    #     elif dic['task'] == 'ejprof':
    #         return self.plot_ejecta_profile(ax, dic)
    #     elif dic['task'] == 'ejband':
    #         return self.plot_ejecta_band_2_objects(ax, dic)
    #     elif dic['task'] == 'nucleo':
    #         return self.plot_nucleo_yeilds_line(ax, dic)
    #     elif dic['task'] == 'mkn model':
    #         return self.plot_mkn_lightcurve(ax, dic)
    #     elif dic['task'] == 'mkn obs':
    #         return self.plot_mkn_obs_data(ax, dic)
    #     elif dic['task'] == 'mkn mismatch':
    #         return self.plot_mkn_mismatch(ax, dic)
    #     elif dic['task'] == 'mkn median':
    #         return self.plot_mkn_model_middle_line(ax, dic)
    #     elif dic['task'] == 'corr_sum':
    #         return self.plot_summed_correlation_with_time(ax, dic)
    #     elif dic['task'] == 'corr_sum_band':
    #         return self.plot_summed_correlation_with_time_band(ax, dic)
    #     elif dic['task'] == 'outflow corr':
    #         return self.plot_outflowed_correlation(ax, dic)
    #
    #     elif dic['task'] == '2d movie xy':
    #         return self.plot_2d_movie_plot_xy(ax, dic)
    #
    #     elif dic['task'] == '2d movie xz':
    #         return self.plot_2d_movie_plot_xz(ax, dic)
    #
    #     elif dic['task'] == 'slice':
    #         return self.plot_d2_slice_for_rl(ax, dic)
    #
    #
    #
    #     else:
    #         raise NameError("dic['task'] is not recognized ({})".format(dic["task"]))

    def plot_mkn_obs_data(self, ax, dic):

        data = dic['data']
        data_list = data.get_obs_data(dic["band"], fname="AT2017gfo.h5")

        for i_, arr in enumerate(data_list):
            if dic["label"] == None:
                self.plot_generic_errorbar(ax, dic, arr[:, 0], arr[:, 1], yerr=arr[:, 2])
            else:
                if i_ == 0:
                    self.plot_generic_errorbar(ax, dic, arr[:, 0], arr[:, 1], yerr=arr[:, 2])
                else:
                    dic['label'] = None
                    self.plot_generic_errorbar(ax, dic, arr[:, 0], arr[:, 1], yerr=arr[:, 2])

    def plot_corr2d(self, ax, dic):

        if "data" in dic.keys():
            corr = dic["data"]
            x_arr = np.array(corr[0, 1:])  # * 6.176269145886162e+17
            y_arr = np.array(corr[1:, 0])
            z_arr = np.array(corr[1:, 1:])
        elif "xarr" in dic.keys() and "yarr" in dic.keys() and "zarr" in dic.keys():
            x_arr = np.array(dic["xarr"])  # * 6.176269145886162e+17
            y_arr = np.array(dic["yarr"])
            z_arr = np.array(dic["zarr"])
        else:
            raise NameError("neither 'data' nor '[x,y,z]arr' found in dic.keys(): {}"
                            .format(dic.keys()))

        if "normalize" in dic.keys():
            if dic["normalize"]:
                z_arr = z_arr / np.sum(z_arr)

        # z_arr = np.maximum(z_arr, 1e-10)
        
        if dic["v_n_x"] in ["theta"]:
            x_arr = 90 - (x_arr * 180 / np.pi)
        if dic["v_n_y"] in ["theta"]:
            y_arr = 90 - (y_arr * 180 / np.pi)
        if dic["v_n_x"] in ["phi"]:
            x_arr =  (x_arr * 180 / np.pi)
        if dic["v_n_y"] in ["phi"]:
            y_arr = (y_arr * 180 / np.pi)
            print(y_arr.min(),y_arr.max())
        im = self.plot_colormesh(ax, dic, x_arr, y_arr, z_arr)
        return im

    def plot_hist1d(self, ax, dic):

        if "data" in dic.keys():
            hist = dic["data"]
            dataarr = hist[:, 0]
            massarr = hist[:, 1]
        elif "xarr" in dic.keys() and "yarr" in dic.keys():
            dataarr = dic["xarr"]
            massarr = dic["yarr"]
        else:
            raise NameError("neither 'data' nor 'xarr' and 'yarr' found in the dic:{}"
                            .format(dic.keys()))
        # normalisation
        if dic['normalize']: massarr /= np.sum(massarr)

        if dic["v_n_x"] == "theta":
            self.plot_generic_line(ax, dic, 90 - (dataarr * 180 / np.pi), massarr)
        elif dic["v_n_x"] == "phi":
            self.plot_generic_line(ax, dic, (dataarr / np.pi * 180.), massarr)
        elif dic["v_n_x"] == "Y_e" or dic["v_n_x"] == "Ye" or dic["v_n_x"] == "ye":
            self.plot_generic_line(ax, dic, dataarr, massarr)
        elif dic["v_n_x"] == "vinf" or dic["v_n_x"] == "vel_inf" or dic["v_n_x"] == "vel inf":
            self.plot_generic_line(ax, dic, dataarr, massarr)
        elif dic["v_n_x"] == "entropy" or dic["v_n_x"] == "s":
            self.plot_generic_line(ax, dic, dataarr, massarr)
        else:
            self.plot_generic_line(ax, dic, dataarr, massarr)
        #
        # if dic["v_n_x"] == "theta":
        #     hist = dic["data"]
        #     tht = hist[:, 0]
        #     M = hist[:, 1]
        #
        #     if dic['normalize']: M /= np.sum(M)
        #
        #     # ax.step(90. - (tht / np.pi * 180.), M, color=dic['color'], where='mid', label=dic['label'])
        #     # ax.plot(90. - (tht / np.pi * 180.), M, color=dic['color'], ls=dic['ls'], drawstyle=dic['ds'], label=dic['label'])
        #     self.plot_generic_line(ax, dic, 90. - (tht / np.pi * 180.), M)
        #     dtht = tht[1] - tht[0]
        #
        #     # ax.set_xlim(xmin=0 - dtht / np.pi * 180, xmax=90.)
        # elif dic["v_n_x"] == "phi":
        #     hist = dic["data"]
        #     tht = hist[:, 0]
        #     M = hist[:, 1]
        #
        #     if dic['normalize']: M /= np.sum(M)
        #
        #     # ax.step(90. - (tht / np.pi * 180.), M, color=dic['color'], where='mid', label=dic['label'])
        #     # ax.plot(90. - (tht / np.pi * 180.), M, color=dic['color'], ls=dic['ls'], drawstyle=dic['ds'], label=dic['label'])
        #     self.plot_generic_line(ax, dic, (tht / np.pi * 180.), M)
        #     dtht = tht[1] - tht[0]
        #
        #     ax.set_xlim(xmin=0 - dtht / np.pi * 180, xmax=180.)
        # elif dic["v_n_x"] == "Y_e" or dic["v_n_x"] == "Ye" or dic["v_n_x"] == "ye":
        #     hist = dic['data']
        #     ye = hist[:, 0]
        #     M = hist[:, 1]
        #
        #     if dic['normalize']: M /= np.sum(M)
        #
        #     # ax.step(ye, M, color=dic['color'], where='mid', label=dic['label'])
        #
        #     # ax.plot(ye, M, color=dic['color'], ls=dic['ls'], drawstyle=dic['ds'], label=dic['label'])
        #     self.plot_generic_line(ax, dic, ye, M)
        # elif dic["v_n_x"] == "vinf" or dic["v_n_x"] == "vel_inf" or dic["v_n_x"] == "vel inf":
        #     hist = dic['data']
        #     vel_inf =hist[:, 0]
        #     M = hist[:, 1]
        #
        #     if dic['normalize']: M /= np.sum(M)
        #
        #     # ax.step(ye, M, color=dic['color'], where='mid', label=dic['label'])
        #
        #     # ax.plot(ye, M, color=dic['color'], ls=dic['ls'], drawstyle=dic['ds'], label=dic['label'])
        #     self.plot_generic_line(ax, dic, vel_inf, M)
        # elif dic["v_n_x"] == "entropy" or dic["v_n_x"] == "s":
        #     hist = dic['data']
        #     s =hist[:, 0]
        #     M = hist[:, 1]
        #
        #     if dic['normalize']: M /= np.sum(M)
        #
        #     # ax.step(ye, M, color=dic['color'], where='mid', label=dic['label'])
        #
        #     # ax.plot(ye, M, color=dic['color'], ls=dic['ls'], drawstyle=dic['ds'], label=dic['label'])
        #     self.plot_generic_line(ax, dic, s, M)
        # else:
        #     hist = dic['data']
        #     s = hist[:, 0]
        #     M = hist[:, 1]
        #
        #     if dic['normalize']: M /= np.sum(M)
        #
        #     self.plot_generic_line(ax, dic, s, M)
        #     print("\tplotting unknown histogram: v_n_x:{} v_n_y:{}".format(dic["v_n_x"], dic["v_n_y"]))
        #     #raise NameError("Plotting method for hist: dic['v_n']:{} is not available".format(dic["v_n"]))
        return 0

    def plot_task(self, ax, dic):

        if "axvline" in dic.keys():
            self.plot_generic_vertical_line(ax, dic)
        if "axhline" in dic.keys():
            self.plot_generic_horisontal_line(ax, dic)

        if "textold" in dic.keys():
            return self.plot_text(ax, dic)

        if dic["task"] == "hist1d":
            return self.plot_hist1d(ax, dic)
        elif dic["task"] == "corr2d":
            return self.plot_corr2d(ax, dic)
        elif dic["task"] == "line":
            assert "xarr" in dic.keys()
            assert "yarr" in dic.keys()
            self.plot_generic_line(ax, dic, dic["xarr"], dic["yarr"])
        elif dic["task"] == "colormesh":
            assert "xarr" in dic.keys()
            assert "yarr" in dic.keys()
            assert "zarr" in dic.keys()
            return self.plot_colormesh(ax, dic, dic["xarr"], dic["yarr"], dic["zarr"])
        elif dic['task'] == 'mkn obs':
            return self.plot_mkn_obs_data(ax, dic)
        elif dic['task'] == 'scatter':
            assert "xarr" in dic.keys()
            assert "yarr" in dic.keys()
            assert "zarr" in dic.keys()
            return self.plot_scatter(ax, dic, np.array(dic["xarr"]), np.array(dic["yarr"]), np.array(dic["zarr"]))
        elif dic['task'] == "textold":
            return self.plot_text(ax, dic)
        elif dic['task'] == "text":
            return self.plot_text2(ax, dic)
        elif dic['task'] == 'contour':
            assert "xarr" in dic.keys()
            assert "yarr" in dic.keys()
            assert "zarr" in dic.keys()
            return self.plot_countour(ax, dic, dic["xarr"], dic["yarr"], dic["zarr"])
        else:
            raise NameError("dic['task'] is not recognized ({})".format(dic["task"]))


class PLOT_MANY_TASKS(PLOT_TASK):

    def __init__(self):

        PLOT_TASK.__init__(self)

        self.gen_set = {
            "figdir": './',
            "dpi": 128,
            "figname": "rename_me.png",
            # "figsize": (13.5, 3.5), # <->, |
            "figsize": (3.8, 3.5),  # <->, |
            "type": "cartesian",
            "subplots_adjust_h": 0.2,
            "subplots_adjust_w": 0.3,
            "fancy_ticks": False,
            "minorticks_on": False,
            "invert_y": False,
            "invert_x": False,
            "sharex": False,
            "sharey": False,
            'style':None,
        }

        self.set_plot_dics = []

    def set_ncols_nrows(self):

        tmp_rows = []
        tmp_cols = []

        for dic in self.set_plot_dics:
            tmp_cols.append(int(dic['position'][1]))
            tmp_rows.append(int(dic['position'][0]))

        max_row = max(tmp_rows)
        max_col = max(tmp_cols)

        for row in range(1, max_row):
            if not row in tmp_rows:
                raise NameError("Please set vertical plot position in a subsequent order: 1,2,3... not 1,3...")

        for col in range(1, max_col):
            if not col in tmp_cols:
                raise NameError("Please set horizontal plot position in a subsequent order: 1,2,3... not 1,3..."
                                "col:{} tmp_cols:{}".format(col, tmp_cols))

        print("\tSet {} rows {} columns (total {}) of plots".format(max_row, max_col, len(self.set_plot_dics)))

        return int(max_row), int(max_col)

    def set_plot_dics_matrix(self):

        plot_dic_matrix = [[0
                             for x in range(self.n_rows)]
                             for y in range(self.n_cols)]

        # get a matrix of dictionaries describing plots (for ease of representation)
        for dic in self.set_plot_dics:
            col, row = int(dic['position'][1]-1), int(dic['position'][0]-1) # -1 as position starts with 1
            # print(col, row)
            for n_row in range(self.n_rows):
                for n_col in range(self.n_cols):
                    if int(col) == int(n_col) and int(row) == int(n_row):
                        plot_dic_matrix[n_col][n_row] = dic
                        # print('adding {} {}'.format(col, row))

            if isinstance(plot_dic_matrix[col][row], int):
                raise ValueError("Dictionary to found for n_row {} n_col {} in "
                                 "creating matrix of dictionaries".format(col, row))

        return plot_dic_matrix

    def set_plot_matrix(self):

         # (<->; v)
        # fig = self.figure

        if self.gen_set['type'] == 'cartesian':
            # initializing the matrix with dummy axis objects
            sbplot_matrix = [[self.figure.add_subplot(self.n_rows, self.n_cols, 1)
                              for x in range(self.n_rows)]
                             for y in range(self.n_cols)]

            i = 1
            for n_row in range(self.n_rows):
                for n_col in range(self.n_cols):

                    if n_col == 0 and n_row == 0:
                        sbplot_matrix[n_col][n_row] = self.figure.add_subplot(self.n_rows, self.n_cols, i)#, aspect=aspect)#, adjustable='box')
                    elif n_col == 0 and n_row > 0:
                        if self.gen_set['sharex']:
                            sbplot_matrix[n_col][n_row] = self.figure.add_subplot(self.n_rows, self.n_cols, i,
                                                                          sharex=sbplot_matrix[n_col][0])#, aspect=aspect)#, adjustable='box')
                        else:
                            sbplot_matrix[n_col][n_row] = self.figure.add_subplot(self.n_rows, self.n_cols, i)#, aspect=aspect)#, adjustable='box')
                    elif n_col > 0 and n_row == 0:
                        if self.gen_set['sharey']:
                            sbplot_matrix[n_col][n_row] = self.figure.add_subplot(self.n_rows, self.n_cols, i,
                                                                          sharey=sbplot_matrix[0][n_row])#, aspect=aspect)#, adjustable='box')
                        else:
                            sbplot_matrix[n_col][n_row] = self.figure.add_subplot(self.n_rows, self.n_cols, i)#, aspect=aspect)#, adjustable='box')
                    else:
                        if self.gen_set['sharex'] and not self.gen_set['sharey']:
                            sbplot_matrix[n_col][n_row] = self.figure.add_subplot(self.n_rows, self.n_cols, i,
                                                                          sharex=sbplot_matrix[n_col][0])#, aspect=aspect)#, adjustable='box')
                        elif not self.gen_set['sharex'] and self.gen_set['sharey']:
                            sbplot_matrix[n_col][n_row] = self.figure.add_subplot(self.n_rows, self.n_cols, i,
                                                                          sharey=sbplot_matrix[0][n_row])#, aspect=aspect)#, adjustable='box')
                        else:
                            sbplot_matrix[n_col][n_row] = self.figure.add_subplot(self.n_rows, self.n_cols, i,
                                                                          sharex=sbplot_matrix[n_col][0],
                                                                          sharey=sbplot_matrix[0][n_row])#, aspect=aspect)#, adjustable='box')


                    plotdic = self.plot_dic_matrix[n_col][n_row]

                    if not isinstance(plotdic, int):
                        if "apsect" in plotdic:
                            if plotdic["aspect"] != None:
                                aspect = float(plotdic["aspect"])
                                sbplot_matrix[n_col][n_row].set_aspect(aspect)
                        # sbplot_matrix[n_col][n_row].axes.get_yaxis().set_visible(False)
                    # sbplot_matrix[n_col][n_row].set_aspect(aspect)#fig.add_subplot(n_rows, n_cols, i)
                    i = i + 1

        elif self.gen_set['type'] == 'polar':
            # initializing the matrix with dummy axis objects
            sbplot_matrix = [[self.figure.add_subplot(self.n_rows, self.n_cols, 1, projection='polar')
                                  for x in range(self.n_rows)]
                                  for y in range(self.n_cols)]

            i = 1
            for n_row in range(self.n_rows):
                for n_col in range(self.n_cols):

                    if n_col == 0 and n_row == 0:
                        sbplot_matrix[n_col][n_row] = self.figure.add_subplot(self.n_rows, self.n_cols, i, projection='polar')
                    elif n_col == 0 and n_row > 0:
                        sbplot_matrix[n_col][n_row] = self.figure.add_subplot(self.n_rows, self.n_cols, i, projection='polar')
                                                                      # sharex=self.sbplot_matrix[n_col][0])
                    elif n_col > 0 and n_row == 0:
                        sbplot_matrix[n_col][n_row] = self.figure.add_subplot(self.n_rows, self.n_cols, i, projection='polar')
                                                                      # sharey=self.sbplot_matrix[0][n_row])
                    else:
                        sbplot_matrix[n_col][n_row] = self.figure.add_subplot(self.n_rows, self.n_cols, i, projection='polar')
                                                                      # sharex=self.sbplot_matrix[n_col][0],
                                                                      # sharey=self.sbplot_matrix[0][n_row])
                    i = i + 1

                        # sbplot_matrix[n_col][n_row].axes.get_yaxis().set_visible(False)
                    # sbplot_matrix[n_col][n_row] = fig.add_subplot(n_rows, n_cols, i)
        else:
            raise NameError("type of the plot is not recognized. Use 'polar' or 'cartesian' ")

        # print(sbplot_matrix)

        return sbplot_matrix

    def plot_images(self):

        # initializing the matrix of images for colorbars (if needed)
        image_matrix = [[0
                        for x in range(self.n_rows)]
                        for y in range(self.n_cols)]


        for n_row in range(self.n_rows):
            for n_col in range(self.n_cols):

                ax = self.sbplot_matrix[n_col][n_row]
                ax.clear()
                ax.cla()
                ax.set_xscale("linear")
                ax.set_yscale("linear")
                ax.set_xlim(0.1, 100.)
                ax.set_ylim(0.1, 100.)

                for dic in self.set_plot_dics:
                    if (n_col + 1) == int(dic['position'][1]) and (n_row + 1) == int(dic['position'][0]):
                        # print("\tPlotting n_row:{} n_col:{}".format(n_row, n_col))

                        # ax
                        self.set_min_max_scale(ax, dic, n_col, n_row)

                        # print("\t\tsubplot:{}".format(ax))
                        # dic = self.plot_dic_matrix[n_col][n_row]
                        if isinstance(dic, int):
                            print("Warning: Dictionary for row:{} col:{} not set".format(n_row, n_col))
                            self.figure.delaxes(ax)  # delets the axis for empty plot
                        else:
                            dic = dict(dic)
                            im = self.plot_task(ax, dic)
                            if not isinstance(im, int):
                                image_matrix[n_col][n_row] = im
                                self.plot_one_cbar(im, dic, n_row, n_col)
                            self.set_plot_title(ax, dic)
                            self.set_xy_labels(ax, dic)
                            self.set_legend(ax, dic)
                            self.remover_some_ticks(ax, dic)
                            # self.plot_text(ax, dic)
                            self.add_fancy_to_ax(ax, dic)

                            # self.set_min_max_scale(ax, dic, n_col, n_row)

                            if 'aspect' in dic.keys():
                                ax.set_aspect(dic['aspect'])
                            # self.sbplot_matrix[n_col][n_row] = ax
                    # ax = self.sbplot_matrix[n_col][n_row]
                    # ax.clear()
                    # ax.cla()
        # for dic in self.set_plot_dics:
        #     for n_row in range(self.n_rows):
        #         for n_col in range(self.n_cols):
        #             if n_col + 1 == int(dic['position'][1]) and n_row + 1 == int(dic['position'][0]):
        #                 ax = self.set_min_max_scale(dic, n_col, n_row)
        #                 break
        #                 break

        return image_matrix

    # def account_for_shared(self, ax, n_row, n_col):
    #
    #     ax.axes.xaxis.set_ticklabels([])
    #     ax.axes.yaxis.set_ticklabels([])
    #
    #     if n_col > 0 and n_row < self.n_rows:
    #         # ax.tick_params(labelbottom=False)
    #         # ax.tick_params(labelleft=False)
    #         #
    #         # ax.set_yticklabels([])
    #         # ax.set_xticklabels([])
    #         #
    #         # ax.get_yaxis().set_ticks([])
    #         #
    #         # ax.set_yticks([])
    #         # ax.set_yticklabels(labels=[])
    #         # # ax.set_yticklabels([]).remove()
    #
    #         # ax.axes.get_xaxis().set_visible(False)
    #         # ax.axes.get_yaxis().set_visible(False)
    #
    #         ax.axes.xaxis.set_ticklabels([])
    #         ax.axes.yaxis.set_ticklabels([])
    #
    #         # ax.tick_params(labelbottom=False)
    #         # ax.tick_params(labelleft=False)
    #         # ax.tick_params(labelright=False)
    #
    #         # ax.get_yaxis().set_visible(False)
    #
    #     if n_col > 0:
    #         ax.set_ylabel('')
    #
    #     if n_row != self.n_rows-1:
    #         ax.set_xlabel('')

    def plot_one_cbar(self, im, idic, n_row, n_col):

        if "cbar" in idic.keys():
            cdic = idic["cbar"]
            if not isinstance(cdic, dict):
                raise NameError("'cbar' must be dic")
            if len(cdic.keys()) > 0 :
                # print("\tColobar for n_row:{} n_col:{}".format(n_row, n_col))
                location = cdic["location"].split(' ')[0]
                shift_h = float(cdic["location"].split(' ')[1])
                shift_w = float(cdic["location"].split(' ')[2])
                cbar_width = 0.02

                if location == 'right':
                    ax_to_use = self.sbplot_matrix[n_col][n_row]
                    pos1 = ax_to_use.get_position()
                    pos2 = [pos1.x0 + pos1.width + shift_h,
                            pos1.y0 + shift_w,
                            cbar_width,
                            pos1.height] # 0.5
                    if 'aspect' in cdic.keys() and cdic['aspect'] != None:
                        cax1 = self.figure.add_axes(pos2, aspect = cdic['aspect'])
                    else: cax1 = self.figure.add_axes(pos2)
                elif location == 'left':
                    ax_to_use = self.sbplot_matrix[n_col][n_row]
                    pos1 = ax_to_use.get_position()
                    pos2 = [pos1.x0 - pos1.width - shift_h,
                            pos1.y0 + shift_w,
                            cbar_width,
                            pos1.height]
                    if 'aspect' in cdic.keys() and cdic['aspect'] != None:
                        cax1 = self.figure.add_axes(pos2, aspect=cdic['aspect'])
                    else:
                        cax1 = self.figure.add_axes(pos2)
                elif location == 'bottom':
                    cbar_width = 0.02
                    ax_to_use = self.sbplot_matrix[n_col][n_row]
                    pos1 = ax_to_use.get_position()
                    pos2 = [pos1.x0 + shift_w,
                            pos1.y0 + shift_h,
                            pos1.width,
                            cbar_width]
                    if 'aspect' in cdic.keys() and cdic['aspect'] != None:
                        cax1 = self.figure.add_axes(pos2, aspect=cdic['aspect'])
                    else:
                        cax1 = self.figure.add_axes(pos2)
                elif location == "right_auto":
                    ax_to_use = self.sbplot_matrix[n_col][n_row]
                    divider = make_axes_locatable(ax_to_use)
                    pos1 = ax_to_use.get_position()
                    pos2 = [pos1.x0 + shift_w,
                            pos1.y0 + shift_h,
                            pos1.width,
                            cbar_width]
                    cax1 = divider.append_axes("right", size="5%", pad=0.05)

                else:
                    raise NameError("cbar location {} not recognized. Use 'right' or 'bottom' "
                                    .format(location))

                if location == 'right':
                    if 'fmt' in cdic.keys() and cdic['fmt'] != None:
                        cbar = plt.colorbar(im, cax=cax1, extend='both', format=cdic['fmt'])
                    else:
                        cbar = plt.colorbar(im, cax=cax1, extend='both')  # , format='%.1e')
                elif location == 'left':
                    if 'fmt' in cdic.keys() and cdic['fmt'] != None:
                        cbar = plt.colorbar(im, cax=cax1, extend='both', format=cdic['fmt'])#, format='%.1e')
                    else:
                        cbar = plt.colorbar(im, cax=cax1, extend='both')
                    cax1.yaxis.set_ticks_position('left')
                    cax1.yaxis.set_label_position('left')
                elif location == 'bottom':
                    if 'fmt' in cdic.keys() and cdic['fmt'] != None:
                        cbar = plt.colorbar(im, cax=cax1, orientation="horizontal", extend='both', format=cdic['fmt'])  # , format='%.1e')
                    else:
                        cbar = plt.colorbar(im, cax=cax1, orientation="horizontal", extend='both')
                    cax1.yaxis.set_ticks_position('left')
                    cax1.yaxis.set_label_position('left')
                elif location == "right_auto":
                    pass
                    # if 'fmt' in cdic.keys() and cdic['fmt'] != None:
                    #     cbar = plt.colorbar(im, cax=cax1, extend='both', format=cdic['fmt'])
                    # else:
                    #     cbar = plt.colorbar(im, cax=cax1, extend='both')  # , format='%.1e')
                else:
                    raise NameError("cbar location {} not recognized. Use 'right' or 'bottom' "
                                    .format(location))
                if 'label' in cdic.keys() and cdic['label'] != None:

                    if location != "bottom":
                        cbar.ax.set_title(cdic['label'], fontsize=cdic["fontsize"])
                    else:
                        cbar.set_label(cdic['label'], fontsize=cdic["fontsize"])

                # else:
                #     if location != "bottom":
                #         cbar.ax.set_title(r"{}".format(str(cdic["v_n"]).replace('_', '\_')), fontsize=cdic["fontsize"])
                #     else:
                #         cbar.set_label(r"{}".format(str(cdic["v_n"]).replace('_', '\_')), fontsize=cdic["fontsize"])

                cbar.ax.tick_params(labelsize=cdic["labelsize"])
    #
    def plot_colobars(self):

        for n_row in range(self.n_rows):
            for n_col in range(self.n_cols):
                for dic in self.set_plot_dics:
                    if n_col + 1 == int(dic['position'][1]) and n_row + 1 == int(dic['position'][0]):
                        # ax  = self.sbplot_matrix[n_col][n_row]
                        # dic = self.plot_dic_matrix[n_col][n_row]
                        im  = self.image_matrix[n_col][n_row]
                        if isinstance(dic, int):
                            print("Warning: Dictionary for row:{} col:{} not set".format(n_row, n_col))
                        else:
                            self.plot_one_cbar(im, dic, n_row, n_col)


        # for n_row in range(self.n_rows):
        #     for n_col in range(self.n_cols):
        #         print("Colobar for n_row:{} n_col:{}".format(n_row, n_col))
        #         # ax  = self.sbplot_matrix[n_col][n_row]
        #         dic = self.plot_dic_matrix[n_col][n_row]
        #         im  = self.image_matrix[n_col][n_row]
        #         if isinstance(dic, int):
        #             Printcolor.yellow("Dictionary for row:{} col:{} not set".format(n_row, n_col))
        #         else:
        #             self.plot_one_cbar(im, dic, n_row, n_col)

    def save_plot(self):

        if self.gen_set["invert_y"]:
            plt.gca().invert_yaxis()

        if self.gen_set["invert_x"]:
            plt.gca().invert_xaxis()

        plt.subplots_adjust(hspace=self.gen_set["subplots_adjust_h"])
        plt.subplots_adjust(wspace=self.gen_set["subplots_adjust_w"])
        # plt.tight_layout()
        plt.savefig('{}{}'.format(self.gen_set["figdir"], self.gen_set["figname"]),
                    bbox_inches='tight', dpi=self.gen_set["dpi"])

        # clean up
        # for n_row in range(self.n_rows):
        #     for n_col in range(self.n_cols):
        #         for dic in self.set_plot_dics:
        #             if n_col + 1 == int(dic['position'][1]) and n_row + 1 == int(dic['position'][0]):
        #                 ax = self.sbplot_matrix[n_col][n_row]
        #                 plt.delaxes(ax)
        plt.close()

    def set_scales_limits(self):

        for n_row in range(self.n_rows):
            for n_col in range(self.n_cols):
                for dic in self.set_plot_dics:
                    if (n_col + 1) == int(dic['position'][1]) and (n_row + 1) == int(dic['position'][0]):
                        ax = self.sbplot_matrix[n_col][n_row]

                        self.set_min_max_scale(ax, dic, n_col, n_row)


    def main(self):

        if len(self.set_plot_dics) == 0:
            raise ValueError("No plot dics have been passed. Exiting")

        self.figure = None # clean figure
        #
        self.figure = plt.figure(figsize=self.gen_set['figsize'])
        if not self.gen_set['style'] == None:
            plt.style.use(self.gen_set['style'])
        # initializing the n_cols, n_rows
        self.n_rows, self.n_cols = self.set_ncols_nrows()
        # initializing the matrix of dictionaries of the
        self.plot_dic_matrix = self.set_plot_dics_matrix()
        # initializing the axis matrix (for all subplots) and image matrix fo colorbars
        self.sbplot_matrix = self.set_plot_matrix()

        # self.set_scales_limits()

        # plotting
        # print(self.sbplot_matrix)
        self.image_matrix = self.plot_images()


        # adding colobars
        # self.plot_colobars()

        # saving the result
        self.save_plot()

        print("\tPlotted:\n\t{}".format(self.gen_set["figdir"] + self.gen_set["figname"]))

        # for n_row in range(self.n_rows):
        #     for n_col in range(self.n_cols):
        #         self.figure.delaxes(self.sbplot_matrix[n_col][n_row])

        # self.figure.delaxes(self.sbplot_matrix)
        self.figure.clear()
        self.figure.clf()