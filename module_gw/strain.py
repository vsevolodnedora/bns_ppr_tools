"""

"""

from __future__ import division
from sys import path
import numpy as np
path.append('modules/')
from math import pi
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import scidata.multipole as multipole
from scidata.utils import diff, fixed_freq_int_1, fixed_freq_int_2, integrate
from scidata.windows import exponential as window
from scipy.signal import detrend

from uutils import Printcolor, Constants

# from uutils import

class STRAIN:
    def __init__(self, indir, outdir):

        self.options = {
            "cutoff.freq": 0.002,
            "detector.radius": 400.0,
            'window.delta': 200.0,
        }

        self.indata_path = indir    # path to collated data
        self.outdata_path = outdir

    @staticmethod
    def factor(m):
        return 1 if m == 0 else 2

    def load_data(self):

        dset = multipole.dataset(basedir=self.indata_path, ignore_negative_m=True)

        Psi4 = {}
        dic_t = {}
        list_t = []
        do_interpol = []
        i = 0
        def_t, def_psi = dset.get(var="Psi4", l=0, m=0, r=self.options["detector.radius"])
        for l, m in dset.modes:
            print("\tReading (%d,%d)..." % (l, m)),
            t, psi = dset.get(var="Psi4", l=l, m=m, r=self.options["detector.radius"])
            # psi = np.unique(psi)
            Psi4[(l, m)] = psi
            dic_t[l, m] = t
            Psi4[l, m] *= self.options['detector.radius']
            print("done!")
            if i > 0:
                if len(t) != len(def_t):
                    do_interpol.append((l, m))
            i = i + 1

        # this is in case different modes have different length. (was found for SLy4)
        if len(do_interpol) != 0:
            from scipy import interpolate
            for l, m in do_interpol:
                print("Warning: Performing re-interpolation ofr l:{} m:{}".format(l, m))
                Psi4[(l, m)] = interpolate.interp1d(dic_t[l, m], Psi4[l, m], kind="linear")(def_t)

        t = def_t

        return t, dset, Psi4

    def main(self):

        Printcolor.blue("Note. The following parameters are used:")
        Printcolor.print_colored_string(["cutoff.freq", "{}".format(self.options["cutoff.freq"])],
                                        ["blue", "green"])
        Printcolor.print_colored_string(["detector.radius", "{}".format(self.options["detector.radius"])],
                                        ["blue", "green"])
        Printcolor.print_colored_string(["window.delta", "{}".format(self.options["window.delta"])],
                                        ["blue", "green"])

        options = self.options

        t, dset, Psi4 = self.load_data()
        dtime = t[1] - t[0]
        win = window(t / t.max(), delta=self.options["window.delta"] / t.max())


        h_2i = {}

        # Strain from fixed-frequency integration
        h_ff = {}

        A_ff = {}
        phi_ff = {}
        omega_ff = {}

        h_dot_ff = {}
        E_GW_dot_ff = {}
        E_GW_ff = {}
        J_GW_dot_ff = {}
        J_GW_ff = {}

        for l, m in dset.modes:
            # print('l:{} m:{}'.format(l, m))
            # print('win: {}'.format(win))
            # print('len(win): {}'.format(len(win)))

            psi4_lm = Psi4[l, m] * win
            h_2i[(l, m)] = detrend(integrate(integrate(psi4_lm))) * (dtime) ** 2
            # Note: we also window the integrated waveform to suppress noise
            # at early times (useful to make hybrid waveforms)
            h_ff[(l, m)] = win * fixed_freq_int_2(psi4_lm,
                                                  2 * self.options["cutoff.freq"] / max(1, abs(m)), dt=dtime)

            A_ff[(l, m)] = np.abs(h_ff[l, m])
            phi_ff[(l, m)] = -np.unwrap(np.angle(h_ff[l, m]))
            omega_ff[(l, m)] = diff(phi_ff[l, m]) / diff(t)

            # Note: we also window the integrated waveform to suppress noise
            # at early times (useful to make hybrid waveforms)
            h_dot_ff[(l, m)] = win * fixed_freq_int_1(psi4_lm,
                                                      2 * self.options["cutoff.freq"] / max(1, abs(m)), dt=dtime)
            E_GW_dot_ff[(l, m)] = 1.0 / (16. * pi) * np.abs(h_dot_ff[l, m]) ** 2
            E_GW_ff[(l, m)] = integrate(E_GW_dot_ff[l, m]) * dtime
            J_GW_dot_ff[(l, m)] = 1.0 / (16. * pi) * m * np.imag(h_ff[l, m] * np.conj(h_dot_ff[l, m]))
            J_GW_ff[(l, m)] = integrate(J_GW_dot_ff[l, m]) * dtime

        E_GW_dot = {}
        E_GW = {}
        J_GW_dot = {}
        J_GW = {}
        for m in dset.mmodes:
            E_GW_dot[m] = np.zeros_like(t)
            E_GW[m] = np.zeros_like(t)
            J_GW_dot[m] = np.zeros_like(t)
            J_GW[m] = np.zeros_like(t)
        for l, m in dset.modes:
            E_GW_dot[m] += self.factor(m) * E_GW_dot_ff[l, m]
            E_GW[m] += self.factor(m) * E_GW_ff[l, m]
            J_GW_dot[m] += self.factor(m) * J_GW_dot_ff[l, m]
            J_GW[m] += self.factor(m) * J_GW_ff[l, m]

        E_GW_dot_all = np.zeros_like(t)
        E_GW_all = np.zeros_like(t)
        J_GW_dot_all = np.zeros_like(t)
        J_GW_all = np.zeros_like(t)
        for m in dset.mmodes:
            E_GW_dot_all += E_GW_dot[m]
            E_GW_all += E_GW[m]
            J_GW_dot_all += J_GW_dot[m]
            J_GW_all += J_GW[m]

        # -------------------------------------------------------
        # Make diagnostic plots
        # -------------------------------------------------------
        l, m = (2, 1)
        plt.figure()
        plt.title("Re h(l=%d,m=%d)" % (l, m))
        plt.plot(t - dtime, np.real(h_2i[l, m]), label="Direct integration")
        plt.plot(t, np.real(h_ff[l, m]), label="Fixed frequency integration")
        plt.xlim(xmin=t.min() + options["window.delta"],
                 xmax=t.max() - options["window.delta"])
        plt.legend(loc='best')
        plt.savefig(self.outdata_path + "Re_l%d_m%d.png" % (l, m))

        plt.figure()
        plt.title("Im h(l=%d,m=%d)" % (l, m))
        plt.plot(t - dtime, np.real(h_2i[l, m]), label="Direct integration")
        plt.plot(t, np.real(h_ff[l, m]), label="Fixed frequency integration")
        plt.xlim(xmin=t.min() + options["window.delta"],
                 xmax=t.max() - options["window.delta"])
        plt.legend(loc='best')
        plt.savefig(self.outdata_path + "Im_l%d_m%d.png" % (l, m))

        l, m = (2, 2)
        plt.figure()
        plt.title("Re h(l=%d,m=%d)" % (l, m))
        plt.plot(t - dtime, np.real(h_2i[l, m]), label="Direct integration")
        plt.plot(t, np.real(h_ff[l, m]), label="Fixed frequency integration")
        plt.xlim(xmin=t.min() + options["window.delta"],
                 xmax=t.max() - options["window.delta"])
        plt.legend(loc='best')
        plt.savefig(self.outdata_path + "Re_l%d_m%d.png" % (l, m))

        plt.figure()
        plt.title("Im h(l=%d,m=%d)" % (l, m))
        plt.plot(t - dtime, np.real(h_2i[l, m]), label="Direct integration")
        plt.plot(t, np.real(h_ff[l, m]), label="Fixed frequency integration")
        plt.xlim(xmin=t.min() + options["window.delta"],
                 xmax=t.max() - options["window.delta"])
        plt.legend(loc='best')
        plt.savefig(self.outdata_path + "Im_l%d_m%d.png" % (l, m))

        plt.figure()
        plt.title("E_GW_dot_l2")
        plt.plot(t, E_GW_dot_all, label='E_GW_dot')
        plt.plot(t, 2 * E_GW_dot_ff[2, 1], label='2 x E_GW_dot(l=2,m=1)')
        plt.plot(t, 2 * E_GW_dot_ff[2, 2], label='2 x E_GW_dot(l=2,m=2)')
        plt.gca().set_yscale("log")
        plt.xlim(xmin=t.min() + options["window.delta"],
                 xmax=t.max() - options["window.delta"])
        plt.ylim(ymin=1e-16)
        plt.legend(loc='best')
        plt.savefig(self.outdata_path + "E_GW_dot_l2.png")

        plt.figure()
        plt.title("E_GW_l2")
        plt.plot(t, E_GW_all, label='E_GW')
        plt.plot(t, 2 * E_GW_ff[2, 1], label='2 x E_GW(l=2,m=1)')
        plt.plot(t, 2 * E_GW_ff[2, 2], label='2 x E_GW(l=2,m=2)')
        plt.gca().set_yscale("log")
        plt.xlim(xmin=t.min() + options["window.delta"],
                 xmax=t.max() - options["window.delta"])
        plt.ylim(ymin=1e-16)
        plt.legend(loc='best')
        plt.savefig(self.outdata_path + "E_GW_l2.png")

        plt.figure()
        plt.title("E_GW_dot")
        plt.plot(t, E_GW_dot_all, label='E_GW_dot')
        for m in dset.mmodes:
            plt.plot(t, 2 * E_GW_dot[m], label='2 x E_GW_dot(m=%d)' % m)
        plt.gca().set_yscale("log")
        plt.legend(loc='best')
        plt.xlim(xmin=t.min() + options["window.delta"],
                 xmax=t.max() - options["window.delta"])
        plt.ylim(ymin=1e-16)
        plt.savefig(self.outdata_path + "E_GW_dot.png")
        plt.figure()

        plt.title("E_GW")
        plt.plot(t, E_GW_all, label='E_GW')
        for m in dset.mmodes:
            plt.plot(t, 2 * E_GW[m], label='2 x E_GW(m=%d)' % m)
        plt.gca().set_yscale("log")
        plt.xlim(xmin=t.min() + options["window.delta"],
                 xmax=t.max() - options["window.delta"])
        plt.ylim(ymin=1e-16)
        plt.legend(loc='best')
        plt.savefig(self.outdata_path + "E_GW.png")

        plt.figure()
        plt.title("J_GW_dot")
        plt.plot(t, J_GW_dot_all, label='J_GW_dot')
        for m in dset.mmodes:
            plt.plot(t, 2 * J_GW_dot[m], label='2 x J_GW_dot(m=%d)' % m)
        plt.gca().set_yscale("log")
        plt.xlim(xmin=t.min() + options["window.delta"],
                 xmax=t.max() - options["window.delta"])
        plt.ylim(ymin=1e-16)
        plt.legend(loc='best')
        plt.savefig(self.outdata_path + "J_GW_dot.png")

        plt.figure()
        plt.title("J_GW")
        plt.plot(t, J_GW_all, label='J_GW')
        for m in dset.mmodes:
            plt.plot(t, 2 * J_GW[m], label='2 x J_GW(m=%d)' % m)
        plt.gca().set_yscale("log")
        plt.xlim(xmin=t.min() + options["window.delta"],
                 xmax=t.max() - options["window.delta"])
        plt.ylim(ymin=1e-16)
        plt.legend(loc='best')
        plt.savefig(self.outdata_path + "J_GW.png")

        # Show plots
        # plt.show()

        # -------------------------------------------------------
        # Output data
        # -------------------------------------------------------
        # Find the merger time from the 22 mode
        tmerger = t[np.argmax(A_ff[2, 2])]
        open(self.outdata_path + "tmerger.dat", "w").write("{}\n".format(tmerger))

        # Output the strain
        for l, m in dset.modes:
            print("\tWriting (%d,%d)..." % (l, m)),

            ofile = open(self.outdata_path + "psi4_l%d_m%d.dat" % (l, m), "w")
            ofile.write("# 1:time 2:Re 3:Im\n")
            for i in range(t.shape[0]):
                ofile.write("{} {} {}\n".format(t[i], Psi4[l, m][i].real,
                                                Psi4[l, m][i].imag))
            ofile.close()

            ofile = open(self.outdata_path + "strain_l%d_m%d.dat" % (l, m), "w")
            ofile.write("# 1:time 2:Re 3:Im 4:phi 5:omega 6:E_dot_GW 7:E_GW 8:J_dot_GW 9:J_GW\n")
            for i in range(t.shape[0]):
                ofile.write("{} {} {} {} {} {} {} {} {}\n".format(t[i],
                                                                  h_ff[l, m][i].real, h_ff[l, m][i].imag,
                                                                  phi_ff[l, m][i],
                                                                  omega_ff[l, m][i], E_GW_dot_ff[l, m][i],
                                                                  E_GW_ff[l, m][i],
                                                                  J_GW_dot_ff[l, m][i], J_GW_ff[l, m][i]))
            ofile.close()

            print("done!")

        outfile = open(self.outdata_path + "EJ.dat", "w")
        outfile.write("# 1:t 2:E_GW_dot 3:E_GW 4:J_GW_dot 5:J_GW\n")
        for i in range(t.shape[0]):
            outfile.write("{} {} {} {} {}\n".format(t[i], E_GW_dot_all[i], E_GW_all[i],
                                                    J_GW_dot_all[i], J_GW_all[i]))
        outfile.close()


def tmerg_tcoll(dens_drop=5., fraction=0.01, indir=None, outdir=None):
    """

    :param sim:
    :param fraction: Fraction of the maximum strain amplitude to be used to set a BH formation.
    :return:
    """


    Printcolor.print_colored_string(["task", "tmerg tcoll", "BH makrs: densistry drop by",
                                     "{}".format(dens_drop), "Strain magnitude drop to ",
                                     "{}".format(fraction), "of the maximum"],
                                    ["blue", "green", "blue", "green", "blue", "green", "blue"])

    c_dir = indir
    w_dir = outdir

    # simdir = Paths.gw170817 + sim + '/'

    dens_fname = "dens.norm1.asc"
    strain_fname = "strain_l2_m2.dat"
    outfile_tmerg = "tmerger2.dat"
    outfile_tcoll = "tcoll.dat"
    outfile_colltime = "colltime.txt"
    outfile_tdens_drop = "tdens_drop.txt"
    plot_name = "tmergtcoll.png"

    assert os.path.isdir(c_dir)
    assert os.path.isdir(w_dir)
    assert os.path.isfile(c_dir + dens_fname)
    assert os.path.isfile(w_dir + strain_fname)

    # --- DENSITY

    _time, dens = np.genfromtxt(c_dir + dens_fname, unpack=True, usecols=[1, 2])
    assert len(_time) == len(dens)
    maxD = np.max(dens)
    _time = np.array(_time)
    dens = np.array(dens)
    # print(np.diff(-1.*np.log(dens)).min(), np.diff(-1.*np.log(dens)).max()); exit(1)

    tmp = np.where(np.diff(-1.*np.log(dens)) > 0.1)

    # tmp = np.where(dens < (maxD / dens_drop))
    # print(tmp); exit(1)
    # print(tmp)
    # exit(1)
    if np.array(tmp, dtype=float).any():
        indx_dens_drop = np.min(tmp)
        tdens_drop = _time[indx_dens_drop]
        Printcolor.blue("\tDensity drops by:{} at time:{} -> BH formation".format(dens_drop, tdens_drop))
    else:
        tdens_drop = np.nan
        Printcolor.blue("\tDensity does not drop by:{} -> BH does not form".format(dens_drop))

    # --- GW ---
    strain_file = w_dir + strain_fname
    time, reh, imh = np.genfromtxt(strain_file, unpack=True, usecols=[0, 1, 2])
    h = reh + 1j * imh
    amp = np.abs(h)
    tmerg = time[np.argmax(amp)]

    # if BH formation is found by the density drop:
    #if True:#not np.isnan(tdens_drop):

    maxA = np.max(amp)
    indx_collapse = np.min(np.where((amp < fraction * maxA) & (time >= tmerg)))
    tcoll = time[indx_collapse]

    # in case, there is an increase of Amp after tcoll
    __time = time[time>tcoll]
    __amp = amp[time>tcoll]
    # if __amp.max() > fraction * maxA:
    #     Printcolor.yellow("Warning. max Amp[(t > tcoll)] > {} maxA".format(fraction))
    #     Printcolor.yellow("Searching for the t[minA] post tcoll...")
    #     ind_pm_max = np.min(np.where(time >= __time[np.min(np.where(__amp >= __amp.max()))]))
    #     indx_collapse2 = np.min(np.where((amp < fraction * maxA) & (time > time[ind_pm_max])))
    #     tcoll2 = time[indx_collapse2]
    #     if tcoll2 > tcoll and tcoll2 < time[-1] and tcoll2 < 0.9 * time[-1]:
    #         Printcolor.red("Found tcoll2 > tcoll. Replacing tcoll...")
    #         tcoll = tcoll2
    #     elif tcoll2 == tcoll:
    #         Printcolor.red("Found tcoll2 == tcoll. Method failed. Using wrong tcoll")
    #         tcoll = tcoll2
    #     elif tcoll2 > tcoll and tcoll2 == time[-1]:
    #         Printcolor.red("Found tcoll2 == t[-1]. Method failed. Using End ofo sim. for tcoll")
    #         tcoll = tcoll2
    #     elif tcoll2 > (0.90 * time[-1]):
    #         Printcolor.red("Found tcoll2 > 0.95*t[-1]. Method failed. Using 1st tcoll")
    #         pass
    #     else:
    #         raise ValueError("no cases set")


    if tcoll > 0.95 * time[-1]:
        Printcolor.red("Warning: tcoll:{:.1f} > 90% of tend:{:.1f}".format(tcoll * Constants.time_constant,
                                                                          time[-1] * Constants.time_constant))
    else:
        pass

    if tcoll > 0.95 * time[-1] and np.isnan(tdens_drop):
        Printcolor.red("Warning: tcoll:> 0.9*tend & tdensdrop == np.nan -> setting tcoll = np.nan ")
        tcoll = np.nan

    tend = time[-1]
    # printing results
    # print("\ttdensdrop: {:.4f}".format(tdens_drop))
    # print("\ttmerg:     {:.4f}".format(tmerg))
    # print("\ttcoll:     {:.4f}".format(tcoll))
    # print("\ttend:      {:.4f}".format(time[-1]))

    Printcolor.print_colored_string(["\ttmerg (GW)", "{:.1f}".format(tmerg), "[geo]",
                                     "{:.1f}".format(tmerg*Constants.time_constant), "[ms]"],
                                    ["blue", "green", "blue", "green", "blue"])
    Printcolor.print_colored_string(["\tdensdrop  ", "{:.1f}".format(tdens_drop), "[geo]",
                                     "{:.1f}".format(tdens_drop*Constants.time_constant), "[ms]"],
                                    ["blue", "green", "blue", "green", "blue"])
    Printcolor.print_colored_string(["\ttcoll (GW)", "{:.1f}".format(tcoll), "[geo]",
                                     "{:.1f}".format(tcoll*Constants.time_constant), "[ms]"],
                                    ["blue", "green", "blue", "green", "blue"])
    Printcolor.print_colored_string(["\ttend      ", "{:.1f}".format(tend), "[geo]",
                                     "{:.1f}".format(tend*Constants.time_constant), "[ms]"],
                                    ["blue", "green", "blue", "green", "blue"])
    if np.abs(tdens_drop-tcoll) > 0.01*tend:
        Printcolor.red("Warning: Time of Dens. drop != tcoll (GW) by {:.2f} [ms]"
                       .format(np.abs(tdens_drop-tcoll)*Constants.time_constant))

    # saving files
    Printcolor.blue("\tSaving tmerg:   {}".format(w_dir + outfile_tmerg))
    open(w_dir + outfile_tmerg, "w").write("{}\n".format(float(tmerg)))
    if not np.isnan(tdens_drop):
        Printcolor.blue("\tSaving tdens_drop:   {}".format(w_dir + outfile_tdens_drop))
        open(w_dir + outfile_tdens_drop, "w").write("{}\n".format(float(tdens_drop)))
    if not np.isnan(tcoll):
        Printcolor.blue("\tSaving t collapse: {}".format(w_dir + outfile_tcoll))
        open(w_dir + outfile_tcoll, "w").write("{}\n".format(float(tcoll)))
        # if os.path.isdir(simdir):
        #     Printcolor.blue("\tSaving {} (for outflowed.cc)".format(simdir + outfile_colltime))
        #     open(simdir + outfile_colltime, "w").write("{}\n".format(float(tcoll)))
        # else:
        #     Printcolor.yellow("\t{} is not saved. Dir: {} is not found".format(outfile_colltime, simdir))

    Printcolor.blue("\tsaving summory plot: {}".format(w_dir + plot_name))
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_ylabel(r'strain [Re]', fontsize=12)
    ax1.set_xlabel(r'time $[M_{\odot}]$', fontsize=12)
    ax1.plot(time, reh, c='k')
    ax1.plot(time, amp, c='red', label="amplitude")
    ax1.axvline(tmerg, ls='-.', c="orange", label="tmerg")
    if not np.isnan(tcoll):
        ax1.axvline(tcoll, ls='-.', c=color, label="tcoll")
    # ax1.tick_params(axis='y', labelcolor=color, fontsize=12)
    ax1.tick_params(
        axis='y', labelleft=True, labelcolor=color,
        # labelright=False, tick1On=True, tick2On=True,
        labelsize=12,
        direction='in',
        # bottom=True, top=True, left=True, right=True
    )
    ax1.minorticks_on()

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel("log(" + dens_fname + ")", color=color)
    ax2.plot(_time, np.log10(dens), color=color)
    if not np.isnan(tdens_drop):
        ax1.axvline(tdens_drop, ls='-.', c=color, label="tdens_drop")
    # ax2.tick_params(axis='y', labelcolor=color, fontsize=12)
    ax2.tick_params(
        axis='y', labelright=True, labelcolor=color,
        # labelright=False, tick1On=True, tick2On=True,
        labelsize=12,
        direction='in',
        # bottom=True, top=True, left=True, right=True
    )
    ax2.minorticks_on()

    # plt.minorticks_on()
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.title('GW analysis', fontsize=20)
    plt.legend(loc='upper right')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(w_dir + plot_name, bbox_inches='tight', dpi=128)
    plt.close()

    Printcolor.yellow("Please note, that {} and {} might be inacurate. Check the plot."
                      .format(outfile_tmerg, outfile_tcoll))