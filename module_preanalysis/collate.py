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
from uutils import Printcolor, Lists, Constants, Tools


from it_time import LOAD_ITTIME

# collate ascii files
class COLLATE_DATA(LOAD_ITTIME):

    def __init__(self, sim, indir, pprdir, usemaxtime=False, maxtime=np.nan, overwrite=False):

        LOAD_ITTIME.__init__(self, sim, pprdir=pprdir)

        self.all_fnames = Lists.collate_list
        self.all_outputs = self.get_list_outputs()
        # print(self.all_outputs); exit(1)
        self.outdir = pprdir+'/collated/'
        self.indir = indir

        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)

        self.tmax = inf         # Maximum time to include (default: inf)
        self.epsilon = 1e-15    # Precision used in comparing timestamps
        self.tidx = 1           # Index of the time column, from 1 (default: 1)
        #
        if usemaxtime:
            if np.isnan(maxtime):
                if not np.isnan(self.maxtime):
                    self.tmax = self.maxtime / (Constants.time_constant * 1.e-3) # [s] -> GEO
            else:
                self.tmax = maxtime / (Constants.time_constant) # [ms] -> GEO
        print("Maximum time is set: {}".format(self.tmax))
        #
        self.collate(overwrite)

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

                tidx = Lists.time_index[fpath.split('/')[-1]]
                tnew = float(dline.split()[tidx - 1])
                if tnew > self.tmax:
                    #print("tnew: {}    tmax: {}".format(tnew, self.tmax))
                    break
                if told is None or tnew > told * (1 + self.epsilon):
                    ofile.write(dline)
                    told = tnew

        ofile.close()

    def collate(self, rewrite=False):
        for fname in self.all_fnames:
            output_files = []
            for output in self.all_outputs:
                fpath = self.indir+output+'/data/'+fname
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