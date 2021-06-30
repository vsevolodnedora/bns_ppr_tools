"""
    top level
"""

from argparse import ArgumentParser
import os
import re
import numpy as np

from module_preanalysis.it_time import LOAD_ITTIME, SIM_STATUS, PRINT_SIM_STATUS
from module_preanalysis.init_data import INIT_DATA, LOAD_INIT_DATA
from module_preanalysis.collate import COLLATE_DATA

from uutils import Printcolor
import config as Paths

TASKLIST = ["update_status", "collate", "print_status", "init_data"]

def do_tasks():
    # do tasks
    for task in glob_tasklist:
        if task == "update_status":
            Printcolor.blue("Task:'{}' Executing...".format(task))
            statis = SIM_STATUS(glob_sim, indir=glob_indir, outdir=glob_outdir)
            Printcolor.blue("Task:'{}' DONE...".format(task))

        elif task == "collate":
            COLLATE_DATA(glob_sim, indir=glob_indir, pprdir=glob_outdir,
                         usemaxtime=glob_usemaxtime, maxtime=glob_usemaxtime, overwrite=glob_overwrite)

        elif task == "print_status":
            Printcolor.blue("Task:'{}' Executing...".format(task))
            statis = PRINT_SIM_STATUS(glob_sim, indir=glob_indir, pprdir=glob_outdir)
            Printcolor.blue("Task:'{}' DONE...".format(task))

        elif task == "init_data":
            Printcolor.blue("Task:'{}' Executing...".format(task))
            statis = INIT_DATA(glob_sim, indir=glob_indir, outdir=glob_outdir)#, lor_archive_fpath=glob_lorene)
            Printcolor.blue("Task:'{}' DONE...".format(task))

        else:
            raise NameError("No method fund for task: {}".format(task))
    print("done")

if __name__ == '__main__':

    # o_ittime = LOAD_ITTIME("BLh_M12591482_M0_LR")
    # _, it, t = o_ittime.get_ittime("overall", "d2")

    # print(it[])

    # LOAD_ITTIME("SFHo_M14521283_M0_LR")




    # exit(1)

    parser = ArgumentParser(description="postprocessing pipeline")
    parser.add_argument("-s", dest="sim", required=True, help="name of the simulation dir")
    parser.add_argument("-t", dest="tasklist", nargs='+', required=False, default=[], help="list of tasks to to")
    #
    parser.add_argument("-o", dest="outdir", required=False, default=None, help="path for output dir")
    parser.add_argument("-i", dest="indir", required=False, default=None, help="path to simulation dir")
    parser.add_argument("--overwrite", dest="overwrite", required=False, default="no", help="overwrite if exists")
    parser.add_argument("--usemaxtime", dest="usemaxtime", required=False, default="no",
                        help=" auto/no to limit data using ittime.h5 or float to overwrite ittime value")
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
    glob_indir = args.indir
    glob_outdir = args.outdir
    glob_tasklist = args.tasklist
    glob_overwrite = args.overwrite
    glob_usemaxtime = args.usemaxtime
    glob_maxtime = np.nan
    # glob_lorene = args.lorene
    glob_tov = args.tov
    # check given data

    if (glob_indir is None):
        glob_indir = Paths.default_data_dir + glob_sim + '/'
        if not os.path.isdir(glob_indir):
            raise NameError("Default simulation data dir: {} does not exist in rootpath: {} ".format(glob_sim, glob_indir))
    if not os.path.isdir(glob_indir):
        raise NameError("simulation dir: {} does not exist in rootpath: {} ".format(glob_sim, glob_indir))
    if len(glob_tasklist) == 0:
        raise NameError("tasklist is empty. Set what tasks to perform with '-t' option")
    else:
        for task in glob_tasklist:
            if task not in TASKLIST:
                raise NameError("task: {} is not among available ones: {}"
                                .format(task, TASKLIST))
    if glob_overwrite == "no":  glob_overwrite = False
    elif glob_overwrite == "yes": glob_overwrite = True
    else: raise NameError("Option '--overwrite' can be 'yes' or 'no' only. Given:{}".format(glob_overwrite))
    #
    if glob_usemaxtime == "no":
        glob_usemaxtime = False
        glob_maxtime = np.nan
    elif glob_usemaxtime == "auto":
        glob_usemaxtime = True
        glob_maxtime = np.nan
    elif re.match(r'^-?\d+(?:\.\d+)?$', glob_usemaxtime):
        glob_maxtime = float(glob_usemaxtime)
        glob_usemaxtime = True
    else: raise NameError("for '--usemaxtime' option use 'yes' or 'no' or float. Given: {}"
                          .format(glob_usemaxtime))

    if (glob_outdir is None):
        glob_outdir = Paths.default_ppr_dir + glob_sim + '/'
        if not os.path.isdir(glob_outdir):
            raise NameError("Default putput data dir: {} does not exist in rootpath: {} ".format(glob_sim, glob_outdir))
    glob_outdir_sim = glob_outdir
    # if not os.path.isdir(glob_outdir_sim):
    #     os.mkdir(glob_outdir_sim)
    # if glob_lorene != None:
    #     if not os.path.isfile(glob_lorene):
    #         raise NameError("Given lorene fpath: {} is not avialable"
    #                         .format(glob_lorene))
    if glob_tov != None:
        if not os.path.isfile(glob_tov):
            raise NameError("Given TOV fpath: {} is not avialable".format(glob_tov))

    # set globals
    # Paths.gw170817 = glob_indir
    # Paths.ppr_sims = glob_outdir

    print("\t sim:    {}".format(glob_sim))
    print("\t indir:  {}".format(glob_indir))
    print("\t outdit: {}".format(glob_outdir))

    do_tasks()