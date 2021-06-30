"""
    pass
"""

from argparse import ArgumentParser
import numpy as np
import os

# from uutils import Paths
import config as Paths

from strain import STRAIN, tmerg_tcoll

__tasklist__ = ['strain', "tmergtcoll"]

def do_stasks():
    for task in glob_tasklist:
        print(task)
        if task == "strain":
            strain = STRAIN(indir=glob_indir, outdir=glob_outdir)
            strain.main()
        if task == "tmergtcoll":
            tmerg_tcoll(indir=glob_indir, outdir=glob_outdir)

if __name__ == '__main__':
    parser = ArgumentParser(description="postprocessing pipeline")
    parser.add_argument("-s", dest="sim", required=False, default='', help="task to perform")
    parser.add_argument("-t", dest="tasklist", required=False, nargs='+', default=[], help="tasks to perform")
    #
    # parser.add_argument("-o", dest="outdir", required=False, default=None, help="path for output dir")
    parser.add_argument("-i", dest="indir", required=False, default=None, help="path to collated data for sim")
    parser.add_argument("--overwrite", dest="overwrite", required=False, default="no", help="overwrite if exists")

    #
    args = parser.parse_args()
    #
    glob_tasklist = args.tasklist
    glob_sim = args.sim
    glob_indir = args.indir
    # glob_outdir = args.outdir
    glob_overwrite = args.overwrite
    # simdir = Paths.gw170817 + glob_sim + '/'
    # resdir = Paths.ppr_sims + glob_sim + '/'
    #
    if glob_overwrite == "no":  glob_overwrite = False
    elif glob_overwrite == "yes": glob_overwrite = True
    #
    if glob_indir is None:
        glob_indir = Paths.default_ppr_dir + glob_sim + '/'
        if not os.path.isdir(glob_indir):
            raise IOError("Input directory not found {}".format(glob_indir))
        if not os.path.isdir(glob_indir+'collated/'):
            raise IOError("Input directory with collated data not found : {}\n"
                          "Run old_preanalysis.py with '-t collate' ".format(glob_indir+'collated/'))
    # if glob_outdir is None:
    #     glob_outdir = Paths.ppr_sims + glob_sim + '/'
    #     if not os.path.isdir(glob_outdir):
    #         raise IOError("Output directory : {} does not exists".format(glob_outdir))

    # if not os.path.isdir(glob_outdir + '/collated/'):
    #     raise IOError(
    #         "Collated data not found. {} \n Run old_preanalysis.py with '-t collate' ".format(glob_outdir + '/collated/'))

    if not os.path.isdir(glob_indir+'waveforms/'):
        os.mkdir(glob_indir+'waveforms/')

    #
    glob_outdir = glob_indir + "waveforms/"
    glob_indir = glob_indir + "collated/"



    #
    # if glob_sim != '' and glob_simdir == Paths.gw170817:
    #     assert os.path.isdir(Paths.ppr_sims)
    #     if not os.path.isdir(Paths.ppr_sims+glob_sim+'/collated/'):
    #         raise IOError("Dir not found. {} \n Run old_preanalysis.py with -t collate "
    #                       .format(Paths.ppr_sims+glob_sim+'/collated/'))
    #     glob_indir = Paths.ppr_sims+glob_sim+'/collated/'
    # elif glob_sim == '' and glob_simdir != Paths.gw170817:
    #     if not os.path.isdir(glob_simdir): raise IOError("simdir does not exist: {}".format(glob_simdir))
    #     assert os.path.isdir(glob_simdir)
    #     glob_indir = glob_simdir
    # else:
    #     raise IOError("Please either provide -s option for simulation, assuming the collated data would be in "
    #                   "{} or provide -i path to inside of the collated data for your simulation. "
    #                   .format(Paths.ppr_sims+'this_simulation/collated/'))
    #
    # if glob_outdir == Paths.ppr_sims:
    #     assert glob_sim != ''
    #     glob_outdir = Paths.ppr_sims+glob_sim+'/waveforms/'
    #     if not os.path.isdir(glob_outdir):
    #         os.mkdir(glob_outdir)
    # else:
    #     # assert glob_sim != ''
    #     glob_outdir = glob_outdir
    #     if not os.path.isdir(glob_outdir):
    #         os.mkdir(glob_outdir)

    assert os.path.isdir(glob_outdir)
    #
    if len(glob_tasklist) == 1 and "all" in glob_tasklist:
        glob_tasklist = __tasklist__
    elif len(glob_tasklist) == 1 and not "all" in glob_tasklist:
        for task in glob_tasklist:
            if not task in __tasklist__:
                raise NameError("task: {} is not recognized. Available:{}"
                                .format(task, __tasklist__))
    elif len(glob_tasklist) > 1 and not "all" in glob_tasklist:
        pass
    else:
        raise IOError("Please provide task to perform in -t option. ")
    #
    # print(glob_indir)
    do_stasks()