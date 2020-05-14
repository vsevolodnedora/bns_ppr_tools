import multiprocessing as mp
from functools import partial
import h5py
import re
from plotting_methods import PLOT_MANY_TASKS
from outflowed import EJECTA_PARS
from utils import *
from math import sqrt
from outflowed import SphericalSurface
from outflowed import EOSTable
import scivis.units as ut
from profile import LOAD_PROFILE_XYXZ, MAINMETHODS_STORE

class LOAD_RESHAPE_SAVE_PARALLEL:

    def __init__(self, sim="LS220_M14691268_M0_LK_SR", det=0, n_proc = 4):

        self.det = det
        self.sim = sim
        n_procs = n_proc
        #
        self.v_ns = ['it', 'time', "fluxdens", "w_lorentz", "eninf", "surface_element",
                "alp", "rho", "vel[0]", "vel[1]", "vel[2]", "Y_e", "entropy", "temperature"]
        self.eos_v_ns = ['eps', 'press']
        #
        self.eos_fpath = Paths.get_eos_fname_from_curr_dir(self.sim)
        #
        self.outdirtmp = Paths.ppr_sims+sim+'/tmp/'
        if not os.path.isdir(self.outdirtmp):
            os.mkdir(self.outdirtmp)
        #
        fname = "outflow_surface_det_%d_fluxdens.asc" % det
        self.flist = glob(Paths.gw170817 + sim + "/" + "output-????" + "/data/" + fname)
        assert len(self.flist) > 0
        #
        self.grid = self.get_grid()
        #
        print("Pool procs = %d" % n_procs)
        pool = mp.Pool(processes=int(n_procs))
        task = partial(serial_load_reshape_save, grid_object=self.grid, outdir=self.outdirtmp)
        result_list = pool.map(task, self.flist)
        #
        tmp_flist = [Paths.ppr_sims + sim + '/tmp/' + outfile.split('/')[-3] + ".h5" for outfile in self.flist]
        tmp_flist = sorted(tmp_flist)
        assert len(tmp_flist) == len(self.flist)
        # load reshaped data
        iterations, times, data_matrix = self.load_tmp_files(tmp_flist)
        # concatenate data into [ntimes, ntheta, nphi] arrays
        self.iterations = np.sort(iterations)
        self.times = np.sort(times)
        concatenated_data = {}
        for v_n in self.v_ns:
            concatenated_data[v_n] = np.stack(([data_matrix[it][v_n] for it in sorted(data_matrix.keys())]))
        # compute EOS quantities
        concatenated_data = self.add_eos_quantities(concatenated_data)
        # save data
        outfname = Paths.ppr_sims + sim + '/' + fname.replace(".asc", ".h5")
        self.save_result(outfname, concatenated_data)
        print("Done. {} is saved".format(outfname))

    def get_grid(self):

        dfile = open(self.flist[0], "r")
        dfile.readline()  # move down one line
        match = re.match('# detector no.=(\d+) ntheta=(\d+) nphi=(\d+)$', dfile.readline())
        assert int(self.det) == int(match.group(1))
        ntheta = int(match.group(2))
        nphi = int(match.group(3))
        dfile.readline()
        dfile.readline()
        line = dfile.readline().split()
        radius = round(sqrt(float(line[2]) ** 2 + float(line[3]) ** 2 + float(line[4]) ** 2))
        # if not self.clean:
        print("\t\tradius = {}".format(radius))
        print("\t\tntheta = {}".format(ntheta))
        print("\t\tnphi   = {}".format(nphi))
        del dfile

        grid = SphericalSurface(ntheta, nphi, radius)
        return grid

    def load_tmp_files(self, tmp_flist):

        iterations = []
        times = []
        data_matrix = {}
        for ifile, fpath in enumerate(tmp_flist):
            assert os.path.isfile(fpath)
            dfile = h5py.File(fpath, "r")
            for v_n in dfile:
                match = re.match('iteration=(\d+)$', v_n)
                it = int(match.group(1))
                if not it in iterations:
                    i_data_matrix = {}
                    for var_name in self.v_ns:
                        data = np.array(dfile[v_n][var_name])
                        i_data_matrix[var_name] = data
                    data_matrix[it] = i_data_matrix
                    times.append(float(i_data_matrix["time"][0, 0]))
                    iterations.append(int(match.group(1)))
                    print(it, fpath)
                else:
                    pass
            dfile.close()
        return iterations, times, data_matrix

    def add_eos_quantities(self, concatenated_data):

        o_eos = EOSTable()
        o_eos.read_table(self.eos_fpath)
        v_n_to_eos_dic = {
            'eps': "internalEnergy",
            'press': "pressure",
            'entropy': "entropy"
        }
        for v_n in self.eos_v_ns:
            print("Evaluating eos: {}".format(v_n))
            data_arr = o_eos.evaluate(v_n_to_eos_dic[v_n], concatenated_data["rho"],
                                      concatenated_data["temperature"],
                                      concatenated_data["Y_e"])
            if v_n == 'eps':
                data_arr = ut.conv_spec_energy(ut.cgs, ut.cactus, data_arr)
            elif v_n == 'press':
                data_arr = ut.conv_press(ut.cgs, ut.cactus, data_arr)
            elif v_n == 'entropy':
                data_arr = data_arr
            else:
                raise NameError("EOS quantity: {}".format(v_n))

            concatenated_data[v_n] = data_arr
        return concatenated_data

    def save_result(self, outfpath, concatenated_data):
        if os.path.isfile(outfpath):
            os.remove(outfpath)

        outfile = h5py.File(outfpath, "w")

        outfile.create_dataset("iterations", data=np.array(self.iterations, dtype=int))
        outfile.create_dataset("times", data=self.times, dtype=np.float32)

        outfile.attrs.create("ntheta", self.grid.ntheta)
        outfile.attrs.create("nphi", self.grid.nphi)
        outfile.attrs.create("radius", self.grid.radius)
        outfile.attrs.create("dphi", 2 * np.pi / self.grid.nphi)
        outfile.attrs.create("dtheta", np.pi / self.grid.ntheta)

        outfile.create_dataset("area", data=self.grid.area(), dtype=np.float32)
        theta, phi = self.grid.mesh()
        outfile.create_dataset("phi", data=phi, dtype=np.float32)
        outfile.create_dataset("theta", data=theta, dtype=np.float32)

        self.v_ns.remove("it")
        self.v_ns.remove("time")

        for v_n in self.v_ns + self.eos_v_ns:
            outfile.create_dataset(v_n, data=concatenated_data[v_n], dtype=np.float32)
        outfile.close()

def serial_load_reshape_save(outflow_ascii_file, outdir, grid_object):
    v_n_to_file_dic = {
        'it': 0,
        'time': 1,
        'fluxdens': 5,
        'w_lorentz': 6,
        'eninf': 7,
        'surface_element': 8,
        'alp': 9,
        'rho': 10,
        'vel[0]': 11,
        'vel[1]': 12,
        'vel[2]': 13,
        'Y_e': 14,
        'entropy': 15,
        'temperature': 16
    }
    data_matrix = {}
    # load ascii
    fdata = np.loadtxt(outflow_ascii_file, usecols=v_n_to_file_dic.values(), unpack=True)
    for i_v_n, v_n in enumerate(v_n_to_file_dic.keys()):
        data = np.array(fdata[i_v_n])
        data_matrix[v_n] = np.array(data)
    iterations = np.sort(np.unique(data_matrix["it"]))
    reshaped_data_matrix = [{} for i in range(len(iterations))]
    # extract the data and reshape to [ntheta, nphi] grid for every iteration
    for i_it, it in enumerate(iterations):
        for i_v_n, v_n in enumerate(v_n_to_file_dic.keys()):
            raw_data = np.array(data_matrix[v_n])
            raw_iterations = np.array(data_matrix["it"], dtype=int)
            tmp = raw_data[np.array(raw_iterations, dtype=int) == int(it)][:grid_object.size()]
            assert len(tmp) > 0
            reshaped_data = grid_object.reshape(tmp)
            reshaped_data_matrix[i_it][v_n] = reshaped_data
    # saving data
    fname = outflow_ascii_file.split('/')[-3]  # output-xxxx
    if os.path.isfile(outdir + fname + ".h5"):
        os.remove(outdir + fname + ".h5")
    dfile = h5py.File(outdir + fname + ".h5", "w")
    for i_it, it in enumerate(iterations):
        gname = "iteration=%d" % it
        dfile.create_group(gname)
        for i_v_n, v_n in enumerate(v_n_to_file_dic.keys()):
            data = reshaped_data_matrix[i_it][v_n]
            dfile[gname].create_dataset(v_n, data=data, dtype=np.float32)
    dfile.close()
    print("Done: {}".format(fname))

class LOAD_EXTRACT_RESHAPE_SAVE:

    def __init__(self, outflow_ascii_file, outdir, grid_object):

        self.infile = outflow_ascii_file
        self.o_grid = grid_object
        assert os.path.isfile(self.infile)

        v_ns = ['it', 'time', "fluxdens", "w_lorentz", "eninf", "surface_element",
               "alp", "rho", "vel[0]", "vel[1]", "vel[2]", "Y_e", "entropy", "temperature"]

        self.v_n_to_file_dic = {
            'it': 0,
            'time': 1,
            'fluxdens': 5,
            'w_lorentz': 6,
            'eninf': 7,
            'surface_element': 8,
            'alp': 9,
            'rho': 10,
            'vel[0]': 11,
            'vel[1]': 12,
            'vel[2]': 13,
            'Y_e': 14,
            'entropy': 15,
            'temperature': 16
        }

        self.data_matrix = {}

        # loading data
        fdata = np.loadtxt(outflow_ascii_file, usecols=self.v_n_to_file_dic.values(), unpack=True)  # dtype=np.float64
        for i_v_n, v_n in enumerate(self.v_n_to_file_dic.keys()):
            data = np.array(fdata[i_v_n])
            self.data_matrix[v_n] = np.array(data)
        #
        iterations = np.sort(np.unique(self.data_matrix["it"]))
        self.reshaped_data_matrix = [{} for i in range(len(iterations))]
        #
        for i_it, it in enumerate(iterations):
            for i_v_n, v_n in enumerate(self.v_n_to_file_dic.keys()):
                raw_data = np.array(self.data_matrix[v_n])
                raw_iterations = np.array(self.data_matrix["it"], dtype=int)
                tmp = raw_data[np.array(raw_iterations, dtype=int) == int(it)][:self.o_grid.size()]
                assert len(tmp) > 0

                reshaped_data = self.o_grid.reshape(tmp)
                self.reshaped_data_matrix[i_it][v_n] = reshaped_data
                # self.reshaped_data_matrix[v_n][i_it] = reshaped_data

        # saving data
        fname = outflow_ascii_file.split('/')[-3] # output-xxxx
        if os.path.isfile(outdir + fname + ".h5"):
            os.remove(outdir + fname + ".h5")
        dfile = h5py.File(outdir + fname + ".h5", "w")
        for i_it, it in enumerate(iterations):
            gname = "iteration=%d" % it
            dfile.create_group(gname)
            for i_v_n, v_n in enumerate(self.v_n_to_file_dic.keys()):
                data = self.reshaped_data_matrix[i_it][v_n]
                # data = self.reshaped_data_matrix[v_n][i_it]
                # print(v_n, np.sum(data))
                dfile[gname].create_dataset(v_n, data=data, dtype=np.float32)
        dfile.close()
        print("Done: {}".format(fname))

def plot_corr(table, v_n1, v_n2, plotfpath):
    # table = o_outflow.get_ejecta_arr(det, mask, "corr2d {} {}".format(v_n1, v_n2))
    # y_arr = table[1:, 0]
    # x_arr = table[0, 1:]
    # z_arr = table[1:, 1:]
    # dfile = h5py.File(fpath, "w")
    # dfile.create_dataset(v_n1, data=x_arr)
    # dfile.create_dataset(v_n2, data=y_arr)
    # dfile.create_dataset("mass", data=z_arr)
    # dfile.close()
    # print(x_arr)
    # exit(1)

    # np.savetxt(outdir + "/hist_{}.dat".format(v_n), X=hist)
    #
    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = plotfpath
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
        'task': 'corr2d', 'dtype': 'table', 'ptype': 'cartesian',
        'data': table,
        'position': (1, 1),
        'v_n_x': v_n1, 'v_n_y': v_n2, 'v_n': 'mass', 'normalize': True,
        'cbar': {
            'location': 'right .03 .0', 'label': Labels.labels("mass"),  # 'fmt': '%.1f',
            'labelsize': 14, 'fontsize': 14},
        'cmap': 'inferno',
        'xlabel': Labels.labels(v_n1), 'ylabel': Labels.labels(v_n2),
        'xmin': 0, 'xmax': 110, 'ymin': 0, 'ymax': 90, 'vmin': 1e-4, 'vmax': 1e-1,
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
    # corr_dic2 = Limits.in_dic(corr_dic2)
    o_plot.set_plot_dics.append(corr_dic2)
    #

    o_plot.main()

def add_q_r_t_to_prof_xyxz():
    glob_sim = "LS220_M14691268_M0_LK_SR"
    glob_profxyxz_path = Paths.ppr_sims+glob_sim+'/profiles/'
    glob_nlevels = 7
    glob_overwrite = False

    from preanalysis import LOAD_ITTIME
    ititme = LOAD_ITTIME(glob_sim)
    _, profit, proft = ititme.get_ittime("profiles", d1d2d3prof="prof")

    if len(profit) == 0:
        Printcolor.yellow("No profiles found. Q R T values are not added to prof.xy.h5")
        return 0

    from slices import COMPUTE_STORE, LOAD_STORE_DATASETS
    from profile import LOAD_PROFILE_XYXZ
    # locate prof.xy and prof xz:
    d2data = COMPUTE_STORE(glob_sim)
    # d3data = LOAD_PROFILE_XYXZ(glob_sim)
    #
    # nu_arr = d2data.get_data(1425408, "xz", "Q_eff_nua")[3]
    # hydro_arr = d3data.get_data(1425408, 3, "xz", "rho")
    # print(nu_arr.shape, hydro_arr.shape)
    # exit(1)

    for it in profit:
        for plane in ["xy", "xz"]:
            fpath = glob_profxyxz_path + str(int(it)) + '/' + "profile.{}.h5".format(plane)
            if os.path.isfile(fpath):
                try:
                    dfile = h5py.File(glob_profxyxz_path + str(int(it)) + '/' + "profile.{}.h5".format(plane), "a")

                    Printcolor.print_colored_string(
                        ["task:", "adding neutrino data to prof. slice", "it:", "{}".format(it), ':', "Adding"],
                        ["blue", "green",                                "blue", "green",        "", "green"]
                    )
                    for rl in range(glob_nlevels):
                        gname = "reflevel=%d" % rl
                        for v_n in LOAD_STORE_DATASETS.list_neut_v_ns:
                            if (v_n in dfile[gname] and glob_overwrite) or not v_n in dfile[gname]:
                                nu_arr = d2data.get_data(it, plane, v_n)[rl]
                                # hydro_arr = d3data.get_data(it, rl, plane, "rho")
                                # assert nu_arr.shape == hydro_arr.shape
                                gname = "reflevel=%d" % rl
                                dfile[gname].create_dataset(v_n, data=np.array(nu_arr, dtype=np.float32))
                            else:
                                pass
                    dfile.close()
                except KeyboardInterrupt:
                    exit(1)
                except ValueError:
                    Printcolor.print_colored_string(
                        ["task:", "adding neutrino data to prof. slice", "it:", "{}".format(it), ':', "ValueError"],
                        ["blue", "green", "blue", "green", "", "red"]
                    )
                except IOError:
                    Printcolor.print_colored_string(
                        ["task:", "adding neutrino data to prof. slice", "it:", "{}".format(it), ':', "IOError"],
                        ["blue", "green", "blue", "green", "", "red"]
                    )
                except:
                    pass
            else:
                Printcolor.print_colored_string(
                    ["task:", "adding neutrino data to prof. slice", "it:", "{}".format(it), ':', "IOError: profile.{}.h5 does not exist".format(plane)],
                    ["blue", "green", "blue", "green", "", "red"]
                )
    # for it in profit:
    #     #
    #     fpathxy = glob_profxyxz_path + str(int(it)) + '/' + "profile.xy.h5"
    #     fpathxz = glob_profxyxz_path + str(int(it)) + '/' + "profile.xz.h5"

''' ------------------------- others ------------------------------- '''

def comparing_mkn_codes():

    from mkn_interface import COMBINE_LIGHTCURVES

    sims = [None, None]
    dirs = ["/data01/numrel/vsevolod.nedora/figs/mkn_test/old_code/",
            "/data01/numrel/vsevolod.nedora/figs/mkn_test/new_code/"]
    colors = ["red", "blue"]
    model_fnames = ["mkn_model.h5", "mkn_model.h5"]
    #
    glob_bands = ["g", "z", "Ks"]
    #
    o_plot = PLOT_MANY_TASKS()
    #
    figname = ''
    for band in glob_bands:
        figname = figname + band
        if band != glob_bands[-1]:
            figname = figname + '_'
    figname = figname + '.png'
    #
    figpath = "/data01/numrel/vsevolod.nedora/figs/mkn_test/"
    #
    o_plot.gen_set["figdir"] = figpath
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (len(glob_bands) * 3.0, 4.3)  # <->, |] # to match hists with (8.5, 2.7)
    o_plot.gen_set["figname"] = figname
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = False
    o_plot.gen_set["subplots_adjust_h"] = 0.3
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []
    fontsize = 14
    labelsize = 14
    #
    for sim, dir, color, model_fname in zip(sims, dirs, colors, model_fnames):

        o_res = COMBINE_LIGHTCURVES(sim, dir)
        #
        for i_plot, band in enumerate(glob_bands):
            i_plot = i_plot + 1

            times, mags = o_res.get_model_median(band, model_fname)

            model = {
                'task': 'line', "ptype": "cartesian",
                'position': (1, i_plot),
                'xarr': times, 'yarr': mags,
                'v_n_x': 'time', 'v_n_y': 'mag',
                'color': color, 'ls': '-', 'lw': 1., 'ds': 'default', 'alpha': 1.,
                'ymin': 25, 'ymax': 15, 'xmin': 3e-1, 'xmax': 3e1,
                'xlabel': r"time [days]", 'ylabel': r"AB magnitude at 40 Mpc",
                'label': None, 'xscale': 'log',
                'fancyticks': True, 'minorticks': True,
                'sharey': False,
                'fontsize': fontsize,
                'labelsize': labelsize,
                'legend': {}  # {'loc': 'best', 'ncol': 2, 'fontsize': 18}
            }

            obs = {
                'task': 'mkn obs', "ptype": "cartesian",
                'position': (1, i_plot),
                'data': o_res, 'band': band, 'obs': True,
                'v_n_x': 'time', 'v_n_y': 'mag',
                'color': 'gray', 'marker': 'o', 'ms': 5., 'alpha': 0.8,
                'ymin': 25, 'ymax': 15, 'xmin': 3e-1, 'xmax': 3e1,
                'xlabel': r"time [days]", 'ylabel': r"AB magnitude at 40 Mpc",
                'label': "AT2017gfo", 'xscale': 'log',
                'fancyticks': True, 'minorticks': True,
                'title': {'text': '{} band'.format(band), 'fontsize': 14},
                'sharey': False,
                'fontsize': fontsize,
                'labelsize': labelsize,
                'legend': {}
            }

            if band != glob_bands[-1]:
                model['label'] = None

            if band != glob_bands[0]:
                model['sharey'] = True
                obs['sharey'] = True

            if band == glob_bands[-1]:
                obs['legend'] = {'loc': 'lower left', 'ncol': 1, 'fontsize': 14}

            o_plot.set_plot_dics.append(obs)
            o_plot.set_plot_dics.append(model)

    o_plot.main()
    exit(1)


def make_grid(dfile, rl, nlevels = 7):
    arr = np.array(dfile["reflevel=%d"%rl]["rho"])
    import scidata.carpet.grid as grid
    L = []
    for il in range(nlevels):
        gname = "reflevel={}".format(il)
        group = dfile[gname]
        level = grid.basegrid()
        level.delta = np.array(group.attrs["delta"])
        # print("delta: {} ".format(np.array(group.attrs["delta"]))); exit(1)
        level.dim = 3
        level.time = group.attrs["time"]
        # level.timestep = group.attrs["timestep"]
        level.directions = range(3)
        level.iorigin = np.array([0, 0, 0], dtype=np.int32)

        origin = np.array(group.attrs["extent"][0::2])
        level.origin = origin

        level.n = np.array(arr.shape, dtype=np.int32)
        level.rlevel = il
        L.append(level)
    return grid.grid(sorted(L, key=lambda x: x.rlevel))
 
def test_it_rl_v_n_structure():


    sim = "LS220_M14691268_M0_LK_SR"
    it = 2228224
    output = "output-0024"
    rl = 4
    v_n = "rho"
    v_n_n = "Q_eff_nua"

    # load and get profile data structure
    fpath = "/data1/numrel/WhiskyTHC/Backup/2018/GW170817/{}/profiles/3d/{}.h5".format(sim, it)
    dfile = h5py.File(fpath, "r")
    arr = np.array(dfile["reflevel=%d"%rl][v_n])
    grid = make_grid(dfile, rl=rl, nlevels=7)
    x, y, z = grid.mesh()[rl]
    print(" ---------- profile -----------")
    print("{} x {} | {} y {} | {} z {}".format(x[0, 0, 0], x[-1, 0, 0],
                                               y[0, 0, 0], y[0, -1, 0],
                                               z[0, 0, 0], z[0, 0, -1]
                                               ))
    print("x.shape {}".format(x.shape))
    print("rho.shape {}".format(arr.shape))

    print(arr[x>38])
    print(arr[((x>38)&(z>45))])
    # exit(1)

    # load 2D data data structure
    from scidata.utils import locate
    import scidata.carpet.hdf5 as h5
    fname = v_n + '.' + "xz" + '.h5'
    path = "/data1/numrel/WhiskyTHC/Backup/2018/GW170817/LS220_M14691268_M0_LK_SR/output-0024/data/"
    files = locate(fname, root=path, followlinks=False)
    assert len(files) > 0
    dset = h5.dataset(files)
    x, z = dset.get_grid(iteration=it).mesh()[rl]
    print("----------- xz ------------------ ")
    print("{} x {} | {} z {} ".format(x[0, 0], x[-1, 0],
                                      z[0, 0], z[0, -1],
                                      ))
    print("x.shape {}".format(x.shape))

    exit(1)








    print("prof: {} {}".format(v_n, arr.shape))

    # load prof_xz data structure
    fpath = "/data01/numrel/vsevolod.nedora/postprocessed4/{}/profiles/{}/profile.xz.h5".format(sim, it)
    dfile = h5py.File(fpath, "r")
    arr = np.array(dfile["reflevel=%d"%rl][v_n])
    print("slice: {} {}".format(v_n, arr.shape))

    # load 2D data data structure
    from scidata.utils import locate
    import scidata.carpet.hdf5 as h5
    fname = v_n + '.' + "xz" + '.h5'
    path = "/data1/numrel/WhiskyTHC/Backup/2018/GW170817/LS220_M14691268_M0_LK_SR/output-0024/data/"
    files = locate(fname, root=path, followlinks=False)
    assert len(files) > 0
    dset = h5.dataset(files)
    grid = dset.get_grid(iteration=it).mesh()
    print(grid);exit(1)
    data = dset.get_grid_data(grid, iteration=it, variable="HYDROBASE::rho")[rl]
    print("2d: {}: data.shape {}".format(v_n, data.shape))
    print(data)

    data = dset.get_component_data(iteration=it, variable="HYDROBASE::rho", reflevel=rl)
    # data = dset.get_reflevel_data(iteration=it, variable="HYDROBASE::rho", reflevel=rl)
    print("2d: {}: data.shape {}".format(v_n, data.shape))
    print(data)

    fname = v_n_n + '.' + "xz" + '.h5'
    path = "/data1/numrel/WhiskyTHC/Backup/2018/GW170817/LS220_M14691268_M0_LK_SR/output-0024/data/"
    files = locate(fname, root=path, followlinks=False)
    assert len(files) > 0
    dset = h5.dataset(files)
    grid = dset.get_grid(iteration=it)
    data = dset.get_reflevel_data(iteration=it, variable="THC_LEAKAGEBASE::R_eff_nua", reflevel=rl)

    print("2d: {}: data.shape {}".format(v_n_n, data.shape))
    # compare

def add_m0_xy_xz_to_prof_slices():

    glob_sim = "BLh_M13641364_M0_LK_SR"
    glob_it = [2187264]
    glob_profxyxz_path = Paths.ppr_sims + glob_sim + '/profiles/'
    glob_planes = ["xy"]
    glob_reflevels = [0, 1, 2, 3, 4, 5, 6]
    glob_overwrite = True

    #
    from scipy import interpolate
    from profile import MODIFY_NU_DATA
    from preanalysis import LOAD_ITTIME
    # iterations = []
    #
    # o_ititme = LOAD_ITTIME(glob_sim)
    nuprof = MODIFY_NU_DATA(glob_sim, symmetry=None)


    for it in glob_it:
        #print("it: {}".format(it))
        for plane in glob_planes:
            #print("\tplane: {}".format(plane))
            fpath = glob_profxyxz_path + str(int(it)) + '/' + "profile.{}.h5".format(plane)
            dfile = h5py.File(fpath, "a")
            #
            if plane == "xz":
                nu_x, nu_z = nuprof.get_x_y_z_grid(it, plane="xz", rmax=512)
                nu_v_ns = nuprof.list_nuprof_v_ns

                for rl in glob_reflevels:
                    #print("\t\trl: {}".format(rl))
                    gname = "reflevel=%d" % rl
                    px, py, pz = dfile[gname]["x"], dfile[gname]["y"], dfile[gname]["z"]
                    #

                    Printcolor.print_colored_string(
                        ["task:", "addm0", "it:", "{}".format(it), "plane", plane, "rl", rl,  ':', "Adding"],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "green"]
                    )

                    for v_n in nu_v_ns:

                        if (v_n in dfile[gname] and glob_overwrite) or not v_n in dfile[gname]:
                            if v_n in dfile[gname]:
                                del dfile[gname][v_n]

                            # print("\t\t\tv_n:{}".format(v_n))
                            nu_data = nuprof.get_nuprof_arr_slice(it, plane, v_n)
                            # print("\t\t\t{} [{} {}]".format(nu_x.shape, nu_x.min(), nu_x.min()))
                            # print("\t\t\t{} [{} {}]".format(nu_z.shape, nu_z.min(), nu_z.max()))
                            # print("\t\t\t{}".format(nu_data.shape))
                            print("\tinterpolating {}".format(v_n))
                            f = interpolate.LinearNDInterpolator( tuple([nu_x.flatten(), nu_z.flatten()]), nu_data.flatten())
                            #
                            nu_arr_interp = f(px, pz)
                            nu_arr_interp = np.reshape(nu_arr_interp, newshape=px.shape)
                            #
                            #print(nu_arr_interp)
                            #
                            dfile[gname].create_dataset(v_n, data=nu_arr_interp, dtype=np.float32)

                            del nu_arr_interp
                            del nu_data
            # xz
            if plane == "xy":
                nu_x, nu_y = nuprof.get_x_y_z_grid(it, plane="xy", rmax=512)
                nu_v_ns = nuprof.list_nuprof_v_ns

                for rl in glob_reflevels:
                    #print("\t\trl: {}".format(rl))
                    gname = "reflevel=%d" % rl
                    px, py, _ = dfile[gname]["x"], dfile[gname]["y"], dfile[gname]["z"]
                    #

                    Printcolor.print_colored_string(
                        ["task:", "addm0", "it:", "{}".format(it), "plane", plane, "rl", rl,  ':', "Adding"],
                        ["blue", "green", "blue", "green", "blue", "green", "blue", "green", "", "green"]
                    )

                    for v_n in nu_v_ns:

                        if (v_n in dfile[gname] and glob_overwrite) or not v_n in dfile[gname]:
                            if v_n in dfile[gname]:
                                del dfile[gname][v_n]

                            # print("\t\t\tv_n:{}".format(v_n))
                            nu_data = nuprof.get_nuprof_arr_slice(it, plane, v_n)
                            # print("\t\t\t{} [{} {}]".format(nu_x.shape, nu_x.min(), nu_x.min()))
                            # print("\t\t\t{} [{} {}]".format(nu_z.shape, nu_z.min(), nu_z.max()))
                            # print("\t\t\t{}".format(nu_data.shape))
                            print("\tinterpolating {}".format(v_n))
                            f = interpolate.LinearNDInterpolator( tuple([nu_x.flatten(), nu_y.flatten()]), nu_data.flatten())
                            #
                            nu_arr_interp = f(px, py)
                            nu_arr_interp = np.reshape(nu_arr_interp, newshape=px.shape)
                            #
                            #print(nu_arr_interp)
                            #
                            dfile[gname].create_dataset(v_n, data=nu_arr_interp, dtype=np.float32)

                            del nu_arr_interp
                            del nu_data
            dfile.close()
    pass


def old_add_m0_to_prof_xyxz(v_ns, rls):
    # glob_sim = "LS220_M14691268_M0_LK_SR"
    glob_profxyxz_path = Paths.ppr_sims+glob_sim+'/profiles/'
    glob_nlevels = 7
    # glob_overwrite = False

    from preanalysis import LOAD_ITTIME
    ititme = LOAD_ITTIME(glob_sim)
    _, profit, proft = ititme.get_ittime("profiles", d1d2d3prof="prof")
    #
    if len(profit) == 0:
        Printcolor.yellow("No profiles found. Q R T values are not added to prof.xy.h5")
        return 0
    #
    d2data = COMPUTE_STORE(glob_sim)
    #
    assert len(glob_reflevels) > 0
    assert len(v_ns) > 0
    #
    for it in glob_it:
        for plane in glob_planes:
            fpath = glob_profxyxz_path + str(int(it)) + '/' + "profile.{}.h5".format(plane)
            if os.path.isfile(fpath):
                try:
                    dfile = h5py.File(glob_profxyxz_path + str(int(it)) + '/' + "profile.{}.h5".format(plane), "a")

                    Printcolor.print_colored_string(
                        ["task:", "addm0", "it:", "{}".format(it), "plane", plane, ':', "Adding"],
                        ["blue", "green", "blue", "green","blue", "green",  "", "green"]
                    )
                    for rl in rls:
                        gname = "reflevel=%d" % rl
                        for v_n in v_ns:
                            if (v_n in dfile[gname] and glob_overwrite) or not v_n in dfile[gname]:
                                if v_n in dfile[gname]:
                                        del dfile[gname][v_n]
                                #
                                prof_rho = dfile[gname]["rho"]
                                rho_arr = d2data.get_data(it, plane, "rho")[rl][3:-3, 3:-3]
                                nu_arr = d2data.get_data(it, plane, v_n)[rl][3:-3, 3:-3]
                                assert rho_arr.shape == nu_arr.shape

                                if prof_rho.shape != nu_arr.shape:
                                    Printcolor.yellow("Size Mismatch. Profile:{} 2D data:{} Filling with nans..."
                                                      .format(prof_rho.shape, nu_arr.shape))
                                    px, py, pz = dfile[gname]["x"], dfile[gname]["y"], dfile[gname]["z"]
                                    nx, nz = d2data.get_grid_v_n_rl(it, plane, rl, "x")[3:-3, 3:-3], \
                                           d2data.get_grid_v_n_rl(it, plane, rl, "z")[3:-3, 3:-3]
                                    # print("mismatch prof_rho:{} nu:{}".format(prof_rho.shape, nu_arr.shape))
                                    # print("mismatch prof x:{} prof z:{}".format(px.shape, pz.shape))
                                    # print("mismatch x:{} z:{}".format(nx.shape, nz.shape))
                                    # arr = np.full(prof_rho[:,0,:].shape, 1)

                                    # tst = np.where((px>=nx.min()) | (px<=nx.max()), arr, nu_arr)
                                    # print(tst)

                                    tmp = np.full(prof_rho.shape, np.nan)
                                    # for ipx in range(len(px)):

                                    for ipx in range(len(px[:, 0])):
                                        for ipz in range(len(pz[0, :])):
                                            if px[ipx] in nx and pz[ipz] in nz:
                                                # print("found: {} {}".format(px[ipx], py[ipz]))
                                                # print(px[(px[ipx] == nx)&(pz[ipz] == nz)])
                                                # print(pz[(px[ipx] == nx) & (pz[ipz] == nz)])
                                                # print(nu_arr[(px[ipx] == nx)&(pz[ipz] == nz)])
                                                # print("x:{} z:{}".format(px[ipx, 0], pz[0,  ipz]))
                                                # print(nu_arr[(px[ipx, 0] == nx)&(pz[0, ipz] == nz)])
                                                # print(float(nu_arr[(px[ipx, 0] == nx) & (pz[0, ipz] == nz)]))
                                                tmp[ipx, ipz] = float(nu_arr[(px[ipx, 0] == nx) & (pz[0, ipz] == nz)])
                                                # print("x:{} z:{} filling with:{}".format(px[ipx, 0], pz[0, ipz], tmp[ipx, ipz]))
                                    #
                                    nu_arr = tmp
                                            # else:
                                                # print("wrong: {}".format(px[ipx], py[ipz]))
                                    # print(tmp)
                                    # print(tmp.shape)
                                    # exit(1)

                                    # UTILS.find_nearest_index()
                                    #
                                    #
                                    #
                                    # for ix in range(len(arr[:, 0])):
                                    #     for iz in range(len(arr[0, :])):
                                    #         x = np.round(px[ix, iz], decimals=1)
                                    #         z = np.round(py[ix, iz], decimals=1)
                                    #
                                    #
                                    #
                                    #         if x in np.round(nx, decimals=1) and z in np.round(nz, decimals=1):
                                    #             arr[ix, iz] = nu_arr[np.where((np.round(nx, decimals=1) == x) & (np.round(nz, decimals=1) == z))]
                                    #             print('\t\treplacing {} {}'.format(ix, iz))
                                    # print(arr)
                                    #
                                    # exit(1)
                                    #
                                    #
                                    # ileft, iright = np.where(px<nx.min()), np.where(px>nx.max())
                                    # print(ileft)  # (axis=0 -- array, axis=1 -- array)
                                    # print(iright)
                                    # ilower, iupper = np.where(pz<nz.min()), np.where(pz>nz.max())
                                    # print(ilower)
                                    # print(iupper)
                                    #
                                    # #
                                    # import copy
                                    # tmp = copy.deepcopy(nu_arr)
                                    # for axis in range(len(ileft)):
                                    #     for element in ileft[axis]:
                                    #         tmp = np.insert(tmp, 0, np.full(len(tmp[0,:]), np.nan), axis=0)
                                    #
                                    # # tmp = copy.deepcopy(nu_arr)
                                    # for axis in range(len(iright)):
                                    #     print("\taxis:{} indexes:{}".format(axis, iright[axis]))
                                    #     for element in iright[axis]:
                                    #         tmp = np.insert(tmp, -1, np.full(len(tmp[0,:]), np.nan), axis=0)
                                    #     print(tmp.shape)
                                    #
                                    # print(prof_rho.shape)
                                    # print(tmp.shape)

                                    # indexmap = np.where((px<nx.min()) | (px>nx.max()), arr, 0)
                                    # arr[indexmap] = nu_arr
                                    # print(indexmap)
                                    # print(arr)
                                    # print(indexmap.shape)

                                    # insert coordinates
                                    # exit(1)


                                    # arr = np.full(prof_rho.shape, np.nan)



                                    # exit(1)
                                    #
                                    #
                                    #
                                    #
                                    # arr = np.full(prof_rho.shape,np.nan)
                                    # for ix in range(len(arr[:, 0])):
                                    #     for iz in range(len(arr[0,:])):
                                    #         x = px[ix, iz]
                                    #         z = py[ix, iz]
                                    #         if x in nx and z in nz:
                                    #             arr[ix, iz] = nu_arr[np.where((nx == x)&(nz == z))]
                                    #             print('\t\treplacing {} {}'.format(ix, iz))
                                    # print(arr);
                                    #
                                    # exit(1)

                                print("\t{} nu:{} prof_rho:{}".format(rl, nu_arr.shape, prof_rho.shape))
                                # nu_arr = nu_arr[3:-3, 3:-3]
                                # hydro_arr = d3data.get_data(it, rl, plane, "rho")
                                # assert nu_arr.shape == hydro_arr.shape
                                gname = "reflevel=%d" % rl
                                dfile[gname].create_dataset(v_n, data=np.array(nu_arr, dtype=np.float32))
                            else:
                                Printcolor.print_colored_string(["\trl:", str(rl), "v_n:", v_n, ':',
                                     "skipping"],
                                    ["blue", "green","blue", "green", "", "blue"]
                                )
                    dfile.close()
                except KeyboardInterrupt:
                    exit(1)
                except ValueError:
                    Printcolor.print_colored_string(
                        ["task:", "addm0", "it:", "{}".format(it), "plane", plane, ':', "ValueError"],
                        ["blue", "green", "blue", "green","blue", "green", "", "red"]
                    )
                except IOError:
                    Printcolor.print_colored_string(
                        ["task:", "addm0", "it:", "{}".format(it), "plane", plane, ':', "IOError"],
                        ["blue", "green", "blue", "green","blue", "green", "", "red"]
                    )
                except:
                    Printcolor.print_colored_string(
                        ["task:", "addm0", "it:", "{}".format(it), "plane", plane, ':', "FAILED"],
                        ["blue", "green", "blue", "green", "blue", "green", "", "red"]
                    )
            else:
                Printcolor.print_colored_string(
                    ["task:", "adding neutrino data to prof. slice", "it:", "{}".format(it), ':', "IOError: profile.{}.h5 does not exist".format(plane)],
                    ["blue", "green", "blue", "green", "", "red"]
                )
    # for it in profit:
    #     #
    #     fpathxy = glob_profxyxz_path + str(int(it)) + '/' + "profile.xy.h5"
    #     fpathxz = glob_profxyxz_path + str(int(it)) + '/' + "profile.xz.h5"


''' -------------------------- neutrin poprofile ---------------------------------------- '''

def test_nu_data():


    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from profile import MODIFY_NU_DATA



    nuprof = MODIFY_NU_DATA("BLh_M13641364_M0_LK_SR")
    it = 2056192

    # arr = nuprof.get_nuprof_arr(516096, "eave_nua")
    # print(arr.shape)
    # rarr = nuprof.get_nuprof_arr_sph(516096, "eave_nua")
    # print(rarr.shape)

    v_n = 'E_nua'

    data = nuprof.get_nuprof_arr_slice(it, "xz", v_n)
    x, z = nuprof.get_x_y_z_grid(it, plane="xz", rmax=512)

    from matplotlib import colors

    fig = plt.figure(figsize=(6.0, 3.0))
    ax = fig.add_subplot(111)
    ax.set_aspect(1.)

    min = data[(x > 20) & (z > 20) & (data > 0.)].min()
    max = data[(x > 20) & (z > 20)].max()

    print("min:{}".format(min))
    print("max:{}".format(max))

    norm = colors.Normalize(vmin=min, vmax=max)
    im = ax.pcolormesh(x, z, data, norm=norm, cmap="inferno_r")

    # divider = make_axes_locatable(ax)
    # cax = divider.new_horizontal(size="5%", pad=0.7, pack_start=True)
    # fig.add_axes(cax)
    # fig.colorbar(im, cax=cax, orientation="vertical")
    fig.colorbar(im, ax=ax)
    # cbar = ax.cax.colorbar(im)
    # cbar = grid.cbar_axes[0].colorbar(im)

    plt.xlim(-70, 70)
    plt.ylim(0, 70)
    plt.title("${}$".format(v_n).replace("_", "\_"))
    figname = Paths.plots + "nu_test/nu_plot_{}_test.png".format(v_n)
    print("saving: {}".format(figname))
    plt.savefig(figname, bbox_inches='tight', dpi=128)
    # print("saved pi_test2.png")
    plt.close()

    # exit(1)
    for v_n in nuprof.list_nuprof_v_ns:
        data = nuprof.get_nuprof_arr_slice(it, "xz", v_n)
        x, z = nuprof.get_x_y_z_grid(it, plane="xz", rmax=512)
        print(x.shape)
        print(z.shape)
        exit(1)

        from matplotlib import colors

        fig = plt.figure(figsize=(6.0, 3.0))
        ax = fig.add_subplot(111)
        ax.set_aspect(1.)

        min = data[(x > 20) & (z > 20) & (data > 0.)].min()
        max = data[(x > 20) & (z > 20)].max()

        print("min:{}".format(min))
        print("max:{}".format(max))

        if v_n.__contains__("E_") or v_n == "flux_fac" or v_n == "abs_nue" or v_n.__contains__("eave_"):
            norm = colors.Normalize(vmin=min, vmax=max)
        else:
            norm = colors.LogNorm(vmin=min, vmax=max)

        # norm = colors.LogNorm(vmin=data[(x > 20) & (z > 20) & (data > 0.)].min(), vmax=data[(x > 20) & (z > 20)].max())
        im = ax.pcolormesh(x, z, data, norm=norm, cmap="inferno_r")

        fig.colorbar(im, ax=ax)

        plt.xlim(-70, 70)
        plt.ylim(0, 70)
        plt.title("M0: {}".format(v_n).replace("_", "\_"))
        figname = '{}'.format(Paths.plots + "nu_test/nu_plot_{}.png".format(v_n))
        plt.savefig(figname, bbox_inches='tight', dpi=128)
        print("saving: {}".format(figname))
        plt.close()

    # print(plane)
    pass

if __name__ == '__main__':

    add_m0_xy_xz_to_prof_slices()
    exit(1)
    test_nu_data()
    exit(1)
    test_it_rl_v_n_structure()
    exit(1)
    # sliceprof = LOAD_PROFILE_XYXZ("BLh_M11041699_M0_LK_LR")
    # data_arr = sliceprof.get_data(1046114, 3, "xz", "entr")
    # print(data_arr)

    d3corr_class = MAINMETHODS_STORE("BLh_M11041699_M0_LK_LR")
    times, iterations, xcs, ycs, modes, rs, mmodes = d3corr_class.get_dens_modes_for_rl(rl=3, mmax=8, nshells=100)

    
    #
    # comparing_mkn_codes()
    # LOAD_RESHAPE_SAVE_PARALLEL()
    # add_q_r_t_to_prof_xyxz()
    # pass
    '''
    o = EJECTA_PARS("LS220_M14691268_M0_LK_SR")
    table = o.get_ejecta_arr(0, "bern_geoend", "timecorr theta")
    table[0,1:]*=Constants.time_constant
    print(table[1:, 0])
    print(table[0, 1:])
    for i in range(len(table[0, 1:])):
        print(table[1:,i])
    plot_corr(table, "time", "theta", Paths.plots+'test_outflow/')
    '''

    # det = 0
    # sim = "LS220_M14691268_M0_LK_SR"
    # n_procs = 8
    #
    # fname = "outflow_surface_det_%d_fluxdens.asc" % det
    # flist = glob(Paths.gw170817 + sim + "/" + "output-????" + "/data/" + fname)
    # assert len(flist) > 0
    #
    # dfile = open(flist[0], "r")
    # dfile.readline()  # move down one line
    # match = re.match('# detector no.=(\d+) ntheta=(\d+) nphi=(\d+)$', dfile.readline())
    # assert int(det) == int(match.group(1))
    # ntheta = int(match.group(2))
    # nphi = int(match.group(3))
    # dfile.readline()
    # dfile.readline()
    # line = dfile.readline().split()
    # radius = round(sqrt(float(line[2]) ** 2 + float(line[3]) ** 2 + float(line[4]) ** 2))
    # # if not self.clean:
    # print("\t\tradius = {}".format(radius))
    # print("\t\tntheta = {}".format(ntheta))
    # print("\t\tnphi   = {}".format(nphi))
    # del dfile
    #
    # grid = SphericalSurface(ntheta, nphi, radius)
    #
    # # print("Pool procs = %d" % n_procs)
    # # pool = mp.Pool(processes=int(n_procs))
    # # task = partial(LOAD_EXTRACT_RESHAPE_SAVE, grid_object=grid, outdir=Paths.ppr_sims+sim+'/tmp/')
    # # result_list = pool.map(task, flist)
    #
    # ''''''
    # v_ns = ['it', 'time', "fluxdens", "w_lorentz", "eninf", "surface_element",
    #         "alp", "rho", "vel[0]", "vel[1]", "vel[2]", "Y_e", "entropy", "temperature"]
    # # load tmp data
    # tmp_flist = [Paths.ppr_sims+sim+'/tmp/' + outfile.split('/')[-3] + ".h5" for outfile in flist]
    # tmp_flist = sorted(tmp_flist)
    # # print(tmp_flist)
    # # exit(1)
    # assert len(tmp_flist) == len(flist)
    # # print(tmp_flist)
    #
    #
    # dfiles = []
    # iterations = []
    # times = []
    # data_matrix = {}
    # for ifile, fpath in enumerate(tmp_flist):
    #     assert os.path.isfile(fpath)
    #     dfile = h5py.File(fpath, "r")
    #     # iterations = []
    #     # times = []
    #     for v_n in dfile:
    #         # print(v_n)
    #         match = re.match('iteration=(\d+)$', v_n)
    #         it = int(match.group(1))
    #         if not it in iterations:
    #             i_data_matrix = {}
    #             for var_name in v_ns:
    #                 data = np.array(dfile[v_n][var_name])
    #                 # print(data.shape); exit(1)
    #                 i_data_matrix[var_name] = data
    #             data_matrix[it] = i_data_matrix
    #             times.append(float(i_data_matrix["time"][0, 0]))
    #             iterations.append(int(match.group(1)))
    #             print(it, fpath)
    #         else:
    #             pass
    #     dfile.close()
    #     # for v_n in v_ns:
    # print(iterations)
    #
    #
    #
    # # data_matrix = [x for _, x in sorted(zip(iterations, data_matrix))]
    #
    # concatenated_data = {}
    #
    # # data_matrix2 = {}
    # # for it in sorted(data_matrix.keys()):
    # #     print(data_matrix[it]["it"].shape)
    # # concatenated_data = {}
    #
    # iterations = np.sort(iterations)
    # times = np.sort(times)# * 1e3 / 0.004925794970773136
    #
    # for v_n in v_ns:
    #     concatenated_data[v_n] = np.stack(([data_matrix[it][v_n] for it in sorted(data_matrix.keys())]))
    #     print(v_n, concatenated_data[v_n].shape)
    #
    # o_eos = EOSTable()
    # o_eos.read_table(Paths.get_eos_fname_from_curr_dir(sim))
    # v_n_to_eos_dic = {
    #     'eps': "internalEnergy",
    #     'press': "pressure",
    #     'entropy': "entropy"
    # }
    # for v_n in ['eps', 'press']:
    #     print("Evaluating eos: {}".format(v_n))
    #     data_arr = o_eos.evaluate(v_n_to_eos_dic[v_n], concatenated_data["rho"],
    #                               concatenated_data["temperature"],
    #                               concatenated_data["Y_e"])
    #     if v_n == 'eps':
    #         data_arr = ut.conv_spec_energy(ut.cgs, ut.cactus, data_arr)
    #     elif v_n == 'press':
    #         data_arr = ut.conv_press(ut.cgs, ut.cactus, data_arr)
    #     elif v_n == 'entropy':
    #         data_arr = data_arr
    #     else:
    #         raise NameError("EOS quantity: {}".format(v_n))
    #
    #     concatenated_data[v_n] = data_arr
    #
    # outfile = h5py.File(Paths.ppr_sims+sim+'/'+fname.replace(".asc",".h5"), "w")
    #
    # outfile.create_dataset("iterations", data=np.array(iterations, dtype=int))
    # outfile.create_dataset("times", data=times, dtype=np.float32)
    #
    # outfile.attrs.create("ntheta", grid.ntheta)
    # outfile.attrs.create("nphi", grid.nphi)
    # outfile.attrs.create("radius", grid.radius)
    # outfile.attrs.create("dphi", 2 * np.pi / grid.nphi)
    # outfile.attrs.create("dtheta", np.pi / grid.ntheta)
    #
    # outfile.create_dataset("area", data=grid.area(), dtype=np.float32)
    # theta, phi = grid.mesh()
    # outfile.create_dataset("phi", data=phi, dtype=np.float32)
    # outfile.create_dataset("theta", data=theta, dtype=np.float32)
    #
    # v_ns.remove("it")
    # v_ns.remove("time")
    #
    # for v_n in v_ns + ['eps', 'press']:
    #     outfile.create_dataset(v_n, data=concatenated_data[v_n], dtype=np.float32)
    # outfile.close()

    # iterations = np.unique(iterations)
    # data_matrix = [{} for i in range(len(iterations))]
    # for ifile, fpath in enumerate(tmp_flist):
    #     dfile = h5py.File(fpath, "r")
    #     for v_n in dfile:
    #         match = re.match('iteration=(\d+)$', v_n)
    #         it = int(match.group(1))




    # For Katy!
    # let the file name be "my_file@ that has two columns and no words or letters. Then do:
    # import numpy as np
    # import matplotlib.pyplot as plt
    # col1, col2 = np.loadtxt("my_file", usecols=(0,1), unpack=True)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(col1, col2, color="black", ls="-", label="my_file")
    # plt.tight_layout()
    # plt.show()