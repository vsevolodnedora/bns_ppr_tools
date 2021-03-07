#######################################################################
#
# Default settings, paths, filenames and constants
# used in the analysis
#
#
#
# (Author Vsevolod Nedora)
#
########################################################################

from math import log
from glob import glob
import numpy as np
import re
import os


# class Paths:
#
#     scripts  =  '/data01/numrel/vsevolod.nedora/scripts_server/'
#     default_ppr_dir = '/data01/numrel/vsevolod.nedora/postprocessed5/'
#     lorene =    '/data/numrel/Lorene/Lorene_TABEOS/GW170817/'
#     # lorene =    '/data01/numrel/vsevolod.nedora/Data/Lorene/'
#     rns =       '/data01/numrel/vsevolod.nedora/Data/RNS/'
#     TOVs =      '/data01/numrel/vsevolod.nedora/Data/TOVs/'
#     default_data_dir = '/data1/numrel/WhiskyTHC/Backup/2018/GW170817/' # "/data01/numrel/vsevolod.nedora/tmp/" # '/data1/numrel/WhiskyTHC/Backup/2018/SLy4_M130130_SR_physics/'
#     skynet =    '/data01/numrel/vsevolod.nedora/Data/skynet/'
#     output =    '/data01/numrel/vsevolod.nedora/output/'
#     plots =     '/data01/numrel/vsevolod.nedora/figs/'
#     mkn =       '/data01/numrel/vsevolod.nedora/macrokilonova_bayes_new/source/'
#     home =      '/data01/numrel/vsevolod.nedora/bns_ppr_tools/'
#     SLy4_hydo=  '/data01/numrel/vsevolod.nedora/Data/EOS/SLy4/SLy4_hydro_14-Dec-2017.h5'
#     # skynet =   '/data01/numrel/vsevolod.nedora/scripts_server/module_ejecta/skynet/'
#
#     @staticmethod
#     def get_eos_fname_from_curr_dir(sim):
#
#         if sim.__contains__("SLy4"):
#             fname = "/data01/numrel/vsevolod.nedora/Data/EOS/SLy4/SLy4_hydro_14-Dec-2017.h5"
#         elif sim.__contains__("LS220"):
#             fname = "/data01/numrel/vsevolod.nedora/Data/EOS/LS220/LS_220_hydro_27-Sep-2014.h5"
#         elif sim.__contains__("DD2"):
#             fname = "/data01/numrel/vsevolod.nedora/Data/EOS/DD2/DD2_DD2_hydro_30-Mar-2015.h5"
#         elif sim.__contains__("SFHo"):
#             fname = "/data01/numrel/vsevolod.nedora/Data/EOS/SFHo/SFHo_hydro_29-Jun-2015.h5"
#         elif sim.__contains__("BLh"):
#             fname = "/data01/numrel/vsevolod.nedora/Data/EOS/SFHo+BL/BLH_new_hydro_10-Jun-2019.h5"
#         elif sim.__contains__("BHBlp"):
#             fname = "/data01/numrel/vsevolod.nedora/Data/EOS/BHB/BHB_lp_hydro_10-May-2016.h5"
#         else:
#             raise NameError("Current dir does not contain a hint to what EOS to use: \n{}"
#                             .format(sim))
#         return fname
#
#     @staticmethod
#     def get_list_iterations_from_res_3d(prodfir):
#         """
#         Checks the /res_3d/ for 12345 folders, (iterations) retunrs their sorted list
#         :param sim:
#         :return:
#         """
#
#         if not os.path.isdir(prodfir):
#             raise IOError("no {} directory found".format(prodfir))
#
#         itdirs = os.listdir(prodfir)
#
#         if len(itdirs) == 0:
#             raise NameError("No iteration-folders found in the {}".format(prodfir))
#
#         # this is a f*cking masterpiece of programming)))
#         list_iterations = np.array(
#             np.sort(np.array(list([int(itdir) for itdir in itdirs if re.match("^[-+]?[0-9]+$", itdir)]))))
#         if len(list_iterations) == 0:
#             raise ValueError("Error extracting the iterations")
#
#         return list(list_iterations)
#
#     @staticmethod
#     def find_itdir_with_grid(sim, gridfname='cyl_grid.h5'):
#         # for dir_ in os.listdir()
#         path = Paths.default_ppr_dir + sim + '/res_3d/' + '*' + '/' + gridfname
#         files = glob(path)
#         # print(files)
#         if len(files) == 0:
#             raise ValueError("No grid file ({}) found for {}".format(gridfname, sim))
#         if len(files) > 1:
#             Printcolor.yellow("More than 1({}) grid file ({}) found for {}"
#                               .format(len(files), gridfname, sim))
#             return files[0]
#         return str(files)
#
#     @staticmethod
#     def get_it_from_itdir(itdir):
#         it = -1
#         try:
#             it = int(itdir.split('/')[-2])
#         except:
#             try:
#                 itdirs = list(itdir)
#                 it = [int(itdir) for itdir in itdirs if re.match("^[-+]?[0-9]+$", itdir)]
#             except:
#                 raise ValueError("failed to extract iteration from itdir:{} ".format(itdir))
#         return it


class Files:
    it_time     = 'collated/rho.maximum.asc'
    models      = 'models.csv'
    models_empty= 'models_tmp2.csv'

    # collated2
    disk_mass   = 'disk_mass.asc'

    # collated
    dens_unb    = 'dens_unbnd.norm1.asc'
    dens_unb_bern = 'dens_unbnd_bernoulli.norm1.asc'
    dens        = 'dens.norm1.asc'

    # module_ejecta
    total_flux  = 'total_flux.dat'
    hist_theta  = 'hist_theta.dat'
    hist_ye     = 'hist_ye.dat'
    hist_entropy= 'hist_entropy.dat'
    hist_vel_inf= 'hist_vel_inf.dat'

    # nucle
    yields      = 'yields.h5'
    solar_r     = 'solar_r.dat'

    # waveforms
    l2_m2       = 'waveform_l2_m2.dat'
    tmerg       = 'tmerger.dat'

    # module_ejecta module_profile:
    ejecta_profile="ejecta_profile.dat"
    ejecta_profile_bern= "ejecta_profile_bern.dat"

    # mkn
    mkn_model   = 'mkn_model.h5'
    filt_at2017gfo= 'AT2017gfo.h5'


class Lists:

    chosen_sims = [
        "DD2_M13641364_M0_SR"
        "DD2_M13641364_M0_LK_SR_R04"
    ]


    eos = ["DD2", "LS220", "SFHo", "SLy4", "BLh"]
    res = ["VLR", "LR", "SR", "HR"]
    neut= ["M0", "M1"]
    visc= ["LK"]
    # q   = [1.0, 1.05, 1.1, 1.11, 1.13, 1.16, 1.19, 1.2, 1.22, 1.29]
    # q   = [1.0, 1.053, 1.102, 1.106, 1.132, 1.159, 1.185, 1.201, 1.222, 1.285]
    colors_q={1.000:"firebrick", 1.053:"red", 1.102:"gold",
              1.106:"darkkhaki", 1.132:"olivedrab", 1.159:"darkgreen",
              1.185:"lightseagreen", 1.201:"darkgreen", 1.222:"royalblue", 1.285:"navy"}

    dyn_not_pas = [
                  "DD2_M13641364_M0_HR_R04", # not in sim2
                  "DD2_M13641364_M0_HR", # not in sim2
                  "DD2_M14861254_M0_HR", # not in sim2
                  "LS220_M14001330_M0_HR", # not in sim2
                  "LS220_M14351298_M0_HR", # in sim2
                  "SLy4_M13641364_M0_HR", # not in sim2
                  "SLy4_M13641364_M0_LR", # in sim2 BUT problem with rocketing dynimical module_ejecta
                  ]

    bern_pass=[
                  "LS220_M13641364_M0_LK_SR",
                  "LS220_M13641364_M0_SR",
                  "SFHo_M14521283_M0_LR",
                  "SFHo_M14521283_M0_LK_SR",
                  "SFHo_M13641364_M0_LK_SR_2019pizza",
                  "SFHo_M13641364_M0_LK_SR",
                  "SFHo_M14521283_M0_HR"
    ]

    collate_list = [
        "bnstrackergen-bns_positions..asc",
        "dens.norm1.asc",
        "dens_unbnd.norm1.asc",
        "dens_unbnd_bernoulli.norm1.asc",
        "dens_unbnd_garching.norm1.asc",
        "H.norm2.asc",
        "luminosity_nua.norm1.asc",
        "luminosity_nue.norm1.asc",
        "luminosity_nux.norm1.asc",
        "mp_Psi4_l0_m0_r400.00.asc",
        "mp_Psi4_l1_m0_r400.00.asc",
        "mp_Psi4_l1_m1_r400.00.asc",
        "mp_Psi4_l2_m0_r400.00.asc",
        "mp_Psi4_l2_m1_r400.00.asc",
        "mp_Psi4_l2_m2_r400.00.asc",
        "mp_Psi4_l3_m0_r400.00.asc",
        "mp_Psi4_l3_m1_r400.00.asc",
        "mp_Psi4_l3_m2_r400.00.asc",
        "mp_Psi4_l3_m3_r400.00.asc",
        "mp_Psi4_l4_m0_r400.00.asc",
        "mp_Psi4_l4_m1_r400.00.asc",
        "mp_Psi4_l4_m2_r400.00.asc",
        "mp_Psi4_l4_m3_r400.00.asc",
        "mp_Psi4_l4_m4_r400.00.asc",
        "outflow_det_0.asc",
        "outflow_det_1.asc",
        "outflow_det_2.asc",
        "outflow_det_3.asc",
        "rho.maximum.asc",
        "temperature.maximum.asc",
        "thc_leakagem0-thc_leakage_m0_flux..asc"
    ]

    time_index = {
        "bnstrackergen-bns_positions..asc":9,
        "dens.norm1.asc":2,
        "dens_unbnd.norm1.asc":2,
        "dens_unbnd_bernoulli.norm1.asc":2,
        "dens_unbnd_garching.norm1.asc":2,
        "H.norm2.asc":2,
        "luminosity_nua.norm1.asc":2,
        "luminosity_nue.norm1.asc":2,
        "luminosity_nux.norm1.asc":2,
        "mp_Psi4_l0_m0_r400.00.asc":1,
        "mp_Psi4_l1_m0_r400.00.asc":1,
        "mp_Psi4_l1_m1_r400.00.asc":1,
        "mp_Psi4_l2_m0_r400.00.asc":1,
        "mp_Psi4_l2_m1_r400.00.asc":1,
        "mp_Psi4_l2_m2_r400.00.asc":1,
        "mp_Psi4_l3_m0_r400.00.asc":1,
        "mp_Psi4_l3_m1_r400.00.asc":1,
        "mp_Psi4_l3_m2_r400.00.asc":1,
        "mp_Psi4_l3_m3_r400.00.asc":1,
        "mp_Psi4_l4_m0_r400.00.asc":1,
        "mp_Psi4_l4_m1_r400.00.asc":1,
        "mp_Psi4_l4_m2_r400.00.asc":1,
        "mp_Psi4_l4_m3_r400.00.asc":1,
        "mp_Psi4_l4_m4_r400.00.asc":1,
        "outflow_det_0.asc":2,
        "outflow_det_1.asc":2,
        "outflow_det_2.asc":2,
        "outflow_det_3.asc":2,
        "rho.maximum.asc":2,
        "temperature.maximum.asc":2,
        "thc_leakagem0-thc_leakage_m0_flux..asc":2
    }

    tarball = [
        # "bnstrackergen - bns_positions..asc",
        "dens.norm1.asc",
        "dens_unbnd.norm1.asc",
        "dens_unbnd_bernoulli.norm1.asc",
        "dens_unbnd_garching.norm1.asc",
        "H.norm2.asc",
        "luminosity_nua.norm1.asc",
        "luminosity_nue.norm1.asc",
        "luminosity_nux.norm1.asc",
        "mp_Psi4_l0_m0_r400.00.asc",
        "mp_Psi4_l1_m0_r400.00.asc",
        "mp_Psi4_l1_m1_r400.00.asc",
        "mp_Psi4_l2_m0_r400.00.asc",
        "mp_Psi4_l2_m1_r400.00.asc",
        "mp_Psi4_l2_m2_r400.00.asc",
        "mp_Psi4_l3_m0_r400.00.asc",
        "mp_Psi4_l3_m1_r400.00.asc",
        "mp_Psi4_l3_m2_r400.00.asc",
        "mp_Psi4_l3_m3_r400.00.asc",
        "mp_Psi4_l4_m0_r400.00.asc",
        "mp_Psi4_l4_m1_r400.00.asc",
        "mp_Psi4_l4_m2_r400.00.asc",
        "mp_Psi4_l4_m3_r400.00.asc",
        "mp_Psi4_l4_m4_r400.00.asc",
        "outflow_det_0.asc",
        "outflow_det_1.asc",
        "outflow_det_2.asc",
        "outflow_det_3.asc",
        "rho.maximum.asc",
        "temperature.maximum.asc",
        # "thc_leakagem0 - thc_leakage_m0_flux..asc",
        "outflow_surface_det_0_fluxdens.asc",
        "outflow_surface_det_1_fluxdens.asc",
        # "BH_diagnostics.ah1.gp"
    ]

    outflow = [
        "corr_vel_inf_bern_theta.h5", #
        "corr_vel_inf_theta.h5",
        "corr_ye_entropy.h5",
        "corr_ye_entropy.png",
        "corr_ye_theta.h5",
        "module_ejecta.h5",
        "ejecta_profile_bern.dat", #
        "ejecta_profile.dat",
        "hist_entropy.dat",
        "hist_entropy.xg",
        "hist_log_rho.dat",
        "hist_log_rho.xg",
        "hist_vel_inf_bern.dat", #
        "hist_temperature.dat",
        "hist_temperature.xg",
        "hist_theta.dat",
        "hist_theta.xg",
        "hist_vel.dat",
        "hist_vel_inf.dat",
        "hist_vel_inf.xg",
        "hist_vel.xg",
        "hist_ye.dat",
        "hist_ye.xg",
        "mass_averages.dat",
        "profile_entropy.xg",
        "profile_flux.xg",
        "profile_rho.xg",
        "profile_temperature.xg",
        "profile_vel_inf.xg",
        "profile_vel.xg",
        "profile_ye.xg",
        "theta_75.dat",
        "total_flux.dat",
        "yields.h5"
    ]

    gw = [
        "E_GW_dot_l2.png",
        "E_GW_dot.png",
        "E_GW_l2.png",
        "E_GW.png",
        "EJ.dat",
        "fpeak.dat",
        "postmerger_psd_l2_m2.dat",
        "postmerger_strain_l2_m2.dat",
        # "psd_l0_m0.dat",
        # "psd_l1_m0.dat",
        # "psd_l1_m1.dat",
        # "psd_l2_m0.dat",
        # "psd_l2_m1.dat",
        "psd_l2_m2.dat",
        # "psd_l3_m0.dat",
        # "psd_l3_m1.dat",
        # "psd_l3_m2.dat",
        # "psd_l3_m3.dat",
        # "psd_l4_m0.dat",
        # "psd_l4_m1.dat",
        # "psd_l4_m2.dat",
        # "psd_l4_m3.dat",
        # "psd_l4_m4.dat",
        # "psi4_l0_m0.dat",
        # "psi4_l1_m0.dat",
        # "psi4_l1_m1.dat",
        # "psi4_l2_m0.dat",
        # "psi4_l2_m1.dat",
        "psi4_l2_m2.dat",
        # "psi4_l3_m0.dat",
        # "psi4_l3_m1.dat",
        # "psi4_l3_m2.dat",
        # "psi4_l3_m3.dat",
        # "psi4_l4_m0.dat",
        # "psi4_l4_m1.dat",
        # "psi4_l4_m2.dat",
        # "psi4_l4_m3.dat",
        # "psi4_l4_m4.dat",
        # "strain_l0_m0.dat",
        # "strain_l1_m0.dat",
        # "strain_l1_m1.dat",
        # "strain_l2_m0.dat",
        # "strain_l2_m1.dat",
        "strain_l2_m2.dat",
        # "strain_l3_m0.dat",
        # "strain_l3_m1.dat",
        # "strain_l3_m2.dat",
        # "strain_l3_m3.dat",
        # "strain_l4_m0.dat",
        # "strain_l4_m1.dat",
        # "strain_l4_m2.dat",
        # "strain_l4_m3.dat",
        # "strain_l4_m4.dat",
        "waveform_l2_m2.dat",
        "tmerger.dat",
        "tcoll.dat"
    ]

    h5_disk = [
        "rho.file *.h5",
        "temperature.file *.h5",
        "Y_e.file *.h5",
        "volform.file *.h5",
        "w_lorentz.file *.h5"
    ]


class Labels:
    def __init__(self):
        pass

    @staticmethod
    def labels(v_n, mask=None):
        # solar

        if v_n == "nsims":
            return r"$N_{\rm{resolutions}}$"

        elif v_n == 'theta':
            return r"Angle from orbital plane"

        elif v_n == 'temp' or v_n == "temperature":
            return r"$T$ [GEO]"

        elif v_n == 'phi':
            return r"Azimuthal angle"

        elif v_n == 'mass':
            return r'normed $M_{\rm{ej}}$'

        elif v_n == 'diskmass':
            return r'$M_{\rm{disk}}$ $[M_{\odot}]$'

        elif v_n == 'Mdisk3Dmax':
            return r'$M_{\rm{disk;max}}$ $[M_{\odot}]$'

        elif v_n == 'ejmass' or v_n == "Mej_tot":

            if mask == "geo_entropy_above_10":
                return r'$M_{\rm{ej;s>10}}$ $[10^{-2}M_{\odot}]$'
            elif mask == "geo_entropy_below_10":
                return r'$M_{\rm{ej;s<10}}$ $[10^{-2}M_{\odot}]$'
            else:
                return r'$M_{\rm{ej}}$ $[10^{-2}M_{\odot}]$'

        elif v_n == "Mej_tot_scaled":
            if mask == "geo_entropy_above_10":
                return r'$M_{\rm{ej;s>10}}/M_{\rm{b;tot}}$ $[10^{-2}M_{\odot}]$'
            elif mask == "geo_entropy_below_10":
                return r'$M_{\rm{ej;s<10}}/M_{\rm{b;tot}}$ $[10^{-2}M_{\odot}]$'
            else:
                return r'$M_{\rm{ej}}/M_{\rm{b;tot}}$ $[10^{-2}M_{\odot}]$'
            # else:
            #     raise NameError("label for v_n:{} mask:{} is not found".format(v_n, mask))

        elif v_n == "Mej_tot_scaled2":

            if mask == "geo_entropy_above_10":
                return r'$M_{\rm{ej;s>10}}/(\eta M_{\rm{b;tot}})$ $[10^{-2}M_{\odot}]$'
            elif mask == "geo_entropy_below_10":
                return r'$M_{\rm{ej;s<10}}/(\eta M_{\rm{b;tot}})$ $[10^{-2}M_{\odot}]$'
            else:
                return r'$M_{\rm{ej}}/(\eta M_{\rm{b;tot}})$ $[10^{-2}M_{\odot}]$'
            # else:
            #     raise NameError("label for v_n:{} mask:{} is not found".format(v_n, mask))

        elif v_n == 'ejmass3':
            return r'$M_{\rm{ej}}$ $[10^{-3}M_{\odot}]$'

        elif v_n == 'ejmass4':
            return r'$M_{\rm{ej}}$ $[10^{-4}M_{\odot}]$'

        elif v_n == "vel_inf":
            return r"$\upsilon_{\infty}$ [c]"

        elif v_n == "vel_inf_ave":
            return r"$<\upsilon_{\infty}>$ [c]"

        elif v_n == "Y_e" or v_n == "ye" or v_n == "Ye":
            return r"$Y_e$"

        elif v_n == "Lambda":
            return r"$\tilde{\Lambda}$"

        elif v_n == "Ye_ave":
            return r"$<Y_e>$"

        elif v_n == 'flux':
            return r'$\dot{M}$'

        elif v_n == 'time':
            return r'$t$ [ms]'

        elif v_n == 't-tmerg':
            return r'$t-t_{\rm{merg}}$ [ms]'

        elif v_n == "Y_final":
            return r'Relative final abundances'

        elif v_n == "A":
            return r"Mass number, A"

        elif v_n == 'entropy' or v_n == 's':
            return r'$s$'

        elif v_n == 't_eff' or v_n == 'T_eff':
            return r'$\log($T$_{eff}/K)$'

        elif v_n == "Mb":
            return r'$M_{\rm{b;tot}}$'

        else:
            return str(v_n).replace('_', '\_')
            # raise NameError("No label found for v_n:{}"
            #                 .format(v_n))


class REFLEVEL_LIMITS:

    @staticmethod
    def get(rl=0):
        if rl == 6:
            xmin, xmax = -14, 14
            ymin, ymax = -14, 14
            zmin, zmax = 0, 14
        elif rl == 5:
            xmin, xmax = -28, 28
            ymin, ymax = -28, 28
            zmin, zmax = 0, 28
        elif rl == 4:
            xmin, xmax = -48, 48
            ymin, ymax = -48, +48
            zmin, zmax = 0, 48
        elif rl == 3:
            xmin, xmax = -88, 88
            ymin, ymax = -88, 88
            zmin, zmax = 0, 88
        elif rl == 2:
            xmin, xmax = -178, 178
            ymin, ymax = -178, +178
            zmin, zmax = 0, 178
        elif rl == 1:
            xmin, xmax = -354, 354
            ymin, ymax = -354, +354
            zmin, zmax = 0, 354
        elif rl == 0:
            xmin, xmax = -1044, 1044
            ymin, ymax = -1044, 1044
            zmin, zmax = 0, 1044
        else:
            raise IOError("Limits not found for rl:{}".format(rl))
        return xmin, xmax, ymin, ymax, zmin, zmax


class Limits:

    @staticmethod
    def lim(v_n):
        if v_n in ["Y_e", "ye", "Ye"]:
            return 0., 0.5
        elif v_n in  ["vel_inf", "vinf", "velinf"]:
            return 0, 1.1
        elif v_n in ["theta"]:
            return 0, 90.
        elif v_n in ["phi"]:
            return 0., 360
        elif v_n in ["entropy", "s"]:
            return 0, 120.
        elif v_n in ["temperature", "temp"]:
            return 0, 5.
        elif v_n in ["rho"]:
            return 1e-10, 1e-4
        elif v_n in ["logrho"]:
            return -16., -8.
        else:
            raise NameError("limit for v_n:{} is not found"
                            .format(v_n))

    @staticmethod
    def in_dic(dic):
        # if "v_n" in dic.keys():
        #     if dic["v_n"] != None:
        #         lim1, lim2 = Limits.lim(dic["v_n"])
        #         dic["zmin"], dic["zmax"] = lim1, lim2

        if "v_n_x" in dic.keys():
            if dic["xmin"] != None and dic["xmax"] != None:
                pass
            else:
                # print(dic["xmin"], dic["xmin"])
                if dic["v_n_x"] != None:
                    try:
                        lim1, lim2 = Limits.lim(dic["v_n_x"])
                    except:
                        raise NameError("X limits for {} are not set and not found".format(dic["v_n_x"]))

                    dic["xmin"], dic["xmax"] = lim1, lim2

        if "v_n_y" in dic.keys():
            if dic["ymin"] != None and dic["ymax"] != None:
                pass
            else:
                if dic["v_n_y"] != None:
                    try:
                        lim1, lim2 = Limits.lim(dic["v_n_y"])
                    except:
                        raise NameError("Y limits for {} are not set and not found".format(dic["v_n_y"]))
                    dic["ymin"], dic["ymax"] = lim1, lim2
        return dic


class Constants:

    ns_rho = 1.6191004634e-5
    time_constant = 0.004925794970773136  # to to to ms
    energy_constant = 1787.5521500932314
    volume_constant = 2048


class Printcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def __init__(self):
        pass

    @staticmethod
    def red(text, comma=False):
        if comma:
            print(Printcolor.FAIL + text + Printcolor.ENDC),
        else:
            print(Printcolor.FAIL + text + Printcolor.ENDC)

    @staticmethod
    def yellow(text, comma=False):
        if comma:
            print(Printcolor.WARNING + text + Printcolor.ENDC),
        else:
            print(Printcolor.WARNING + text + Printcolor.ENDC)

    @staticmethod
    def blue(text, comma=False):
        if comma:
            print(Printcolor.OKBLUE + text + Printcolor.ENDC),
        else:
            print(Printcolor.OKBLUE + text + Printcolor.ENDC)

    @staticmethod
    def green(text, comma=False):
        if comma:
            print(Printcolor.OKGREEN + text + Printcolor.ENDC),
        else:
            print(Printcolor.OKGREEN + text + Printcolor.ENDC)

    @staticmethod
    def bold(text, comma=False):
        if comma:
            print(Printcolor.BOLD + text + Printcolor.ENDC),
        else:
            print(Printcolor.BOLD + text + Printcolor.ENDC)

    @staticmethod
    def print_colored_string(parts, colors, comma=False):
        assert len(parts) == len(colors)
        for color in colors:
            assert color in ["", "blue", "red", "yellow", "green"]

        for part, color in zip(parts, colors):
            part = str(part)
            if color == "":
                print(part),
            elif color == "blue":
                Printcolor.blue(part, comma=True)
            elif color == "green":
                Printcolor.green(part, comma=True)
            elif color == "red":
                Printcolor.red(part, comma=True)
            elif color == "yellow":
                Printcolor.yellow(part, comma=True)
            else:
                raise NameError("wrong color: {}".format(color))
        if comma:
            print(''),
        else:
            print('')


class FORMULAS:

    def __init__(self):
        pass

    @staticmethod
    def r(x, y):
        return np.sqrt(x ** 2 + y ** 2)# + z ** 2)

    @staticmethod
    def density(rho, w_lorentz, vol):
        return rho * w_lorentz * vol

    @staticmethod
    def vup(velx, vely, velz):
        return [velx, vely, velz]

    @staticmethod
    def metric(gxx, gxy, gxz, gyy, gyz, gzz):
        return [[gxx, gxy, gxz], [gxy, gyy, gyz], [gxz, gyz, gzz]]

    @staticmethod
    def enthalpy(eps, press, rho):
        return 1 + eps + (press / rho)

    @staticmethod
    def shift(betax, betay, betaz):
        return [betax, betay, betaz]

    @staticmethod
    def shvel(shift, vlow):
        shvel = np.zeros(shift[0].shape)
        for i in range(len(shift)):
            shvel += shift[i] * vlow[i]
        return shvel

    @staticmethod
    def u_0(w_lorentz, shvel, lapse):
        return w_lorentz * (shvel - lapse)

    @staticmethod
    def vlow(metric, vup):
        vlow = [np.zeros_like(vv) for vv in [vup[0], vup[1], vup[2]]]
        for i in range(3):  # for x, y
            for j in range(3):
                vlow[i][:] += metric[i][j][:] * vup[j][:]  # v_i = g_ij * v^j (lowering index) for x y
        return vlow

    @staticmethod
    def vphi(x, y, vlow):
        return -y * vlow[0] + x * vlow[1]

    @staticmethod
    def vr(x, y, r, vup):
        # r = np.sqrt(x ** 2 + y ** 2)
        # print("x: {}".format(x.shape))
        # print("y: {}".format(y.shape))
        # print("r: {}".format(y.shape))
        # print("vup[0]: {}".format(vup[0].shape))

        return (x / r) * vup[0] + (y / r) * vup[1]

    @staticmethod
    def theta(r, z):
        # r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return np.arccos(z/r)

    @staticmethod
    def phi(x, y):
        return np.arctan2(y, x)

    @staticmethod
    def dens_unb_geo(u_0, rho, w_lorentz, vol):

        c_geo = -u_0 - 1.0
        mask_geo = (c_geo > 0).astype(int)  # 1 or 0
        rho_unbnd_geo = rho * mask_geo  # if 1 -> same, if 0 -> masked
        dens_unbnd_geo = rho_unbnd_geo * w_lorentz * vol

        return dens_unbnd_geo

    @staticmethod
    def dens_unb_bern(enthalpy, u_0, rho, w_lorentz, vol):

        c_ber = -enthalpy * u_0 - 1.0
        mask_ber = (c_ber > 0).astype(int)
        rho_unbnd_bernoulli = rho * mask_ber
        density_unbnd_bernoulli = rho_unbnd_bernoulli * w_lorentz * vol

        return density_unbnd_bernoulli

    @staticmethod
    def dens_unb_garch(enthalpy, u_0, lapse, press, rho, w_lorentz, vol):

        c_ber = -enthalpy * u_0 - 1.0
        c_gar = c_ber - (lapse / w_lorentz) * (press / rho)
        mask_gar = (c_gar > 0).astype(int)
        rho_unbnd_garching = rho * mask_gar
        density_unbnd_garching = rho_unbnd_garching * w_lorentz * vol

        return density_unbnd_garching

    @staticmethod
    def ang_mom(rho, eps, press, w_lorentz, vol, vphi):
        return (rho * (1 + eps) + press) * w_lorentz * w_lorentz * vol * vphi

    @staticmethod
    def ang_mom_flux(ang_mom, lapse, vr):
        return ang_mom * lapse * vr

    # data manipulation methods
    @staticmethod
    def get_slice(x3d, y3d, z3d, data3d, slice='xy'):

        if slice == 'yz':
            ix0 = np.argmin(np.abs(x3d[:, 0, 0]))
            if abs(x3d[ix0, 0, 0]) < 1e-15:
                res = data3d[ix0, :, :]
            else:
                if x3d[ix0, 0, 0] > 0:
                    ix0 -= 1
                res = 0.5 * (data3d[ix0, :, :] + data3d[ix0 + 1, :, :])
        elif slice == 'xz':
            iy0 = np.argmin(np.abs(y3d[0, :, 0]))
            if abs(y3d[0, iy0, 0]) < 1e-15:
                res = data3d[:, iy0, :]
            else:
                if y3d[0, iy0, 0] > 0:
                    iy0 -= 1
                res = 0.5 * (data3d[:, iy0, :] + data3d[:, iy0 + 1, :])
        elif slice == 'xy':
            iz0 = np.argmin(np.abs(z3d[0, 0, :]))
            if abs(z3d[0, 0, iz0]) < 1e-15:
                res = data3d[:, :, iz0]
            else:
                if z3d[0, 0, iz0] > 0 and iz0 > 0:
                    iz0 -= 1
                res = 0.5 * (data3d[:, :, iz0] + data3d[:, :, iz0 + 1])
        else:
            raise ValueError("slice:{} not recognized. Use 'xy', 'yz' or 'xz' to get a slice")
        return res


    # --------- OUTFLOWED -----

    @staticmethod
    def vinf(eninf):
        return np.sqrt(2. * eninf)

    @staticmethod
    def vinf_bern(eninf, enthalpy):
        return np.sqrt(2.*(enthalpy*(eninf + 1.) - 1.))

    @staticmethod
    def vel(w_lorentz):
        return np.sqrt(1. - 1. / (w_lorentz**2))

    @staticmethod
    def get_tau(rho, vel, radius, lrho_b):

        rho_b = 10 ** lrho_b
        tau_0 = 0.5 * 2.71828182845904523536 * (radius / vel) * (0.004925794970773136) # in ms
        tau_b = tau_0 * ((rho/rho_b) ** (1.0 / 3.0))
        return tau_b # ms


class Tools:

    @staticmethod
    def combine(x, y, xy, corner_val=None):
        '''creates a 2d array  1st raw    [0, 1:] -- x -- density     (log)
                               1st column [1:, 0] -- y -- lemperature (log)
                               Matrix     [1:,1:] -- xy --Opacity     (log)
           0th element in 1st raw (column) - can be used a corner value

        '''
        x = np.array(x)
        y = np.array(y)
        xy = np.array((xy))

        res = np.insert(xy, 0, x, axis=0)
        new_y = np.insert(y, 0, 0, axis=0)  # inserting a 0 to a first column of a
        res = np.insert(res, 0, new_y, axis=1)

        if corner_val != None:
            res[0, 0] = corner_val

        return res

    @staticmethod
    def combine3d(x, y, z, xyz, corner_val=None):
        '''creates a 2d array  1st raw    [0, 1:] -- x -- density     (log)
                               1st column [1:, 0] -- y -- lemperature (log)
                               Matrix     [1:,1:] -- xy --Opacity     (log)
           0th element in 1st raw (column) - can be used a corner value

        '''

        print(xyz.shape, x.shape, y.shape, z.shape)

        tmp = np.zeros((len(xyz[:, 0, 0])+1, len(xyz[0, :, 0])+1, len(xyz[0, 0, :])+1))
        tmp[1:, 1:, 1:] = xyz
        tmp[1:, 0, 0] = x
        tmp[0, 1:, 0] = y
        tmp[0, 0, 1:] = z
        return tmp

    @staticmethod
    def find_nearest_index(array, value):
        ''' Finds index of the value in the array that is the closest to the provided one '''
        idx = (np.abs(array - value)).argmin()
        return idx

    @staticmethod
    def get_xmin_xmax_ymin_ymax_zmin_zmax(rl):
        if rl == 6:
            xmin, xmax = -14, 14
            ymin, ymax = -14, 14
            zmin, zmax = 0, 14
        elif rl == 5:
            xmin, xmax = -28, 28
            ymin, ymax = -28, 28
            zmin, zmax = 0, 28
        elif rl == 4:
            xmin, xmax = -48, 48
            ymin, ymax = -48, +48
            zmin, zmax = 0, 48
        elif rl == 3:
            xmin, xmax = -88, 88
            ymin, ymax = -88, 88
            zmin, zmax = 0, 88
        elif rl == 2:
            xmin, xmax = -178, 178
            ymin, ymax = -178, +178
            zmin, zmax = 0, 178
        elif rl == 1:
            xmin, xmax = -354, 354
            ymin, ymax = -354, +354
            zmin, zmax = 0, 354
        elif rl == 0:
            xmin, xmax = -1044, 1044
            ymin, ymax = -1044, 1044
            zmin, zmax = 0, 1044
        else:
            # pass
            raise IOError("Set limits for rl:{}".format(rl))

        return xmin, xmax, ymin, ymax, zmin, zmax

    @staticmethod
    def x_y_z_sort(x_arr, y_arr, z_arr=np.empty(0, ), sort_by_012=0):
        '''
        RETURNS x_arr, y_arr, (z_arr) sorted as a matrix by a row, given 'sort_by_012'
        :param x_arr:
        :param y_arr:
        :param z_arr:
        :param sort_by_012:
        :return:
        '''

        # ind = np.lexsort((x_arr, y_arr))
        # tmp = [(x_arr[i],y_arr[i]) for i in ind]
        #
        # print(tmp); exit(1)

        if len(z_arr) == 0 and sort_by_012 < 2:
            if len(x_arr) != len(y_arr):
                raise ValueError('len(x)[{}]!= len(y)[{}]'.format(len(x_arr), len(y_arr)))

            x_y_arr = []
            for i in range(len(x_arr)):
                x_y_arr = np.append(x_y_arr, [x_arr[i], y_arr[i]])
            # print(x_y_arr.shape)

            x_y_sort = np.sort(x_y_arr.view('float64, float64'), order=['f{}'.format(sort_by_012)], axis=0).view(
                np.float)
            x_y_arr_shaped = np.reshape(x_y_sort, (int(len(x_y_sort) / 2), 2))
            # print(x_y_arr_shaped.shape)
            _x_arr = x_y_arr_shaped[:, 0]
            _y_arr = x_y_arr_shaped[:, 1]
            assert len(_x_arr) == len(x_arr)
            assert len(_y_arr) == len(y_arr)

            return _x_arr, _y_arr

        if len(z_arr) > 0 and len(z_arr) == len(y_arr):
            if len(x_arr) != len(y_arr) or len(x_arr) != len(z_arr):
                raise ValueError('len(x)[{}]!= len(y)[{}]!=len(z_arr)[{}]'.format(len(x_arr), len(y_arr), len(z_arr)))

            x_y_z_arr = []
            for i in range(len(x_arr)):
                x_y_z_arr = np.append(x_y_z_arr, [x_arr[i], y_arr[i], z_arr[i]])

            x_y_z_sort = np.sort(x_y_z_arr.view('float64, float64, float64'), order=['f{}'.format(sort_by_012)],
                                 axis=0).view(np.float)
            x_y_z_arr_shaped = np.reshape(x_y_z_sort, (int(len(x_y_z_sort) / 3), 3))
            return x_y_z_arr_shaped[:, 0], x_y_z_arr_shaped[:, 1], x_y_z_arr_shaped[:, 2]

    @staticmethod
    def fit_polynomial(x, y, order, depth, new_x=np.empty(0, ), print_formula=True):
        '''
        RETURNS new_x, f(new_x)
        :param x:
        :param y:
        :param order: 1-4 are supported
        :return:
        '''

        x = np.array(x)
        y = np.array(y)

        f = None
        lbl = None

        if not new_x.any():
            new_x = np.mgrid[x.min():x.max():depth * 1j]

        if order == 1:
            fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)
            lbl = '({}) + ({}*x)'.format(
                "%.4f" % f.coefficients[1],
                "%.4f" % f.coefficients[0]
            )
            # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
            # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')

        if order == 2:
            fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)
            lbl = '({}) + ({}*x) + ({}*x**2)'.format(
                "%.4f" % f.coefficients[2],
                "%.4f" % f.coefficients[1],
                "%.4f" % f.coefficients[0]
            )
            # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
            # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')
        if order == 3:
            fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)
            lbl = '({}) + ({}*x) + ({}*x**2) + ({}*x**3)'.format(
                "%.4f" % f.coefficients[3],
                "%.4f" % f.coefficients[2],
                "%.4f" % f.coefficients[1],
                "%.4f" % f.coefficients[0]
            )
            # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
            # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')
        if order == 4:
            fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)
            lbl = '({}) + ({}*x) + ({}*x**2) + ({}*x**3) + ({}*x**4)'.format(
                "%.4f" % f.coefficients[4],
                "%.4f" % f.coefficients[3],
                "%.4f" % f.coefficients[2],
                "%.4f" % f.coefficients[1],
                "%.4f" % f.coefficients[0]
            )
            # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
            # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')

        if order == 5:
            fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)
            lbl = '({}) + ({}*x) + ({}*x**2) + ({}*x**3) + ({}*x**4) + ({}*x**5)'.format(
                "%.4f" % f.coefficients[5],
                "%.4f" % f.coefficients[4],
                "%.4f" % f.coefficients[3],
                "%.4f" % f.coefficients[2],
                "%.4f" % f.coefficients[1],
                "%.4f" % f.coefficients[0]
            )
            # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
            # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')

        if order == 6:
            fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)
            lbl = '({}) + ({}*x) + ({}*x**2) + ({}*x**3) + ({}*x**4) + ({}*x**5) + ({}*x**6)'.format(
                "%.4f" % f.coefficients[6],
                "%.4f" % f.coefficients[5],
                "%.4f" % f.coefficients[4],
                "%.4f" % f.coefficients[3],
                "%.4f" % f.coefficients[2],
                "%.4f" % f.coefficients[1],
                "%.4f" % f.coefficients[0]
            )
            # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
            # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')

        if not order in [1, 2, 3, 4, 5, 6]:
            fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)
            # raise ValueError('Supported orders: 1,2,3,4 only')

        if print_formula:
            print(lbl)

        return new_x, f(new_x)

    @staticmethod
    def cart2pol(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return (rho, phi)

    @staticmethod
    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)


class PHYSICS:

    def __init__(self):
        pass

    @staticmethod
    def get_dens_decomp_2d(dens_2d, phi_2d, dphi_2d, dr_2d, m=1):
        '''
        Uses a 2d slice at z=0
        Returns complex arrays [\int(d\phi)] and [\int(dr d\phi)]
        '''
        # dens_2d = dens_3d[:, :, 0]
        # phi_2d  = phi_3d[:, :, 0]
        # dr_2d   = dr_3d[:, :, 0]
        # dphi_2d = dphi_3d[:, :, 0]

        integ_over_phi = np.sum(dens_2d * np.exp(1j * m * phi_2d) * dphi_2d, axis=1)

        integ_over_phi_r = np.sum(integ_over_phi * dr_2d[:, 0])

        return integ_over_phi, integ_over_phi_r

    @staticmethod
    def get_dens_decomp_3d(dens_3d, r, phi_3d, dphi_3d, dr_3d, dz_3d, m=1):
        '''
        Integrates density over 'z'
        Returns complex arrays [\int(d\phi)] and [\int(dr d\phi)]
        '''

        integ_dens_z = np.sum(dens_3d * dz_3d[:, :, :], axis=2) # -> 2d array

        integ_over_z_phi = np.sum(integ_dens_z * np.exp(1j * m * phi_3d[:, :, 0]) * dphi_3d[:, :, 0], axis=1) # -> 1d array

        integ_over_z_phi_r = np.sum(integ_over_z_phi * dr_3d[:, 0, 0] * r[:, 0, 0]) # -> number

        return integ_over_z_phi, integ_over_z_phi_r

    @staticmethod
    def get_retarded_time(t, M_Inf=2.7, R_GW=400.0):
        R = R_GW * (1 + M_Inf / (2 * R_GW)) ** 2
        rstar = R + 2 * M_Inf * log(R / (2 * M_Inf) - 1)
        return t - rstar

